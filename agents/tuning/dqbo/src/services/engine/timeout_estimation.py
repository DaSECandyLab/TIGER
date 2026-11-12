"""Workload-corrected timeout feedback, following workflow section 5.4.

Use only finished-request histograms, never the engine's incremental token
counters or TTFT histogram (which also contains unfinished requests).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .flock.trial.metrics import parse_prometheus

WORKLOAD_HISTOGRAMS = {
    "input_tokens": "vllm:request_prompt_tokens",
    "output_tokens": "vllm:request_generation_tokens",
    "prefill_time_s": "vllm:request_prefill_time_seconds",
    "decode_time_s": "vllm:request_decode_time_seconds",
}
METHOD = "workload_corrected_timeout_v1"


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)


def _series(samples, name):
    rows = {}
    for sample in samples:
        if sample.name == name:
            key = tuple(sorted(sample.labels.items()))
            if key in rows:
                raise ValueError(f"Duplicate metric series: {name}")
            rows[key] = _number(sample.value, name)
    return rows


def _delta(before, after, name):
    start, end = _series(before, name), _series(after, name)
    if not end or not start.keys() <= end.keys():
        raise ValueError(f"Missing/disappeared metric series: {name}")
    result = {key: value - start.get(key, 0.0) for key, value in end.items()}
    if any(value < 0 or not math.isfinite(value) for value in result.values()):
        raise ValueError(f"Counter reset or invalid delta: {name}")
    return result


def finished_request_workload(before_text: str, after_text: str) -> dict[str, Any]:
    """All four sums must describe the same successful cohort, per engine.

    vLLM 0.20 updates these four histograms in its finished-request loop;
    prefill_time is first-token minus scheduled time, decode_time is last-token
    minus first-token. Any abort/error, reset or non-atomic scrape mismatch
    makes the estimate unavailable rather than contaminating the cohort.
    """
    try:
        before, after = parse_prometheus(before_text), parse_prometheus(after_text)
        count_name = next((name for name in ("vllm:request_success_total", "vllm:request_success")
                           if any(s.name == name for s in after)), None)
        if count_name is None:
            raise ValueError("Missing finished-request counters")
        successes = {}
        for labels, value in _delta(before, after, count_name).items():
            reason = dict(labels).get("finished_reason")
            if not value.is_integer():
                raise ValueError("Non-integer request count")
            if value and reason not in {"stop", "length"}:
                raise ValueError("Failed, aborted or unknown finished requests in snapshot")
            key = tuple((k, v) for k, v in labels if k != "finished_reason")
            successes[key] = successes.get(key, 0.0) + value
        count = sum(successes.values())
        _number(count, "completed_requests", positive=True)
        totals = {}
        for field, metric in WORKLOAD_HISTOGRAMS.items():
            counts = _delta(before, after, metric + "_count")
            sums = _delta(before, after, metric + "_sum")
            if counts.keys() != sums.keys():
                raise ValueError(f"Histogram label mismatch: {metric}")
            keys = counts.keys() | successes.keys()
            if any(counts.get(k, 0) != successes.get(k, 0) for k in keys):
                raise ValueError(f"Finished-request cohort mismatch: {metric}")
            totals[field] = sum(sums.values())
        for field in ("input_tokens", "output_tokens"):
            if not totals[field].is_integer():
                raise ValueError(f"Non-integer {field}")
        _number(totals["input_tokens"], "input_tokens", positive=True)
        if totals["output_tokens"] < count:
            raise ValueError("Finished requests have fewer than one output token each")
        return {"status": "measured", "completed_requests": int(count), **totals,
                "failed_requests": 0, "scope": "same_finished_successful_requests",
                "prefill_scope": "first_token_minus_first_scheduled_excludes_queue",
                "decode_scope": "last_token_minus_first_token",
                "metric_sources": WORKLOAD_HISTOGRAMS}
    except (TypeError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _union(intervals):
    total, right = 0.0, -math.inf
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


def baseline_timeout_reference(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a frozen reference from a complete result and its raw snapshots.

    Older results may be reused if their query intervals and finished-request
    metrics are present. Missing evidence is never replaced by incremental
    prompt/generation counters or total process wall time.
    """
    try:
        saved = result.get("query_timeout_reference")
        if isinstance(saved, Mapping) and saved.get("status") == "measured":
            reference = dict(saved)
        else:
            if result.get("sembench_returncode", 0) != 0:
                raise ValueError("Baseline workload did not complete successfully")
            workload = result.get("flock_workload_metrics") or {}
            queries = [q for t in workload.get("terminals", []) for q in t.get("queries", [])]
            if not queries or any(q.get("status") != "success" for q in queries):
                raise ValueError("Baseline lacks complete successful SQL intervals")
            intervals = []
            for q in queries:
                # New workers store the exact execute_query interval. Old ones
                # expose their enclosing SQL execution window with provenance.
                window = q.get("sql_execution_interval") or {"start": q.get("started_at"), "end": q.get("completed_at")}
                start, end = _number(window.get("start"), "query_start"), _number(window.get("end"), "query_end")
                if end < start:
                    raise ValueError("Invalid baseline query interval")
                intervals.append((start, end))
            if len({bool(q.get("sql_execution_interval")) for q in queries}) != 1:
                raise ValueError("Baseline mixes SQL interval clock domains")
            logs = result.get("logs") or {}
            stats = finished_request_workload(Path(logs["metrics_before"]).read_text(),
                                              Path(logs["metrics_after"]).read_text())
            if stats["status"] != "measured":
                raise ValueError("Baseline request workload unavailable: " + stats["reason"])
            if result.get("successful_query_count") != len(queries):
                raise ValueError("Baseline SQL count does not match complete intervals")
            if result.get("request_success_total_delta") != stats["completed_requests"]:
                raise ValueError("Baseline request count does not match finished cohort")
            reference = {
                "status": "measured", "query_execution_s": _union(intervals),
                "query_span_s": max(end for _, end in intervals) - min(start for start, _ in intervals),
                "request_count": stats["completed_requests"], "sql_count": len(queries),
                "mean_input_tokens": stats["input_tokens"] / stats["completed_requests"],
                "mean_output_tokens": stats["output_tokens"] / stats["completed_requests"],
                "mean_query_latency_ms": result.get("mean_query_execution_latency_ms"),
                "p95_query_latency_ms": result.get("p95_query_execution_latency_ms"),
                "interval_source": ("exact_execute_query_intervals" if all(q.get("sql_execution_interval") for q in queries)
                                    else "recorded_sql_execution_windows"),
                "request_workload": stats,
            }
        for field in ("query_execution_s", "query_span_s", "request_count", "sql_count",
                      "mean_input_tokens", "mean_output_tokens", "mean_query_latency_ms", "p95_query_latency_ms"):
            _number(reference.get(field), "baseline " + field, positive=True)
        if reference["query_execution_s"] > reference["query_span_s"] + 1e-6:
            raise ValueError("Baseline execution union exceeds its span")
        if reference["mean_output_tokens"] < 1:
            raise ValueError("Baseline output length is below one token")
        return reference
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def extrapolate_timeout(reference, cohort, *, elapsed_s, lower_bound_s):
    unavailable = {"status": "unavailable", "method": METHOD, "is_measurement": False}
    try:
        if reference.get("status") != "measured" or cohort.get("status") != "measured":
            raise ValueError("Need measured baseline and finished-request workload")
        for field in ("query_execution_s", "query_span_s", "request_count", "sql_count",
                      "mean_input_tokens", "mean_output_tokens", "mean_query_latency_ms", "p95_query_latency_ms"):
            _number(reference.get(field), "baseline " + field, positive=True)
        t = _number(elapsed_s, "snapshot span", positive=True)
        lower = _number(lower_bound_s, "query execution lower bound", positive=True)
        m = _number(cohort.get("completed_requests"), "completed requests", positive=True)
        n_in = _number(cohort.get("input_tokens"), "input tokens", positive=True)
        n_out = _number(cohort.get("output_tokens"), "output tokens", positive=True)
        prefill = _number(cohort.get("prefill_time_s"), "prefill time")
        decode = _number(cohort.get("decode_time_s"), "decode time")
        if cohort.get("failed_requests") != 0:
            raise ValueError("Request failures prevent successful workload projection")
        if n_out <= m:
            raise ValueError("No non-first output tokens to estimate beta")
        if m > reference["request_count"]:
            raise ValueError("Completed requests exceed frozen baseline workload")
        if lower > t + 1e-6:
            raise ValueError("Query execution lower bound exceeds snapshot span")
        alpha, beta = prefill / n_in, decode / (n_out - m)
        w_done = alpha * n_in / m + beta * max(n_out / m - 1, 0)
        w_all = alpha * reference["mean_input_tokens"] + beta * max(reference["mean_output_tokens"] - 1, 0)
        _number(w_done, "completed workload", positive=True)
        _number(w_all, "baseline workload under candidate weights", positive=True)
        raw = t * reference["request_count"] / m * w_all / w_done * reference["query_execution_s"] / reference["query_span_s"]
        total = max(raw, lower)
        k = total / reference["query_execution_s"]
        metrics = {"query_throughput_qps": reference["sql_count"] / total,
                   "mean_query_execution_latency_ms": k * reference["mean_query_latency_ms"],
                   "p95_query_execution_latency_ms": k * reference["p95_query_latency_ms"]}
        for field, value in {"estimated_query_execution_s": total, **metrics}.items():
            _number(value, field, positive=True)
        return {"status": "estimated", "method": METHOD, "is_measurement": False,
                "alpha_seconds_per_input_token": alpha, "beta_seconds_per_nonfirst_output_token": beta,
                "w_done": w_done, "w_all": w_all, "snapshot_span_s": t,
                "query_execution_lower_bound_s": lower, "unclamped_query_execution_s": raw,
                "estimated_query_execution_s": total, "slowdown_vs_baseline": k, "metrics": metrics,
                "assumptions": ["stable_remaining_weighted_workload_rate", "latency_distribution_scaled_by_k"],
                "baseline_reference": dict(reference), "finished_request_workload": dict(cohort)}
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
        return {**unavailable, "reason": str(exc)}


def timeout_record_fields(censoring):
    """Keep estimates separate from measured targets and measured quality."""
    if not isinstance(censoring, Mapping):
        return {}
    estimate = censoring.get("performance_estimate") or {}
    fields = {"performance_estimate": dict(estimate)}
    if estimate.get("status") != "estimated" or estimate.get("method") != METHOD:
        return fields
    checked = extrapolate_timeout(
        estimate.get("baseline_reference") or {}, estimate.get("finished_request_workload") or {},
        elapsed_s=estimate.get("snapshot_span_s"), lower_bound_s=estimate.get("query_execution_lower_bound_s"),
    )
    if checked.get("status") != "estimated" or checked.get("metrics") != estimate.get("metrics"):
        return fields
    try:
        metrics = estimate["metrics"]
        vector = [_number(metrics["query_throughput_qps"], "qps", positive=True),
                  -_number(metrics["mean_query_execution_latency_ms"], "mean latency", positive=True),
                  -_number(metrics["p95_query_execution_latency_ms"], "p95 latency", positive=True)]
    except (TypeError, ValueError, KeyError):
        return fields
    return {**fields, "estimated_objective_vector": vector, "performance_source": "timeout_extrapolation",
            "used_for_performance_surrogates": True, "used_for_hidden_constraint": False}
