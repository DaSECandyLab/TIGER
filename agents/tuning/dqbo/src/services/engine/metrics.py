"""Trial 完整性、三性能目标与质量指标提取。"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

OBJECTIVE_NAMES = ("qps", "negative_mean_latency", "negative_p95_latency")
QUALITY_GATE_METRIC = "sembench_summary.quality_gate_passed"
AGGREGATE_QUALITY_THRESHOLD_SOURCE = "explicit_aggregate_quality_gate"
RELATIVE_QUALITY_THRESHOLD_SOURCE = "single_baseline_multiplicative_retention"
RELATIVE_QUALITY_THRESHOLD_FORMULAS = {
    "relative_higher_is_better": "baseline_query_metric * retention_factor",
    "absolute": "fixed absolute threshold",
}
AGGREGATION_REFERENCE_SCALE_METRIC = "aggregation_reference_scale"
QUALITY_THRESHOLD_SOURCE = "single_baseline_additive_non_inferiority"
QUALITY_THRESHOLD_FORMULAS = {
    "bounded_higher_is_better": "max(metric_min, baseline - alpha)",
    "spearman": "max(-1, baseline - alpha)",
    "relative_error": "baseline + alpha",
}
HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "precision",
        "recall",
        "spearman_correlation",
        "ranking_item_recall",
        "accuracy",
        "within1_accuracy",
    }
)
RETRIEVAL_METRICS = frozenset({"precision", "recall", "f1_score"})
AGGREGATION_METRICS = frozenset(
    {"absolute_error", "relative_error", "mean_absolute_percentage_error"}
)
RANKING_METRICS = frozenset(
    {
        "spearman_correlation",
        "kendall_tau",
        "ranking_item_precision",
        "ranking_item_recall",
        "ranking_item_f1_score",
    }
)


def nested_metric(payload: Mapping[str, Any], path: str) -> Any:
    """读取点分隔路径；单段质量名会回退到 sembench_summary。"""

    parts = [part for part in str(path).split(".") if part]
    current: Any = payload
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            current = None
            break
        current = current[part]
    if current is None and len(parts) == 1:
        summary = payload.get("sembench_summary")
        if isinstance(summary, Mapping):
            current = summary.get(parts[0])
    return current


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _quality_metric_value(value: Any, *, metric_name: str, name: str) -> float:
    """读取代表指标，并拒绝 evaluator 合同之外的数值。"""
    numeric = _finite_number(value, name=name)
    if metric_name == "spearman_correlation":
        valid = -1.0 <= numeric <= 1.0
        expected = "in [-1, 1]"
    elif metric_name in HIGHER_IS_BETTER_METRICS:
        valid = 0.0 <= numeric <= 1.0
        expected = "in [0, 1]"
    else:
        valid = numeric >= 0.0
        expected = "non-negative"
    if not valid:
        raise ValueError(f"{name} must be {expected}")
    return numeric


def _representative_metric_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    metric_names = set(row)
    if {"accuracy", "classification_sample_count"}.issubset(metric_names):
        return ("accuracy",)
    if metric_names & RETRIEVAL_METRICS:
        return ("precision", "recall")
    if metric_names & AGGREGATION_METRICS:
        # Aggregation values have query-specific units and scales.  Keep
        # absolute_error in the evaluator output for diagnosis, but use only
        # the scale-free relative error as the quality gate.
        return ("relative_error",)
    if metric_names & RANKING_METRICS:
        return ("spearman_correlation", "ranking_item_recall")
    if "accuracy" in metric_names:
        return ("accuracy",)
    return ()


def _query_id(metric_key: object, row: Mapping[str, Any]) -> str:
    raw_query_id = row.get("query_id")
    if isinstance(raw_query_id, int) and not isinstance(raw_query_id, bool):
        return f"Q{raw_query_id}"
    match = re.search(r"Q(\d+)", str(metric_key), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot determine query id from quality row {metric_key}")
    return f"Q{int(match.group(1))}"


def query_representative_metric_means(
    result: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """按 query_id 汇总 baseline/candidate 的代表指标均值。"""
    raw_metrics = result.get("sembench_metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise ValueError("trial result lacks per-query sembench_metrics")

    grouped: dict[str, dict[str, list[float]]] = {}
    for metric_key, raw_row in raw_metrics.items():
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"quality row {metric_key} must be an object")
        status = raw_row.get("status", raw_row.get("evaluation_status"))
        if status is not None and status != "success":
            raise ValueError(f"quality row {metric_key} has status={status}")
        nested_metrics = raw_row.get("metrics")
        row = nested_metrics if isinstance(nested_metrics, Mapping) else raw_row
        representative_names = _representative_metric_names(row)
        if not representative_names:
            raise ValueError(f"quality row {metric_key} lacks representative metrics")
        query_id = _query_id(metric_key, raw_row)
        query_group = grouped.setdefault(query_id, {})
        for metric_name in representative_names:
            value = _quality_metric_value(
                row.get(metric_name),
                metric_name=metric_name,
                name=f"{query_id}.{metric_name}",
            )
            query_group.setdefault(metric_name, []).append(value)

    return {
        query_id: {
            metric_name: sum(values) / len(values)
            for metric_name, values in sorted(metric_groups.items())
        }
        for query_id, metric_groups in sorted(grouped.items())
    }


def query_configured_metric_means(
    result: Mapping[str, Any],
    configured_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Average exactly the per-query metrics named by a frozen gate.

    This intentionally differs from ``query_representative_metric_means``:
    Movie Q1-Q8 expose logical-sample accuracy in addition to their legacy
    retrieval/aggregation diagnostics, and the 0.95/0.8 experiment gates that
    accuracy directly.
    """

    raw_metrics = result.get("sembench_metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise ValueError("trial result lacks per-query sembench_metrics")
    requested = {
        str(query_id).upper(): {str(name) for name in metrics}
        for query_id, metrics in configured_metrics.items()
    }
    grouped: dict[str, dict[str, list[float]]] = {}
    for metric_key, raw_row in raw_metrics.items():
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"quality row {metric_key} must be an object")
        status = raw_row.get("status", raw_row.get("evaluation_status"))
        if status is not None and status != "success":
            raise ValueError(f"quality row {metric_key} has status={status}")
        query_id = _query_id(metric_key, raw_row)
        if query_id not in requested:
            continue
        nested_metrics = raw_row.get("metrics")
        row = nested_metrics if isinstance(nested_metrics, Mapping) else raw_row
        query_group = grouped.setdefault(query_id, {})
        for metric_name in requested[query_id]:
            if metric_name not in row:
                continue
            value = _quality_metric_value(
                row.get(metric_name),
                metric_name=metric_name,
                name=f"{query_id}.{metric_name}",
            )
            query_group.setdefault(metric_name, []).append(value)
    return {
        query_id: {
            metric_name: sum(values) / len(values)
            for metric_name, values in sorted(metric_groups.items())
        }
        for query_id, metric_groups in sorted(grouped.items())
    }


def derive_relative_quality_thresholds_from_baseline(
    baseline_result: Mapping[str, Any],
    relative_factors: Mapping[str, Mapping[str, Any]],
    absolute_thresholds: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Freeze multiplicative per-SQL retention gates from one baseline.

    Relative factors are supported only for higher-is-better metrics.  The
    resulting payload is the ordinary absolute threshold JSON consumed by the
    workload, so no baseline data is consulted after context preparation.
    """

    if not isinstance(relative_factors, Mapping) or not relative_factors:
        raise ValueError("relative quality threshold factors must be non-empty")
    normalized_factors: dict[str, dict[str, float]] = {}
    for raw_query_id, raw_metrics in relative_factors.items():
        query_id = str(raw_query_id).strip().upper()
        if not re.fullmatch(r"Q\d+", query_id):
            raise ValueError(f"invalid relative quality query id: {raw_query_id}")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValueError(f"{query_id} relative quality factors must be non-empty")
        normalized_factors[query_id] = {}
        for raw_metric_name, raw_factor in raw_metrics.items():
            metric_name = str(raw_metric_name)
            if metric_name not in HIGHER_IS_BETTER_METRICS:
                raise ValueError(
                    f"relative retention requires higher-is-better metric: "
                    f"{query_id}.{metric_name}"
                )
            factor = _finite_number(
                raw_factor, name=f"{query_id}.{metric_name} retention factor"
            )
            if not 0.0 <= factor <= 1.0:
                raise ValueError(
                    f"{query_id}.{metric_name} retention factor must be in [0, 1]"
                )
            normalized_factors[query_id][metric_name] = factor

    baseline_metrics = query_configured_metric_means(
        baseline_result, normalized_factors
    )
    if set(baseline_metrics) != set(normalized_factors):
        missing = sorted(set(normalized_factors) - set(baseline_metrics))
        raise ValueError(f"baseline lacks configured quality queries: {missing}")
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for query_id, metric_factors in normalized_factors.items():
        observed = baseline_metrics[query_id]
        if set(observed) != set(metric_factors):
            missing = sorted(set(metric_factors) - set(observed))
            raise ValueError(
                f"baseline lacks configured metrics for {query_id}: {missing}"
            )
        thresholds[query_id] = {
            metric_name: {"min": observed[metric_name] * factor}
            for metric_name, factor in metric_factors.items()
        }

    for raw_query_id, raw_metrics in (absolute_thresholds or {}).items():
        query_id = str(raw_query_id).strip().upper()
        if not re.fullmatch(r"Q\d+", query_id):
            raise ValueError(f"invalid absolute quality query id: {raw_query_id}")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValueError(f"{query_id} absolute thresholds must be non-empty")
        query_thresholds = thresholds.setdefault(query_id, {})
        for raw_metric_name, raw_rule in raw_metrics.items():
            metric_name = str(raw_metric_name)
            if metric_name in query_thresholds:
                raise ValueError(
                    f"duplicate relative/absolute threshold: {query_id}.{metric_name}"
                )
            if not isinstance(raw_rule, Mapping) or set(raw_rule) not in (
                {"min"},
                {"max"},
            ):
                raise ValueError(
                    f"{query_id}.{metric_name} must contain exactly one of min/max"
                )
            operator = next(iter(raw_rule))
            expected = "min" if metric_name in HIGHER_IS_BETTER_METRICS else "max"
            if operator != expected:
                raise ValueError(f"{query_id}.{metric_name} must use {expected}")
            threshold = _finite_number(
                raw_rule[operator], name=f"{query_id}.{metric_name}.{operator}"
            )
            query_thresholds[metric_name] = {operator: threshold}

    # Validate that the baseline itself satisfies every frozen absolute rule.
    baseline_all = query_configured_metric_means(baseline_result, thresholds)
    if set(baseline_all) != set(thresholds):
        missing = sorted(set(thresholds) - set(baseline_all))
        raise ValueError(f"baseline lacks configured quality queries: {missing}")
    for query_id, query_thresholds in thresholds.items():
        observed = baseline_all.get(query_id, {})
        if set(observed) != set(query_thresholds):
            missing = sorted(set(query_thresholds) - set(observed))
            raise ValueError(
                f"baseline lacks configured metrics for {query_id}: {missing}"
            )
        for metric_name, rule in query_thresholds.items():
            value = observed[metric_name]
            if "min" in rule and value < rule["min"]:
                raise ValueError(f"baseline violates {query_id}.{metric_name} minimum")
            if "max" in rule and abs(value) > rule["max"]:
                raise ValueError(f"baseline violates {query_id}.{metric_name} maximum")
    return thresholds


def query_aggregation_reference_scale_means(
    result: Mapping[str, Any],
) -> dict[str, float]:
    """读取 aggregation query 的单次 baseline ground-truth 尺度。"""
    raw_metrics = result.get("sembench_metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise ValueError("trial result lacks per-query sembench_metrics")

    grouped: dict[str, list[float]] = {}
    for metric_key, raw_row in raw_metrics.items():
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"quality row {metric_key} must be an object")
        nested_metrics = raw_row.get("metrics")
        row = nested_metrics if isinstance(nested_metrics, Mapping) else raw_row
        if not (set(row) & AGGREGATION_METRICS):
            continue
        query_id = _query_id(metric_key, raw_row)
        if AGGREGATION_REFERENCE_SCALE_METRIC not in row:
            raise ValueError(
                f"{query_id} lacks {AGGREGATION_REFERENCE_SCALE_METRIC}; "
                "run one baseline with the current evaluator"
            )
        scale = _finite_number(
            row.get(AGGREGATION_REFERENCE_SCALE_METRIC),
            name=f"{query_id}.{AGGREGATION_REFERENCE_SCALE_METRIC}",
        )
        if scale <= 0.0:
            raise ValueError(
                f"{query_id}.{AGGREGATION_REFERENCE_SCALE_METRIC} must be positive"
            )
        grouped.setdefault(query_id, []).append(scale)

    return {
        query_id: sum(values) / len(values)
        for query_id, values in sorted(grouped.items())
    }


def derive_quality_thresholds_from_baseline(
    baseline_result: Mapping[str, Any],
    alpha: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """由一次 baseline 和统一的 additive alpha 冻结 candidate 门禁。"""
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("quality alpha must satisfy 0 <= alpha < 1")
    baseline_metrics = query_representative_metric_means(baseline_result)
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for query_id, metrics in baseline_metrics.items():
        query_thresholds: dict[str, dict[str, float]] = {}
        for metric_name, baseline_value in metrics.items():
            if metric_name == "spearman_correlation":
                query_thresholds[metric_name] = {
                    "min": max(-1.0, baseline_value - float(alpha))
                }
            elif metric_name in HIGHER_IS_BETTER_METRICS:
                query_thresholds[metric_name] = {
                    "min": max(0.0, baseline_value - float(alpha))
                }
            elif metric_name == "relative_error":
                query_thresholds[metric_name] = {"max": baseline_value + float(alpha)}
            else:
                raise ValueError(f"unsupported representative metric: {metric_name}")
        thresholds[query_id] = query_thresholds
    return thresholds


def _passes_quality_thresholds(
    result: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> bool:
    candidate_metrics = query_configured_metric_means(result, thresholds)
    if set(candidate_metrics) != set(thresholds):
        return False
    for query_id, query_thresholds in thresholds.items():
        observed = candidate_metrics.get(query_id, {})
        if set(observed) != set(query_thresholds):
            return False
        for metric_name, rule in query_thresholds.items():
            if set(rule) == {"min"}:
                if observed[metric_name] < float(rule["min"]):
                    return False
            elif set(rule) == {"max"}:
                if abs(observed[metric_name]) > float(rule["max"]):
                    return False
            else:
                raise ValueError(f"invalid quality rule for {query_id}.{metric_name}")
    return True


def aggregate_quality_metrics(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Read legacy aggregate gates, deriving them from immutable query rows if needed."""

    summary = result.get("sembench_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    raw_mean_f1 = summary.get("mean_f1_score")
    raw_max_relative_error = summary.get("max_relative_error")
    raw_metrics = result.get("sembench_metrics")
    rows = list(raw_metrics.values()) if isinstance(raw_metrics, Mapping) else []
    if raw_mean_f1 is None:
        f1_scores = [
            _finite_number(row.get("f1_score"), name="f1_score")
            for row in rows
            if isinstance(row, Mapping) and row.get("f1_score") is not None
        ]
        raw_mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
    if raw_max_relative_error is None:
        relative_errors = [
            abs(_finite_number(row.get("relative_error"), name="relative_error"))
            for row in rows
            if isinstance(row, Mapping) and row.get("relative_error") is not None
        ]
        raw_max_relative_error = max(relative_errors) if relative_errors else None
    return {
        "mean_f1_score": (
            _finite_number(raw_mean_f1, name="mean_f1_score")
            if raw_mean_f1 is not None
            else None
        ),
        "max_relative_error": (
            abs(_finite_number(raw_max_relative_error, name="max_relative_error"))
            if raw_max_relative_error is not None
            else None
        ),
    }


def quality_margin_diagnostics(
    result: Mapping[str, Any],
    *,
    quality_metric: str,
    quality_thresholds_json: str | None = None,
    min_mean_f1: float | None = None,
    max_relative_error: float | None = None,
    tau_q: float | None = None,
    quality: float | None = None,
) -> dict[str, Any]:
    """Describe measured distances from the configured hard quality gates.

    Every returned margin uses the same convention: non-negative passes and a
    negative value fails.  These values are diagnostics for case presentation
    and post-run auditing only; RepairBO never fits or queries a quality
    surrogate from them.  Hard feasibility is still evaluated metric by metric
    against the original thresholds.
    """

    margins: list[dict[str, Any]] = []

    def add(
        *,
        name: str,
        value: float,
        operator: str,
        threshold: float,
    ) -> None:
        scale = abs(float(threshold))
        if scale <= 1e-12:
            scale = 1.0
        if operator == "min":
            raw_margin = float(value) - float(threshold)
        elif operator == "max":
            raw_margin = float(threshold) - abs(float(value))
        else:  # pragma: no cover - all callers below use a closed enum.
            raise ValueError(f"unsupported quality margin operator: {operator}")
        normalized_margin = raw_margin / scale
        margins.append(
            {
                "metric": name,
                "value": float(value),
                "operator": operator,
                "threshold": float(threshold),
                "raw_margin": float(raw_margin),
                "normalized_margin": float(normalized_margin),
                "passed": bool(raw_margin >= 0.0),
            }
        )

    if quality_metric == QUALITY_GATE_METRIC and quality_thresholds_json is not None:
        thresholds = json.loads(quality_thresholds_json)
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise ValueError("quality thresholds must be a non-empty object")
        observed_by_query = query_configured_metric_means(result, thresholds)
        for raw_query_id, query_thresholds in sorted(thresholds.items()):
            query_id = str(raw_query_id).upper()
            observed = observed_by_query.get(query_id, {})
            if not isinstance(query_thresholds, Mapping):
                raise TypeError(f"quality thresholds for {query_id} must be an object")
            for metric_name, rule in sorted(query_thresholds.items()):
                if not isinstance(rule, Mapping) or set(rule) not in ({"min"}, {"max"}):
                    raise ValueError(
                        f"invalid quality rule for {query_id}.{metric_name}"
                    )
                if metric_name not in observed:
                    raise ValueError(
                        f"trial lacks configured quality metric {query_id}.{metric_name}"
                    )
                operator = next(iter(rule))
                add(
                    name=f"{query_id}.{metric_name}",
                    value=float(observed[metric_name]),
                    operator=operator,
                    threshold=float(rule[operator]),
                )
    elif quality_metric == QUALITY_GATE_METRIC and (
        min_mean_f1 is not None or max_relative_error is not None
    ):
        aggregate = aggregate_quality_metrics(result)
        if min_mean_f1 is not None:
            value = aggregate["mean_f1_score"]
            if value is None:
                raise ValueError("trial lacks F1 measurements required by min_mean_f1")
            add(
                name="mean_f1_score",
                value=float(value),
                operator="min",
                threshold=float(min_mean_f1),
            )
        if max_relative_error is not None:
            value = aggregate["max_relative_error"]
            if value is None:
                raise ValueError(
                    "trial lacks aggregation measurements required by max_relative_error"
                )
            add(
                name="max_relative_error",
                value=float(value),
                operator="max",
                threshold=float(max_relative_error),
            )
    else:
        resolved_quality = (
            float(quality)
            if quality is not None
            else _finite_number(
                nested_metric(result, quality_metric),
                name=f"quality metric {quality_metric}",
            )
        )
        threshold = float(tau_q if tau_q is not None else 0.5)
        # An unconfigured boolean gate has no meaningful distance to its
        # boundary.  Preserve direction and expose the limitation explicitly.
        if quality_metric == QUALITY_GATE_METRIC:
            passed = bool(resolved_quality >= threshold)
            margins.append(
                {
                    "metric": quality_metric,
                    "value": resolved_quality,
                    "operator": "boolean",
                    "threshold": threshold,
                    "raw_margin": None,
                    "normalized_margin": 1.0 if passed else -1.0,
                    "passed": passed,
                }
            )
        else:
            add(
                name=quality_metric,
                value=resolved_quality,
                operator="min",
                threshold=threshold,
            )

    if not margins:
        raise ValueError("quality margin diagnostics produced no constraints")
    worst = min(float(item["normalized_margin"]) for item in margins)
    return {
        "quality_margins": margins,
        "worst_quality_margin": worst,
        "quality_feasible": all(bool(item["passed"]) for item in margins),
    }


def extract_complete_trial_metrics(
    result: Mapping[str, Any],
    *,
    quality_metric: str,
    quality_thresholds_json: str | None = None,
    min_mean_f1: float | None = None,
    max_relative_error: float | None = None,
) -> tuple[list[float], float, dict[str, float]]:
    """完整 Trial 转换为用户定义的最大化目标向量和质量标量。"""

    summary = result.get("sembench_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    failed_queries = int(summary.get("failed_query_count") or 0)
    if failed_queries > 0:
        raise ValueError(f"trial has failed_query_count={failed_queries}")
    return_code = result.get("sembench_returncode", 0)
    if return_code not in {None, 0}:
        raise ValueError(f"trial has sembench_returncode={return_code}")
    qps = _finite_number(result.get("query_throughput_qps"), name="QPS")
    mean_latency = _finite_number(
        result.get("mean_query_execution_latency_ms"), name="MeanLatency"
    )
    p95_latency = _finite_number(
        result.get("p95_query_execution_latency_ms"), name="P95Latency"
    )
    raw_quality = nested_metric(result, quality_metric)
    if quality_metric == QUALITY_GATE_METRIC:
        if min_mean_f1 is not None or max_relative_error is not None:
            aggregate = aggregate_quality_metrics(result)
            if min_mean_f1 is not None and aggregate["mean_f1_score"] is None:
                raise ValueError("trial lacks F1 measurements required by min_mean_f1")
            if (
                max_relative_error is not None
                and aggregate["max_relative_error"] is None
            ):
                raise ValueError(
                    "trial lacks aggregation measurements required by max_relative_error"
                )
            quality = float(
                (
                    min_mean_f1 is None
                    or float(aggregate["mean_f1_score"]) >= float(min_mean_f1)
                )
                and (
                    max_relative_error is None
                    or float(aggregate["max_relative_error"])
                    <= float(max_relative_error)
                )
            )
        elif quality_thresholds_json is not None:
            thresholds = json.loads(quality_thresholds_json)
            if not isinstance(thresholds, Mapping) or not thresholds:
                raise ValueError("quality thresholds must be a non-empty object")
            independently_passed = _passes_quality_thresholds(result, thresholds)
            if (
                isinstance(raw_quality, bool)
                and raw_quality is not independently_passed
            ):
                raise ValueError(
                    "reported quality_gate_passed conflicts with per-query metrics"
                )
            quality = float(independently_passed)
        elif isinstance(raw_quality, bool):
            quality = float(raw_quality)
        else:
            raise TypeError(
                f"quality gate {quality_metric} must be a configured boolean"
            )
    else:
        quality = _finite_number(raw_quality, name=f"quality metric {quality_metric}")
    objectives = [qps, -mean_latency, -p95_latency]
    return (
        objectives,
        quality,
        {
            "query_throughput_qps": qps,
            "mean_query_execution_latency_ms": mean_latency,
            "p95_query_execution_latency_ms": p95_latency,
        },
    )


def quality_threshold(baseline_quality: float, alpha: float) -> float:
    baseline = _finite_number(baseline_quality, name="baseline quality")
    if baseline <= 0:
        raise ValueError(
            "baseline quality must be positive for a relative quality gate"
        )
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("quality alpha must satisfy 0 <= alpha < 1")
    return baseline * (1.0 - float(alpha))
