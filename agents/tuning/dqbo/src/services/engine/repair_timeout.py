"""Baseline-relative query execution cutoff for native and repair Trials.

Reuse the interval-union monitor already exercised by mechanism screening.
Truncated measurements remain runtime failures with explicit censoring evidence;
no partial result is presented as a complete performance/quality observation.
"""

from __future__ import annotations

import math
import json
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
from urllib.request import build_opener, ProxyHandler

from agents.tuning.mechanism_screening.baseline_only.combination_timeout import QueryTimeCutoff
from agents.tuning.mechanism_screening.baseline_only.combination_timeout import execution_lower_bound
from agents.tuning.mechanism_screening.baseline_only.combination_timeout import request_counts
from .timeout_estimation import baseline_timeout_reference, finished_request_workload, extrapolate_timeout


def read_metrics_snapshot(url):
    # Match the owned Trial HTTP client: loopback inference traffic must not
    # enter HTTP_PROXY, which can otherwise silently make every estimate absent.
    with build_opener(ProxyHandler({})).open(url, timeout=10) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("Metrics snapshot exceeds 8 MiB")
    return payload.decode("utf-8")


def query_timeout_policy(config: Any, baseline: Mapping[str, Any]) -> dict[str, Any]:
    policy = {
        "enabled": config.repair_query_timeout_enabled,
        "multiplier": config.repair_query_timeout_multiplier,
        "scope": "union_of_query_execution_intervals",
        "excludes": ["engine_startup", "preparation", "inter_query_gaps", "quality_evaluation"],
    }
    if not policy["enabled"]:
        return policy
    reference = baseline_timeout_reference(baseline)
    if reference["status"] != "measured":
        raise ValueError("RepairBO query timeout needs a complete baseline reference: " + reference["reason"])
    seconds = reference["query_execution_s"]
    cutoff = float(seconds) * policy["multiplier"]
    if not math.isfinite(cutoff):
        raise ValueError("RepairBO query execution cutoff must be finite")
    return {**policy, "baseline_query_execution_s": float(seconds), "cutoff_s": cutoff,
            "baseline_reference": reference, "performance_feedback": "workload_corrected_timeout_v1"}


def sync_query_timeout_policy(history, policy):
    previous = history.get("query_timeout_policy")
    if previous == policy:
        return False
    if previous is not None or policy["enabled"]:
        history.setdefault("query_timeout_policy_changes", []).append({
            "after_trial_count": len(history.get("trials", [])),
            "previous": previous or {"enabled": False},
            "current": policy,
        })
    history["query_timeout_policy"] = policy
    history.setdefault("optimizer_config", {}).update(
        repair_query_timeout_enabled=policy["enabled"],
        repair_query_timeout_multiplier=policy["multiplier"],
    )
    return True


class RepairQueryTimeCutoff(QueryTimeCutoff):
    def __init__(self, directory: Path, policy: Mapping[str, Any], baseline: Mapping[str, Any]):
        self.policy = dict(policy)
        self._attempts: dict[str, str] = {}
        super().__init__(
            directory, policy["cutoff_s"],
            expected_sql_count=policy["baseline_reference"]["sql_count"],
            expected_request_count=policy["baseline_reference"]["request_count"],
        )

    def environment(self, trial_id):
        # Resume/retry must never reuse a dead worker's interval or counters.
        attempt = f"{trial_id}/{uuid4().hex}"
        self._attempts[trial_id] = attempt
        return super().environment(attempt)

    def __call__(self, trial_id):
        attempt = self._attempts.get(trial_id)
        if attempt is None:
            return None
        evidence = super().__call__(attempt)
        if evidence is None:
            return None
        requests = evidence["progress"]["requests"]
        # Sampling itself takes time. Re-read intervals so completed queries do
        # not remain artificially active through the metrics HTTP request.
        try:
            records = [json.loads(f.read_text()) for f in (self.directory / attempt).glob("*.json")]
            ended = requests.get("snapshot_end_monotonic", evidence["recorded_monotonic"])
            current = execution_lower_bound(records, ended)
            if current is not None:
                evidence["lower_bound_s"] = max(evidence["lower_bound_s"], current)
            if (not any(row.get("end") is None for row in records)
                    and sum(row.get("success") is True for row in records) >= self.expected_sql_count):
                return None  # SQL finished during sampling; retain its full quality evaluation.
        except (OSError, ValueError, KeyError, TypeError):
            pass
        estimate = extrapolate_timeout(
            self.policy["baseline_reference"], requests.get("finished_request_workload") or {},
            elapsed_s=requests.get("elapsed_s"), lower_bound_s=evidence["lower_bound_s"],
        )
        return {
            **evidence,
            "trigger": "baseline_query_execution_multiplier",
            "baseline_query_execution_s": self.policy["baseline_query_execution_s"],
            "multiplier": self.policy["multiplier"],
            "progress_directory": str((self.directory / attempt).resolve()),
            "performance_estimate": estimate,
        }

    def request_progress(self, trial_id, first_query_start):
        telemetry = self.directory / trial_id / "telemetry"
        progress = {"status": "unavailable", "successful_requests": None, "failed_requests": None,
                    "expected_requests": self.expected_request_count,
                    "finished_request_workload": {"status": "unavailable", "reason": "Request snapshot unavailable"},
                    "rate_scope": "since_first_query_start_to_snapshot_end; includes inter-query gaps; not instantaneous"}
        for attempt in range(2):
            try:
                source = json.loads((telemetry / "source.json").read_text())
                before = (telemetry / "before.prom").read_text()
                started = time.monotonic()
                snapshot = read_metrics_snapshot(source["metrics_url"])
                ended = time.monotonic()
                (telemetry / "cutoff.prom").write_text(snapshot)
                (telemetry / f"cutoff_attempt_{attempt + 1}.prom").write_text(snapshot)
                counts = request_counts(before, snapshot)
                elapsed = ended - first_query_start
                if elapsed <= 0:
                    raise ValueError("Invalid snapshot span")
                cohort = finished_request_workload(before, snapshot)
                progress = {**progress, **counts, "status": "measured", "elapsed_s": elapsed,
                            "snapshot_start_monotonic": started, "snapshot_end_monotonic": ended,
                            "snapshot_path": str(telemetry / "cutoff.prom"),
                            "snapshot_attempts": attempt + 1,
                            "finished_request_workload": cohort,
                            "average_successful_requests_per_second": counts["successful_requests"] / elapsed,
                            "successful_request_fraction": (counts["successful_requests"] / self.expected_request_count
                                                            if self.expected_request_count else None)}
                if cohort["status"] == "measured":
                    break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                progress["last_snapshot_error"] = str(exc)
        return progress


def configure_repair_query_timeout(executor, config, baseline, output_dir):
    policy = query_timeout_policy(config, baseline)
    if executor is not None and isinstance(getattr(executor, "stop_condition", None), RepairQueryTimeCutoff):
        if policy == executor.stop_condition.policy:
            return policy
        executor.stop_condition = None
        executor.trial_environment = None
    if not policy["enabled"] or executor is None:
        return policy
    from .executor import BOTrialExecutor

    if not isinstance(executor, BOTrialExecutor):
        raise TypeError("RepairBO query timeout requires BOTrialExecutor monitoring support")
    if executor.stop_condition is not None or executor.trial_environment is not None:
        raise ValueError("RepairBO query timeout cannot replace an existing execution monitor")
    # Reject unsupported drivers instead of silently advertising an inactive cutoff.
    argv = executor.trial_argv
    driver = next((argv[i + 1] for i, arg in enumerate(argv[:-1])
                   if arg == "--flock-workload-driver"), None)
    if driver != "tpcc":
        raise ValueError("RepairBO query timeout requires the tpcc workload progress writer")
    monitor = RepairQueryTimeCutoff(Path(output_dir) / "query_progress", policy, baseline)
    executor.stop_condition = monitor
    executor.trial_environment = monitor.environment
    return policy
