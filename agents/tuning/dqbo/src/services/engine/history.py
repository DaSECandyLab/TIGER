"""Candidate Optimizer 的可恢复权威历史与结果投影。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pareto import pareto_mask

HISTORY_SCHEMA = "sembench.candidate-optimizer-history.v1"
RESULT_SCHEMA = "sembench.candidate-optimizer-result.v1"


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_history(path: str | Path) -> dict[str, Any] | None:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != HISTORY_SCHEMA:
        raise ValueError(f"unsupported Candidate Optimizer history: {target}")
    if not isinstance(payload.get("trials"), list):
        raise TypeError("Candidate Optimizer history.trials must be an array")
    return payload


def evaluation_count(history: Mapping[str, Any]) -> int:
    """Count every real execution, including baseline and repair children."""

    baseline = history.get("baseline_observation")
    baseline_count = int(
        isinstance(baseline, Mapping)
        and baseline.get("trial_id") == "baseline"
        and baseline.get("status") == "complete"
    )
    return baseline_count + sum(
        1 for item in history.get("trials", []) if isinstance(item, Mapping)
    )


def current_run_execution_count(history: Mapping[str, Any]) -> int:
    """Count trials physically executed by the run owning this history.

    Older native histories lack the explicit marker and are treated as locally
    executed.  Replay imports set it to false so historical observations can
    train the models without being reported as newly executed work.
    """

    return sum(
        1
        for item in history.get("trials", [])
        if isinstance(item, Mapping)
        and item.get("executed_in_current_run", True) is not False
    )


def native_bo_iteration_count(history: Mapping[str, Any]) -> int:
    """Count cold-start/native acquisitions but never repair children.

    Histories written before RepairBO have no ``trial_role`` and are treated as
    native acquisitions for backward-compatible resume semantics.
    """

    return sum(
        1
        for item in history.get("trials", [])
        if isinstance(item, Mapping) and item.get("trial_role") != "repair"
    )


def is_timeout_record(record: Mapping[str, Any]) -> bool:
    """Timeouts have unknown quality and are not repair outcome evidence."""

    return (record.get("execution_status") == "censored"
            or record.get("status") == "censored"
            or record.get("failure_kind") == "timeout_failure"
            or isinstance(record.get("censoring"), Mapping))


def complete_records(
    history: Mapping[str, Any], *, include_baseline: bool = True
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline = history.get("baseline_observation")
    if (
        include_baseline
        and isinstance(baseline, Mapping)
        and baseline.get("used_for_surrogates")
    ):
        records.append(dict(baseline))
    records.extend(
        dict(item)
        for item in history.get("trials", [])
        if isinstance(item, Mapping) and item.get("status") == "complete"
    )
    return records


def feasible_records(
    history: Mapping[str, Any], *, include_baseline: bool = True
) -> list[dict[str, Any]]:
    return [
        record
        for record in complete_records(history, include_baseline=include_baseline)
        if record.get("quality_feasible") is True
    ]


def estimated_performance_records(history: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Estimated performance is an independent training set, not a full Trial."""
    return [dict(row) for row in history.get("trials", [])
            if isinstance(row, Mapping) and row.get("status") != "complete"
            and row.get("used_for_performance_surrogates") is True
            and row.get("performance_source") == "timeout_extrapolation"
            and row.get("estimated_objective_vector") is not None]


def feasible_pareto_records(
    history: Mapping[str, Any], *, include_baseline: bool = True
) -> list[dict[str, Any]]:
    feasible = feasible_records(history, include_baseline=include_baseline)
    if not feasible:
        return []
    # Re-executing the same full configuration is a replicate, not a new search
    # point. Keep the earliest authoritative measurement so stochastic noise cannot
    # manufacture multiple Pareto anchors for one configuration.
    unique: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for record in feasible:
        candidate = record.get("candidate")
        if isinstance(candidate, Mapping):
            identity = json.dumps(
                dict(candidate), sort_keys=True, separators=(",", ":"), default=str
            )
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
        unique.append(record)
    feasible = unique
    points = np.asarray([item["objective_vector"] for item in feasible], dtype=float)
    mask = pareto_mask(points)
    return [record for record, keep in zip(feasible, mask, strict=True) if keep]


def result_from_history(history: Mapping[str, Any]) -> dict[str, Any]:
    trials = [dict(item) for item in history.get("trials", [])]
    complete = [item for item in trials if item.get("status") == "complete"]
    feasible = [item for item in complete if item.get("quality_feasible") is True]
    failures = [item for item in trials if item.get("status") == "runtime_failed"]
    repairs = [item for item in trials if item.get("trial_role") == "repair"]
    native = [item for item in trials if item.get("trial_role") != "repair"]
    episodes = [
        dict(item)
        for item in history.get("repair_episodes", [])
        if isinstance(item, Mapping)
    ]
    pareto = feasible_pareto_records(history)
    repair_enabled = isinstance(history.get("repair_scope"), Mapping)
    dataset_counts = {
        "complete": len(complete),
        "quality_feasible": len(feasible),
        "runtime_failed": len(failures),
    }
    censored = [row for row in trials if row.get("execution_status") == "censored"]
    if censored:
        dataset_counts.update(censored=len(censored), estimated_performance=len(estimated_performance_records(history)))
    if repair_enabled:
        dataset_counts.update(
            {
                "native_bo": len(native),
                "repair": len(repairs),
                "repair_quality_converted": sum(
                    1 for item in repairs if item.get("quality_feasible") is True
                ),
            }
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "agent": "candidate_optimizer",
        "source_parameter_results": history["source_parameter_results"],
        "source_domain_spec": history["source_domain_spec"],
        "source_baseline_config": history["source_baseline_config"],
        "source_baseline_trial_result": history["source_baseline_trial_result"],
        "rounds_requested": history["rounds_requested"],
        "rounds_completed": len(trials),
        "objective": {
            "direction": "maximize",
            "names": ["qps", "negative_mean_latency", "negative_p95_latency"],
            "quality_is_objective": False,
        },
        "quality_constraint": dict(history["quality_constraint"]),
        "surrogate_models": list(history["surrogate_models"]),
        "conditional_space": history.get("conditional_space"),
        "bo_algorithm_changes": list(history.get("bo_algorithm_changes", [])),
        "acquisition": dict(history["acquisition"]),
        "dataset_counts": dataset_counts,
        "complete_trial_ids": [item["trial_id"] for item in complete],
        "quality_feasible_trial_ids": [item["trial_id"] for item in feasible],
        "runtime_failure_trial_ids": [item["trial_id"] for item in failures],
        "feasible_pareto_front": [
            {
                "trial_id": item["trial_id"],
                "candidate": item["candidate"],
                "objective_vector": item["objective_vector"],
                "quality": item["quality"],
                "quality_ratio_to_baseline": item["quality_ratio_to_baseline"],
            }
            for item in pareto
        ],
        "history_path": history["history_path"],
    }
    if "quality_probability_gate" in history:
        result["quality_probability_gate"] = dict(history["quality_probability_gate"])
    if repair_enabled:
        completed_repairs = [item for item in repairs if item.get("status") == "complete"]
        converted_repairs = [
            item for item in completed_repairs if item.get("quality_feasible") is True
        ]
        strict_success_repairs = [
            item for item in completed_repairs if item.get("repair_success") is True
        ]
        triggered_episodes = [
            item
            for item in episodes
            if item.get("status")
            in {
                "running", "succeeded", "partially_succeeded",
                "failed_after_round_2", "budget_truncated",
                "llm_recommendation_failed", "candidate_exhausted",
                "no_measured_repair_candidate",
            }
        ]
        repair_decisions = [
            decision
            for episode in episodes
            for round_record in episode.get("rounds", [])
            if isinstance(round_record, Mapping)
            for decision in round_record.get("decisions", [])
            if isinstance(decision, Mapping)
        ]
        llm_attempts = [
            attempt
            for decision in repair_decisions
            for attempt in decision.get("llm_attempts", [])
            if isinstance(attempt, Mapping)
        ]
        invalid_llm_attempts = [
            item for item in llm_attempts if item.get("valid") is not True
        ]
        proposal_attempts = [a for e in episodes for c in e.get("proposal_cycles", [])
                             for a in c.get("attempts", [])]
        proposal_reviews = [a for a in proposal_attempts if "assessment" in a]

        result.update(
            {
                "evaluations_completed": evaluation_count(history),
                "observations_available": evaluation_count(history),
                "executions_completed_in_current_run": current_run_execution_count(
                    history
                ),
                "baseline_evaluations_completed": 1,
                "baseline_reused": True,
                "bo_iterations_completed": len(native),
                "repair_trial_ids": [item["trial_id"] for item in repairs],
                "repair_episode_count": len(episodes),
                "repair_episodes": episodes,
                "repair_trigger_events": [
                    dict(item)
                    for item in history.get("repair_trigger_events", [])
                    if isinstance(item, Mapping)
                ],
                "repair_scope": history.get("repair_scope"),
                "repair_metrics": {
                    "completed_repair_trials": len(completed_repairs),
                    "proposal_count": sum("proposal" in a for a in proposal_attempts),
                    "proposal_llm_attempt_count": sum(len(a.get("llm_attempts", [])) for a in proposal_attempts),
                    "proposal_review_count": len(proposal_reviews),
                    "audit_route_counts": {
                        route: sum(route in a["assessment"].get("audits", {}) for a in proposal_reviews)
                        for route in ("runtime", "performance")
                    },
                    "audit_route_rejections": {
                        route: sum(a["assessment"].get("audits", {}).get(route, {}).get("verdict") == "revise"
                                   for a in proposal_reviews)
                        for route in ("runtime", "performance")
                    },
                    "proposal_revision_count": sum(a.get("revision", 0) > 0 and "proposal" in a for a in proposal_attempts),
                    "proposal_review_unavailable_count": sum(a["assessment"].get("status") == "unavailable" for a in proposal_reviews),
                    "a_review_skipped_count": sum(bool(e.get("round_1_a_skipped")) for e in episodes),
                    "quality_conversion_rate": (
                        len(converted_repairs) / len(completed_repairs)
                        if completed_repairs
                        else None
                    ),
                    "strict_repair_success_rate": (
                        sum(any(t.get("episode_id") == item.get("episode_id")
                                for t in strict_success_repairs) for item in triggered_episodes)
                        / len(triggered_episodes)
                        if triggered_episodes
                        else None
                    ),
                    "strict_success_trial_count": len(strict_success_repairs),
                    "repair_hv_gain_total": sum(
                        float(item.get("repair_actual_hv_gain") or 0.0)
                        for item in completed_repairs
                    ),
                    "strategy_success_contribution": {
                        strategy: sum(
                            item.get("repair_success") is True
                            and item.get("repair_strategy") == strategy
                            for item in completed_repairs
                        )
                        for strategy in sorted(
                            {
                                str(item.get("repair_strategy"))
                                for item in repairs
                                if item.get("repair_strategy")
                            }
                            | {"A", "B", "R2"}
                        )
                    },
                    "duplicate_execution_count": 0,
                    "llm_attempt_count": len(llm_attempts),
                    "llm_invalid_output_count": len(invalid_llm_attempts),
                    "llm_correction_count": sum(
                        any(
                            isinstance(item, Mapping) and item.get("valid") is True
                            for item in decision.get("llm_attempts", [])[1:]
                        )
                        and bool(decision.get("llm_attempts"))
                        and isinstance(decision.get("llm_attempts", [])[0], Mapping)
                        and decision.get("llm_attempts", [])[0].get("valid") is not True
                        for decision in repair_decisions
                    ),
                },
            }
        )
    return result
