"""Optional, measured-label quality screening independent of BO acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .history import write_json_atomic
from .metrics import QUALITY_GATE_METRIC, nested_metric, quality_margin_diagnostics


def measured_quality_label(result: Mapping[str, Any] | None, config: Any, tau_q: float) -> int | None:
    """Extract a quality label even when performance measurements are missing."""
    if not isinstance(result, Mapping) or result.get("sembench_returncode", 0) not in (None, 0):
        return None
    summary = result.get("sembench_summary") or {}
    if not isinstance(summary, Mapping) or summary.get("failed_query_count", 0):
        return None
    quality = None
    if (config.quality_metric == QUALITY_GATE_METRIC and config.quality_thresholds_json is None
            and config.min_mean_f1 is None and config.max_relative_error is None):
        measured_gate = nested_metric(result, config.quality_metric)
        if type(measured_gate) is not bool:
            return None
        quality = float(measured_gate)
    try:
        diagnostics = quality_margin_diagnostics(
            result, quality_metric=config.quality_metric,
            quality_thresholds_json=config.quality_thresholds_json,
            min_mean_f1=config.min_mean_f1, max_relative_error=config.max_relative_error, tau_q=tau_q, quality=quality,
        )
    except (TypeError, ValueError, KeyError):
        return None
    return int(diagnostics["quality_feasible"])


class QualityProbabilityModel(Protocol):
    """Replaceable classifier on the optimizer's encoded configurations."""

    def fit(self, x: np.ndarray, labels: np.ndarray) -> Any: ...

    def predict_probability(self, x: np.ndarray) -> np.ndarray: ...


class RandomForestQualityModel:
    """Learn P(pass) from unweighted binary measured labels."""

    def __init__(self, *, random_seed: int = 0) -> None:
        self.classifier = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=random_seed,
        )

    def fit(self, x: np.ndarray, labels: np.ndarray) -> RandomForestQualityModel:
        self.classifier.fit(x, labels)
        return self

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        index = list(self.classifier.classes_).index(1)
        return self.classifier.predict_proba(x)[:, index]


class QualityProbabilityGateError(RuntimeError):
    """No executable candidate was selected; no trial budget was consumed."""

    def __init__(self, reason: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.receipt = dict(receipt)


def sync_quality_gate_history(history: dict[str, Any], fresh: Mapping[str, Any]) -> bool:
    """Update plug-in metadata on resume without relabeling old observations."""
    current = fresh["quality_probability_gate"]
    previous = history.get("quality_probability_gate", {})
    if not current["enabled"] and not previous.get("enabled"):
        return False
    def measured_contract(contract):
        return {key: value for key, value in contract.items()
                if key not in {"model_target", "stored_constraint_column"}}

    if measured_contract(history["quality_constraint"]) != measured_contract(fresh["quality_constraint"]):
        raise ValueError("quality probability gate cannot reuse labels from a different quality contract")
    if current == previous:
        return False
    history.setdefault("bo_algorithm_changes", []).append({
        "after_trial_count": len(history["trials"]),
        "previous_quality_probability_gate": previous,
        "quality_probability_gate": dict(current),
    })
    for field in ("quality_probability_gate", "quality_constraint", "implementation", "acquisition", "surrogate_models"):
        if field in fresh:
            history[field] = fresh[field]
    history["optimizer_config"].update({
        key: value for key, value in fresh["optimizer_config"].items()
        if key.startswith("quality_probability_")
    })
    return True


class QualityProbabilityGate:
    """Fit once per recommendation, reject low probabilities, audit separately.

    ``propose`` returns a mechanically repaired, executable recommendation.
    Rejections never become observations. The caller owns candidate generation
    and can exclude previously returned configurations without changing history.
    """

    def __init__(
        self, *, encode: Callable[[Sequence[Mapping[str, Any]]], np.ndarray],
        key: Callable[[Mapping[str, Any]], str], threshold: float,
        max_attempts: int, audit_path: str | Path,
        model: QualityProbabilityModel | None = None, random_seed: int = 0,
        start_after_rounds: int = 20,
    ) -> None:
        if type(start_after_rounds) is not int or start_after_rounds < 20:
            raise ValueError("quality probability gating requires start_after_rounds >= 20")
        self.encode = encode
        self.key = key
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.start_after_rounds = start_after_rounds
        self.audit_path = Path(audit_path)
        self.model = model if model is not None else RandomForestQualityModel(random_seed=random_seed)

    def _audit(self, receipt: Mapping[str, Any]) -> None:
        payload = {"schema_version": "sembench.quality-probability-gate.v1", "recommendations": []}
        if self.audit_path.is_file():
            payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        payload["recommendations"].append(dict(receipt))
        write_json_atomic(self.audit_path, payload)

    def select(
        self, *, history: Mapping[str, Any], cold_start: bool,
        propose: Callable[[set[str]], dict[str, Any]],
    ) -> dict[str, Any]:
        records = [history.get("baseline_observation"), *history.get("trials", [])]
        candidates, labels, trial_ids, skipped = [], [], [], []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            measured = record.get("quality_feasible")
            if type(measured) is bool:
                label = int(measured)
            elif (record.get("quality_probability_label_source") == "measured_quality_gate"
                  and type(record.get("quality_probability_label")) is int
                  and record["quality_probability_label"] in (0, 1)):
                label = record["quality_probability_label"]
            else:
                continue
            candidate = record.get("candidate")
            if not isinstance(candidate, Mapping):
                continue
            # The baseline can legitimately be outside the current fixed domain.
            if record.get("trial_id") == "baseline" and record.get("used_for_surrogates") is False:
                skipped.append(record.get("trial_id"))
                continue
            candidates.append(dict(candidate))
            labels.append(label)
            trial_ids.append(record.get("trial_id"))
        receipt = {
            "enabled": True, "threshold": self.threshold, "max_attempts": self.max_attempts,
            "exhaustion_policy": "execute_first_recommendation_after_max_attempts",
            "start_after_rounds": self.start_after_rounds,
            "current_round": len(history.get("trials", [])) + 1,
            "model": type(self.model).__name__, "target": "measured_quality_feasible_0_or_1",
            "training_trial_ids": trial_ids, "training_rows": len(labels),
            "positive_rows": sum(labels), "negative_rows": len(labels) - sum(labels),
            "skipped_trial_ids": skipped, "history_trial_count": len(history.get("trials", [])),
            "applied": False, "checks": [], "retry_count": 0,
        }
        round_warmup = receipt["current_round"] <= self.start_after_rounds
        if round_warmup or cold_start or len(set(labels)) < 2:
            receipt["reason"] = ("round_warmup" if round_warmup else
                                 "cold_start" if cold_start else "insufficient_label_classes")
            recommendation = propose(set())
            receipt["p_quality"] = None
            receipt["outcome"] = "round_warmup_bypass" if round_warmup else "cold_start_bypass"
            recommendation["selection"]["quality_probability_gate"] = receipt
            self._audit(receipt)
            return recommendation
        receipt["applied"] = True
        try:
            x = self.encode(candidates)
            if not np.isfinite(x).all():
                raise ValueError("non-finite quality training features")
            self.model.fit(x, np.asarray(labels, dtype=int))
        except Exception as exc:
            receipt.update(outcome="quality_probability_model_unavailable", error=f"{type(exc).__name__}: {exc}")
            self._audit(receipt)
            raise QualityProbabilityGateError(receipt["outcome"], receipt) from exc
        excluded: set[str] = set()
        first_recommendation = None
        for attempt in range(self.max_attempts):
            receipt["retry_count"] = attempt
            try:
                recommendation = propose(excluded)
            except RuntimeError as exc:
                receipt.update(outcome="quality_probability_gate_exhausted", error=str(exc))
                self._audit(receipt)
                raise QualityProbabilityGateError(receipt["outcome"], receipt) from exc
            candidate = recommendation["candidate"]
            identity = self.key(candidate)
            check = {"candidate": dict(candidate), "performance_score": recommendation["selection"].get("score")}
            if identity in excluded:
                check.update(p_quality=None, accepted=False, reason="duplicate_recommendation")
                receipt["checks"].append(check)
                continue
            excluded.add(identity)
            try:
                probabilities = np.asarray(self.model.predict_probability(self.encode([candidate])), dtype=float)
                if probabilities.shape != (1,) or not np.isfinite(probabilities).all() or not 0 <= probabilities[0] <= 1:
                    raise ValueError("classifier must return one finite probability in [0, 1] per candidate")
                probability = float(probabilities[0])
            except Exception as exc:
                receipt.update(outcome="quality_probability_model_unavailable", error=f"{type(exc).__name__}: {exc}")
                self._audit(receipt)
                raise QualityProbabilityGateError(receipt["outcome"], receipt) from exc
            accepted = probability > self.threshold
            check.update(p_quality=probability, accepted=accepted,
                         reason="above_threshold" if accepted else "at_or_below_threshold")
            receipt["checks"].append(check)
            if first_recommendation is None:
                first_recommendation = recommendation
            if accepted:
                receipt.update(p_quality=probability, outcome="accepted")
                recommendation["selection"]["quality_probability_gate"] = receipt
                self._audit(receipt)
                return recommendation
        # The measured quality contract remains unchanged.  This explicit
        # execution fallback is not a predicted pass and does not relabel checks.
        receipt.update(
            outcome="attempt_limit_fallback", reason="max_attempts_reached",
            p_quality=receipt["checks"][0]["p_quality"],
            fallback_selected_check_index=0,
        )
        first_recommendation["selection"]["quality_probability_gate"] = receipt
        self._audit(receipt)
        return first_recommendation
