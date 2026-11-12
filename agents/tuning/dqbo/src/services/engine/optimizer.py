"""独立的质量约束三目标 Candidate Optimizer 主循环。"""

from __future__ import annotations

from agents.tuning.dqbo.src.services.failure_taxonomy import summarize_runtime_failure

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.stats import qmc

from .acquisition import joint_mc_constrained_ehvi_scores, performance_mc_ehvi_scores
from .constraints import KnownConstraintHandler
from .executor import TrialExecution, TrialExecutor
from .history import (
    HISTORY_SCHEMA,
    complete_records,
    estimated_performance_records,
    feasible_pareto_records,
    load_history,
    result_from_history,
    write_json_atomic,
)
from .metrics import (
    AGGREGATE_QUALITY_THRESHOLD_SOURCE,
    QUALITY_THRESHOLD_FORMULAS,
    QUALITY_THRESHOLD_SOURCE,
    RELATIVE_QUALITY_THRESHOLD_FORMULAS,
    RELATIVE_QUALITY_THRESHOLD_SOURCE,
    extract_complete_trial_metrics,
    quality_threshold,
)
from .models import FourSurrogateModels, PerformanceSurrogateModels, RunFeasibilityModel, SURROGATE_NAMES
from .quality_probability import (
    QualityProbabilityGate, QualityProbabilityModel, measured_quality_label, sync_quality_gate_history,
)
from .space import FixedSearchSpace


@dataclass(frozen=True)
class OptimizerConfig:
    rounds: int
    # 每个代表指标已在 workload 内逐项判定；优化器只消费最终逻辑门禁。
    quality_metric: str = "sembench_summary.quality_gate_passed"
    quality_alpha: float = 0.05
    quality_thresholds_json: str | None = None
    quality_threshold_source: str | None = None
    min_mean_f1: float | None = None
    max_relative_error: float | None = None
    random_seed: int = 0
    # None 严格沿用原 HEBO：rand_sample = 1 + logical search parameters。
    # 显式整数仅保留给小型测试或专项实验覆盖。
    initial_random_trials: int | None = None
    candidate_pool_size: int = 128
    # Deliberately use a small posterior sample count so candidate ranking keeps
    # useful Monte-Carlo variation instead of becoming overly deterministic.
    monte_carlo_samples: int = 10
    reference_margin: float = 0.05
    # False: quality is still measured/gated, but its posterior probability does
    # not affect candidate ranking; acquisition is performance-only EHVI.
    quality_probability_affects_acquisition: bool = True
    # Opt-in classifier gate. Enabling it keeps performance acquisition as EHVI.
    quality_probability_gate_enabled: bool = False
    quality_probability_threshold: float = 0.25
    quality_probability_max_attempts: int = 32
    quality_probability_start_after_rounds: int = 20
    # RepairBO is opt-in so existing BO experiments retain their exact
    # acquisition and preflight semantics.
    repair_bo_enabled: bool = False
    # Opt-in query execution cutoff; the independent Trial wall-clock limit remains.
    repair_query_timeout_enabled: bool = False
    repair_query_timeout_multiplier: float = 3.0
    # Explicit ablation mode is independent of LLM RepairBO. None preserves prior runs.
    bo_mode: str | None = None
    repair_conditional_space_enabled: bool = True
    repair_candidate_supplement_per_source: int = 2
    repair_scope_fields: tuple[str, ...] = (
        "quantization",
        "kv_cache_dtype",
        "dtype",
    )
    repair_scope_version: str = "repair-scope.triplet.v1"
    # Total measured cold-start executions, including the pre-measured baseline.
    # v1 uses evaluation 1 as baseline and evaluations 2..8 as seven Sobol trials.
    repair_cold_start_evaluations: int = 8
    repair_llm_cold_start_enabled: bool = True
    repair_historical_memory_enabled: bool = False
    repair_reflect_enabled: bool = False
    repair_adaptive_threshold_enabled: bool = True
    repair_hv_epsilon: float = 1e-12
    repair_boundary_utility_epsilon: float = 1e-12
    repair_max_rounds: int = 2
    # Compatibility option; nearest cases now cover every signature without top-K.
    repair_context_limit: int = 32
    # Round 2 executes exactly one candidate.  When round 1 recovers quality but
    # misses the Pareto front, this pool explores the full non-representation Range.
    repair_performance_candidate_pool_size: int = 64
    repair_proposal_review_enabled: bool = True
    repair_proposal_max_revisions: int = 1

    def __post_init__(self) -> None:
        if self.quality_probability_gate_enabled is True:
            object.__setattr__(self, "quality_probability_affects_acquisition", False)

    def quality_probability_gate_contract(self) -> dict[str, Any]:
        return {
            "enabled": self.quality_probability_gate_enabled,
            "threshold": self.quality_probability_threshold,
            "max_attempts": self.quality_probability_max_attempts,
            "start_after_rounds": self.quality_probability_start_after_rounds,
            "round_rule": "len(history.trials) + 1 > start_after_rounds; baseline excluded",
            "target": "measured_quality_feasible_0_or_1",
            "default_model": "RandomForestQualityModel",
            "pass_rule": "p_quality > threshold",
            "exhaustion_policy": "execute_first_recommendation_after_max_attempts",
            "scope": "native_bo_recommendations_after_preflight",
            "affects_acquisition": False,
        }

    @property
    def conditional_model_enabled(self) -> bool:
        return self.bo_mode == "conditional" or (
            self.bo_mode is None and self.repair_bo_enabled
            and self.repair_conditional_space_enabled
        )

    @property
    def project_before_acquisition(self) -> bool:
        return self.bo_mode != "post_recommendation"

    def validate(self) -> None:
        if type(self.quality_probability_gate_enabled) is not bool:
            raise TypeError("quality_probability_gate_enabled must be boolean")
        if (isinstance(self.quality_probability_threshold, bool)
                or not isinstance(self.quality_probability_threshold, (int, float))
                or not 0 <= self.quality_probability_threshold < 1):
            raise ValueError("quality_probability_threshold must be in [0, 1)")
        if type(self.quality_probability_max_attempts) is not int or self.quality_probability_max_attempts < 1:
            raise ValueError("quality_probability_max_attempts must be a positive integer")
        if type(self.quality_probability_start_after_rounds) is not int or self.quality_probability_start_after_rounds < 20:
            raise ValueError("quality_probability_start_after_rounds must be an integer >= 20")
        if self.bo_mode not in (None, "conditional", "post_recommendation"):
            raise ValueError("bo_mode must be conditional, post_recommendation, or None")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if not 0.0 <= self.quality_alpha < 1.0:
            raise ValueError("quality_alpha must satisfy 0 <= alpha < 1")
        if self.quality_thresholds_json is not None:
            thresholds = json.loads(self.quality_thresholds_json)
            if not isinstance(thresholds, dict) or not thresholds:
                raise ValueError("quality_thresholds_json must be a non-empty object")
        if self.quality_threshold_source is not None and not str(
            self.quality_threshold_source
        ).strip():
            raise ValueError("quality_threshold_source must be non-empty")
        if (
            self.quality_threshold_source is not None
            and self.quality_thresholds_json is None
        ):
            raise ValueError(
                "quality_threshold_source requires quality_thresholds_json"
            )
        if self.min_mean_f1 is not None and not 0.0 <= self.min_mean_f1 <= 1.0:
            raise ValueError("min_mean_f1 must be in [0, 1]")
        if self.max_relative_error is not None and self.max_relative_error < 0.0:
            raise ValueError("max_relative_error must be non-negative")
        if self.quality_thresholds_json is not None and (
            self.min_mean_f1 is not None or self.max_relative_error is not None
        ):
            raise ValueError(
                "per-query and aggregate quality thresholds are mutually exclusive"
            )
        if (
            self.min_mean_f1 is not None or self.max_relative_error is not None
        ) and self.quality_metric != "sembench_summary.quality_gate_passed":
            raise ValueError(
                "aggregate quality thresholds require quality_gate_passed"
            )
        if self.initial_random_trials is not None and self.initial_random_trials < 1:
            raise ValueError("initial_random_trials must be positive")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        if self.monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be positive")
        if not 0.0 < self.reference_margin < 1.0:
            raise ValueError("reference_margin must be between 0 and 1")
        if not isinstance(self.quality_probability_affects_acquisition, bool):
            raise TypeError(
                "quality_probability_affects_acquisition must be boolean"
            )
        if not isinstance(self.repair_bo_enabled, bool):
            raise TypeError("repair_bo_enabled must be boolean")
        if type(self.repair_query_timeout_enabled) is not bool:
            raise TypeError("repair_query_timeout_enabled must be boolean")
        if (isinstance(self.repair_query_timeout_multiplier, bool)
                or not isinstance(self.repair_query_timeout_multiplier, (int, float))
                or not math.isfinite(self.repair_query_timeout_multiplier)
                or self.repair_query_timeout_multiplier <= 0):
            raise ValueError("repair_query_timeout_multiplier must be positive and finite")
        if self.repair_query_timeout_enabled and not self.repair_bo_enabled:
            raise ValueError("repair_query_timeout_enabled requires repair_bo_enabled")
        for name in ("repair_llm_cold_start_enabled", "repair_historical_memory_enabled", "repair_reflect_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.repair_conditional_space_enabled, bool):
            raise TypeError("repair_conditional_space_enabled must be boolean")
        if type(self.repair_candidate_supplement_per_source) is not int or not 0 <= self.repair_candidate_supplement_per_source <= 8:
            raise ValueError("repair_candidate_supplement_per_source must be an integer in [0, 8]")
        fields = tuple(str(item).strip() for item in self.repair_scope_fields)
        if not fields or any(not item for item in fields) or len(set(fields)) != len(fields):
            raise ValueError("repair_scope_fields must contain unique non-empty names")
        if not str(self.repair_scope_version).strip():
            raise ValueError("repair_scope_version must be non-empty")
        if self.repair_cold_start_evaluations < 1:
            raise ValueError(
                "repair_cold_start_evaluations must include at least the baseline"
            )
        if type(self.repair_adaptive_threshold_enabled) is not bool:
            raise ValueError("repair_adaptive_threshold_enabled must be boolean")
        if self.repair_hv_epsilon < 0.0:
            raise ValueError("repair_hv_epsilon must be non-negative")
        if self.repair_boundary_utility_epsilon < 0.0:
            raise ValueError("repair_boundary_utility_epsilon must be non-negative")
        if self.repair_max_rounds != 2:
            raise ValueError("RepairBO v1 requires exactly two repair rounds")
        if type(self.repair_proposal_max_revisions) is not int or not 0 <= self.repair_proposal_max_revisions <= 2:
            raise ValueError("repair_proposal_max_revisions must be an integer between 0 and 2")
        if self.repair_context_limit < 1:
            raise ValueError("repair_context_limit must be positive")
        if self.repair_performance_candidate_pool_size < 1:
            raise ValueError(
                "repair_performance_candidate_pool_size must be positive"
            )

    @property
    def repair_cold_start_native_trials(self) -> int:
        """Post-baseline warm-up slots, including optional memory candidates."""

        return self.repair_cold_start_evaluations - 1


class CandidateOptimizer:
    """只在给定固定域中执行 n 轮质量约束多目标 BO。"""

    def __init__(
        self,
        *,
        search_space: FixedSearchSpace,
        executor: TrialExecutor,
        baseline_config_path: str | Path,
        baseline_result_path: str | Path,
        parameter_results_path: str | Path,
        output_dir: str | Path,
        config: OptimizerConfig,
        constraint_handler: KnownConstraintHandler | None = None,
        candidate_repairer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        quality_probability_model: QualityProbabilityModel | None = None,
    ) -> None:
        config.validate()
        self.space = search_space
        self.executor = executor
        self.baseline_config_path = str(
            Path(baseline_config_path).expanduser().resolve()
        )
        self.baseline_result_path = str(
            Path(baseline_result_path).expanduser().resolve()
        )
        self.parameter_results_path = str(
            Path(parameter_results_path).expanduser().resolve()
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "bo_history.json"
        self.result_path = self.output_dir / "candidate_optimizer_result.json"
        self.config = config
        self.constraint_handler = constraint_handler or KnownConstraintHandler(
            search_space
        )
        self.candidate_repairer = candidate_repairer
        self.quality_probability_model = quality_probability_model
        self.initial_design_trials = (
            int(config.initial_random_trials)
            if config.initial_random_trials is not None
            else 1 + len(search_space.tuned_parameters)
        )
        self.rng = np.random.default_rng(config.random_seed)
        self.baseline_result = self._load_object(self.baseline_result_path)
        from .repair_timeout import configure_repair_query_timeout

        self.query_timeout_policy = configure_repair_query_timeout(
            self.executor, config, self.baseline_result, self.output_dir,
        )
        self.baseline_objectives, self.baseline_quality, self.baseline_metrics = (
            extract_complete_trial_metrics(
                self.baseline_result,
                quality_metric=config.quality_metric,
                quality_thresholds_json=config.quality_thresholds_json,
                min_mean_f1=config.min_mean_f1,
                max_relative_error=config.max_relative_error,
            )
        )
        self.tau_q = (
            1.0 - config.quality_alpha
            if config.quality_metric == "sembench_summary.quality_gate_passed"
            else quality_threshold(self.baseline_quality, config.quality_alpha)
        )

    @staticmethod
    def _load_object(path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSON object: {path}")
        return payload

    def _baseline_observation(self) -> dict[str, Any]:
        candidate = self.space.bo_to_canonical(self.baseline_result)
        constraint_check = self.constraint_handler.repair(candidate)
        candidate = constraint_check.candidate
        validation_errors = list(constraint_check.validation_errors)
        used_for_surrogates = not validation_errors
        return {
            "trial_id": "baseline",
            "source": "baseline",
            "status": "complete",
            "candidate": candidate,
            "candidate_bo_names": self.space.canonical_to_bo(candidate),
            "objective_vector": list(self.baseline_objectives),
            "metrics": dict(self.baseline_metrics),
            "quality": self.baseline_quality,
            "quality_ratio_to_baseline": (
                1.0 if self.baseline_quality > 0.0 else None
            ),
            "quality_feasible": self.baseline_quality >= self.tau_q,
            "used_for_surrogates": used_for_surrogates,
            "used_for_feasible_pareto": used_for_surrogates
            and self.baseline_quality >= self.tau_q,
            "used_for_hidden_constraint": used_for_surrogates,
            "result_path": self.baseline_result_path,
            "domain_validation_errors": validation_errors,
            "known_constraint_check": constraint_check.to_dict(),
        }

    def _new_history(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORY_SCHEMA,
            "agent": "candidate_optimizer",
            "source_parameter_results": self.parameter_results_path,
            "source_domain_spec": self.space.source_path,
            "source_baseline_config": self.baseline_config_path,
            "source_baseline_trial_result": self.baseline_result_path,
            "history_path": str(self.history_path),
            "rounds_requested": self.config.rounds,
            "optimizer_config": asdict(self.config),
            "quality_probability_gate": self.config.quality_probability_gate_contract(),
            "query_timeout_policy": self.query_timeout_policy,
            "objective": {
                "direction": "maximize",
                "names": ["qps", "negative_mean_latency", "negative_p95_latency"],
                "quality_is_objective": False,
            },
            "quality_constraint": {
                "metric": self.config.quality_metric,
                "baseline_quality": self.baseline_quality,
                "baseline_run_count": 1,
                "alpha": self.config.quality_alpha,
                "tau_q": self.tau_q,
                "threshold_source": (
                    self.config.quality_threshold_source
                    or QUALITY_THRESHOLD_SOURCE
                    if self.config.quality_thresholds_json is not None
                    else (
                        AGGREGATE_QUALITY_THRESHOLD_SOURCE
                        if self.config.min_mean_f1 is not None
                        or self.config.max_relative_error is not None
                        else None
                    )
                ),
                "aggregate_thresholds": {
                    "min_mean_f1": self.config.min_mean_f1,
                    "max_relative_error": self.config.max_relative_error,
                },
                "query_metric_thresholds": (
                    json.loads(self.config.quality_thresholds_json)
                    if self.config.quality_thresholds_json is not None
                    else None
                ),
                "threshold_formulas": (
                    dict(
                        RELATIVE_QUALITY_THRESHOLD_FORMULAS
                        if self.config.quality_threshold_source
                        == RELATIVE_QUALITY_THRESHOLD_SOURCE
                        else QUALITY_THRESHOLD_FORMULAS
                    )
                    if self.config.quality_thresholds_json is not None
                    else None
                ),
                "rule": (
                    "mean_f1_score >= min_mean_f1 and max_relative_error <= max_relative_error"
                    if self.config.min_mean_f1 is not None
                    or self.config.max_relative_error is not None
                    else (
                        "configured per-query metrics pass their frozen "
                        "baseline-relative and absolute thresholds"
                    )
                    if self.config.quality_threshold_source
                    == RELATIVE_QUALITY_THRESHOLD_SOURCE
                    else "quality_gate_passed is true"
                    if self.config.quality_metric == "sembench_summary.quality_gate_passed"
                    else "quality / baseline_quality >= 1 - alpha"
                ),
                "inclusive": True,
            },
            "surrogate_models": list(SURROGATE_NAMES[:3] if self.config.quality_probability_gate_enabled else SURROGATE_NAMES),
            "acquisition": {
                "name": (
                    "joint_mc_constrained_ehvi"
                    if self.config.quality_probability_affects_acquisition
                    else "performance_only_mc_ehvi"
                ),
                "formula": (
                    "E[delta_HV(f(x); P_F) * I(q(x) >= tau_q)]"
                    if self.config.quality_probability_affects_acquisition
                    else "E[delta_HV(f(x); P_F)]"
                ),
                "quality_feasibility": "P(q(x) >= tau_q)",
                "quality_mc_indicator": (
                    self.config.quality_probability_affects_acquisition
                ),
                "scalar_ehvi_times_p_feasible": False,
                "quality_mean_hard_threshold": False,
                "quality_surrogate_affects_final_ranking": (
                    self.config.quality_probability_affects_acquisition
                ),
                "p_run_affects_final_score": False,
                "pareto_source": "quality_feasible_complete_trials_only",
            },
            "initial_design": {
                "method": "scrambled_sobol",
                "required_complete_observations": self.initial_design_trials,
                "rule": (
                    "explicit_override"
                    if self.config.initial_random_trials is not None
                    else "1 + logical_tuned_parameter_count"
                ),
                "logical_tuned_parameter_count": len(self.space.tuned_parameters),
                "baseline_counts_when_inside_domain": True,
                "runtime_failures_count": False,
            },
            "baseline_observation": self._baseline_observation(),
            "trials": [],
        }

    def _load_or_create_history(self) -> dict[str, Any]:
        history = load_history(self.history_path)
        if history is None:
            history = self._new_history()
            write_json_atomic(self.history_path, history)
            return history
        expected = {
            "source_parameter_results": self.parameter_results_path,
            "source_domain_spec": self.space.source_path,
            "source_baseline_config": self.baseline_config_path,
            "source_baseline_trial_result": self.baseline_result_path,
            "rounds_requested": self.config.rounds,
        }
        conflicts = {
            name: {"history": history.get(name), "requested": value}
            for name, value in expected.items()
            if history.get(name) != value
        }
        if conflicts:
            raise ValueError(f"existing BO history conflicts with this run: {conflicts}")
        from .repair_timeout import sync_query_timeout_policy

        if sync_query_timeout_policy(history, self.query_timeout_policy):
            write_json_atomic(self.history_path, history)
        if self.config.quality_probability_gate_enabled or history.get("quality_probability_gate", {}).get("enabled"):
            if sync_quality_gate_history(history, self._new_history()):
                write_json_atomic(self.history_path, history)
        return history

    def initialize_history(self) -> dict[str, Any]:
        """Materialize the empty, baseline-backed history during prepare."""

        return self._load_or_create_history()

    def _training_arrays(
        self, history: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        records = complete_records(history)
        estimates = estimated_performance_records(history) if self.config.repair_bo_enabled else []
        candidates = [record["candidate"] for record in records]
        candidates.extend(record["candidate"] for record in estimates)
        return (
            self.space.encode(candidates),
            np.asarray([record["objective_vector"] for record in records]
                       + [record["estimated_objective_vector"] for record in estimates], dtype=float),
            np.asarray([record["quality"] for record in records] + [np.nan] * len(estimates), dtype=float),
        )

    def _run_model(self, history: Mapping[str, Any]) -> RunFeasibilityModel:
        attempts: list[Mapping[str, Any]] = []
        baseline = history.get("baseline_observation")
        if isinstance(baseline, Mapping) and baseline.get("used_for_hidden_constraint"):
            attempts.append(baseline)
        attempts.extend(item for item in history.get("trials", []) if isinstance(item, Mapping)
                        and item.get("used_for_hidden_constraint") is not False)
        model = RunFeasibilityModel(random_seed=self.config.random_seed)
        if not attempts:
            return model
        return model.fit(
            self.space.encode([item["candidate"] for item in attempts]),
            [item.get("status") == "complete" for item in attempts],
        )

    def _reference_point(self, objectives: np.ndarray) -> np.ndarray:
        values = np.vstack(
            [np.asarray(self.baseline_objectives, dtype=float).reshape(1, 3), objectives]
        )
        minima = np.min(values, axis=0)
        spans = np.maximum(np.ptp(values, axis=0), np.maximum(np.abs(minima), 1.0))
        return minima - self.config.reference_margin * spans

    def _candidate_pool(self, history: Mapping[str, Any], *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
        excluded = {
            self.space.key(item["candidate"])
            for item in history.get("trials", [])
            if isinstance(item, Mapping) and isinstance(item.get("candidate"), Mapping)
        }
        baseline = history.get("baseline_observation")
        if isinstance(baseline, Mapping) and baseline.get("used_for_surrogates"):
            excluded.add(self.space.key(baseline["candidate"]))
        excluded.update(exclude or ())
        requested = self.config.candidate_pool_size
        while requested >= 1:
            try:
                return self.constraint_handler.sample(
                    self.rng, count=requested, exclude=excluded
                )
            except RuntimeError:
                if requested == 1:
                    raise
                requested = max(1, requested // 2)
        raise RuntimeError("fixed search space has no unobserved candidate")

    def _sobol_candidate(
        self, history: Mapping[str, Any], *, exclude: set[str] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """复现原 HEBO 的 scrambled Sobol 初始探索，并支持确定性恢复。"""

        excluded = {
            self.space.key(item["candidate"])
            for item in history.get("trials", [])
            if isinstance(item, Mapping) and isinstance(item.get("candidate"), Mapping)
        }
        baseline = history.get("baseline_observation")
        if isinstance(baseline, Mapping) and baseline.get("used_for_surrogates"):
            excluded.add(self.space.key(baseline["candidate"]))
        excluded.update(exclude or ())
        used_indices = [
            int(selection["sobol_draw_index"])
            for item in history.get("trials", [])
            if isinstance(item, Mapping)
            and isinstance((selection := item.get("selection")), Mapping)
            and isinstance(selection.get("sobol_draw_index"), int)
        ]
        start_index = max(used_indices, default=-1) + 1
        attempt_limit = max(4096, self.config.candidate_pool_size * 32)
        required_rows = start_index + attempt_limit
        exponent = max(0, math.ceil(math.log2(max(1, required_rows))))
        points = qmc.Sobol(
            d=len(self.space.tuned_parameters),
            scramble=True,
            seed=self.config.random_seed,
        ).random_base2(exponent)
        for draw_index in range(start_index, min(required_rows, len(points))):
            raw_candidate = self.space.from_unit_cube(points[draw_index])
            repaired = self.constraint_handler.repair(raw_candidate)
            if not repaired.valid:
                continue
            key = self.space.key(repaired.candidate)
            if key in excluded:
                continue
            return repaired.candidate, draw_index
        raise RuntimeError(
            "scrambled Sobol initial design could not produce an unobserved "
            f"constraint-safe candidate after {attempt_limit} draws"
        )

    def _suggest(
        self, history: Mapping[str, Any], *, exclude: set[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.config.quality_probability_gate_enabled:
            ranked = getattr(self, "_quality_probability_ranked_pool", [])
            while ranked:
                candidate, selection = ranked.pop(0)
                if self.space.key(candidate) not in (exclude or ()):
                    return candidate, selection
        complete = complete_records(history)
        estimate_count = len(estimated_performance_records(history)) if self.config.repair_bo_enabled else 0
        if len(complete) + estimate_count < self.initial_design_trials:
            candidate, draw_index = self._sobol_candidate(history, exclude=exclude)
            return candidate, {
                "method": "initial_scrambled_sobol",
                "complete_training_rows": len(complete),
                "estimated_training_rows": estimate_count,
                "required_complete_observations": self.initial_design_trials,
                "sobol_draw_index": draw_index,
                "candidate_pool_size": None,
                "ehvi": None,
                "p_feasible": None,
                "p_run": self._run_model(history).predict_probability(
                    self.space.encode([candidate])
                )[0],
                "score": None,
            }
        pool = self._candidate_pool(history, exclude=exclude)
        x_train, objective_train, quality_train = self._training_arrays(history)
        if self.config.quality_probability_gate_enabled or self.config.repair_bo_enabled:
            surrogate = PerformanceSurrogateModels(random_seed=self.config.random_seed).fit(x_train, objective_train)
        else:
            surrogate = FourSurrogateModels(random_seed=self.config.random_seed).fit(
                x_train, objective_train, quality_train
            )
        run_model = self._run_model(history)
        feasible_front = np.asarray(
            [item["objective_vector"] for item in feasible_pareto_records(history)],
            dtype=float,
        )
        if feasible_front.size == 0:
            feasible_front = np.empty((0, 3), dtype=float)
        reference = self._reference_point(objective_train)
        acquisition_kwargs = dict(
            surrogate_models=surrogate,
            run_model=run_model,
            encoded_candidates=self.space.encode(pool),
            feasible_pareto=feasible_front,
            reference_point=reference,
            monte_carlo_samples=self.config.monte_carlo_samples,
            rng=self.rng,
        )
        if self.config.quality_probability_gate_enabled or self.config.repair_bo_enabled:
            scores = performance_mc_ehvi_scores(**acquisition_kwargs)
        else:
            scores = joint_mc_constrained_ehvi_scores(
                **acquisition_kwargs, tau_q=self.tau_q,
                quality_probability_affects_acquisition=self.config.quality_probability_affects_acquisition,
            )
        # 质量不参与 acquisition 时，连 tie-break 也不能读取 P_feasible。
        ranking_key = (
            (lambda index: (
                    float(scores.score[index]),
                    float(scores.p_feasible[index]),
                    -index,
            ))
            if self.config.quality_probability_affects_acquisition
            else (lambda index: (float(scores.score[index]), -index))
        )
        best = max(range(len(pool)), key=ranking_key)
        selection = {
            "method": (
                "joint_mc_constrained_ehvi"
                if self.config.quality_probability_affects_acquisition
                else "performance_only_mc_ehvi"
            ),
            "complete_training_rows": len(complete),
            "candidate_pool_size": len(pool),
            "monte_carlo_samples": self.config.monte_carlo_samples,
            "reference_point": reference.tolist(),
            "feasible_pareto_size": int(feasible_front.shape[0]),
            "ehvi": float(scores.ehvi[best]),
            "constrained_ehvi": None if self.config.quality_probability_gate_enabled else float(scores.constrained_ehvi[best]),
            "p_feasible": None if self.config.quality_probability_gate_enabled else float(scores.p_feasible[best]),
            "quality_feasible_mc_samples": int(
                scores.quality_feasible_mc_samples[best]
            ),
            "p_run": float(scores.p_run[best]),
            "p_run_affects_final_score": False,
            "quality_mean_hard_threshold": False,
            "quality_mc_indicator": (
                self.config.quality_probability_affects_acquisition
            ),
            "scalar_ehvi_times_p_feasible": False,
            "quality_surrogate_affects_final_ranking": (
                self.config.quality_probability_affects_acquisition
            ),
            "score": float(scores.score[best]),
        }
        if self.config.quality_probability_gate_enabled:
            self._quality_probability_ranked_pool = [
                (pool[i], {**selection, "ehvi": float(scores.ehvi[i]),
                           "p_run": float(scores.p_run[i]), "score": float(scores.score[i])})
                for i in sorted(range(len(pool)), key=lambda i: (-scores.score[i], i)) if i != best
            ]
        return pool[best], selection

    def _record_execution(
        self,
        *,
        candidate: Mapping[str, Any],
        selection: Mapping[str, Any],
        constraint_check: Mapping[str, Any],
        execution: TrialExecution,
        trial_id: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        base = {
            "trial_id": trial_id,
            "source": "candidate_optimizer",
            "candidate": dict(candidate),
            "candidate_bo_names": self.space.canonical_to_bo(candidate),
            "active_parameters": sorted(self.space.active_parameter_names(candidate)),
            "selection": dict(selection),
            "known_constraint_check": dict(constraint_check),
            "elapsed_seconds": elapsed_seconds,
            "result_path": execution.result_path,
            "executor": {
                "command": list(execution.command),
                "return_code": execution.return_code,
            },
        }
        if self.config.quality_probability_gate_enabled:
            base["quality_probability_label"] = (measured_quality_label(execution.result, self.config, self.tau_q)
                                                 if execution.status == "complete" else None)
            base["quality_probability_label_source"] = "measured_quality_gate"
        if execution.status != "complete" or not isinstance(execution.result, Mapping):
            from .timeout_estimation import timeout_record_fields

            return {
                **base,
                **({"censoring": dict(execution.censoring), "execution_status": execution.status}
                   if execution.censoring is not None else {}),
                "status": "runtime_failed",
                "failure_kind": execution.failure_kind or "runtime_failure",
                "failure_reason": execution.failure_reason,
                "objective_vector": None,
                "quality": None,
                "quality_feasible": None,
                "used_for_performance_surrogates": False,
                "used_for_quality_surrogate": False,
                "used_for_feasible_pareto": False,
                "used_for_hidden_constraint": execution.censoring is None,
                **timeout_record_fields(execution.censoring),
            }
        try:
            objectives, quality, metrics = extract_complete_trial_metrics(
                execution.result,
                quality_metric=self.config.quality_metric,
                quality_thresholds_json=self.config.quality_thresholds_json,
                min_mean_f1=self.config.min_mean_f1,
                max_relative_error=self.config.max_relative_error,
            )
        except (TypeError, ValueError) as exc:
            return {
                **base,
                "status": "runtime_failed",
                "failure_kind": "incomplete_measurement",
                "failure_reason": summarize_runtime_failure(str(exc), fallback=f"Measurement extraction failed: {exc}"),
                "objective_vector": None,
                "quality": None,
                "quality_feasible": None,
                "used_for_performance_surrogates": False,
                "used_for_quality_surrogate": False,
                "used_for_feasible_pareto": False,
                "used_for_hidden_constraint": True,
            }
        feasible = quality >= self.tau_q
        return {
            **base,
            "status": "complete",
            "failure_kind": None,
            "failure_reason": None,
            "objective_vector": objectives,
            "metrics": metrics,
            "quality": quality,
            "quality_ratio_to_baseline": (
                quality / self.baseline_quality
                if self.baseline_quality > 0.0
                else None
            ),
            "quality_feasible": feasible,
            # 完整但质量不达标的 Trial 仍训练全部四个代理。
            "used_for_performance_surrogates": True,
            "used_for_quality_surrogate": True,
            "used_for_feasible_pareto": feasible,
            "used_for_hidden_constraint": True,
        }

    def run(self) -> dict[str, Any]:
        history = self._load_or_create_history()
        while len(history["trials"]) < self.config.rounds:
            iteration = len(history["trials"]) + 1
            trial_id = f"trial_{iteration:04d}"
            self._quality_probability_ranked_pool = []
            def propose(excluded):
                candidate, selection = (self._suggest(history, exclude=excluded)
                                        if self.config.quality_probability_gate_enabled
                                        else self._suggest(history))
                if self.candidate_repairer is None:
                    constraint_result = self.constraint_handler.repair(candidate)
                    constraint_check = constraint_result.to_dict()
                    repair_valid = constraint_result.valid
                    repair_errors = list(constraint_result.validation_errors)
                    candidate = constraint_result.candidate
                else:
                    constraint_check = dict(self.candidate_repairer(candidate))
                    repair_valid = constraint_check.get("valid") is True
                    repair_errors = list(constraint_check.get("validation_errors") or [])
                    candidate = dict(constraint_check.get("candidate") or {})
                if not repair_valid:
                    raise RuntimeError(
                        "selected candidate failed the pre-execution known-constraint "
                        f"check: {repair_errors}"
                    )
                return {"candidate": candidate, "selection": selection,
                        "known_constraint_check": constraint_check}

            if self.config.quality_probability_gate_enabled:
                gate = QualityProbabilityGate(
                    encode=self.space.encode, key=self.space.key,
                    threshold=self.config.quality_probability_threshold,
                    max_attempts=self.config.quality_probability_max_attempts,
                    start_after_rounds=self.config.quality_probability_start_after_rounds,
                    audit_path=self.output_dir / "quality_probability_gate.json",
                    model=self.quality_probability_model, random_seed=self.config.random_seed,
                )
                recommendation = gate.select(history=history,
                    cold_start=len(complete_records(history)) < self.initial_design_trials,
                    propose=propose)
            else:
                recommendation = propose(set())
            candidate = recommendation["candidate"]
            selection = recommendation["selection"]
            constraint_check = recommendation["known_constraint_check"]
            started = time.monotonic()
            try:
                execution = self.executor.execute(
                    candidate,
                    active_parameters=sorted(
                        self.space.active_parameter_names(candidate)
                    ),
                    trial_id=trial_id,
                )
            except Exception as exc:
                execution = TrialExecution(
                    status="runtime_failed",
                    result=None,
                    result_path=None,
                    failure_kind="executor_exception",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            record = self._record_execution(
                candidate=candidate,
                selection=selection,
                constraint_check=constraint_check,
                execution=execution,
                trial_id=trial_id,
                elapsed_seconds=time.monotonic() - started,
            )
            history["trials"].append(record)
            write_json_atomic(self.history_path, history)
        result = result_from_history(history)
        result["result_path"] = str(self.result_path)
        write_json_atomic(self.result_path, result)
        return result
