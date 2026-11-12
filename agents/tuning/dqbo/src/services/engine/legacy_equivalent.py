"""Agent 内置的原 BO 等价执行器；联合采样质量约束 cEHVI。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from agents.tuning.dqbo.src.services.failure_taxonomy import summarize_runtime_failure

from ._hebo.optimizers.general import GeneralBO
from ._legacy_bo.arguments import add_args
from ._legacy_bo.constraints import fit_hidden_constraint_model
from ._legacy_bo.history import filter_history_to_space
from ._legacy_bo.parameters import (
    default_combination,
    normalize_conditional_combination,
    rec_to_combination,
)
from ._legacy_bo.search_space import build_design_space
from .constraints import (
    COMPRESSED_KV_CACHE_DTYPES,
    MIN_COMPRESSED_KV_BLOCK_SIZE,
    KnownConstraintHandler,
)
from .executor import TrialExecution, TrialExecutor
from .flock.trial.arguments import parse_args as parse_trial_args
from .flock.trial.configuration import apply_baseline_config_to_namespace
from .history import (
    HISTORY_SCHEMA,
    current_run_execution_count,
    evaluation_count,
    estimated_performance_records,
    load_history,
    native_bo_iteration_count,
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
    quality_margin_diagnostics,
    quality_threshold,
)
from .optimizer import OptimizerConfig
from .quality_probability import (
    QualityProbabilityGate, QualityProbabilityModel, measured_quality_label, sync_quality_gate_history,
)
from .space import FixedSearchSpace


SURROGATE_NAMES = (
    "qps",
    "mean_latency",
    "p95_latency",
    "quality",
)


def _load_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _json_value(value: Any) -> Any:
    """把 pandas/NumPy 标量转换为可稳定落盘的 Python 值。"""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(name): _json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _temporary_filter_fixed_inputs(
    space: Any,
    predicate: Mapping[str, Any],
    *,
    numeric_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Push safe equality bounds into native HEBO sampling.

    The temporary predicate remains the authoritative post-repair constraint.
    This optimization fixes categorical, boolean and exact numeric HEBO
    coordinates that the predicate already requires, avoiding exponentially
    sparse rejection sampling for narrow one-round guidance regions.
    """

    aliases = {"tensor_parallel_size": "tp"}
    fixable = set(space.enum_names)
    fixable.update(
        name
        for name in space.numeric_names
        if type(space.paras[name]).__name__ == "BoolPara"
    )
    fixed: dict[str, Any] = {}
    for raw_name, raw_condition in predicate.items():
        name = aliases.get(str(raw_name), str(raw_name))
        if name not in fixable or not isinstance(raw_condition, Mapping):
            continue
        if set(raw_condition) == {"eq"}:
            fixed[name] = _json_value(raw_condition["eq"])
    for name, bounds in (numeric_bounds or {}).items():
        lower, upper = map(float, bounds)
        if not math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
            continue
        parameter = space.paras[name]
        raw_value = np.asarray(
            parameter.inverse_transform(np.asarray([lower], dtype=float))
        ).reshape(-1)[0]
        fixed[name] = _json_value(raw_value)
    return fixed


def _temporary_filter_numeric_bounds(
    space: Any, predicate: Mapping[str, Any]
) -> dict[str, tuple[float, float]]:
    """Project temporary min/max conditions into native HEBO coordinates.

    This lets scrambled-Sobol initial design draw directly from the temporary
    numeric subspace.  The full predicate remains the authoritative check after
    conditional repair, so unsupported operators still fail closed there.
    """

    aliases = {"tensor_parallel_size": "tp"}
    bounded: dict[str, tuple[float, float]] = {}
    for raw_name, raw_condition in predicate.items():
        name = aliases.get(str(raw_name), str(raw_name))
        if name not in space.numeric_names or not isinstance(raw_condition, Mapping):
            continue
        parameter = space.paras[name]
        if type(parameter).__name__ == "BoolPara":
            continue
        if not ({"eq", "min", "max"} & set(raw_condition)):
            continue

        lower = float(parameter.opt_lb)
        upper = float(parameter.opt_ub)

        def transformed(value: Any) -> float:
            # The canonical executable configuration stores GPU memory as a
            # fraction, while the legacy-equivalent HEBO coordinate stores an
            # integer percentage.  Observation filters intentionally use the
            # canonical/Range value so post-repair validation and source Trial
            # evidence stay on the same scale.
            native_value = (
                float(value) * 100.0
                if name == "gpu_memory_utilization"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and abs(float(value)) <= 1.0
                else value
            )
            result = parameter.transform(np.asarray([native_value]))
            return float(np.asarray(result, dtype=float).reshape(-1)[0])

        if "eq" in raw_condition:
            lower = upper = transformed(raw_condition["eq"])
        else:
            if "min" in raw_condition:
                lower = max(lower, transformed(raw_condition["min"]))
            if "max" in raw_condition:
                upper = min(upper, transformed(raw_condition["max"]))
        if parameter.is_discrete_after_transform:
            lower = float(math.ceil(lower))
            upper = float(math.floor(upper))
        lower = max(float(parameter.opt_lb), lower)
        upper = min(float(parameter.opt_ub), upper)
        if lower > upper:
            raise ValueError(
                f"temporary numeric interval is empty after domain projection: {name}"
            )
        bounded[name] = (lower, upper)
    return bounded


def _temporary_numeric_bounds_receipt(
    space: Any,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Expose both user-facing and transformed native Sobol bounds."""

    receipt: dict[str, dict[str, Any]] = {}
    for name, (lower, upper) in bounds.items():
        parameter = space.paras[name]
        raw_values = parameter.inverse_transform(
            np.asarray([lower, upper], dtype=float)
        )
        raw_values = np.asarray(raw_values).reshape(-1)
        receipt[name] = {
            "lower": _json_value(raw_values[0]),
            "upper": _json_value(raw_values[1]),
            "native_transformed_lower": lower,
            "native_transformed_upper": upper,
        }
    return receipt


def build_legacy_namespace(
    *,
    trial_argv: Sequence[str],
    parameter_results_path: str | Path,
    domain_spec_path: str | Path,
    baseline_config_path: str | Path,
    gpu_count: int = 2,
) -> argparse.Namespace:
    """用 Agent 内部参数实现构造与原 BO 相同的 search-space namespace。"""

    parser = add_args(argparse.ArgumentParser(add_help=False))
    namespace = parser.parse_args([])
    trial_namespace = parse_trial_args([str(value) for value in trial_argv])
    # A fixed context length is still part of the measured execution contract,
    # even when it is not a searched BO coordinate.
    namespace.max_model_len = getattr(trial_namespace, "max_model_len", None)
    # Preserve the baseline executor even when it is not a searched dimension.
    # Otherwise baseline application only writes an extra CLI argument, while
    # default_combination sees None and loses the fixed execution contract.
    namespace.distributed_executor_backend = getattr(
        trial_namespace, "distributed_executor_backend", None
    )
    for name, value in vars(trial_namespace).items():
        if hasattr(namespace, name):
            setattr(namespace, name, value)
    baseline_path = Path(baseline_config_path).expanduser().resolve()
    apply_baseline_config_to_namespace(namespace, _load_object(baseline_path))
    namespace.screened_parameter_results = str(
        Path(parameter_results_path).expanduser().resolve()
    )
    namespace.parameter_domain_spec = str(Path(domain_spec_path).expanduser().resolve())
    namespace.baseline_config = str(baseline_path)
    namespace.gpu_num = int(gpu_count)
    namespace.detected_gpu_num = int(gpu_count)
    namespace.num_obj = 3
    return namespace


def _row_inside_space(space: Any, row: Mapping[str, Any]) -> bool:
    try:
        frame = pd.DataFrame.from_records([dict(row)], columns=space.para_names)
        numeric, categorical = space.transform(frame)
        encoded = np.hstack([numeric.numpy(), categorical.numpy()])
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    lower = space.opt_lb.numpy().reshape(1, -1)
    upper = space.opt_ub.numpy().reshape(1, -1)
    return bool(
        np.isfinite(encoded).all()
        and (encoded >= lower).all()
        and (encoded <= upper).all()
    )


def _hebo_row_key(space: Any, row: Mapping[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in space.para_names:
        value = _json_value(row[name])
        if isinstance(value, float):
            value = round(value, 12)
        values.append(value)
    return tuple(values)


def _resolved_attention_backend(result: Mapping[str, Any]) -> str | None:
    """Resolve the measured backend needed by deterministic runtime constraints."""

    explicit = result.get("resolved_attention_backend")
    if explicit:
        return str(explicit).strip().upper()
    logs = result.get("logs")
    log_path = logs.get("vllm") if isinstance(logs, Mapping) else None
    if not isinstance(log_path, str):
        return None
    path = Path(log_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"Using\s+([A-Z0-9_]+)\s+attention backend", line)
            if match:
                return match.group(1)
    except OSError:
        return None
    return None


def _constraint_check(
    raw: Mapping[str, Any], repaired: Mapping[str, Any]
) -> dict[str, Any]:
    changes = [
        {
            "parameter": name,
            "before": _json_value(raw.get(name)),
            "after": _json_value(value),
        }
        for name, value in repaired.items()
        if raw.get(name) != value
    ]
    return {
        "valid": True,
        "validation_errors": [],
        "repairs": changes,
        "implementation": "agent_internal_legacy_normalize_conditional_combination",
    }


_REPRESENTATION_ARROW = re.compile(
    r"^\s*quantization\s*==\s*(['\"])(?P<quantization>[^'\"]+)\1\s*"
    r"->\s*dtype\s+must\s+be\s*(['\"])(?P<dtype>[^'\"]+)\3\s*$"
)


def _apply_declared_representation_template(
    candidate: Mapping[str, Any], domain_spec_path: str | Path
) -> dict[str, Any]:
    """Apply the quantization/dtype arrows already frozen by Range.

    The legacy normalizer resolves ``auto`` and quantized model dtypes, but it
    historically did not enforce the unquantized ``none -> bfloat16`` arrow.
    Read the declaration instead of duplicating a hardware-specific policy.
    """

    prepared = dict(candidate)
    payload = _load_object(domain_spec_path)
    parameters = payload.get("parameters")
    if not isinstance(parameters, list):
        return prepared
    declared: dict[str, str] = {}
    allowed_dtypes: set[Any] = set()
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or parameter.get("name") != "dtype":
            continue
        domain = parameter.get("domain")
        if isinstance(domain, Mapping) and isinstance(domain.get("values"), list):
            allowed_dtypes.update(domain["values"])
        conditions = parameter.get("activation_conditions")
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            match = _REPRESENTATION_ARROW.fullmatch(str(condition))
            if match is not None:
                declared[match.group("quantization")] = match.group("dtype")
    quantization = str(prepared.get("quantization") or "none")
    expected = declared.get(quantization)
    if expected is not None and (not allowed_dtypes or expected in allowed_dtypes):
        prepared["dtype"] = expected
    return prepared


def candidate_matches_filter(
    candidate: Mapping[str, Any], predicate: Mapping[str, Any]
) -> bool:
    """Evaluate an Observation predicate against a canonical candidate."""

    def same(left: Any, right: Any) -> bool:
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        return left == right

    for name, raw in predicate.items():
        if name not in candidate:
            return False
        actual = candidate[name]
        condition = raw if isinstance(raw, Mapping) else {"eq": raw}
        if set(condition) - {"eq", "in", "not_in", "min", "max"}:
            return False
        if "eq" in condition and not same(actual, condition["eq"]):
            return False
        if "in" in condition and not any(
            same(actual, value) for value in condition["in"]
        ):
            return False
        if "not_in" in condition and any(
            same(actual, value) for value in condition["not_in"]
        ):
            return False
        if "min" in condition and float(actual) < float(condition["min"]):
            return False
        if "max" in condition and float(actual) > float(condition["max"]):
            return False
    return True


def candidate_filter_violations(
    candidate: Mapping[str, Any], predicate: Mapping[str, Any]
) -> list[str]:
    """Return the predicate fields a candidate does not satisfy."""

    return sorted(
        str(name)
        for name, condition in predicate.items()
        if not candidate_matches_filter(candidate, {str(name): condition})
    )


def repair_with_legacy_constraints(
    *,
    trial_argv: Sequence[str],
    parameter_results_path: str | Path,
    domain_spec_path: str | Path,
    baseline_config_path: str | Path,
    candidate: Mapping[str, Any],
    gpu_count: int = 2,
) -> dict[str, Any]:
    """公开约束 tool 与真实 BO 共用原 ``normalize``/``rec_to`` 实现。"""

    namespace = build_legacy_namespace(
        trial_argv=trial_argv,
        parameter_results_path=parameter_results_path,
        domain_spec_path=domain_spec_path,
        baseline_config_path=baseline_config_path,
        gpu_count=gpu_count,
    )
    space, _ = build_design_space(namespace, int(gpu_count), 4096)
    original_raw = {
        ("tp" if str(name) == "tensor_parallel_size" else str(name)): value
        for name, value in candidate.items()
    }
    prepared_candidate = _apply_declared_representation_template(
        candidate, domain_spec_path
    )
    raw = {
        ("tp" if str(name) == "tensor_parallel_size" else str(name)): value
        for name, value in prepared_candidate.items()
    }
    if set(space.para_names).issubset(raw):
        # The composite repair callback receives the executable vLLM
        # combination (gpu_memory_utilization=0.9), while public callers may
        # also supply a HEBO row (gpu_memory_utilization=90).  Normalize both
        # through the same history-to-space conversion before rec_to_combination
        # so aliases and scaled parameters have one representation boundary.
        observed_input = filter_history_to_space([raw], space, namespace)[0]
        recommendation = pd.DataFrame.from_records(
            [observed_input],
            columns=space.para_names,
        )
        repaired = rec_to_combination(recommendation, namespace)
    else:
        repaired = normalize_conditional_combination(
            {**default_combination(namespace), **raw}, namespace
        )
    observed_row = filter_history_to_space([repaired], space, namespace)[0]
    valid = _row_inside_space(space, observed_row)
    check = _constraint_check(original_raw, repaired)
    check["valid"] = valid
    check["validation_errors"] = (
        [] if valid else ["repaired candidate is outside the fixed HEBO domain"]
    )
    return {
        **check,
        "candidate": _json_value(repaired),
        "candidate_bo_names": _json_value(observed_row),
        "effective_hebo_parameter_names": list(space.para_names),
    }


class LegacyEquivalentQualityBO:
    """沿用 GeneralBO 主循环，并以配置的 MC 数计算 EHVI acquisition。"""

    def __init__(
        self,
        *,
        executor: TrialExecutor,
        trial_argv: Sequence[str],
        baseline_config_path: str | Path,
        baseline_result_path: str | Path,
        parameter_results_path: str | Path,
        domain_spec_path: str | Path,
        output_dir: str | Path,
        config: OptimizerConfig,
        gpu_count: int = 2,
        max_sequence_length: int = 4096,
        warmup_complete_observations: int | None = None,
        candidate_repairer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        quality_probability_model: QualityProbabilityModel | None = None,
    ) -> None:
        config.validate()
        if config.initial_random_trials is not None:
            raise ValueError(
                "legacy-equivalent BO forbids initial_random_trials overrides; "
                "the original rule is 1 + HEBO search dimensions"
            )
        self.executor = executor
        self.trial_argv = [str(value) for value in trial_argv]
        self.baseline_config_path = str(Path(baseline_config_path).resolve())
        self.baseline_result_path = str(Path(baseline_result_path).resolve())
        self.parameter_results_path = str(Path(parameter_results_path).resolve())
        self.domain_spec_path = str(Path(domain_spec_path).resolve())
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "bo_history.json"
        self.result_path = self.output_dir / "candidate_optimizer_result.json"
        self.rec_history_path = self.output_dir / "rec_history.json"
        self.config = config
        if (
            warmup_complete_observations is not None
            and int(warmup_complete_observations) < 1
        ):
            raise ValueError("warmup_complete_observations must be positive")
        self.warmup_complete_observations = (
            int(warmup_complete_observations)
            if warmup_complete_observations is not None
            else None
        )
        self.candidate_repairer = candidate_repairer
        self.quality_probability_model = quality_probability_model
        self.max_sequence_length = int(max_sequence_length)
        self.args = build_legacy_namespace(
            trial_argv=self.trial_argv,
            parameter_results_path=self.parameter_results_path,
            domain_spec_path=self.domain_spec_path,
            baseline_config_path=self.baseline_config_path,
            gpu_count=gpu_count,
        )
        self.space, self.min_world_size = build_design_space(
            self.args, int(gpu_count), self.max_sequence_length
        )
        self.baseline_result = _load_object(self.baseline_result_path)
        from .repair_timeout import configure_repair_query_timeout

        self.query_timeout_policy = configure_repair_query_timeout(
            self.executor, config, self.baseline_result, self.output_dir,
        )
        (
            self.baseline_objectives,
            self.baseline_quality,
            self.baseline_metrics,
        ) = extract_complete_trial_metrics(
            self.baseline_result,
            quality_metric=config.quality_metric,
            quality_thresholds_json=config.quality_thresholds_json,
            min_mean_f1=config.min_mean_f1,
            max_relative_error=config.max_relative_error,
        )
        self.fixed_search_space = FixedSearchSpace.from_json(self.domain_spec_path)
        self.cacheblend_require_eager = any(
            c.get("id") == "cacheblend_compatibility" and c.get("require_enforce_eager")
            for c in self.fixed_search_space.joint_constraints
        )
        attention_backend = _resolved_attention_backend(self.baseline_result)
        self.repair_scope = None
        if self.config.repair_bo_enabled:
            from agents.tuning.repair.src.services.repair_bo import RepairScope, repair_scope_runtime_constraints

            self.repair_scope = RepairScope.from_space(
                self.fixed_search_space,
                fields=self.config.repair_scope_fields,
                scope_version=self.config.repair_scope_version,
                runtime_constraints=repair_scope_runtime_constraints(
                    {
                        "resolved_attention_backend": attention_backend,
                        "source": "baseline_result_or_vllm_log",
                        "uncompressed_kv_model_dtype_alignment": (
                            attention_backend == "FLASH_ATTN"
                        ),
                    }
                ),
            )
        self.known_constraint_handler = KnownConstraintHandler(
            self.fixed_search_space,
            model_path=self.args.model_path,
            awq_model_path=self.args.awq_model_path,
            gptq_model_path=self.args.gptq_model_path,
            uncompressed_kv_model_dtype_alignment=(
                attention_backend == "FLASH_ATTN"
            ),
            alignment_scope=attention_backend,
        )
        self.tau_q = (
            1.0 - config.quality_alpha
            if config.quality_metric == "sembench_summary.quality_gate_passed"
            else quality_threshold(self.baseline_quality, config.quality_alpha)
        )
        self.baseline_combination = normalize_conditional_combination(
            default_combination(self.args), self.args
        )
        self.baseline_row = filter_history_to_space(
            [self.baseline_combination], self.space, self.args
        )[0]
        self.baseline_inside_domain = _row_inside_space(self.space, self.baseline_row)
        self._representation_hebo_dtype: dict[str, Any] = {}
        for quantization, declared_dtype in (
            self.fixed_search_space.representation_dtype_template().items()
        ):
            normalized = normalize_conditional_combination(
                {
                    **default_combination(self.args),
                    "quantization": quantization,
                    "dtype": declared_dtype,
                },
                self.args,
            )
            observed = filter_history_to_space(
                [normalized], self.space, self.args
            )[0]
            if "dtype" in observed:
                self._representation_hebo_dtype[quantization] = observed["dtype"]
        self._automatic_kv_hebo_dtype: dict[tuple[Any, Any], Any] = {}
        if all(name in self.space.paras for name in ("quantization", "dtype", "kv_cache_dtype")):
            quantizations = self.space.paras["quantization"].categories
            dtypes = self.space.paras["dtype"].categories
            for quantization in quantizations:
                for dtype in dtypes:
                    normalized = normalize_conditional_combination(
                        {
                            **default_combination(self.args),
                            "quantization": quantization,
                            "dtype": dtype,
                            "kv_cache_dtype": "auto",
                        },
                        self.args,
                    )
                    observed = filter_history_to_space(
                        [normalized], self.space, self.args
                    )[0]
                    self._automatic_kv_hebo_dtype[(quantization, dtype)] = observed[
                        "kv_cache_dtype"
                    ]

        if config.conditional_model_enabled:
            from .conditional_space import ConditionalSpace
            self.space.conditional_space = ConditionalSpace(
                self.space, defaults={**self.baseline_combination, "max_model_len": self.baseline_combination.get("max_model_len") or getattr(self.args, "max_model_len", None)},
                gpu_count=int(self.args.detected_gpu_num or self.args.gpu_num),
                ray_v2=os.environ.get("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0") == "1",
                align_uncompressed=self.known_constraint_handler.uncompressed_kv_model_dtype_alignment,
                representation_dtypes=self._representation_hebo_dtype,
                cacheblend_allow_prefix_caching=self.args.cacheblend_allow_prefix_caching,
                cacheblend_require_eager=self.cacheblend_require_eager,
            )

    def _project_one_feasible_row(
        self, raw_row: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Map one HEBO genotype to its declared, executable equivalence class."""

        frame = pd.DataFrame.from_records(
            [dict(raw_row)], columns=self.space.para_names
        )
        executable = rec_to_combination(frame, self.args)
        declared = self.known_constraint_handler.repair(
            self.fixed_search_space.bo_to_canonical(executable)
        )
        if not declared.valid:
            # Screening constrains whole tuples. Decode an arbitrary HEBO row
            # onto a retained alternative, including its background bindings;
            # independent category sampling must not resurrect deleted tuples.
            for constraint in self.fixed_search_space.joint_constraints:
                if constraint.get("id") != "screened_mechanism_combinations":
                    continue
                alternatives = constraint.get("alternatives", [])
                def distance(alternative):
                    values = {**alternative.get("mechanisms", {}), **alternative.get("required_bindings", {})}
                    return sum(declared.candidate.get(k) != v for k, v in values.items())
                for alternative in sorted(alternatives, key=distance):
                    values = {**alternative.get("mechanisms", {}), **alternative.get("required_bindings", {})}
                    trial = self.known_constraint_handler.repair({**declared.candidate, **values})
                    if trial.valid:
                        declared = trial
                        break
                if declared.valid:
                    break
        if not declared.valid:
            raise RuntimeError(
                "fixed feasible-space projection has no Range-valid result: "
                f"{list(declared.validation_errors)}"
            )
        prepared = dict(executable)
        prepared.update(
            self.fixed_search_space.canonical_to_bo(declared.candidate)
        )
        executable = normalize_conditional_combination(prepared, self.args)
        observed = filter_history_to_space(
            [executable], self.space, self.args
        )[0]
        if not _row_inside_space(self.space, observed):
            raise RuntimeError(
                "fixed feasible-space projection produced an out-of-domain HEBO row"
            )
        final_check = self.known_constraint_handler.repair(
            self.fixed_search_space.bo_to_canonical(executable)
        )
        if not final_check.valid:
            raise RuntimeError(
                "legacy projection reintroduced a declared constraint violation: "
                f"errors={list(final_check.validation_errors)}, "
                f"repairs={[dict(item) for item in final_check.repairs]}"
            )
        return dict(observed)

    def _project_feasible_candidates_once(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorized decoder for the explicit hard-constraint manifold."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("feasible candidate projection expects a DataFrame")
        missing = [name for name in self.space.para_names if name not in frame.columns]
        if missing:
            raise ValueError(
                "feasible candidate projection omitted HEBO columns: "
                + ", ".join(missing)
            )
        result = frame.loc[:, self.space.para_names].copy().reset_index(drop=True)
        screened = [c for c in self.fixed_search_space.joint_constraints
                    if c.get("id") == "screened_mechanism_combinations"]
        single_screened_tuple = False
        if len(screened) == 1 and len(screened[0].get("alternatives", [])) == 1:
            alternative = screened[0]["alternatives"][0]
            if not alternative.get("required_bindings"):
                aliases = {"tensor_parallel_size": "tp", "chunked_prefill": "enable_chunked_prefill"}
                assignments = {}
                single_screened_tuple = True
                for name, value in alternative["mechanisms"].items():
                    parameter = self.fixed_search_space.by_name[name]
                    column = aliases.get(parameter.bo_name, parameter.bo_name)
                    if column in result.columns:
                        assignments[column] = value
                    elif parameter.decision != "fixed" or parameter.fixed_value != value:
                        single_screened_tuple = False
                        break
                if single_screened_tuple:
                    # Exactly the same singleton constraint as the row decoder;
                    # keep this frequent case vectorized during acquisition.
                    for column, value in assignments.items():
                        result[column] = value

        # The legacy HEBO coordinate is an integer percentage even when Range
        # expresses the executable value as a [0,1] float.
        if "gpu_memory_utilization" in result.columns:
            result["gpu_memory_utilization"] = (
                result["gpu_memory_utilization"].astype(float).round().astype(int)
            )

        if {"quantization", "dtype"}.issubset(result.columns):
            for quantization, dtype in self._representation_hebo_dtype.items():
                result.loc[result["quantization"] == quantization, "dtype"] = dtype

        if {"quantization", "dtype", "kv_cache_dtype"}.issubset(result.columns):
            automatic = result["kv_cache_dtype"] == "auto"
            for (quantization, dtype), kv_dtype in (
                self._automatic_kv_hebo_dtype.items()
            ):
                result.loc[
                    automatic
                    & (result["quantization"] == quantization)
                    & (result["dtype"] == dtype),
                    "kv_cache_dtype",
                ] = kv_dtype

        if {"speculative_method", "num_speculative_tokens"}.issubset(result.columns):
            result.loc[
                result["speculative_method"] == "none", "num_speculative_tokens"
            ] = self.baseline_combination.get("num_speculative_tokens") or int(
                self.args.num_speculative_tokens_lb
            )

        if "cacheblend_mode" in result.columns:
            enabled = result["cacheblend_mode"] == "on"
            if getattr(self, "cacheblend_require_eager", False) and "enforce_eager" in result.columns:
                result.loc[enabled, "enforce_eager"] = True
            if "enable_chunked_prefill" in result.columns:
                result.loc[enabled, "enable_chunked_prefill"] = False
            for child in ("lmcache_chunk_size", "cacheblend_blend_min_tokens",
                          "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios"):
                if child in result.columns:
                    result.loc[~enabled, child] = self.baseline_combination[child]
            if "enable_prefix_caching" in result.columns and not self.args.cacheblend_allow_prefix_caching:
                result.loc[enabled, "enable_prefix_caching"] = False
            if {"dtype", "kv_cache_dtype"}.issubset(result.columns):
                half = result["dtype"].isin({"half", "float16"})
                result.loc[enabled & half, "kv_cache_dtype"] = "float16"
                result.loc[enabled & ~half, "kv_cache_dtype"] = "bfloat16"
            if "block_size" in result.columns:
                result.loc[enabled & (result["block_size"].astype(int) < 16), "block_size"] = 16

        if "max_num_batched_tokens" in result.columns:
            full_prefill = pd.Series(False, index=result.index)
            if "cacheblend_mode" in result.columns:
                full_prefill |= result["cacheblend_mode"] == "on"
            if "enable_chunked_prefill" in result.columns:
                full_prefill |= ~result["enable_chunked_prefill"].astype(bool)
            limit = self.baseline_combination.get("max_model_len") or getattr(self.args, "max_model_len", None) or 32768
            limits = result["max_model_len"].fillna(limit) if "max_model_len" in result.columns else pd.Series(limit, index=result.index)
            result.loc[full_prefill, "max_num_batched_tokens"] = np.maximum(result.loc[full_prefill, "max_num_batched_tokens"], limits.loc[full_prefill])

        # DP-only child settings are inactive when TP occupies every GPU.
        if "tp" in result.columns:
            total_gpu_count = int(
                self.args.detected_gpu_num
                if self.args.detected_gpu_num is not None
                else self.args.gpu_num
            )
            dp_inactive = result["tp"].astype(int) >= total_gpu_count
            if "data_parallel_backend" in result.columns:
                result.loc[dp_inactive, "data_parallel_backend"] = "mp"
            if "disable_nccl_for_dp_synchronization" in result.columns:
                result.loc[
                    dp_inactive, "disable_nccl_for_dp_synchronization"
                ] = False

        if "async_scheduling" in result.columns and os.environ.get(
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0"
        ) != "1":
            ray = pd.Series(False, index=result.index)
            for backend in ("data_parallel_backend", "distributed_executor_backend"):
                if backend in result.columns:
                    ray |= result[backend] == "ray"
            result.loc[ray, "async_scheduling"] = False

        if {
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
        }.issubset(result.columns):
            result.loc[
                ~result["enable_prefix_caching"].astype(bool),
                "prefix_caching_hash_algo",
            ] = "sha256"

        if {"max_num_batched_tokens", "max_num_seqs"}.issubset(result.columns):
            invalid_budget = (
                result["max_num_batched_tokens"].astype(int)
                < result["max_num_seqs"].astype(int)
            )
            result.loc[invalid_budget, "max_num_batched_tokens"] = result.loc[
                invalid_budget, "max_num_seqs"
            ].astype(int)

        if {"kv_cache_dtype", "block_size"}.issubset(result.columns):
            compressed = result["kv_cache_dtype"].isin(
                COMPRESSED_KV_CACHE_DTYPES
            )
            too_small = result["block_size"].astype(int) < MIN_COMPRESSED_KV_BLOCK_SIZE
            result.loc[compressed & too_small, "block_size"] = (
                MIN_COMPRESSED_KV_BLOCK_SIZE
            )

        if (
            self.known_constraint_handler.uncompressed_kv_model_dtype_alignment
            and {"dtype", "kv_cache_dtype"}.issubset(result.columns)
        ):
            half_model = result["dtype"].isin({"half", "float16"})
            bfloat_model = result["dtype"] == "bfloat16"
            result.loc[
                half_model & (result["kv_cache_dtype"] == "bfloat16"),
                "kv_cache_dtype",
            ] = "float16"
            result.loc[
                bfloat_model & (result["kv_cache_dtype"] == "float16"),
                "kv_cache_dtype",
            ] = "bfloat16"
        if screened and not single_screened_tuple:
            result = pd.DataFrame.from_records(
                [self._project_one_feasible_row(row) for row in result.to_dict(orient="records")],
                columns=self.space.para_names,
            )
        return result.loc[:, self.space.para_names].reset_index(drop=True)

    def project_feasible_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Canonicalize every candidate before surrogate/acquisition evaluation."""

        projected = self._project_feasible_candidates_once(frame)
        # This cheap vectorized second pass guards the decoder's defining fixed-
        # point contract. The complete legacy implementation is checked once on
        # the final returned candidate by ``_assert_feasible_observed_row``.
        second_pass = self._project_feasible_candidates_once(projected)
        if [
            _hebo_row_key(self.space, row)
            for row in projected.to_dict(orient="records")
        ] != [
            _hebo_row_key(self.space, row)
            for row in second_pass.to_dict(orient="records")
        ]:
            raise RuntimeError("feasible candidate projection is not idempotent")
        return projected

    def _assert_feasible_observed_row(self, row: Mapping[str, Any]) -> None:
        original = pd.DataFrame.from_records(
            [dict(row)], columns=self.space.para_names
        ).reset_index(drop=True)
        projected = self.project_feasible_candidates(original)
        if _hebo_row_key(
            self.space, original.iloc[0].to_dict()
        ) != _hebo_row_key(self.space, projected.iloc[0].to_dict()):
            raise RuntimeError(
                "BO returned a non-canonical point outside the feasible search space"
            )
        authoritative = self._project_one_feasible_row(
            original.iloc[0].to_dict()
        )
        if _hebo_row_key(self.space, original.iloc[0].to_dict()) != _hebo_row_key(
            self.space, authoritative
        ):
            raise RuntimeError(
                "fast feasible decoder disagrees with authoritative legacy constraints: "
                + str({name: (original.iloc[0][name], authoritative[name])
                       for name in self.space.para_names
                       if original.iloc[0][name] != authoritative[name]})
            )

    @property
    def quality_surrogate_enabled(self) -> bool:
        """RepairBO uses measured quality only, including when resuming old configs."""
        return (
            not self.config.repair_bo_enabled
            and self.config.quality_probability_affects_acquisition
        )

    def _new_optimizer(self) -> GeneralBO:
        return GeneralBO(
            space=self.space,
            num_obj=3,
            num_model_constr=1,
            num_hard_constr=2 if self.config.project_before_acquisition else 0,
            num_hidden_constr=1,
            ref_point=np.asarray(
                [-value for value in self.baseline_objectives], dtype=float
            ),
            model_config={
                "optimizer": "adam",
                "base_model_name": "gpy",
                "space": self.space,
            },
            use_noise=True,
            max_sequence_length=math.ceil(math.log2(self.max_sequence_length)),
            rand_sample=self.warmup_complete_observations,
            model_constraints_affect_candidate_generation=False,
            constrained_ehvi_mc_samples=self.config.monte_carlo_samples,
            quality_probability_affects_acquisition=(
                self.quality_surrogate_enabled
            ),
            candidate_supplement_per_source=(self.config.repair_candidate_supplement_per_source
                                             if self.config.repair_bo_enabled else 0),
            candidate_transform=(self.project_feasible_candidates
                                 if self.config.project_before_acquisition else None),
        )

    def _baseline_observation(self) -> dict[str, Any]:
        margin = quality_margin_diagnostics(
            self.baseline_result,
            quality_metric=self.config.quality_metric,
            quality_thresholds_json=self.config.quality_thresholds_json,
            min_mean_f1=self.config.min_mean_f1,
            max_relative_error=self.config.max_relative_error,
            tau_q=self.tau_q,
            quality=self.baseline_quality,
        )
        observation = {
            "trial_id": "baseline",
            "source": "baseline",
            "status": "complete",
            "candidate": dict(self.baseline_combination),
            "candidate_bo_names": dict(self.baseline_row),
            "objective_vector": list(self.baseline_objectives),
            "metrics": dict(self.baseline_metrics),
            "quality": self.baseline_quality,
            "quality_ratio_to_baseline": (
                1.0 if self.baseline_quality > 0.0 else None
            ),
            "quality_feasible": self.baseline_quality >= self.tau_q,
            "quality_margin": margin["worst_quality_margin"],
            "quality_margins": margin["quality_margins"],
            "repair_signature": (
                self.repair_scope.signature(
                    self.baseline_combination, self.fixed_search_space
                )
                if self.repair_scope is not None
                else None
            ),
            "used_for_surrogates": self.baseline_inside_domain,
            "used_for_feasible_pareto": (
                self.baseline_inside_domain
                and self.baseline_quality >= self.tau_q
            ),
            "used_for_hidden_constraint": False,
            "result_path": self.baseline_result_path,
            "domain_validation_errors": (
                []
                if self.baseline_inside_domain
                else ["baseline is outside fixed HEBO domain"]
            ),
            "known_constraint_check": _constraint_check(
                self.baseline_combination, self.baseline_combination
            ),
        }
        if self.config.repair_bo_enabled:
            observation.update(
                {
                    "evaluation_index": 1,
                    "run_execution_index": None,
                    "bo_iteration": None,
                    "trial_role": "baseline",
                    "execution_mode": "reused_precomputed_result",
                    "executed_in_current_run": False,
                }
            )
        return observation

    def _new_history(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORY_SCHEMA,
            "agent": "candidate_optimizer",
            "implementation": (
                "agent_internal_legacy_general_bo_plus_joint_mc_cehvi"
                if self.quality_surrogate_enabled
                else "agent_internal_legacy_general_bo_plus_performance_only_mc_ehvi"
            ),
            "source_parameter_results": self.parameter_results_path,
            "source_domain_spec": self.domain_spec_path,
            "source_baseline_config": self.baseline_config_path,
            "source_baseline_trial_result": self.baseline_result_path,
            "baseline_execution_policy": {
                "mode": "reuse_precomputed_result",
                "rerun_in_current_tuning_run": False,
                "counts_as_cold_start_observation": True,
                "consumes_round_budget": False,
            },
            "history_path": str(self.history_path),
            "rounds_requested": self.config.rounds,
            "quality_probability_gate": self.config.quality_probability_gate_contract(),
            "query_timeout_policy": self.query_timeout_policy,
            "optimizer_config": {
                "rounds": self.config.rounds,
                "quality_metric": self.config.quality_metric,
                "quality_alpha": self.config.quality_alpha,
                "quality_thresholds_json": self.config.quality_thresholds_json,
                "quality_threshold_source": self.config.quality_threshold_source,
                "min_mean_f1": self.config.min_mean_f1,
                "max_relative_error": self.config.max_relative_error,
                "monte_carlo_samples": self.config.monte_carlo_samples,
                "quality_probability_affects_acquisition": (
                    self.quality_surrogate_enabled
                ),
                "quality_probability_gate_enabled": self.config.quality_probability_gate_enabled,
                "quality_probability_threshold": self.config.quality_probability_threshold,
                "quality_probability_max_attempts": self.config.quality_probability_max_attempts,
                "quality_probability_start_after_rounds": self.config.quality_probability_start_after_rounds,
                "repair_bo_enabled": self.config.repair_bo_enabled,
                "repair_query_timeout_enabled": self.config.repair_query_timeout_enabled,
                "repair_query_timeout_multiplier": self.config.repair_query_timeout_multiplier,
                "bo_mode": self.config.bo_mode,
                "repair_conditional_space_enabled": self.config.repair_conditional_space_enabled,
                "repair_candidate_supplement_per_source": self.config.repair_candidate_supplement_per_source,
                "repair_scope_fields": list(self.config.repair_scope_fields),
                "repair_scope_version": self.config.repair_scope_version,
                "repair_cold_start_evaluations": (
                    self.config.repair_cold_start_evaluations
                ),
                "repair_cold_start_native_trials": (
                    self.config.repair_cold_start_native_trials
                ),
                "repair_hv_epsilon": self.config.repair_hv_epsilon,
                "repair_boundary_utility_epsilon": (
                    self.config.repair_boundary_utility_epsilon
                ),
                "repair_max_rounds": self.config.repair_max_rounds,
                "repair_context_limit": self.config.repair_context_limit,
                "repair_proposal_review_enabled": self.config.repair_proposal_review_enabled,
                "repair_reflect_enabled": self.config.repair_reflect_enabled,
                "repair_proposal_max_revisions": self.config.repair_proposal_max_revisions,
                "repair_performance_candidate_pool_size": (
                    self.config.repair_performance_candidate_pool_size
                ),
            },
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
                "model_target": (
                    "tau_q - quality <= 0" if self.quality_surrogate_enabled else None
                ),
                "stored_constraint_column": "tau_q - quality <= 0; observed Pareto membership",
                "repair_quality_decision": "measured_hard_thresholds_only",
            },
            "surrogate_models": list(
                SURROGATE_NAMES if self.quality_surrogate_enabled else SURROGATE_NAMES[:3]
            ),
            "conditional_space": (
                self.space.conditional_space.contract()
                if hasattr(self.space, "conditional_space") else None
            ),
            "surrogate_implementation": {
                "wrapper": "internal_MultiTaskModel",
                "per_output_model": "internal_GPyGP",
                "independent_outputs": True,
            },
            "acquisition": {
                "name": (
                    "joint_mc_constrained_ehvi"
                    if self.quality_surrogate_enabled
                    else "performance_only_mc_ehvi"
                ),
                "formula": (
                    "E[delta_HV(f(x); P_F) * I(q(x) >= tau_q)]"
                    if self.quality_surrogate_enabled
                    else "E[delta_HV(f(x); P_F)]"
                ),
                "candidate_generator": (
                    "GeneralAcq LCB + EvolutionOpt(pop=100,iters=200) over "
                    + ("canonical feasible configurations" if self.config.project_before_acquisition
                       else "raw configurations")
                ),
                "known_constraint_enforcement": (
                    "deterministic projection before every surrogate/acquisition evaluation and before return"
                    if self.config.project_before_acquisition
                    else "mechanical repair only after complete BO recommendation"
                ),
                "equivalent_conditional_points_deduplicated": self.config.project_before_acquisition,
                "monte_carlo_samples": self.config.monte_carlo_samples,
                "ehvi_matches_original": False,
                "ehvi_formula_matches_original": True,
                "monte_carlo_sample_count_modified_from_original": {
                    "original": 64,
                    "configured": self.config.monte_carlo_samples,
                    "intent": "increase_candidate_ranking_variation",
                },
                "quality_mc_indicator": (
                    self.quality_surrogate_enabled
                ),
                "scalar_ehvi_times_p_feasible": False,
                "quality_mean_hard_threshold": False,
                "quality_surrogate_affects_candidate_generation": False,
                "quality_surrogate_affects_final_ranking": (
                    self.quality_surrogate_enabled
                ),
                "p_run_affects_final_score": False,
                "quality_surrogate_role": (
                    "paired posterior-sample feasibility indicator"
                    if self.quality_surrogate_enabled
                    else "disabled_no_quality_model"
                ),
                "observed_quality_role": "feasible Pareto membership only",
                "pareto_source": "quality_feasible_complete_trials_only",
            },
            "initial_design": {
                "method": "torch_scrambled_sobol_engine",
                "required_complete_observations": (
                    self.warmup_complete_observations or 1 + self.space.num_paras
                ),
                "rule": (
                    "baseline_plus_fixed_7_cold_start_trials"
                    if self.warmup_complete_observations is not None
                    else "1 + effective_HEBO_dimension_count"
                ),
                "effective_hebo_dimension_count": self.space.num_paras,
                "baseline_counts_when_inside_domain": True,
                "runtime_failures_count": False,
            },
            "hidden_constraint": {
                "model": "original_random_forest_regressor",
                "initial_threshold": 0.5,
                "threshold_range": [0.25, 0.75],
                "refit_rule": "original_full_normalized_trial_configuration",
            },
            "baseline_observation": self._baseline_observation(),
            "trials": [],
            "repair_scope": (
                self.repair_scope.contract() if self.repair_scope is not None else None
            ),
            "repair_episodes": [],
            "repair_trigger_events": [],
        }

    def _load_or_create_history(self) -> dict[str, Any]:
        history = load_history(self.history_path)
        if history is None:
            history = self._new_history()
            write_json_atomic(self.history_path, history)
        from .repair_timeout import sync_query_timeout_policy

        if sync_query_timeout_policy(history, self.query_timeout_policy):
            write_json_atomic(self.history_path, history)
        contract = (self.space.conditional_space.contract()
                    if hasattr(self.space, "conditional_space") else None)
        if history.get("conditional_space") != contract:
            history.setdefault("bo_algorithm_changes", []).append({
                "after_trial_count": len(history.get("trials", [])),
                "previous_conditional_space": history.get("conditional_space"),
                "conditional_space": contract,
            })
            history["conditional_space"] = contract
            write_json_atomic(self.history_path, history)
        if self.config.quality_probability_gate_enabled or history.get("quality_probability_gate", {}).get("enabled"):
            if sync_quality_gate_history(history, self._new_history()):
                write_json_atomic(self.history_path, history)
        return history

    def initialize_history(self) -> dict[str, Any]:
        """Materialize the empty, baseline-backed history during prepare."""

        return self._load_or_create_history()

    @staticmethod
    def _rec_history_from_trials(
        trials: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "rec": [dict(trial["candidate_bo_names"])],
                "executed_combination": dict(trial["candidate"]),
                "obj": trial.get("model_target"),
                "rec_time": trial.get("suggest_seconds"),
                "run_time": trial.get("elapsed_seconds"),
            }
            for trial in trials
            if trial.get("used_for_hidden_constraint") is not False
        ]

    def _restore_optimizer(self, history: Mapping[str, Any]) -> GeneralBO:
        optimizer = self._new_optimizer()
        x_rows: list[dict[str, Any]] = []
        y_rows: list[list[float]] = []
        baseline = history.get("baseline_observation")
        if isinstance(baseline, Mapping) and baseline.get("used_for_surrogates"):
            x_rows.append(dict(baseline["candidate_bo_names"]))
            y_rows.append(
                [-float(value) for value in baseline["objective_vector"]]
                + [self.tau_q - float(baseline["quality"])]
            )
        for trial in history.get("trials", []):
            if isinstance(trial, Mapping) and trial.get("status") == "complete":
                x_rows.append(dict(trial["candidate_bo_names"]))
                y_rows.append([float(value) for value in trial["model_target"]])
        if x_rows:
            optimizer.observe(
                pd.DataFrame.from_records(x_rows, columns=self.space.para_names),
                np.asarray(y_rows, dtype=float),
            )
        estimates = estimated_performance_records(history)
        if estimates:
            optimizer.observe_performance(
                pd.DataFrame.from_records([row["candidate_bo_names"] for row in estimates], columns=self.space.para_names),
                np.asarray([[-float(v) for v in row["estimated_objective_vector"]] for row in estimates]),
            )
        attempted_rows = [
            dict(trial["candidate_bo_names"])
            for trial in history.get("trials", [])
            if isinstance(trial, Mapping)
            and isinstance(trial.get("candidate_bo_names"), Mapping)
        ]
        if attempted_rows:
            optimizer.register_attempted(
                pd.DataFrame.from_records(
                    attempted_rows, columns=self.space.para_names
                )
            )
        return optimizer

    def predict_current_performance(self, *, history, candidates):
        """Predict repair configurations with the current native BO posterior.

        Accept executable BO aliases (such as tp) as well as canonical names.
        Parents and Pareto-front points remain measured evidence at audit sites.
        """
        key = json.dumps([
            history.get("baseline_observation"),
            [row for row in history.get("trials", []) if row.get("status") == "complete"
             or row.get("used_for_performance_surrogates") is True],
        ], sort_keys=True, default=str)
        if getattr(self, "_audit_posterior_key", None) != key:
            posterior = self._restore_optimizer(history)
            if posterior.X.empty and posterior.performance_X.empty:
                raise ValueError("No observations for current BO posterior")
            posterior.fit_current_models()
            self._audit_posterior = posterior
            self._audit_posterior_key = key
        posterior = self._audit_posterior
        configurations = [self.fixed_search_space.canonical_to_bo(
            self.fixed_search_space.bo_to_canonical(c)) for c in candidates]
        frame = pd.DataFrame.from_records(
            filter_history_to_space(configurations, self.space, self.args),
            columns=self.space.para_names,
        )
        mean, variance = posterior.model.predict(*posterior.space.transform(frame))
        mean = mean.detach().cpu().numpy()[:, :3]
        variance = np.maximum(variance.detach().cpu().numpy()[:, :3], 0)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Current native BO returned non-finite performance predictions")
        return [{
            "predicted_objectives": (-mean[i]).tolist(),
            "objective_variance": variance[i].tolist(),
            "objective_uncertainty": np.sqrt(variance[i]).tolist(),
            "prediction_source": "current_native_bo_posterior",
            "training_trial_ids": [row["trial_id"] for row in history.get("trials", [])
                                   if row.get("status") == "complete" or row.get("used_for_performance_surrogates") is True],
            "complete_training_rows": int(posterior.X.shape[0]),
            "estimated_training_rows": len(posterior.performance_X),
        } for i in range(len(candidates))]

    def _selection(
        self,
        optimizer: GeneralBO,
        *,
        initial: bool,
        suggest_seconds: float,
        rf_model: Any,
        rf_threshold: float,
    ) -> dict[str, Any]:
        common = {
            "complete_training_rows": int(optimizer.X.shape[0]),
            "estimated_training_rows": len(optimizer.performance_X),
            "required_complete_observations": int(optimizer.rand_sample),
            "suggest_seconds": suggest_seconds,
            "hidden_constraint_threshold": rf_threshold,
            "hidden_constraint_model_fitted": rf_model is not None,
        }
        if initial:
            return {
                **common,
                "method": "initial_scrambled_sobol",
                "implementation": "torch.quasirandom.SobolEngine",
                "sobol_draw_index": int(optimizer.sobol.num_generated - 1),
                "ehvi": None,
                "constrained_ehvi": None,
                "p_feasible": None,
                "p_run": None,
                "score": None,
            }
        acquisition = dict(getattr(optimizer, "last_quality_acquisition", {}))
        if not acquisition:
            raise RuntimeError(
                "post-warmup native BO returned a candidate without acquisition "
                "diagnostics"
            )
        return {
            **common,
            "method": (
                "joint_mc_constrained_ehvi"
                if self.quality_surrogate_enabled
                else "performance_only_mc_ehvi"
            ),
            "implementation": (
                "GeneralAcq_LCB_EvolutionOpt_then_"
                + (
                    f"{self.config.monte_carlo_samples}MC_HVI_times_I_quality"
                    if self.quality_surrogate_enabled
                    else f"{self.config.monte_carlo_samples}MC_HVI_only"
                )
            ),
            "candidate_pool_size": acquisition.get("candidate_count"),
            "monte_carlo_samples": acquisition.get(
                "monte_carlo_samples", self.config.monte_carlo_samples
            ),
            "ehvi": acquisition.get("ehvi", 0.0),
            "constrained_ehvi": acquisition.get("constrained_ehvi", 0.0),
            "p_feasible": acquisition.get("p_feasible", 0.0),
            "p_run": acquisition.get("p_run", 1.0),
            "p_run_affects_final_score": acquisition.get(
                "p_run_affects_final_score", False
            ),
            "score": acquisition.get("score", 0.0),
            "quality_mc_indicator": acquisition.get(
                "quality_mc_indicator",
                self.quality_surrogate_enabled,
            ),
            "quality_feasible_mc_samples": acquisition.get(
                "quality_feasible_mc_samples", 0
            ),
            "scalar_ehvi_times_p_feasible": acquisition.get(
                "scalar_ehvi_times_p_feasible", False
            ),
            "quality_mean_hard_threshold": acquisition.get(
                "quality_mean_hard_threshold", False
            ),
            "quality_surrogate_affects_candidate_generation": acquisition.get(
                "quality_surrogate_affects_candidate_generation", False
            ),
            "quality_surrogate_affects_final_ranking": acquisition.get(
                "quality_surrogate_affects_final_ranking",
                self.quality_surrogate_enabled,
            ),
            "quality_constraint_mean": acquisition.get(
                "quality_constraint_mean"
            ),
            "quality_constraint_variance": acquisition.get(
                "quality_constraint_variance"
            ),
            "conditional_space": (self.space.conditional_space.contract()
                                  if hasattr(self.space, "conditional_space") else None),
            "selected_candidate_source": acquisition.get("selected_candidate_source"),
            "candidate_generation": acquisition.get(
                "candidate_generation"
            ),
        }

    def _hidden_constraint_state(
        self, history: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], Any, float]:
        rec_history = self._rec_history_from_trials(history["trials"])
        rf_model = fit_hidden_constraint_model(rec_history, self.space, self.args)
        rf_threshold = 0.5
        continuous_right = 0
        start_index = self.space.num_paras
        for index in range(start_index, len(rec_history)):
            if rec_history[index]["obj"] is None:
                rf_threshold = round(min(0.75, max(rf_threshold + 0.05, 0.5)), 3)
                continuous_right = 0
            else:
                continuous_right += 1
                if continuous_right > 5:
                    continuous_right -= 5
                    rf_threshold = round(max(0.25, rf_threshold - 0.05), 3)
        return rec_history, rf_model, rf_threshold

    def _repair_selected_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        raw_candidate: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply the same composite repair used by the public repair tool."""

        if self.candidate_repairer is None:
            repaired = dict(candidate)
            check = _constraint_check(raw_candidate, repaired)
        else:
            check = dict(self.candidate_repairer(candidate))
            if check.get("valid") is not True:
                raise RuntimeError(
                    "candidate repair could not satisfy the manual constraint boundary: "
                    f"{check.get('validation_errors')}"
                )
            repaired = dict(check.get("candidate") or {})
        observed_row = filter_history_to_space([repaired], self.space, self.args)[0]
        if not _row_inside_space(self.space, observed_row):
            raise RuntimeError("candidate repair produced an out-of-domain BO row")
        return repaired, dict(observed_row), check

    def recommend_next(
        self,
        *,
        temporary_candidate_filter: Mapping[str, Any] | None = None,
        guidance_source: str | None = None,
        guidance_observation_keys: Sequence[str] | None = None,
        candidate_override: Mapping[str, Any] | None = None,
        guidance_context: Mapping[str, Any] | None = None,
        direct_candidate_kind: str = "cold_start",
    ) -> dict[str, Any]:
        """Optionally screen native recommendations before any trial executes."""
        kwargs = dict(
            temporary_candidate_filter=temporary_candidate_filter,
            guidance_source=guidance_source,
            guidance_observation_keys=guidance_observation_keys,
            candidate_override=candidate_override,
            guidance_context=guidance_context,
            direct_candidate_kind=direct_candidate_kind,
        )
        if not self.config.quality_probability_gate_enabled or direct_candidate_kind == "repair":
            return self._recommend_next_unfiltered(**kwargs)
        history = self._load_or_create_history()
        optimizer = self._restore_optimizer(history)
        optimizer.retain_ranked_candidate_pool = True
        hidden_state = self._hidden_constraint_state(history)

        def encode(candidates):
            if hasattr(self.space, "conditional_space"):
                rows = filter_history_to_space(candidates, self.space, self.args)
                frame = pd.DataFrame.from_records(rows, columns=self.space.para_names)
                values, _groups = self.space.conditional_space.embed(*self.space.transform(frame))
                return np.asarray(values, dtype=float)
            return self.fixed_search_space.encode(candidates)

        gate = QualityProbabilityGate(
            encode=encode, key=self.fixed_search_space.key,
            threshold=self.config.quality_probability_threshold,
            max_attempts=self.config.quality_probability_max_attempts,
            start_after_rounds=self.config.quality_probability_start_after_rounds,
            audit_path=self.output_dir / "quality_probability_gate.json",
            model=self.quality_probability_model, random_seed=self.config.random_seed,
        )
        proposed_once = False

        def propose(excluded):
            nonlocal proposed_once
            # A rejected direct cold-start proposal falls back to native BO.
            current = dict(kwargs)
            if proposed_once and candidate_override is not None:
                current["candidate_override"] = None
            proposed_once = True
            recommendation = self._recommend_next_unfiltered(
                **current, _optimizer=optimizer, _history=history, _hidden_state=hidden_state,
            )
            # Exclude the executed representation after mandatory preflight as
            # well as the original BO row. These are local attempted keys only.
            optimizer.register_attempted(pd.DataFrame.from_records(
                [recommendation["candidate_bo_names"]], columns=self.space.para_names,
            ))
            return recommendation

        return gate.select(
            history=history, cold_start=optimizer.num_performance_observations < optimizer.rand_sample,
            propose=propose,
        )

    def _recommend_next_unfiltered(
        self,
        *,
        temporary_candidate_filter: Mapping[str, Any] | None = None,
        guidance_source: str | None = None,
        guidance_observation_keys: Sequence[str] | None = None,
        candidate_override: Mapping[str, Any] | None = None,
        guidance_context: Mapping[str, Any] | None = None,
        direct_candidate_kind: str = "cold_start",
        _optimizer: GeneralBO | None = None,
        _history: Mapping[str, Any] | None = None,
        _hidden_state: tuple[list[dict[str, Any]], Any, float] | None = None,
    ) -> dict[str, Any]:
        """Select one point by native BO or accept one cold-start Agent point."""

        if temporary_candidate_filter and candidate_override is not None:
            raise ValueError(
                "temporary_candidate_filter and candidate_override are mutually exclusive"
            )

        applied_guidance_source = (
            str(guidance_source or "unspecified_guidance")
            if temporary_candidate_filter or candidate_override is not None
            else None
        )
        history = self._load_or_create_history() if _history is None else _history
        optimizer = self._restore_optimizer(history) if _optimizer is None else _optimizer
        _rec_history, rf_model, rf_threshold = (self._hidden_constraint_state(history)
                                              if _hidden_state is None else _hidden_state)
        constraint = None
        temporary_fix_input: dict[str, Any] = {}
        temporary_numeric_bounds: dict[str, tuple[float, float]] = {}
        if temporary_candidate_filter:
            predicate = dict(temporary_candidate_filter)
            temporary_numeric_bounds = _temporary_filter_numeric_bounds(
                self.space, predicate
            )
            temporary_fix_input = _temporary_filter_fixed_inputs(
                self.space,
                predicate,
                numeric_bounds=temporary_numeric_bounds,
            )

            def constraint(frame: pd.DataFrame) -> np.ndarray:
                decisions: list[bool] = []
                for index in range(len(frame)):
                    try:
                        # Acquisition may evaluate hundreds of rows.  Capacity
                        # guidance is intentionally reserved for the single
                        # selected candidate; here we only apply the legacy
                        # deterministic normalization needed to evaluate the
                        # temporary one-round predicate.
                        repaired = normalize_conditional_combination(
                            rec_to_combination(frame.iloc[[index]], self.args),
                            self.args,
                        )
                        decisions.append(candidate_matches_filter(repaired, predicate))
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        decisions.append(False)
                return np.asarray(decisions, dtype=bool)

        initial = optimizer.num_performance_observations < optimizer.rand_sample
        if candidate_override is not None:
            suggest_seconds = 0.0
            raw_candidate = dict(candidate_override)
            selected_candidate = normalize_conditional_combination(
                raw_candidate, self.args
            )
            candidate, observed_row, candidate_repair = self._repair_selected_candidate(
                selected_candidate,
                raw_candidate=raw_candidate,
            )
        else:
            suggest_started = time.monotonic()
            recommendation = None
            if _optimizer is not None:
                pool = getattr(optimizer, "ranked_candidate_pool", [])
                while pool:
                    frame, diagnostics = pool.pop(0)
                    frame = optimizer._filter_new_candidates(frame)
                    if not frame.empty:
                        recommendation = optimizer._mark_returned(frame)
                        optimizer.last_quality_acquisition = diagnostics
                        break
            if recommendation is None:
                recommendation = optimizer.suggest(
                    n_suggestions=1,
                    fix_input=temporary_fix_input or None,
                    rf_with_thres=(rf_model, rf_threshold),
                    temporary_candidate_constraint=constraint,
                    temporary_numeric_bounds=temporary_numeric_bounds or None,
                )
            suggest_seconds = time.monotonic() - suggest_started
            # Keep the public optimizer boundary fail-safe even when a test or
            # alternate caller replaces GeneralBO.suggest. In the native path
            # this is an idempotent second pass over the already-scored point.
            if self.config.project_before_acquisition:
                recommendation = self.project_feasible_candidates(recommendation)
            raw_candidate = {
                name: _json_value(recommendation.iloc[0][name])
                for name in self.space.para_names
            }
            if self.config.project_before_acquisition:
                self._assert_feasible_observed_row(raw_candidate)
            selected_candidate = rec_to_combination(recommendation, self.args)
            candidate, observed_row, candidate_repair = self._repair_selected_candidate(
                selected_candidate,
                raw_candidate=raw_candidate,
            )
        if self.config.project_before_acquisition:
            self._assert_feasible_observed_row(observed_row)
        selected_candidate_satisfies_filter = (
            candidate_matches_filter(selected_candidate, temporary_candidate_filter)
            if temporary_candidate_filter
            else None
        )
        if temporary_candidate_filter and not selected_candidate_satisfies_filter:
            raise RuntimeError(
                "native BO recommendation escaped the temporary Observation search space"
            )
        post_preflight_satisfies_filter = (
            candidate_matches_filter(candidate, temporary_candidate_filter)
            if temporary_candidate_filter
            else None
        )
        post_preflight_outside_parameters = (
            candidate_filter_violations(candidate, temporary_candidate_filter)
            if temporary_candidate_filter
            else []
        )
        if candidate_override is not None:
            selection = {
                "complete_training_rows": int(optimizer.X.shape[0]),
                "required_complete_observations": int(optimizer.rand_sample),
                "suggest_seconds": 0.0,
                "hidden_constraint_threshold": rf_threshold,
                "hidden_constraint_model_fitted": rf_model is not None,
                "method": (
                    "repair_complete_candidate"
                    if direct_candidate_kind == "repair"
                    else "cold_start_agent_complete_candidate"
                ),
                "implementation": (
                    "direct_structured_completion_complete_candidate_then_"
                    "mandatory_preflight_repair"
                ),
                "sobol_draw_index": None,
                "ehvi": None,
                "constrained_ehvi": None,
                "p_feasible": None,
                "p_run": None,
                "score": None,
            }
        else:
            selection = self._selection(
                optimizer,
                initial=initial,
                suggest_seconds=suggest_seconds,
                rf_model=rf_model,
                rf_threshold=rf_threshold,
            )
        selection["feasible_space_enforcement"] = {
            "stage": ("before_surrogate_and_acquisition_evaluation"
                      if self.config.project_before_acquisition else "after_complete_recommendation"),
            "candidate_transform": (
                "fixed_range_dependency_and_known_constraint_projection"
            ),
            "idempotence_checked": self.config.project_before_acquisition,
            "selected_and_executed_rows_share_canonical_representation": self.config.project_before_acquisition,
            "objective_count": 3,
            "pareto_logic_changed": False,
        }
        selection["temporary_search_space"] = {
            "active": bool(temporary_candidate_filter),
            "candidate_filter": dict(temporary_candidate_filter or {}),
            "source": applied_guidance_source,
            "source_provenance_complete": bool(
                temporary_candidate_filter and guidance_source
            ),
            "scope": "single_next_bo_recommendation",
            "selected_by": (
                "independent_cold_start_agent"
                if candidate_override is not None
                else "native_bo_acquisition"
            ),
            "native_sampler_fixed_inputs": dict(temporary_fix_input),
            "native_sampler_numeric_bounds": _temporary_numeric_bounds_receipt(
                self.space, temporary_numeric_bounds
            ),
            "enforcement_policy": (
                "hard_during_native_bo_selection_soft_after_mandatory_preflight"
                if temporary_candidate_filter
                else None
            ),
            "selected_candidate_satisfies_filter": (
                selected_candidate_satisfies_filter
            ),
            "candidate_after_preflight_satisfies_filter": (
                post_preflight_satisfies_filter
            ),
            "candidate_after_preflight_outside_parameters": (
                post_preflight_outside_parameters
            ),
            "candidate_after_preflight_escape_accepted": bool(
                temporary_candidate_filter
                and selected_candidate_satisfies_filter
                and not post_preflight_satisfies_filter
            ),
        }
        selection["direct_complete_candidate"] = {
            "active": candidate_override is not None,
            "source": applied_guidance_source,
            "source_provenance_complete": bool(
                candidate_override is not None and guidance_source
            ),
                "selected_by": (
                    "repair_bo_reasoner"
                    if candidate_override is not None
                    and direct_candidate_kind == "repair"
                    else "independent_cold_start_agent"
                    if candidate_override is not None
                    else None
                ),
            "agent_candidate": (
                dict(candidate_override) if candidate_override is not None else None
            ),
            "candidate_after_preflight": (
                dict(candidate) if candidate_override is not None else None
            ),
            "mandatory_preflight_repair": candidate_override is not None,
            "native_bo_suggest_called": candidate_override is None,
        }
        selection["guidance_decision"] = {
            "chosen_path": (
                "repair_bo"
                if direct_candidate_kind == "repair"
                else "cold_start_guidance"
                if guidance_source in {"adjust_cold_start_search_space", "repair_cold_start"}
                else (
                    "llm_recommendation"
                    if guidance_source == "guide_search_space_with_llm"
                    else (
                        "llm_guidance"
                        if guidance_source == "llm_guidance_agent"
                        else (
                            "guidance_failed"
                            if guidance_source == "guidance_failed"
                            else (
                                "dynamic_observation_guidance"
                                if guidance_source == "dynamic_observation_agent"
                                else "no_guidance"
                            )
                        )
                    )
                )
            ),
            "attempted_source": guidance_source,
            "temporary_filter_applied": bool(temporary_candidate_filter),
            "complete_candidate_applied": candidate_override is not None,
            "observation_keys": sorted(
                {
                    str(item).strip()
                    for item in (guidance_observation_keys or [])
                    if str(item).strip()
                }
            ),
            "context": dict(guidance_context or {}),
        }
        return {
            "candidate": dict(candidate),
            "candidate_bo_names": dict(observed_row),
            "raw_bo_candidate": raw_candidate,
            "selection": selection,
            "known_constraint_check": candidate_repair,
            "selected_by": (
                "repair_bo_reasoner"
                if candidate_override is not None and direct_candidate_kind == "repair"
                else "independent_cold_start_agent"
                if candidate_override is not None
                else "native_bo_acquisition"
            ),
        }

    def run_one(
        self,
        *,
        temporary_candidate_filter: Mapping[str, Any] | None = None,
        guidance_source: str | None = None,
        guidance_observation_keys: Sequence[str] | None = None,
        candidate_override: Mapping[str, Any] | None = None,
        guidance_context: Mapping[str, Any] | None = None,
        direct_candidate_kind: str = "cold_start",
        trial_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute exactly one BO-selected point and durably observe it."""

        history = self._load_or_create_history()
        if len(history["trials"]) >= self.config.rounds:
            raise RuntimeError("Candidate Optimizer already completed all rounds")
        recommendation = self.recommend_next(
            temporary_candidate_filter=temporary_candidate_filter,
            guidance_source=guidance_source,
            guidance_observation_keys=guidance_observation_keys,
            candidate_override=candidate_override,
            guidance_context=guidance_context,
            direct_candidate_kind=direct_candidate_kind,
        )
        duplicate_trial_ids = [
            str(trial.get("trial_id") or "")
            for trial in history.get("trials", [])
            if isinstance(trial, Mapping)
            and isinstance(trial.get("candidate"), Mapping)
            and dict(trial["candidate"]) == dict(recommendation["candidate"])
        ]
        if duplicate_trial_ids:
            raise RuntimeError(
                "post-preflight candidate duplicates an already executed complete "
                "configuration; regenerate a non-duplicate BO point instead of treating "
                "measurement noise as exploration; duplicate_trial_ids="
                + json.dumps(duplicate_trial_ids)
            )
        trial_id = f"trial_{len(history['trials']) + 1:04d}"
        started = time.monotonic()
        try:
            execution = self.executor.execute(
                recommendation["candidate"],
                active_parameters=sorted(recommendation["candidate"]),
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
        record = self._record(
            trial_id=trial_id,
            raw_candidate=recommendation["raw_bo_candidate"],
            candidate=recommendation["candidate"],
            observed_row=recommendation["candidate_bo_names"],
            selection=recommendation["selection"],
            candidate_repair=recommendation["known_constraint_check"],
            execution=execution,
            elapsed_seconds=time.monotonic() - started,
            evaluation_index=(
                evaluation_count(history) + 1
                if self.config.repair_bo_enabled
                else len(history["trials"]) + 1
            ),
            run_execution_index=current_run_execution_count(history) + 1,
            bo_iteration=(
                int(trial_metadata.get("bo_iteration"))
                if isinstance(trial_metadata, Mapping)
                and trial_metadata.get("bo_iteration") is not None
                else native_bo_iteration_count(history) + 1
                if not isinstance(trial_metadata, Mapping)
                or trial_metadata.get("trial_role") != "repair"
                else None
            ),
            trial_metadata=trial_metadata,
        )
        history["trials"].append(record)
        write_json_atomic(self.history_path, history)
        result = result_from_history(history)
        result["implementation"] = history["implementation"]
        result["result_path"] = str(self.result_path)
        write_json_atomic(self.result_path, result)
        return {
            "status": "round_executed",
            "trial": record,
            "history_path": str(self.history_path),
            "result_path": str(self.result_path),
            "rounds_completed": len(history["trials"]),
            "rounds_requested": self.config.rounds,
            "temporary_search_space": recommendation["selection"][
                "temporary_search_space"
            ],
        }

    def _record(
        self,
        *,
        trial_id: str,
        raw_candidate: Mapping[str, Any],
        candidate: Mapping[str, Any],
        observed_row: Mapping[str, Any],
        selection: Mapping[str, Any],
        candidate_repair: Mapping[str, Any] | None,
        execution: TrialExecution,
        elapsed_seconds: float,
        evaluation_index: int,
        run_execution_index: int,
        bo_iteration: int | None,
        trial_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(trial_metadata or {})
        trial_role = str(
            metadata.get("trial_role")
            or ("cold_start" if selection.get("direct_complete_candidate", {}).get("active") else "native_bo")
        )
        base = {
            "trial_id": trial_id,
            "source": "candidate_optimizer",
            "evaluation_index": int(evaluation_index),
            "run_execution_index": int(run_execution_index),
            "execution_mode": "executed_in_current_run",
            "executed_in_current_run": True,
            "bo_iteration": bo_iteration,
            "parent_bo_iteration": metadata.get("parent_bo_iteration"),
            "trial_role": trial_role,
            "episode_id": metadata.get("episode_id"),
            "parent_trial_id": metadata.get("parent_trial_id"),
            "repair_anchor_trial_id": metadata.get("repair_anchor_trial_id"),
            "repair_strategy": metadata.get("repair_strategy"),
            "repair_stage": metadata.get("repair_stage"),
            "repair_strategy_mode": metadata.get("repair_strategy_mode"),
            "repair_mode": metadata.get("repair_mode"),
            "repair_round": metadata.get("repair_round"),
            "repair_signature": metadata.get("repair_signature"),
            "repair_scope_version": metadata.get("repair_scope_version"),
            "repair_allowed_fields": metadata.get("repair_allowed_fields"),
            "workload_snapshot": metadata.get("workload_snapshot"),
            "repair_prediction": metadata.get("repair_prediction"),
            **({'repair_cold_start': metadata['repair_cold_start']}
               if metadata.get('repair_cold_start') is not None else {}),
            "raw_bo_candidate": dict(raw_candidate),
            "candidate": dict(candidate),
            "candidate_bo_names": dict(observed_row),
            "active_parameters": sorted(candidate),
            "selection": dict(selection),
            "known_constraint_check": (
                dict(candidate_repair)
                if isinstance(candidate_repair, Mapping)
                else _constraint_check(raw_candidate, candidate)
            ),
            "suggest_seconds": selection.get("suggest_seconds"),
            "elapsed_seconds": elapsed_seconds,
            "result_path": execution.result_path,
            "executor": {
                "command": list(execution.command),
                "return_code": execution.return_code,
            },
            "execution_control": {
                key: execution.result.get(key)
                for key in (
                    "validator",
                    "validation",
                    "validator_rejections",
                    "recovery",
                    "recoveries",
                    "rollback",
                    "rollbacks",
                )
                if isinstance(execution.result, Mapping) and key in execution.result
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
                "model_target": None,
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
                "model_target": None,
                "used_for_performance_surrogates": False,
                "used_for_quality_surrogate": False,
                "used_for_feasible_pareto": False,
                "used_for_hidden_constraint": True,
            }
        margin = quality_margin_diagnostics(
            execution.result,
            quality_metric=self.config.quality_metric,
            quality_thresholds_json=self.config.quality_thresholds_json,
            min_mean_f1=self.config.min_mean_f1,
            max_relative_error=self.config.max_relative_error,
            tau_q=self.tau_q,
            quality=quality,
        )
        feasible = bool(margin["quality_feasible"])
        repair_signature = metadata.get("repair_signature")
        if repair_signature is None and self.repair_scope is not None:
            repair_signature = self.repair_scope.signature(
                candidate, self.fixed_search_space
            )
        return {
            **base,
            "status": "complete",
            "failure_kind": None,
            "failure_reason": None,
            "objective_vector": objectives,
            "model_target": [-value for value in objectives] + [self.tau_q - quality],
            "metrics": metrics,
            "quality": quality,
            "quality_margin": margin["worst_quality_margin"],
            "quality_margins": margin["quality_margins"],
            "repair_signature": repair_signature,
            "quality_ratio_to_baseline": (
                quality / self.baseline_quality
                if self.baseline_quality > 0.0
                else None
            ),
            "quality_feasible": feasible,
            "used_for_performance_surrogates": True,
            "used_for_quality_surrogate": True,
            "used_for_feasible_pareto": feasible,
            "used_for_hidden_constraint": True,
        }

    def run(self) -> dict[str, Any]:
        if self.config.quality_probability_gate_enabled:
            # Share the same gate and budget accounting as the round-wise API.
            history = self._load_or_create_history()
            while len(history["trials"]) < self.config.rounds:
                self.run_one()
                history = self._load_or_create_history()
            result = result_from_history(history)
            result["implementation"] = history["implementation"]
            result["result_path"] = str(self.result_path)
            write_json_atomic(self.result_path, result)
            return result
        history = self._load_or_create_history()
        optimizer = self._restore_optimizer(history)
        rec_history = self._rec_history_from_trials(history["trials"])
        rf_model = fit_hidden_constraint_model(rec_history, self.space, self.args)
        rf_threshold = 0.5
        continuous_right = 0
        start_index = self.space.num_paras
        for index in range(start_index, len(rec_history)):
            if rec_history[index]["obj"] is None:
                rf_threshold = round(min(0.75, max(rf_threshold + 0.05, 0.5)), 3)
                continuous_right = 0
            else:
                continuous_right += 1
                if continuous_right > 5:
                    continuous_right -= 5
                    rf_threshold = round(max(0.25, rf_threshold - 0.05), 3)

        while len(history["trials"]) < self.config.rounds:
            iteration_index = len(history["trials"])
            trial_id = f"trial_{iteration_index + 1:04d}"
            initial = optimizer.num_performance_observations < optimizer.rand_sample
            suggest_started = time.monotonic()
            recommendation = optimizer.suggest(
                n_suggestions=1,
                rf_with_thres=(rf_model, rf_threshold),
            )
            suggest_seconds = time.monotonic() - suggest_started
            recommendation = self.project_feasible_candidates(recommendation)
            raw_candidate = {
                name: _json_value(recommendation.iloc[0][name])
                for name in self.space.para_names
            }
            self._assert_feasible_observed_row(raw_candidate)
            selected_candidate = rec_to_combination(recommendation, self.args)
            candidate, observed_row, candidate_repair = self._repair_selected_candidate(
                selected_candidate,
                raw_candidate=raw_candidate,
            )
            self._assert_feasible_observed_row(observed_row)
            selection = self._selection(
                optimizer,
                initial=initial,
                suggest_seconds=suggest_seconds,
                rf_model=rf_model,
                rf_threshold=rf_threshold,
            )
            selection["feasible_space_enforcement"] = {
                "stage": "before_surrogate_and_acquisition_evaluation",
                "candidate_transform": (
                    "fixed_range_dependency_and_known_constraint_projection"
                ),
                "idempotence_checked": True,
                "selected_and_executed_rows_share_canonical_representation": True,
                "objective_count": 3,
                "pareto_logic_changed": False,
            }
            started = time.monotonic()
            try:
                execution = self.executor.execute(
                    candidate,
                    active_parameters=sorted(candidate),
                    trial_id=trial_id,
                )
            except Exception as exc:  # executor failures are hidden constraints
                execution = TrialExecution(
                    status="runtime_failed",
                    result=None,
                    result_path=None,
                    failure_kind="executor_exception",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            record = self._record(
                trial_id=trial_id,
                raw_candidate=raw_candidate,
                candidate=candidate,
                observed_row=observed_row,
                selection=selection,
                candidate_repair=candidate_repair,
                execution=execution,
                elapsed_seconds=time.monotonic() - started,
                evaluation_index=iteration_index + 1,
                run_execution_index=iteration_index + 1,
                bo_iteration=iteration_index + 1,
                trial_metadata={"trial_role": "native_bo"},
            )
            if record["status"] == "complete":
                optimizer.observe(
                    pd.DataFrame.from_records(
                        [observed_row], columns=self.space.para_names
                    ),
                    np.asarray([record["model_target"]], dtype=float),
                )
                if iteration_index > self.space.num_paras:
                    continuous_right += 1
                    if continuous_right >= 5:
                        continuous_right -= 5
                        rf_threshold = round(max(0.25, rf_threshold - 0.05), 3)
            elif record.get("performance_source") == "timeout_extrapolation":
                optimizer.observe_performance(
                    pd.DataFrame.from_records([observed_row], columns=self.space.para_names),
                    np.asarray([[-v for v in record["estimated_objective_vector"]]], dtype=float),
                )
            elif iteration_index > self.space.num_paras:
                rf_threshold = round(min(0.75, max(rf_threshold + 0.05, 0.5)), 3)
                continuous_right = 0
            history["trials"].append(record)
            if record.get("used_for_hidden_constraint") is not False:
                rec_history.append(
                    {
                        "rec": [dict(observed_row)],
                        "executed_combination": dict(candidate),
                        "obj": record.get("model_target"),
                        "rec_time": suggest_seconds,
                        "run_time": record["elapsed_seconds"],
                    }
                )
            rf_model = fit_hidden_constraint_model(rec_history, self.space, self.args)
            write_json_atomic(self.history_path, history)
            self.rec_history_path.write_text(
                json.dumps(rec_history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        result = result_from_history(history)
        result["implementation"] = history["implementation"]
        result["result_path"] = str(self.result_path)
        write_json_atomic(self.result_path, result)
        return result
