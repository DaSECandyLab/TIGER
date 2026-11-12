"""Candidate Optimizer 本地工作流、来源绑定和机械审计。"""

from __future__ import annotations

import json
import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from agents.audit import AuditReport
from agents.tuning.dqbo.src.services.workload_facts import build_static_workload_context
from agents.tuning.dqbo.src.services.failure_taxonomy import CAPACITY_FAILURE

from agents.tuning.dqbo.src.services.engine.constraints import KNOWN_CONSTRAINT_IDS, KnownConstraintHandler
from agents.tuning.repair.src.services import repair_cold_start
from agents.tuning.dqbo.src.services.engine.executor import BOTrialExecutor
from agents.tuning.dqbo.src.services.engine.history import (
    HISTORY_SCHEMA,
    RESULT_SCHEMA,
    current_run_execution_count,
    evaluation_count,
    feasible_pareto_records,
    native_bo_iteration_count,
    write_json_atomic,
)
from agents.tuning.dqbo.src.services.engine.metrics import (
    AGGREGATE_QUALITY_THRESHOLD_SOURCE,
    QUALITY_GATE_METRIC,
    RELATIVE_QUALITY_THRESHOLD_FORMULAS,
    RELATIVE_QUALITY_THRESHOLD_SOURCE,
    derive_relative_quality_thresholds_from_baseline,
    derive_quality_thresholds_from_baseline,
    query_configured_metric_means,
    QUALITY_THRESHOLD_FORMULAS,
    QUALITY_THRESHOLD_SOURCE,
    quality_threshold,
)
from agents.tuning.dqbo.src.services.engine.models import SURROGATE_NAMES
from agents.tuning.dqbo.src.services.engine.optimizer import CandidateOptimizer, OptimizerConfig
from agents.tuning.dqbo.src.services.engine.legacy_equivalent import (
    LegacyEquivalentQualityBO,
    build_legacy_namespace,
    repair_with_legacy_constraints,
)
from agents.tuning.dqbo.src.services.engine._legacy_bo.search_space import build_design_space
from agents.tuning.dqbo.src.services.engine._legacy_bo.screened_space import (
    SUPPORTED_SCREENED_PARAMETERS,
    canonical_screened_name,
)
from agents.tuning.dqbo.src.services.engine.space import FixedSearchSpace
from .direct_inference import default_candidate_structured_inference_client
from agents.tuning.dqbo.src.services.engine.flock.trial.arguments import parse_args as parse_trial_args
from agents.tuning.repair.src.services.repair_bo import (
    REPAIR_BO_V1_FIELDS,
    RepairCoordinator,
    RepairScope,
    _repair_frozen_field_changes,
    repair_scope_runtime_constraints,
)


CONTEXT_SCHEMA = "sembench.candidate-optimizer-context.v1"
RESULT_NAME = "candidate_optimizer_result.json"
GUIDANCE_SOURCES = frozenset(
    {
        "adjust_cold_start_search_space",
        "repair_cold_start",
        "guide_search_space_with_llm",
        "dynamic_observation_agent",
        "llm_guidance_agent",
        "guidance_failed",
    }
)
GUIDANCE_MODES = frozenset(
    {"agentic", "dynamic", "full", "cold_start_only", "disabled"}
)


def _guidance_policy(mode: str = "disabled") -> dict[str, Any]:
    if mode != "disabled":
        raise ValueError("External-memory BO guidance has been removed")
    return {"mode": "disabled", "cold_start_enabled": False, "llm_guidance_enabled": False,
            "dynamic_observation_guidance_enabled": False,
            "historical_performance_observations_enabled": False,
            "candidate_preflight_repair_enabled": False}




def _validated_guidance_source(
    temporary_candidate_filter: Mapping[str, Any] | None,
    guidance_source: str | None,
) -> str | None:
    """Validate optional one-round guidance provenance without breaking old callers."""

    if guidance_source is not None and guidance_source not in GUIDANCE_SOURCES:
        raise ValueError(f"unsupported BO guidance source: {guidance_source!r}")
    return guidance_source


def _validate_guidance_round(
    context: Mapping[str, Any],
    guidance_source: str | None,
    *,
    agentic_mode: bool = False,
) -> None:
    """Enforce the frozen guidance policy and cold-start round window."""

    if guidance_source == 'repair_cold_start':
        history = _load_object(context['history_path'])
        if not repair_cold_start.enabled(context) or native_bo_iteration_count(history) + 1 not in context['repair_cold_start']['trial_slots']:
            raise ValueError('RepairBO cold-start guidance is disabled or outside its slots')
        return
    policy = context.get("guidance_policy")
    cold_start_enabled = not isinstance(policy, Mapping) or bool(
        policy.get("cold_start_enabled", True)
    )
    llm_guidance_enabled = not isinstance(policy, Mapping) or bool(
        policy.get("llm_guidance_enabled", True)
    )
    dynamic_guidance_enabled = isinstance(policy, Mapping) and bool(
        policy.get("dynamic_observation_guidance_enabled", False)
    )
    if (
        guidance_source == "adjust_cold_start_search_space"
        and not cold_start_enabled
        and not agentic_mode
    ):
        raise ValueError("cold-start guidance is disabled by the frozen BO context")
    if (
        guidance_source == "guide_search_space_with_llm"
        and not llm_guidance_enabled
        and not agentic_mode
    ):
        raise ValueError("LLM BO guidance is disabled by the frozen BO context")
    if guidance_source == "dynamic_observation_agent" and not dynamic_guidance_enabled:
        raise ValueError(
            "dynamic Observation guidance is disabled by the frozen BO context"
        )

    if guidance_source != "adjust_cold_start_search_space":
        return
    history_path = context.get("history_path")
    history = (
        _load_object(str(history_path))
        if isinstance(history_path, str) and Path(history_path).is_file()
        else {"trials": []}
    )
    trials = history.get("trials")
    if not isinstance(trials, list):
        raise TypeError("Candidate Optimizer history.trials must be an array")
    current_round = len(trials) + 1
    cold_start = context.get("cold_start_guidance")
    initial = context.get("initial_design")
    round_limit = (
        2
        if agentic_mode
        else int(cold_start.get("round_limit") or 0)
        if isinstance(cold_start, Mapping)
        else (
            int(initial.get("required_complete_observations") or 0)
            if isinstance(initial, Mapping)
            else 0
        )
    )
    if current_round > round_limit:
        raise ValueError(
            "cold-start guidance is outside its tuning-round window: "
            f"current_round={current_round}, round_limit={round_limit}"
        )


def _resolve(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _load_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {resolved}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_constraint_scope(
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze backend-specific hard dependencies from measured Baseline logs."""

    explicit_backend = baseline_result.get("resolved_attention_backend")
    backend = str(explicit_backend).strip().upper() if explicit_backend else None
    evidence_path: str | None = None
    evidence_line: str | None = None
    logs = baseline_result.get("logs")
    vllm_log = logs.get("vllm") if isinstance(logs, Mapping) else None
    if backend is None and isinstance(vllm_log, str):
        path = Path(vllm_log).expanduser().resolve()
        if path.is_file():
            try:
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    match = re.search(
                        r"Using\s+([A-Z0-9_]+)\s+attention backend", line
                    )
                    if match:
                        backend = match.group(1)
                        evidence_path = str(path)
                        evidence_line = line.strip()
                        break
            except OSError:
                pass
    return {
        "resolved_attention_backend": backend,
        "source": (
            "measured_baseline_vllm_log"
            if evidence_path is not None
            else "baseline_result_field" if backend is not None else "unresolved"
        ),
        "evidence_path": evidence_path,
        "evidence_line": evidence_line,
        "uncompressed_kv_model_dtype_alignment": backend == "FLASH_ATTN",
        "alignment_semantics": (
            "FLASH_ATTN query/key tensors must share dtype; only explicit "
            "float16/bfloat16 KV storage is aligned, while quantized KV formats "
            "remain independently selectable"
        ),
    }


def _validate_frozen_context(context: Mapping[str, Any]) -> None:
    if context.get("schema_version") != CONTEXT_SCHEMA:
        raise ValueError("unsupported Candidate Optimizer context")
    source_paths = {
        "parameter_results": context.get("source_parameter_results"),
        "domain_spec": context.get("source_domain_spec"),
        "baseline_config": context.get("source_baseline_config"),
        "baseline_trial_result": context.get("source_baseline_trial_result"),
    }
    source_hashes = context.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("BO context lacks frozen source hashes")
    for name, source_path in source_paths.items():
        if not isinstance(source_path, str) or _sha256(
            source_path
        ) != source_hashes.get(name):
            raise ValueError(f"frozen Candidate source changed after prepare: {name}")
    workload = context.get("workload_contract")
    if (
        not isinstance(workload, Mapping)
        or workload.get("matches_baseline_trial") is not True
    ):
        raise ValueError("BO context workload does not match the measured Baseline")
    static_workload = context.get("static_workload")
    if isinstance(static_workload, Mapping):
        baseline_result = _load_object(str(context["source_baseline_trial_result"]))
        expected_static_workload = build_static_workload_context(baseline_result)
        if dict(static_workload) != expected_static_workload:
            raise ValueError("BO context static workload changed after prepare")
    baseline_result = _load_object(str(context["source_baseline_trial_result"]))
    expected_runtime_scope = _runtime_constraint_scope(baseline_result)
    recorded_runtime_scope = context.get("runtime_constraint_scope")
    if (
        recorded_runtime_scope is not None
        and recorded_runtime_scope != expected_runtime_scope
    ):
        raise ValueError("BO context runtime constraint scope changed after prepare")
    policy = context.get("guidance_policy")
    if isinstance(policy, Mapping):
        expected_policy = _guidance_policy(str(policy.get("mode") or "full"))
        for name in (
            "cold_start_enabled",
            "llm_guidance_enabled",
            "dynamic_observation_guidance_enabled",
            "historical_performance_observations_enabled",
            "candidate_preflight_repair_enabled",
        ):
            if name in policy and policy.get(name) is not expected_policy[name]:
                raise ValueError(
                    f"BO context has an inconsistent guidance policy: {name}"
                )


def _known_constraint_handler(
    context: Mapping[str, Any], space: FixedSearchSpace
) -> KnownConstraintHandler:
    executor_spec = context.get("executor")
    if not isinstance(executor_spec, Mapping):
        raise TypeError("BO context executor must be an object")
    namespace = parse_trial_args(
        [str(item) for item in executor_spec.get("trial_argv", [])]
    )
    runtime_scope = context.get("runtime_constraint_scope")
    if not isinstance(runtime_scope, Mapping):
        baseline_result = _load_object(
            str(context["source_baseline_trial_result"])
        )
        runtime_scope = _runtime_constraint_scope(baseline_result)
    return KnownConstraintHandler(
        space,
        model_path=namespace.model_path,
        awq_model_path=namespace.awq_model_path,
        gptq_model_path=namespace.gptq_model_path,
        uncompressed_kv_model_dtype_alignment=bool(
            runtime_scope.get("uncompressed_kv_model_dtype_alignment")
        ),
        alignment_scope=(
            str(runtime_scope.get("resolved_attention_backend"))
            if runtime_scope.get("resolved_attention_backend")
            else None
        ),
    )


def _workload_contract(
    trial_argv: Sequence[str], baseline_result: Mapping[str, Any]
) -> dict[str, Any]:
    """证明 Candidate 延用 Baseline 实际执行过的 workload，而非另换负载。"""

    namespace = parse_trial_args([str(item) for item in trial_argv])
    requested = {
        "sembench_use_case": namespace.use_case,
        "sembench_files_override_root": namespace.sembench_files_override_root,
        "flock_queries": [str(item) for item in namespace.queries],
        "flock_workload_driver": namespace.flock_workload_driver,
        "flock_terminal_count": namespace.flock_terminal_count,
        "flock_terminal_queries": namespace.flock_terminal_queries,
        "flock_request_rate": namespace.flock_request_rate,
        "flock_request_arrival": namespace.flock_request_arrival,
        "flock_max_in_flight": namespace.flock_max_in_flight,
        "flock_max_in_flight_strategy": namespace.flock_max_in_flight_strategy,
        "flock_http_max_connections": namespace.flock_http_max_connections,
        "flock_query_concurrency": namespace.flock_query_concurrency,
        "flock_query_start_stagger_ms": namespace.flock_query_start_stagger_ms,
        "flock_request_control_mode": namespace.flock_request_control_mode,
        "flock_vllm_batch_size": namespace.flock_vllm_batch_size,
        "flock_failure_policy": namespace.flock_failure_policy,
        "cacheblend_mode": namespace.cacheblend_mode,
        "vllm_api_server_count": namespace.vllm_api_server_count,
    }
    observed = {name: baseline_result.get(name) for name in requested}
    mismatches: dict[str, dict[str, Any]] = {}
    for name, requested_value in requested.items():
        observed_value = observed[name]
        if name == "flock_queries":
            equal = sorted(set(requested_value)) == sorted(set(observed_value or []))
        else:
            equal = requested_value == observed_value
        if not equal:
            mismatches[name] = {
                "baseline": observed_value,
                "candidate": requested_value,
            }
    return {
        "matches_baseline_trial": not mismatches,
        "requested": requested,
        "baseline_observed": observed,
        "mismatches": mismatches,
        "scale_factor": namespace.scale_factor,
    }


def _with_quality_thresholds(
    trial_argv: Sequence[str], quality_thresholds_json: str
) -> list[str]:
    """用 baseline 派生门禁替换 trial argv 中可能存在的旧手工阈值。"""
    normalized: list[str] = []
    index = 0
    values = [str(item) for item in trial_argv]
    while index < len(values):
        value = values[index]
        if value == "--quality-thresholds-json":
            index += 2
            continue
        if value.startswith("--quality-thresholds-json="):
            index += 1
            continue
        normalized.append(value)
        index += 1
    normalized.extend(["--quality-thresholds-json", quality_thresholds_json])
    return normalized


def _without_quality_thresholds(trial_argv: Sequence[str]) -> list[str]:
    """Remove a stale per-query gate when an explicit aggregate gate is used."""

    normalized: list[str] = []
    index = 0
    values = [str(item) for item in trial_argv]
    while index < len(values):
        value = values[index]
        if value == "--quality-thresholds-json":
            index += 2
            continue
        if value.startswith("--quality-thresholds-json="):
            index += 1
            continue
        normalized.append(value)
        index += 1
    return normalized


def prepare_bo_context(
    *,
    parameter_results: str,
    domain_spec: str,
    baseline_config: str,
    baseline_trial_result: str,
    output_dir: str,
    rounds: int,
    quality_metric: str,
    quality_alpha: float,
    trial_argv: Sequence[str],
    optimizer_options: Mapping[str, Any] | None = None,
    python_executable: str | None = None,
    trial_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """验证上游来源和固定域，生成 execute_bo 唯一可消费的上下文。"""

    # 第 1 步：解析并验证 Knob、Range、Baseline 配置和 Baseline 实测结果。
    parameter_path = _resolve(parameter_results)
    domain_path = _resolve(domain_spec)
    baseline_path = _resolve(baseline_config)
    baseline_result_path = _resolve(baseline_trial_result)
    for path in (parameter_path, domain_path, baseline_path, baseline_result_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    parameter_payload = _load_object(parameter_path)
    domain_payload = _load_object(domain_path)
    baseline_payload = _load_object(baseline_path)
    baseline_result = _load_object(baseline_result_path)
    space = FixedSearchSpace.from_json(domain_path)

    if domain_payload.get("screening_source") == "fixed_config":
        from agents.tuning.dqbo.ranges import validate_fixed_domain
        validate_fixed_domain(domain_payload)

    # 第 2 步：要求 Range 精确消费本轮 Knob 结果和本轮 Baseline，不允许跨 run 混用。
    domain_parameter_source = domain_payload.get("source_parameter_results")
    if (
        not domain_parameter_source
        or _resolve(str(domain_parameter_source)) != parameter_path
    ):
        raise ValueError(
            "Range domain source_parameter_results conflicts with Knob output"
        )
    domain_baseline_source = domain_payload.get("source_baseline_config")
    if (
        not domain_baseline_source
        or _resolve(str(domain_baseline_source)) != baseline_path
    ):
        raise ValueError(
            "Range domain source_baseline_config conflicts with Knob Baseline"
        )
    selected_names = [
        str(item["name"])
        for item in parameter_payload.get("parameters", [])
        if isinstance(item, Mapping) and item.get("enters_bo") is True
    ]
    if domain_payload.get("screening_source") == "simple":
        raise ValueError("Legacy Range domains are not supported in tiger; regenerate inputs with agents.tuning.dqbo")
    if selected_names != [parameter.name for parameter in space.parameters]:
        raise ValueError(
            "Knob-selected parameter order differs from the fixed Range domain"
        )
    if baseline_payload.get("purpose") != "pre_measurement_baseline_configuration":
        raise ValueError("baseline config has an unsupported purpose")
    recorded_baseline = baseline_result.get("baseline_config_path")
    if recorded_baseline and _resolve(str(recorded_baseline)) != baseline_path:
        raise ValueError(
            "baseline trial result was produced from a different baseline config"
        )

    # 第 3 步：按 query 从单次 Baseline 派生并冻结门禁。
    from agents.tuning.dqbo.src.services.engine.metrics import extract_complete_trial_metrics

    trial_argv = [str(item) for item in trial_argv]
    optimizer_option_values = dict(optimizer_options or {})
    relative_quality_threshold_factors = optimizer_option_values.pop(
        "relative_quality_threshold_factors", None
    )
    absolute_quality_thresholds = optimizer_option_values.pop(
        "absolute_quality_thresholds", None
    )
    min_mean_f1 = optimizer_option_values.get("min_mean_f1")
    max_relative_error = optimizer_option_values.get("max_relative_error")
    aggregate_quality_gate = (
        min_mean_f1 is not None or max_relative_error is not None
    )
    quality_thresholds_json: str | None = None
    quality_threshold_source: str | None = None
    quality_threshold_formulas: dict[str, str] | None = None
    if (
        absolute_quality_thresholds is not None
        and relative_quality_threshold_factors is None
    ):
        raise ValueError(
            "absolute_quality_thresholds require relative_quality_threshold_factors"
        )
    if relative_quality_threshold_factors is not None:
        if quality_metric != QUALITY_GATE_METRIC:
            raise ValueError(
                "relative per-query thresholds require quality_gate_passed"
            )
        if aggregate_quality_gate or optimizer_option_values.get(
            "quality_thresholds_json"
        ) is not None:
            raise ValueError(
                "relative, explicit per-query, and aggregate quality gates "
                "are mutually exclusive"
            )
        quality_thresholds = derive_relative_quality_thresholds_from_baseline(
            baseline_result,
            relative_quality_threshold_factors,
            absolute_quality_thresholds,
        )
        quality_thresholds_json = json.dumps(
            quality_thresholds,
            sort_keys=True,
            separators=(",", ":"),
        )
        quality_threshold_source = RELATIVE_QUALITY_THRESHOLD_SOURCE
        quality_threshold_formulas = dict(RELATIVE_QUALITY_THRESHOLD_FORMULAS)
        trial_argv = _with_quality_thresholds(
            trial_argv,
            quality_thresholds_json,
        )
    elif quality_metric == QUALITY_GATE_METRIC and not aggregate_quality_gate:
        quality_thresholds = derive_quality_thresholds_from_baseline(
            baseline_result,
            quality_alpha,
        )
        quality_thresholds_json = json.dumps(
            quality_thresholds,
            sort_keys=True,
            separators=(",", ":"),
        )
        trial_argv = _with_quality_thresholds(
            trial_argv,
            quality_thresholds_json,
        )
        quality_threshold_source = QUALITY_THRESHOLD_SOURCE
        quality_threshold_formulas = dict(QUALITY_THRESHOLD_FORMULAS)
    elif aggregate_quality_gate:
        trial_argv = _without_quality_thresholds(trial_argv)
    _, baseline_quality, _ = extract_complete_trial_metrics(
        baseline_result,
        quality_metric=quality_metric,
        quality_thresholds_json=quality_thresholds_json,
        min_mean_f1=min_mean_f1,
        max_relative_error=max_relative_error,
    )
    tau_q = (
        1.0 - quality_alpha
        if quality_metric == QUALITY_GATE_METRIC
        else quality_threshold(baseline_quality, quality_alpha)
    )
    config_values = {
        "rounds": int(rounds),
        "quality_metric": str(quality_metric),
        "quality_alpha": float(quality_alpha),
        **optimizer_option_values,
    }
    if quality_thresholds_json is not None:
        config_values["quality_thresholds_json"] = quality_thresholds_json
        config_values["quality_threshold_source"] = quality_threshold_source
    if config_values.get("repair_bo_enabled"):
        config_values["quality_probability_affects_acquisition"] = False
    optimizer_config = OptimizerConfig(**config_values)
    optimizer_config.validate()
    from agents.tuning.dqbo.src.services.engine.repair_timeout import query_timeout_policy

    timeout_policy = query_timeout_policy(optimizer_config, baseline_result)
    if (
        optimizer_config.repair_bo_enabled
        and optimizer_config.repair_scope_version == "repair-scope.triplet.v1"
        and tuple(optimizer_config.repair_scope_fields) != REPAIR_BO_V1_FIELDS
    ):
        raise ValueError(
            "RepairBO v1 must instantiate exactly "
            "[quantization, kv_cache_dtype, dtype]; additional repair groups "
            "are an architectural extension, not a v1 experimental claim"
        )
    workload_contract = _workload_contract(trial_argv, baseline_result)
    if not workload_contract["matches_baseline_trial"]:
        raise ValueError(
            "Candidate workload differs from the measured Baseline trial: "
            f"{workload_contract['mismatches']}"
        )

    # 第 4 步：冻结 BO executor、搜索域和代理模型契约并原子写入本轮目录。
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    context_path = output_path / "bo_context.json"
    guidance_policy = _guidance_policy("disabled")
    runtime_constraint_scope = _runtime_constraint_scope(baseline_result)
    repair_scope = (
        RepairScope.from_space(
            space,
            fields=optimizer_config.repair_scope_fields,
            scope_version=optimizer_config.repair_scope_version,
            runtime_constraints=repair_scope_runtime_constraints(
                runtime_constraint_scope
            ),
        )
        if optimizer_config.repair_bo_enabled
        else None
    )
    if repair_scope is not None:
        # The former per-round search-space guidance is outside RepairBO v1.
        # Cold-start generation remains available, but post-warmup native BO is
        # measured without temporary LLM range guidance.
        guidance_policy = {
            **guidance_policy,
            "cold_start_enabled": False,
            "llm_guidance_enabled": False,
            "dynamic_observation_guidance_enabled": False,
            "repair_bo_owns_post_measurement_llm_reasoning": True,
        }
    legacy_contract_available = all(
        canonical_screened_name(name) in SUPPORTED_SCREENED_PARAMETERS
        for name in selected_names
    )
    if optimizer_config.repair_bo_enabled and not legacy_contract_available:
        raise ValueError(
            "RepairBO v1 requires the executable legacy-equivalent vLLM domain"
        )
    legacy_space = None
    if legacy_contract_available:
        legacy_args = build_legacy_namespace(
            trial_argv=trial_argv,
            parameter_results_path=parameter_path,
            domain_spec_path=domain_path,
            baseline_config_path=baseline_path,
            gpu_count=2,
        )
        legacy_space, _ = build_design_space(legacy_args, 2, 4096)
    effective_dimension_count = (
        legacy_space.num_paras
        if legacy_space is not None
        else len(space.tuned_parameters)
    )
    effective_parameter_names = (
        list(legacy_space.para_names)
        if legacy_space is not None
        else [item.name for item in space.tuned_parameters]
    )
    runtime_hard_constraints = ["max_num_seqs <= max_num_batched_tokens"]
    if runtime_constraint_scope["uncompressed_kv_model_dtype_alignment"]:
        runtime_hard_constraints.append(
            "measured FLASH_ATTN backend requires model dtype alignment only "
            "for explicit float16/bfloat16 KV cache"
        )
    context = {
        "schema_version": CONTEXT_SCHEMA,
        "agent": "candidate_optimizer",
        "implementation": (
            "candidate_optimizer_local.internal_legacy_general_bo_plus_joint_mc_cehvi"
            if optimizer_config.quality_probability_affects_acquisition
            else "candidate_optimizer_local.internal_legacy_general_bo_plus_performance_only_mc_ehvi"
        ),
        "legacy_contract_available": legacy_contract_available,
        "source_parameter_results": parameter_path,
        "source_domain_spec": domain_path,
        "source_baseline_config": baseline_path,
        "source_baseline_trial_result": baseline_result_path,
        "source_sha256": {
            "parameter_results": _sha256(parameter_path),
            "domain_spec": _sha256(domain_path),
            "baseline_config": _sha256(baseline_path),
            "baseline_trial_result": _sha256(baseline_result_path),
        },
        "output_dir": str(output_path),
        "result_path": str(output_path / RESULT_NAME),
        "history_path": str(output_path / "bo_history.json"),
        "optimizer_config": {
            name: getattr(optimizer_config, name)
            for name in optimizer_config.__dataclass_fields__
        },
        "quality_probability_gate": optimizer_config.quality_probability_gate_contract(),
        "query_timeout_policy": timeout_policy,
        "quality_constraint": {
            "metric": quality_metric,
            "baseline_quality": baseline_quality,
            "baseline_run_count": 1,
            "alpha": float(quality_alpha),
            "tau_q": tau_q,
            "threshold_source": (
                quality_threshold_source
                if quality_thresholds_json is not None
                else (
                    AGGREGATE_QUALITY_THRESHOLD_SOURCE
                    if aggregate_quality_gate
                    else None
                )
            ),
            "aggregate_thresholds": {
                "min_mean_f1": min_mean_f1,
                "max_relative_error": max_relative_error,
            },
            "query_metric_thresholds": (
                json.loads(quality_thresholds_json)
                if quality_thresholds_json is not None
                else None
            ),
            "relative_threshold_factors": relative_quality_threshold_factors,
            "absolute_thresholds": absolute_quality_thresholds,
            "baseline_query_metric_values": (
                query_configured_metric_means(
                    baseline_result, json.loads(quality_thresholds_json)
                )
                if quality_thresholds_json is not None
                else None
            ),
            "threshold_formulas": (
                quality_threshold_formulas
                if quality_thresholds_json is not None
                else None
            ),
            "rule": (
                "mean_f1_score >= min_mean_f1 and max_relative_error <= max_relative_error"
                if aggregate_quality_gate
                else (
                    "all configured per-query metrics retain their baseline "
                    "fractions; absolute metrics pass their configured floors"
                )
                if quality_threshold_source == RELATIVE_QUALITY_THRESHOLD_SOURCE
                else "quality_gate_passed is true"
                if quality_metric == "sembench_summary.quality_gate_passed"
                else "quality / baseline_quality >= 1 - alpha"
            ),
        },
        "fixed_search_space": {
            "parameter_count": len(space.parameters),
            "tuned_parameter_count": len(space.tuned_parameters),
            "encoded_dimension": space.dimension,
            "parameter_names": [item.name for item in space.parameters],
            "domains_are_rewritten": False,
            "effective_hebo_dimension_count": effective_dimension_count,
            "effective_hebo_parameter_names": effective_parameter_names,
            "runtime_hard_constraints": runtime_hard_constraints,
        },
        "runtime_constraint_scope": runtime_constraint_scope,
        "repair_bo": (
            {
                "enabled": True,
                "scope": repair_scope.contract(),
                "quality_decision": "measured_hard_thresholds_only",
                "quality_surrogate_used_by_repair": False,
                "candidate_scoring": "proposal_then_pre_execution_review_with_labeled_performance_predictions",
                "proposal_review_enabled": optimizer_config.repair_proposal_review_enabled,
                "proposal_max_revisions": optimizer_config.repair_proposal_max_revisions,
                "native_semantic_quality_preflight": False,
                "repair_child_preflight": "mechanical_hard_constraints_only",
                "legacy_round_guidance_enabled": False,
                "max_repair_rounds": optimizer_config.repair_max_rounds,
                "repair_executions_per_round": [2, 1],
                "repair_executions_per_round_are_maxima": True,
                "round_1_stop_policy": "A_review_rejection_after_revision_skips_execution_and_informs_B; otherwise_execute_A_then_B; any_measured_quality_feasible_and_positive_frozen_front_hvi_skips_round_2",
                "case_retrieval": "nearest_quality_success_and_failure_per_signature_by_non_triplet_configuration_only_no_global_top_k; inject_all_runtime_failures_separately_without_signature_grouping",
                "round_2_policy": (
                    "target_one_unsuccessful_branch; quality_repair_for_quality_or_runtime_failure; "
                    "local_performance_recovery_with_frozen_triplet_for_quality_feasible_branch"
                ),
                "performance_candidate_pool_size": (
                    optimizer_config.repair_performance_candidate_pool_size
                ),
                "performance_mutable_field_limit": None,
                "cold_start_total_evaluations": (
                    optimizer_config.repair_cold_start_evaluations
                ),
                "cold_start_native_sobol_trials": (
                    optimizer_config.repair_cold_start_native_trials
                ),
                "baseline_execution_policy": "reuse_precomputed_result_without_rerun",
            }
            if repair_scope is not None
            else {"enabled": False}
        ),
        "known_constraint_handler": {
            "tool": "repair_bo_candidate",
            "compatibility_alias": "repair_candidate_known_constraints",
            "invoked_before_every_trial": True,
            "invoked_during_candidate_generation": True,
            "candidate_generation_stage": (
                "before_surrogate_and_acquisition_evaluation"
            ),
            "canonical_equivalent_points_are_deduplicated": True,
            "post_selection_role": "idempotent_validation",
            "repairs_stay_within_fixed_range": True,
            "constraint_ids": list(KNOWN_CONSTRAINT_IDS),
        },
        "candidate_repair": {
            "tool": "repair_bo_candidate",
            "sources": ["manual_constraints"],
            "manual_constraints_are_hard": True,
            "invoked_before_every_trial": True,
            "repairs_stay_within_fixed_range": True,
            "llm_preflight_enabled": False,
        },
        "initial_design": {
            "method": "torch_scrambled_sobol_engine",
            "required_complete_observations": (
                8 if repair_scope is not None else 1 + effective_dimension_count
            ),
            "rule": (
                "baseline_plus_fixed_7_native_sobol_trials"
                if repair_scope is not None
                else "1 + effective_HEBO_dimension_count"
            ),
            "baseline_counts_when_inside_domain": True,
            "runtime_failures_count": False,
        },
        "cold_start_guidance": {"round_limit": 0, "rule": "current_tuning_round <= round_limit", "uses_complete_observation_count": False, "automatic_every_cold_start_round": False, "invocation_policy": "native_sobol_only"},
        "agentic_workflow": {
            "schema": "sembench.bo.agentic-workflow-policy.v3",
            "default_entry": "execute_bo",
            "cold_start_trial_count": optimizer_config.repair_cold_start_native_trials,
            "hebo_required_complete_observations": optimizer_config.repair_cold_start_evaluations,
            "cold_start_llm_rounds": [],
            "cold_start_sobol_rounds": list(range(1, optimizer_config.repair_cold_start_native_trials + 1)),
            "guidance_required_every_round": False,
            "repair_bo_enabled": repair_scope is not None,
        },
        "guidance_policy": guidance_policy,
        "objective": {
            "direction": "maximize",
            "vector": ["QPS", "-MeanLatency", "-P95Latency"],
            "quality_is_objective": False,
        },
        "surrogate_models": list(
            SURROGATE_NAMES[:3]
            if optimizer_config.quality_probability_gate_enabled or (legacy_contract_available and not optimizer_config.quality_probability_affects_acquisition)
            else SURROGATE_NAMES
        ),
        "surrogate_implementation": {
            "performance": "three independent internal GPyGP models",
            "conditional_space_enabled": (
                optimizer_config.conditional_model_enabled
            ),
            "quality_extension": (
                "disabled_no_quality_model"
                if optimizer_config.quality_probability_gate_enabled or (legacy_contract_available and not optimizer_config.quality_probability_affects_acquisition)
                else "one independent internal GPyGP model"
            ),
            "quality_probability_gate": optimizer_config.quality_probability_gate_contract(),
        },
        "acquisition": {
            "name": (
                "joint_mc_constrained_ehvi"
                if optimizer_config.quality_probability_affects_acquisition
                else "performance_only_mc_ehvi"
            ),
            "formula": (
                "E[delta_HV(f(x); P_F) * I(q(x) >= tau_q)]"
                if optimizer_config.quality_probability_affects_acquisition
                else "E[delta_HV(f(x); P_F)]"
            ),
            "candidate_generator": (
                "internal GeneralAcq LCB + EvolutionOpt(pop=100,iters=200) "
                + ("over canonical feasible configurations" if optimizer_config.project_before_acquisition
                   else "over raw configurations")
            ),
            "known_constraint_enforcement": (
                "deterministic projection before every surrogate/acquisition evaluation and before return"
                if optimizer_config.project_before_acquisition
                else "mechanical repair only after complete BO recommendation"
            ),
            "equivalent_conditional_points_deduplicated": optimizer_config.project_before_acquisition,
            "monte_carlo_samples": optimizer_config.monte_carlo_samples,
            "ehvi_matches_original": False,
            "ehvi_formula_matches_original": True,
            "monte_carlo_sample_count_modified_from_original": {
                "original": 64,
                "configured": optimizer_config.monte_carlo_samples,
                "intent": "increase_candidate_ranking_variation",
            },
            "quality_mc_indicator": (
                optimizer_config.quality_probability_affects_acquisition
            ),
            "scalar_ehvi_times_p_feasible": False,
            "quality_mean_hard_threshold": False,
            "quality_surrogate_affects_candidate_generation": False,
            "quality_surrogate_affects_final_ranking": (
                optimizer_config.quality_probability_affects_acquisition
            ),
            "p_run_affects_final_score": False,
            "quality_surrogate_role": (
                "paired posterior-sample feasibility indicator"
                if optimizer_config.quality_probability_affects_acquisition
                else "disabled_no_quality_model" if legacy_contract_available
                else "diagnostic_only_during_candidate_ranking"
            ),
            "observed_quality_role": "feasible Pareto membership only",
        },
        "allowed_algorithm_difference": [
            (
                "independent binary quality classifier for post-recommendation probability gate"
                if optimizer_config.quality_probability_gate_enabled
                else "measured quality label only; no quality surrogate"
                if legacy_contract_available and not optimizer_config.quality_probability_affects_acquisition
                else "fourth independent quality surrogate target tau_q - quality"
            ),
            (
                "paired quality posterior indicator in joint MC constrained EHVI"
                if optimizer_config.quality_probability_affects_acquisition
                else "quality posterior excluded from candidate ranking"
            ),
            "quality feasibility in observed Pareto construction",
            "paired posterior MC sample count reduced from 64 to 10",
        ],
        "legacy_algorithm_contract": {
            "initial_random_trials": "1 + effective_HEBO_dimension_count",
            "sobol": "torch.quasirandom.SobolEngine(scramble=True, seed=None)",
            "model_wrapper": "MultiTaskModel",
            "per_output_surrogate": "GPyGP",
            "general_acquisition": "LCB",
            "evolution_population": 100,
            "evolution_iterations": 200,
            "ehvi_monte_carlo_samples": optimizer_config.monte_carlo_samples,
            "joint_mc_quality_indicator": (
                optimizer_config.quality_probability_affects_acquisition
            ),
            "scalar_ehvi_times_p_feasible": False,
            "quality_mean_hard_threshold": False,
            "quality_surrogate_affects_candidate_generation": False,
            "quality_surrogate_affects_final_ranking": (
                optimizer_config.quality_probability_affects_acquisition
            ),
            "p_run_affects_final_score": False,
            "hidden_constraint": "RandomForestRegressor with original dynamic threshold",
            "optimizer_compatibility_options_ignored": [
                "candidate_pool_size",
                "reference_margin",
                "random_seed",
            ],
        },
        "executor": {
            "implementation": "candidate_optimizer_local.owned_flock_trial",
            "trial_argv": [str(item) for item in trial_argv],
            "python_executable": python_executable,
            "timeout_seconds": float(trial_timeout_seconds),
        },
        "workload_contract": workload_contract,
        "static_workload": build_static_workload_context(baseline_result),
    }
    if repair_scope is not None:
        slots = (list(range(1, min(2, optimizer_config.repair_cold_start_native_trials) + 1))
                 if optimizer_config.repair_llm_cold_start_enabled else [])
        context['repair_cold_start'] = {**repair_cold_start.POLICY, 'trial_slots': slots,
                                       'enabled': optimizer_config.repair_llm_cold_start_enabled}
        context['repair_bo']['cold_start_native_sobol_trials'] = optimizer_config.repair_cold_start_native_trials - len(slots)
        context['repair_bo']['cold_start_llm_trials'] = len(slots)
        context['agentic_workflow']['cold_start_llm_rounds'] = slots
        context['agentic_workflow']['cold_start_sobol_rounds'] = list(range(len(slots) + 1, optimizer_config.repair_cold_start_native_trials + 1))
        context['initial_design'].update(
            method=('memory_candidates_then_torch_scrambled_sobol_engine' if slots
                    else 'torch_scrambled_sobol_engine'),
            rule=('baseline_plus_memory_candidates_and_native_sobol_slots' if slots
                  else 'baseline_plus_native_sobol_slots'),
        )
    write_json_atomic(context_path, context)

    # A ReAct round always begins with inspect_bo_round_state.  Materialize the
    # baseline-backed empty history here so that first inspection is genuinely
    # read-only and never has to bootstrap through a failed execute call.
    bootstrap_executor = BOTrialExecutor(
        output_dir=str(output_path),
        baseline_config=baseline_path,
        trial_argv=[str(item) for item in trial_argv],
        python_executable=python_executable,
        timeout_seconds=float(trial_timeout_seconds),
    )
    if legacy_contract_available:
        bootstrap_optimizer: LegacyEquivalentQualityBO | CandidateOptimizer = (
            LegacyEquivalentQualityBO(
                executor=bootstrap_executor,
                trial_argv=[str(item) for item in trial_argv],
                baseline_config_path=baseline_path,
                baseline_result_path=baseline_result_path,
                parameter_results_path=parameter_path,
                domain_spec_path=domain_path,
                output_dir=str(output_path),
                config=optimizer_config,
                    candidate_repairer=None,
            )
        )
    else:
        bootstrap_optimizer = CandidateOptimizer(
            search_space=space,
            executor=bootstrap_executor,
            baseline_config_path=baseline_path,
            baseline_result_path=baseline_result_path,
            parameter_results_path=parameter_path,
            output_dir=str(output_path),
            config=optimizer_config,
            constraint_handler=_known_constraint_handler(context, space),
            candidate_repairer=None,
        )
    initial_history = bootstrap_optimizer.initialize_history()
    if optimizer_config.repair_bo_enabled:
        initial_history["workload_contract"] = workload_contract
        initial_history["static_workload"] = context["static_workload"]
        if optimizer_config.repair_historical_memory_enabled:
            initial_history['repair_cold_start_memory'] = repair_cold_start.load_historical_memory(context)
        write_json_atomic(context["history_path"], initial_history)
    return {
        "status": "ready",
        "context_path": str(context_path),
        "result_path": context["result_path"],
        "history_path": context["history_path"],
        "rounds": optimizer_config.rounds,
        "tau_q": tau_q,
        "fixed_search_space": context["fixed_search_space"],
        "guidance_policy": context["guidance_policy"],
        "default_execution_entry": "execute_bo",
        "agentic_workflow": context["agentic_workflow"],
        "repair_bo": context["repair_bo"],
        "history_initialized": True,
        "initial_trial_count": len(initial_history["trials"]),
    }


def _repair_manual_constraints(
    context: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the declared/manual runtime constraints in the frozen domain."""

    if context.get("legacy_contract_available") is True:
        executor_spec = context.get("executor")
        if not isinstance(executor_spec, Mapping):
            raise TypeError("BO context executor must be an object")
        result = repair_with_legacy_constraints(
            trial_argv=[str(item) for item in executor_spec.get("trial_argv", [])],
            parameter_results_path=context["source_parameter_results"],
            domain_spec_path=context["source_domain_spec"],
            baseline_config_path=context["source_baseline_config"],
            candidate=candidate,
        )
        return result
    space = FixedSearchSpace.from_json(context["source_domain_spec"])
    result = _known_constraint_handler(context, space).repair(candidate)
    return {
        "candidate_bo_names": space.canonical_to_bo(result.candidate),
        **result.to_dict(),
    }


_STARTUP_KV_CAPACITY_ERRORS = (
    "No available memory for the cache blocks",
    "larger than the available KV cache memory",
)


_STARTUP_CAPACITY_SIGNATURE_FIELDS = (
    "quantization",
    "dtype",
    "tp",
    "data_parallel_size",
    "model_impl",
    "kv_cache_dtype",
    "block_size",
    "cpu_offload_gb",
    "kv_offloading_backend",
    "kv_offloading_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enforce_eager",
    "enable_chunked_prefill",
    "speculative_method",
    "num_speculative_tokens",
)


def _capacity_group(candidate: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Build a narrow startup-capacity signature for mechanical reuse.

    A startup KV-capacity failure is monotone in ``gpu_memory_utilization`` only
    while the remaining memory-affecting configuration is held fixed.  In
    particular, KV dtype, scheduler profile sizes, eager mode, offloading, and
    speculative decoding can all alter the amount of memory available at
    engine startup.  They must therefore not be generalized merely because the
    representation family and TP happen to match.

    The LLM capacity preflight may still reason across related configurations;
    this signature deliberately scopes only the final mechanical projection.
    """

    signature: list[tuple[str, Any]] = []
    for name in _STARTUP_CAPACITY_SIGNATURE_FIELDS:
        if name == "tp":
            value = candidate.get("tp", candidate.get("tensor_parallel_size"))
        else:
            value = candidate.get(name)
        if isinstance(value, np.generic):
            value = value.item()
        # These fields are executable scalar values.  Fall back to a stable
        # JSON representation if a future field becomes structured so the
        # cache key remains hashable without silently dropping that evidence.
        if isinstance(value, (Mapping, list, tuple, set)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        signature.append((name, value))
    return tuple(signature)


@lru_cache(maxsize=32)
def _startup_memory_failure_floors_cached(
    history_path: str,
    history_mtime_ns: int,
    history_size: int,
    trial_log_dir: str,
) -> tuple[tuple[tuple[tuple[str, Any], ...], float], ...]:
    """Derive floors only from vLLM's deterministic pre-ready KV errors.

    ``mtime`` and ``size`` are cache-key inputs, so a newly committed trial
    invalidates the receipt without rereading the multi-megabyte history for
    every acquisition candidate.
    """

    del history_mtime_ns, history_size
    payload = _load_object(history_path)
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return ()
    failures: dict[tuple[tuple[str, Any], ...], float] = {}
    logs = Path(trial_log_dir)
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        if trial.get("status") != "runtime_failed" or trial.get(
            "failure_kind"
        ) != CAPACITY_FAILURE:
            continue
        trial_id = str(trial.get("trial_id") or "")
        candidate = trial.get("candidate")
        if not trial_id or not isinstance(candidate, Mapping):
            continue
        log_path = logs / f"{trial_id}.vllm.log"
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(message in log_text for message in _STARTUP_KV_CAPACITY_ERRORS):
            continue
        raw_utilization = candidate.get("gpu_memory_utilization")
        try:
            failed_utilization = float(raw_utilization)
        except (TypeError, ValueError):
            continue
        group = _capacity_group(candidate)
        failures[group] = max(failures.get(group, 0.0), failed_utilization)
    return tuple(
        (group, failed_utilization)
        for group, failed_utilization in sorted(
            failures.items(), key=lambda item: repr(item[0])
        )
    )


def _startup_memory_failure_floors(
    context: Mapping[str, Any],
) -> dict[tuple[tuple[str, Any], ...], float]:
    """Load the latest committed startup-capacity evidence for this run."""

    output_dir = Path(str(context["output_dir"])).expanduser().resolve()
    agentic_history = output_dir / "agentic_bo_history.json"
    configured_history = Path(str(context["history_path"])).expanduser().resolve()
    history_path = agentic_history if agentic_history.is_file() else configured_history
    if not history_path.is_file():
        return {}
    stat = history_path.stat()
    records = _startup_memory_failure_floors_cached(
        str(history_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(output_dir / "trials" / "logs"),
    )
    return dict(records)


def _apply_learned_startup_memory_floor(
    context: Mapping[str, Any],
    space: FixedSearchSpace,
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Project a repeated startup-capacity region to an in-domain safe side.

    The Range reference is preferred because it is the pre-established stable
    operating point.  If a later failure reaches that point, advance one
    percentage point while remaining inside the frozen domain.
    """

    repaired = dict(candidate)
    failed_utilization = _startup_memory_failure_floors(context).get(
        _capacity_group(repaired)
    )
    if failed_utilization is None:
        return repaired, None
    parameter = space.by_name.get("gpu_memory_utilization")
    if parameter is None or not isinstance(parameter.domain, Mapping):
        return repaired, None
    raw_current = repaired.get("gpu_memory_utilization")
    try:
        current = float(raw_current)
        upper = float(parameter.domain["upper"])
        reference = float(parameter.reference_value)
    except (KeyError, TypeError, ValueError):
        return repaired, None
    target = min(upper, max(reference, round(failed_utilization + 0.01, 2)))
    # ``failed_utilization + 0.01`` may still be below the Range reference
    # chosen as the stable operating point.  Treat ``target`` as a true floor,
    # not merely as an escape from the exact observed failure coordinate.
    if current >= target - 1e-9:
        return repaired, None
    if not parameter.contains(target):
        return repaired, None
    repaired[parameter.bo_name] = target
    return repaired, {
        "constraint_id": "learned_vllm_startup_memory_floor",
        "parameter": parameter.bo_name,
        "before": raw_current,
        "after": target,
        "failed_at_or_below": failed_utilization,
        "capacity_group": dict(_capacity_group(repaired)),
        "reason": (
            "same representation group previously failed before vLLM ready "
            "with an explicit insufficient-KV-capacity error"
        ),
    }


def _repair_manual_constraints_canonical(
    context: Mapping[str, Any],
    space: FixedSearchSpace,
    candidate: Mapping[str, Any],
    *,
    apply_learned_startup_floor: bool = True,
) -> dict[str, Any]:
    """Apply declared hard constraints and bridge Legacy/Range representations.

    The Legacy normalizer remains the executable compatibility authority, but
    its historical repair receipts do not identify the constraint that caused
    each edit.  Run the explicit frozen-Range constraint handler first so the
    same edits retain their stable ``constraint_id`` in the preflight audit.
    A second explicit pass below verifies that Legacy projection did not
    reintroduce a declared conflict.
    """

    constraint_handler = _known_constraint_handler(context, space)
    declared_result = constraint_handler.repair(space.bo_to_canonical(candidate))
    prepared_candidate = dict(candidate)
    prepared_candidate.update(space.canonical_to_bo(declared_result.candidate))
    result = dict(_repair_manual_constraints(context, prepared_candidate))
    result["repairs"] = [
        dict(item) for item in declared_result.repairs
    ] + [
        dict(item)
        for item in result.get("repairs", [])
        if isinstance(item, Mapping)
    ]
    result["known_constraint_ids"] = list(KNOWN_CONSTRAINT_IDS)
    result["validation_errors"] = list(
        dict.fromkeys(
            [str(item) for item in declared_result.validation_errors]
            + [str(item) for item in result.get("validation_errors", [])]
        )
    )
    result["valid"] = not result["validation_errors"]
    executable_candidate = dict(result.get("candidate") or {})
    projected_candidate, learned_repair = (
        _apply_learned_startup_memory_floor(
            context, space, executable_candidate
        )
        if apply_learned_startup_floor
        else (dict(executable_candidate), None)
    )
    if learned_repair is not None:
        initial_validation_errors = [
            str(item) for item in result.get("validation_errors", [])
        ]
        initial_repairs = [
            dict(item)
            for item in result.get("repairs", [])
            if isinstance(item, Mapping)
        ]
        result = dict(_repair_manual_constraints(context, projected_candidate))
        result["repairs"] = initial_repairs + [learned_repair] + [
            dict(item)
            for item in result.get("repairs", [])
            if isinstance(item, Mapping)
        ]
        result["implementation"] = (
            f"{result.get('implementation', 'manual_constraints')}"
            "+learned_vllm_startup_memory_floor"
        )
        result["known_constraint_ids"] = list(KNOWN_CONSTRAINT_IDS)
        result["validation_errors"] = list(
            dict.fromkeys(
                initial_validation_errors
                + [str(item) for item in result.get("validation_errors", [])]
            )
        )
        result["valid"] = not result["validation_errors"]
        executable_candidate = dict(result.get("candidate") or {})

    # Legacy normalization owns executable aliases and conditional defaults.
    # Verify its projected candidate against every predefined constraint once
    # more.  If it introduced a conflict, apply that deterministic correction
    # and project the result through Legacy one final time.
    post_legacy = constraint_handler.repair(
        space.bo_to_canonical(executable_candidate)
    )
    if post_legacy.repairs:
        previous_repairs = [
            dict(item)
            for item in result.get("repairs", [])
            if isinstance(item, Mapping)
        ]
        corrected_executable = dict(executable_candidate)
        corrected_executable.update(space.canonical_to_bo(post_legacy.candidate))
        projected_result = dict(
            _repair_manual_constraints(context, corrected_executable)
        )
        projected_result["repairs"] = previous_repairs + [
            dict(item) for item in post_legacy.repairs
        ] + [
            dict(item)
            for item in projected_result.get("repairs", [])
            if isinstance(item, Mapping)
        ]
        projected_result["implementation"] = (
            f"{projected_result.get('implementation', 'manual_constraints')}"
            "+post_legacy_known_constraint_verification"
        )
        projected_result["known_constraint_ids"] = list(KNOWN_CONSTRAINT_IDS)
        projected_result["validation_errors"] = list(
            dict.fromkeys(
                [str(item) for item in result.get("validation_errors", [])]
                + [str(item) for item in post_legacy.validation_errors]
                + [
                    str(item)
                    for item in projected_result.get("validation_errors", [])
                ]
            )
        )
        projected_result["valid"] = not projected_result["validation_errors"]
        result = projected_result
        executable_candidate = dict(result.get("candidate") or {})
    elif post_legacy.validation_errors:
        result["validation_errors"] = list(
            dict.fromkeys(
                [str(item) for item in result.get("validation_errors", [])]
                + [str(item) for item in post_legacy.validation_errors]
            )
        )
        result["valid"] = False

    canonical_candidate = space.bo_to_canonical(executable_candidate)
    result["candidate"] = canonical_candidate
    result["candidate_canonical_names"] = dict(canonical_candidate)
    result["executable_candidate"] = executable_candidate
    return result


def _restore_executable_candidate(
    result: Mapping[str, Any], space: FixedSearchSpace
) -> dict[str, Any]:
    """Return the executable candidate while retaining canonical repair evidence."""

    restored = dict(result)
    # The canonical repair path has already converted ``candidate`` and stores
    # the executable alias form separately. Converting that canonical object as
    # though it still used BO aliases silently corrupts fields whose names differ.
    explicit_canonical = restored.get("candidate_canonical_names")
    if isinstance(explicit_canonical, Mapping):
        canonical_candidate = dict(explicit_canonical)
    else:
        candidate_payload = dict(restored.get("candidate") or {})
        canonical_names = {parameter.name for parameter in space.parameters}
        canonical_candidate = (
            candidate_payload
            if canonical_names.issubset(candidate_payload)
            else space.bo_to_canonical(candidate_payload)
        )
    executable_candidate = dict(
        restored.pop(
            "executable_candidate",
            space.canonical_to_bo(canonical_candidate),
        )
        or {}
    )
    restored["candidate"] = executable_candidate
    restored["candidate_canonical_names"] = dict(canonical_candidate)
    # Recompute this audit view from the final canonical candidate instead of
    # retaining a possibly stale pre-repair snapshot.
    restored["candidate_bo_names"] = space.canonical_to_bo(canonical_candidate)
    return restored


def _repair_candidate_mechanical_only(
    context_path: str,
    context: Mapping[str, Any],
    space: FixedSearchSpace,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only frozen-Range and declared runtime constraints.

    Candidate preflight applies only declared mechanical validity constraints.
    Measured quality repair and local reflection belong to RepairCoordinator.
    """

    result = _restore_executable_candidate(
        _repair_manual_constraints_canonical(
            context,
            space,
            candidate,
            apply_learned_startup_floor=False,
        ),
        space,
    )
    result["repair_sources"] = ["manual_constraints"]
    result["known_constraint_ids"] = list(KNOWN_CONSTRAINT_IDS)
    result["status"] = "ready" if result.get("valid") is True else "constraint_failed"
    return {
        "context_path": _resolve(context_path),
        "input_candidate": dict(candidate),
        **result,
    }


def repair_bo_candidate(context_path: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Repair declared mechanical constraints before executing a candidate."""
    context = _load_object(context_path)
    _validate_frozen_context(context)
    space = FixedSearchSpace.from_json(context["source_domain_spec"])
    return _repair_candidate_mechanical_only(context_path, context, space, candidate)




def repair_candidate_known_constraints(
    context_path: str, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Backward-compatible alias for the composite BO candidate repair tool."""

    return repair_bo_candidate(context_path, candidate)






def _cached_candidate_repairer(
    context_path: str, *, guidance_candidate_filter: Mapping[str, Any] | None = None,
    guidance_source: str | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    context = _load_object(context_path)
    _validate_frozen_context(context)
    space = FixedSearchSpace.from_json(context["source_domain_spec"])
    return lambda candidate: _repair_candidate_mechanical_only(context_path, context, space, candidate)




def propose_next_bo_candidate(
    context_path: str,
    *,
    temporary_candidate_filter: Mapping[str, Any] | None = None,
    guidance_source: str | None = None,
) -> dict[str, Any]:
    """Ask BO for the next point in an optional one-round temporary region.

    This is read-only with respect to BO history and never executes a Trial.  A
    non-empty filter is injected as an additional acquisition-optimizer
    constraint, so the actual configuration is still selected by the native BO
    posterior/acquisition rather than copied from an Observation.
    """

    context = _load_object(context_path)
    _validate_frozen_context(context)
    guidance_source = _validated_guidance_source(
        temporary_candidate_filter, guidance_source
    )
    _validate_guidance_round(context, guidance_source)
    if context.get("legacy_contract_available") is not True:
        raise ValueError(
            "temporary Observation search spaces currently require the real "
            "legacy-equivalent Candidate domain"
        )
    config = OptimizerConfig(**dict(context["optimizer_config"]))
    executor_spec = context.get("executor")
    if not isinstance(executor_spec, Mapping):
        raise TypeError("BO context executor must be an object")
    # The optimizer constructor shares the exact runtime configuration, but its
    # recommend_next method never invokes this executor.
    executor = BOTrialExecutor(
        output_dir=context["output_dir"],
        baseline_config=context["source_baseline_config"],
        trial_argv=list(executor_spec.get("trial_argv", [])),
        python_executable=executor_spec.get("python_executable"),
        timeout_seconds=float(executor_spec.get("timeout_seconds", 3600.0)),
    )
    optimizer = LegacyEquivalentQualityBO(
        executor=executor,
        trial_argv=list(executor_spec.get("trial_argv", [])),
        baseline_config_path=context["source_baseline_config"],
        baseline_result_path=context["source_baseline_trial_result"],
        parameter_results_path=context["source_parameter_results"],
        domain_spec_path=context["source_domain_spec"],
        output_dir=context["output_dir"],
        config=config,
        candidate_repairer=_cached_candidate_repairer(
            context_path,
            guidance_candidate_filter=temporary_candidate_filter,
            guidance_source=guidance_source,
        ),
    )
    recommendation = optimizer.recommend_next(
        temporary_candidate_filter=temporary_candidate_filter,
        guidance_source=guidance_source,
    )
    safety = dict(recommendation["known_constraint_check"])
    if safety.get("status") != "ready":
        raise RuntimeError(
            "native BO recommendation failed the mandatory composite candidate repair"
        )
    return {
        "status": "recommended",
        "context_path": _resolve(context_path),
        "temporary_search_space": {
            "active": bool(temporary_candidate_filter),
            "candidate_filter": dict(temporary_candidate_filter or {}),
            "source": guidance_source,
            "lifetime": "one_bo_round",
            "restored_after_recommendation": True,
        },
        "candidate": dict(safety["candidate"]),
        "candidate_bo_names": dict(recommendation.get("candidate_bo_names") or {}),
        "selection": dict(recommendation["selection"]),
        "known_constraint_check": safety,
        "selected_by": "native_bo_acquisition",
        "executes_trial": False,
    }


def execute_bo(context_path: str) -> dict[str, Any]:
    """Run the native BO and local Repair loop, resuming persisted Trials."""
    context = _load_object(context_path)
    _validate_frozen_context(context)
    config = OptimizerConfig(**dict(context["optimizer_config"]))
    if config.repair_bo_enabled or context.get("legacy_contract_available") is True:
        from .execution_loop import execute_native_bo
        return execute_native_bo(context_path)
    executor_spec = context["executor"]
    executor = BOTrialExecutor(
        output_dir=context["output_dir"], baseline_config=context["source_baseline_config"],
        trial_argv=list(executor_spec.get("trial_argv", [])),
        python_executable=executor_spec.get("python_executable"),
        timeout_seconds=float(executor_spec.get("timeout_seconds", 3600.0)),
    )
    space = FixedSearchSpace.from_json(context["source_domain_spec"])
    optimizer = CandidateOptimizer(
        search_space=space, executor=executor,
        baseline_config_path=context["source_baseline_config"],
        baseline_result_path=context["source_baseline_trial_result"],
        parameter_results_path=context["source_parameter_results"],
        output_dir=context["output_dir"], config=config,
        constraint_handler=_known_constraint_handler(context, space),
        candidate_repairer=_cached_candidate_repairer(context_path),
    )
    return {"status": "executed", "context_path": _resolve(context_path), **optimizer.run()}




def execute_bo_round(
    context_path: str,
    *,
    temporary_candidate_filter: Mapping[str, Any] | None = None,
    guidance_source: str | None = None,
    guidance_observation_keys: Sequence[str] | None = None,
    cold_start_candidate: Mapping[str, Any] | None = None,
    guidance_context: Mapping[str, Any] | None = None,
    warmup_complete_observations: int | None = None,
    agentic_mode: bool = False,
    structured_inference_client: Any | None = None,
) -> dict[str, Any]:
    """Execute one native-BO point or one complete cold-start Agent candidate."""

    context = _load_object(context_path)
    _validate_frozen_context(context)
    guidance_source = _validated_guidance_source(
        temporary_candidate_filter, guidance_source
    )
    if temporary_candidate_filter and cold_start_candidate is not None:
        raise ValueError(
            "temporary_candidate_filter and cold_start_candidate are mutually exclusive"
        )
    if (
        cold_start_candidate is not None
        and guidance_source not in {"adjust_cold_start_search_space", "repair_cold_start"}
    ):
        raise ValueError("cold_start_candidate requires cold-start guidance provenance")
    _validate_guidance_round(context, guidance_source, agentic_mode=agentic_mode)
    normalized_observation_keys = sorted(
        {
            str(item).strip()
            for item in (guidance_observation_keys or [])
            if str(item).strip()
        }
    )
    if normalized_observation_keys and guidance_source is None:
        raise ValueError(
            "guidance_observation_keys require an explicit guidance_source"
        )
    if context.get("legacy_contract_available") is not True:
        raise ValueError(
            "round-wise Observation guidance currently requires the real "
            "legacy-equivalent Candidate domain"
        )
    config = OptimizerConfig(**dict(context["optimizer_config"]))
    executor_spec = context.get("executor")
    if not isinstance(executor_spec, Mapping):
        raise TypeError("BO context executor must be an object")
    executor = BOTrialExecutor(
        output_dir=context["output_dir"],
        baseline_config=context["source_baseline_config"],
        trial_argv=list(executor_spec.get("trial_argv", [])),
        python_executable=executor_spec.get("python_executable"),
        timeout_seconds=float(executor_spec.get("timeout_seconds", 3600.0)),
    )
    optimizer = LegacyEquivalentQualityBO(
        executor=executor,
        trial_argv=list(executor_spec.get("trial_argv", [])),
        baseline_config_path=context["source_baseline_config"],
        baseline_result_path=context["source_baseline_trial_result"],
        parameter_results_path=context["source_parameter_results"],
        domain_spec_path=context["source_domain_spec"],
        output_dir=context["output_dir"],
        config=config,
        candidate_repairer=_cached_candidate_repairer(
            context_path,
            guidance_candidate_filter=temporary_candidate_filter,
            guidance_source=guidance_source,
        ),
        warmup_complete_observations=warmup_complete_observations,
    )
    history_before = (
        _load_object(optimizer.history_path)
        if optimizer.history_path.is_file()
        else {"trials": []}
    )
    next_bo_iteration = native_bo_iteration_count(history_before) + 1
    cold_receipt = None
    if (repair_cold_start.enabled(context)
        and next_bo_iteration in context['repair_cold_start']['trial_slots']
        and cold_start_candidate is None):
        cold_receipt = repair_cold_start.recommend(
            context=context, history=history_before, space=optimizer.fixed_search_space,
            validator=lambda c: _repair_candidate_mechanical_only(context_path, context, optimizer.fixed_search_space, c),
            slot=next_bo_iteration, client=structured_inference_client,
        )
        cold_start_candidate = cold_receipt.get('recommended_candidate')
        if cold_start_candidate is not None:
            guidance_source = 'repair_cold_start'
            normalized_observation_keys = cold_receipt.get('observation_ids', [])
        guidance_context = {**dict(guidance_context or {}), 'repair_cold_start': cold_receipt}
        history_before['repair_cold_start_memory'] = repair_cold_start.repair_memory(context['output_dir'])
        write_json_atomic(optimizer.history_path, history_before)
    trial_role = (
        "cold_start"
        if next_bo_iteration <= config.repair_cold_start_native_trials
        else "native_bo"
    )
    execution = optimizer.run_one(
        temporary_candidate_filter=temporary_candidate_filter,
        guidance_source=guidance_source,
        guidance_observation_keys=normalized_observation_keys,
        candidate_override=cold_start_candidate,
        guidance_context=guidance_context,
        trial_metadata={
            "trial_role": trial_role,
            "bo_iteration": next_bo_iteration,
            **({'repair_cold_start': cold_receipt} if cold_receipt is not None else {}),
        },
    )
    repair_outcome: dict[str, Any] | None = None
    if config.repair_bo_enabled and trial_role == "native_bo":
        space = optimizer.fixed_search_space
        scope = optimizer.repair_scope or RepairScope.from_space(
            space,
            fields=config.repair_scope_fields,
            scope_version=config.repair_scope_version,
            runtime_constraints=repair_scope_runtime_constraints(
                context.get("runtime_constraint_scope")
                if isinstance(context.get("runtime_constraint_scope"), Mapping)
                else None
            ),
        )

        def mechanical(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            return _repair_candidate_mechanical_only(
                context_path, context, space, candidate
            )

        repair_optimizer = LegacyEquivalentQualityBO(
            executor=executor,
            trial_argv=list(executor_spec.get("trial_argv", [])),
            baseline_config_path=context["source_baseline_config"],
            baseline_result_path=context["source_baseline_trial_result"],
            parameter_results_path=context["source_parameter_results"],
            domain_spec_path=context["source_domain_spec"],
            output_dir=context["output_dir"],
            config=config,
                candidate_repairer=mechanical,
            warmup_complete_observations=warmup_complete_observations,
        )

        def execute_repair(
            candidate: Mapping[str, Any], metadata: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return repair_optimizer.run_one(
                candidate_override=candidate,
                guidance_source="repair_bo",
                guidance_context={
                    "episode_id": metadata.get("episode_id"),
                    "parent_trial_id": metadata.get("parent_trial_id"),
                    "repair_anchor_trial_id": metadata.get(
                        "repair_anchor_trial_id"
                    ),
                    "strategy": metadata.get("repair_strategy"),
                    "repair_mode": metadata.get("repair_mode"),
                    "repair_round": metadata.get("repair_round"),
                    "repair_allowed_fields": metadata.get(
                        "repair_allowed_fields"
                    ),
                },
                direct_candidate_kind="repair",
                trial_metadata={
                    **dict(metadata),
                    "bo_iteration": None,
                    "parent_bo_iteration": next_bo_iteration,
                },
            )

        repair_client = structured_inference_client
        repair_client_error: str | None = None
        if repair_client is None:
            try:
                repair_client = default_candidate_structured_inference_client()
            except Exception as exc:
                repair_client_error = f"{type(exc).__name__}: {exc}"
        coordinator = RepairCoordinator(
            history_path=optimizer.history_path,
            result_path=optimizer.result_path,
            output_dir=context["output_dir"],
            space=space,
            scope=scope,
            evaluation_budget=config.rounds,
            cold_start_evaluations=config.repair_cold_start_evaluations,
            adaptive_threshold_enabled=config.repair_adaptive_threshold_enabled,
            hv_epsilon=config.repair_hv_epsilon,
            boundary_utility_epsilon=config.repair_boundary_utility_epsilon,
            monte_carlo_samples=config.monte_carlo_samples,
            reference_margin=config.reference_margin,
            random_seed=config.random_seed,
            context_limit=config.repair_context_limit,
            performance_candidate_pool_size=(
                config.repair_performance_candidate_pool_size
            ),
            proposal_review_enabled=config.repair_proposal_review_enabled,
            proposal_max_revisions=config.repair_proposal_max_revisions,
            reflect_enabled=config.repair_reflect_enabled,
            validator=mechanical,
            execute_candidate=execute_repair,
            prediction_provider=repair_optimizer.predict_current_performance,
            structured_client=repair_client,
        )
        repair_outcome = coordinator.maybe_repair(execution["trial"]["trial_id"])
        if repair_client_error is not None:
            repair_outcome["llm_client_error"] = repair_client_error
            repair_outcome["llm_fallback"] = "strategy_specific_deterministic_fallback"
    history = _load_object(execution["history_path"])
    return {
        **execution,
        "rounds_completed": len(history.get("trials", [])),
        "evaluations_completed": (
            evaluation_count(history)
            if config.repair_bo_enabled
            else len(history.get("trials", []))
        ),
        "observations_available": (
            evaluation_count(history)
            if config.repair_bo_enabled
            else len(history.get("trials", []))
        ),
        "executions_completed_in_current_run": current_run_execution_count(history),
        "baseline_reused": bool(config.repair_bo_enabled),
        "bo_iterations_completed": native_bo_iteration_count(history),
        "repair_bo": repair_outcome,
        "context_path": _resolve(context_path),
        "actual_candidate_selected_by": (
            "independent_cold_start_agent"
            if cold_start_candidate is not None
            else "native_bo_acquisition"
        ),
        "temporary_search_space_restored": True,
    }


def audit_candidate_result(
    result_path: str,
    *,
    expected_context_path: str | None = None,
) -> dict[str, Any]:
    """独立重算数据分流、质量阈值、Pareto 与 acquisition 公式。"""

    # 第 1 步：加载最终结果、权威历史和可选冻结上下文。
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {}
    history: dict[str, Any] = {}
    path = Path(result_path).expanduser().resolve()
    try:
        result = _load_object(path)
        if result.get("schema_version") != RESULT_SCHEMA:
            errors.append("unsupported Candidate Optimizer result schema")
        history_path = result.get("history_path")
        history = _load_object(str(history_path)) if history_path else {}
        if history.get("schema_version") != HISTORY_SCHEMA:
            errors.append("unsupported Candidate Optimizer history schema")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot load Candidate Optimizer artifacts: {exc}")

    context: dict[str, Any] = {}
    if expected_context_path:
        try:
            context = _load_object(expected_context_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load expected BO context: {exc}")

    # 第 2 步：核对固定域、三目标、四代理和 Baseline 相对质量阈值。
    if result:
        objective = result.get("objective")
        expected_names = ["qps", "negative_mean_latency", "negative_p95_latency"]
        if (
            not isinstance(objective, Mapping)
            or objective.get("names") != expected_names
        ):
            errors.append("objective vector must be [QPS, -MeanLatency, -P95Latency]")
        if (
            isinstance(objective, Mapping)
            and objective.get("quality_is_objective") is not False
        ):
            errors.append("quality must be a constraint, not a fourth objective")
        declared_models = result.get("surrogate_models")
        declared_acquisition = result.get("acquisition")
        performance_only = (
            isinstance(declared_acquisition, Mapping)
            and declared_acquisition.get("name") == "performance_only_mc_ehvi"
        )
        if declared_models != list(SURROGATE_NAMES) and not (
            performance_only and declared_models == list(SURROGATE_NAMES[:3])
        ):
            errors.append(
                "Candidate Optimizer surrogate declarations do not match its acquisition"
            )
        acquisition = result.get("acquisition")
        if not isinstance(acquisition, Mapping) or acquisition.get("name") not in {
            "constrained_ehvi_times_p_run",
            "feasible_front_ehvi_times_p_run",
            "original_ehvi_times_quality_feasibility",
            "joint_mc_constrained_ehvi",
            "performance_only_mc_ehvi",
        }:
            errors.append("Candidate Optimizer acquisition is unsupported")
        constraint = result.get("quality_constraint")
        if isinstance(constraint, Mapping):
            try:
                expected_tau = (
                    1.0 - float(constraint["alpha"])
                    if constraint.get("metric") == QUALITY_GATE_METRIC
                    else quality_threshold(
                        float(constraint["baseline_quality"]),
                        float(constraint["alpha"]),
                    )
                )
                if not math.isclose(
                    expected_tau, float(constraint["tau_q"]), rel_tol=0, abs_tol=1e-12
                ):
                    errors.append("tau_q does not match the configured quality rule")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"invalid quality constraint: {exc}")
        else:
            errors.append("quality_constraint must be an object")
    if context and result:
        for name in (
            "source_parameter_results",
            "source_domain_spec",
            "source_baseline_config",
            "source_baseline_trial_result",
        ):
            if result.get(name) != context.get(name):
                errors.append(f"result {name} conflicts with frozen BO context")
        fixed = context.get("fixed_search_space")
        if (
            not isinstance(fixed, Mapping)
            or fixed.get("domains_are_rewritten") is not False
        ):
            errors.append("BO context does not prove the Range domains stayed fixed")
        if isinstance(
            fixed, Mapping
        ) and "max_num_seqs <= max_num_batched_tokens" not in fixed.get(
            "runtime_hard_constraints", []
        ):
            errors.append("Candidate Optimizer lacks the scheduler hard constraint")
        runtime_scope = context.get("runtime_constraint_scope")
        alignment_required = bool(
            isinstance(runtime_scope, Mapping)
            and runtime_scope.get("uncompressed_kv_model_dtype_alignment")
        )
        alignment_contract = (
            "measured FLASH_ATTN backend requires model dtype alignment only "
            "for explicit float16/bfloat16 KV cache"
        )
        if (
            alignment_required
            and isinstance(fixed, Mapping)
            and alignment_contract not in fixed.get("runtime_hard_constraints", [])
        ):
            errors.append(
                "Candidate Optimizer lacks the measured attention-backend dtype constraint"
            )
        handler = context.get("known_constraint_handler")
        if (
            not isinstance(handler, Mapping)
            or handler.get("tool") != "repair_bo_candidate"
            or handler.get("invoked_before_every_trial") is not True
            or handler.get("repairs_stay_within_fixed_range") is not True
        ):
            errors.append("BO context lacks the pre-execution known-constraint tool")
        repair_contract = context.get("repair_bo")
        repair_enabled = isinstance(repair_contract, Mapping) and repair_contract.get("enabled") is True
        candidate_repair = context.get("candidate_repair")
        candidate_repair_sources = (
            candidate_repair.get("sources")
            if isinstance(candidate_repair, Mapping)
            else None
        )
        guidance_policy = context.get("guidance_policy")
        preflight_guidance_enabled = bool(
            isinstance(guidance_policy, Mapping)
            and guidance_policy.get("candidate_preflight_repair_enabled") is True
        )
        expected_candidate_repair_sources = (
            (
                ["capacity_failure_guidance", "manual_constraints"],
                [
                    "quality_failure_guidance",
                    "capacity_failure_guidance",
                    "manual_constraints",
                ],
            )
            if preflight_guidance_enabled
            else (["manual_constraints"],)
        )
        if (
            not isinstance(candidate_repair, Mapping)
            or candidate_repair.get("tool") != "repair_bo_candidate"
            or candidate_repair_sources not in expected_candidate_repair_sources
            or candidate_repair.get("manual_constraints_are_hard") is not True
            or not (
                candidate_repair.get("invoked_before_every_trial") is True
                or (repair_enabled and candidate_repair.get("invoked_before_every_native_trial") is True)
            )
            or candidate_repair.get("repairs_stay_within_fixed_range") is not True
        ):
            errors.append("BO context lacks the ordered candidate repair contract")
        source_hashes = context.get("source_sha256")
        if not isinstance(source_hashes, Mapping):
            errors.append("BO context lacks frozen upstream hashes")
        else:
            for source_name, result_name in (
                ("parameter_results", "source_parameter_results"),
                ("domain_spec", "source_domain_spec"),
                ("baseline_config", "source_baseline_config"),
                ("baseline_trial_result", "source_baseline_trial_result"),
            ):
                try:
                    actual_hash = _sha256(str(context[result_name]))
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    errors.append(f"cannot verify frozen source {source_name}: {exc}")
                    continue
                if actual_hash != source_hashes.get(source_name):
                    errors.append(f"frozen source hash changed: {source_name}")
        workload = context.get("workload_contract")
        if (
            not isinstance(workload, Mapping)
            or workload.get("matches_baseline_trial") is not True
        ):
            errors.append("Candidate workload differs from measured Baseline")
        cold_start = context.get("cold_start_guidance")
        if (not isinstance(cold_start, Mapping)
            or cold_start.get("round_limit") != 0
            or cold_start.get("invocation_policy") != "native_sobol_only"):
            errors.append("BO context lacks the native cold-start contract")
        if context.get("legacy_contract_available") is True:
            optimizer_context = context.get("optimizer_config")
            quality_affects_acquisition = bool(
                dict(optimizer_context or {}).get(
                    "quality_probability_affects_acquisition", True
                )
            )
            expected_context_implementation = (
                "candidate_optimizer_local.internal_legacy_general_bo_plus_joint_mc_cehvi"
                if quality_affects_acquisition
                else "candidate_optimizer_local.internal_legacy_general_bo_plus_performance_only_mc_ehvi"
            )
            if context.get("implementation") != expected_context_implementation:
                errors.append(
                    "real vLLM Candidate run did not use internal legacy GeneralBO"
                )
            contract = context.get("legacy_algorithm_contract")
            expected_contract = {
                "initial_random_trials": "1 + effective_HEBO_dimension_count",
                "sobol": "torch.quasirandom.SobolEngine(scramble=True, seed=None)",
                "model_wrapper": "MultiTaskModel",
                "per_output_surrogate": "GPyGP",
                "general_acquisition": "LCB",
                "evolution_population": 100,
                "evolution_iterations": 200,
                "ehvi_monte_carlo_samples": int(
                    dict(optimizer_context or {}).get("monte_carlo_samples", 10)
                ),
                "joint_mc_quality_indicator": quality_affects_acquisition,
                "scalar_ehvi_times_p_feasible": False,
                "quality_mean_hard_threshold": False,
                "quality_surrogate_affects_candidate_generation": False,
                "quality_surrogate_affects_final_ranking": quality_affects_acquisition,
                "p_run_affects_final_score": False,
            }
            if not isinstance(contract, Mapping) or any(
                contract.get(name) != value for name, value in expected_contract.items()
            ):
                errors.append("internal BO algorithm contract differs from configured BO")
            if (
                not isinstance(optimizer_context, Mapping)
                or optimizer_context.get("initial_random_trials") is not None
            ):
                errors.append("original Sobol warmup was overridden")
            expected_result_implementation = (
                "agent_internal_legacy_general_bo_plus_joint_mc_cehvi"
                if quality_affects_acquisition
                else "agent_internal_legacy_general_bo_plus_performance_only_mc_ehvi"
            )
            if result.get("implementation") != expected_result_implementation:
                errors.append(
                    "result was not produced by the internal legacy-equivalent engine"
                )
            result_acquisition = result.get("acquisition")
            if (
                not isinstance(result_acquisition, Mapping)
                or result_acquisition.get("ehvi_matches_original") is not False
                or result_acquisition.get("ehvi_formula_matches_original") is not True
                or result_acquisition.get("monte_carlo_samples")
                != int(dict(optimizer_context or {}).get("monte_carlo_samples", 10))
                or result_acquisition.get("quality_mc_indicator")
                is not quality_affects_acquisition
                or result_acquisition.get("scalar_ehvi_times_p_feasible") is not False
                or result_acquisition.get("quality_mean_hard_threshold") is not False
                or result_acquisition.get(
                    "quality_surrogate_affects_candidate_generation"
                )
                is not False
                or result_acquisition.get("quality_surrogate_affects_final_ranking")
                is not quality_affects_acquisition
                or result_acquisition.get("p_run_affects_final_score") is not False
            ):
                errors.append(
                    "selection acquisition differs from the frozen quality-probability policy"
                )

    # 第 3 步：逐 Trial 验证完整/低质量/运行失败三类数据的模型分流。
    trials = history.get("trials", []) if history else []
    if not isinstance(trials, list):
        errors.append("history trials must be an array")
        trials = []
    rounds_requested = result.get("rounds_requested") if result else None
    early_termination = result.get("termination") if result else None
    valid_early_stop = (
        result.get("rounds_are_maximum_budget") is True
        and isinstance(early_termination, Mapping)
        and early_termination.get("reason") == "early_stop"
        and result.get("rounds_completed") == len(trials)
        and isinstance(rounds_requested, int)
        and len(trials) < rounds_requested
    )
    if (
        result
        and not valid_early_stop
        and (
            result.get("rounds_completed") != rounds_requested
            or len(trials) != rounds_requested
        )
    ):
        errors.append(
            "BO did not reach its maximum budget and lacks a valid early-stop receipt"
        )
    tau_q = (
        float(result["quality_constraint"]["tau_q"])
        if result and isinstance(result.get("quality_constraint"), Mapping)
        else math.nan
    )
    complete_ids: list[str] = []
    feasible_ids: list[str] = []
    failure_ids: list[str] = []
    cold_start_spec = context.get("cold_start_guidance") if context else None
    cold_start_round_limit = (
        int(cold_start_spec.get("round_limit") or 0)
        if isinstance(cold_start_spec, Mapping)
        else None
    )
    repair_contract = context.get("repair_bo") if context else None
    repair_enabled = (
        isinstance(repair_contract, Mapping)
        and repair_contract.get("enabled") is True
    )
    for round_index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            errors.append("every trial record must be an object")
            continue
        trial_id = str(trial.get("trial_id"))
        status = trial.get("status")
        constraint_check = trial.get("known_constraint_check")
        if (
            not isinstance(constraint_check, Mapping)
            or constraint_check.get("valid") is not True
            or constraint_check.get("validation_errors") != []
        ):
            errors.append(f"{trial_id} did not pass the known-constraint tool")
        elif context.get("candidate_repair"):
            expected_sources = (
                ["manual_constraints"]
                if trial.get("trial_role") == "repair"
                else context["candidate_repair"].get("sources")
            )
            if constraint_check.get("repair_sources") != expected_sources:
                errors.append(
                    f"{trial_id} did not record all candidate repair sources"
                )
        if status == "complete":
            complete_ids.append(trial_id)
            objectives = trial.get("objective_vector")
            quality = trial.get("quality")
            if not isinstance(objectives, list) or len(objectives) != 3:
                errors.append(f"{trial_id} complete trial lacks three objectives")
            if not isinstance(quality, (int, float)):
                errors.append(f"{trial_id} complete trial lacks quality")
                continue
            feasible = float(quality) >= tau_q
            if trial.get("quality_feasible") is not feasible:
                errors.append(f"{trial_id} has incorrect quality_feasible")
            quality_margin = trial.get("quality_margin")
            if repair_enabled and (
                not isinstance(quality_margin, (int, float))
                or (float(quality_margin) >= 0.0) is not feasible
            ):
                errors.append(
                    f"{trial_id} quality margin disagrees with the measured hard gate"
                )
            if trial.get("used_for_performance_surrogates") is not True:
                errors.append(
                    f"{trial_id} complete trial excluded from performance models"
                )
            if trial.get("used_for_quality_surrogate") is not True:
                errors.append(f"{trial_id} complete trial excluded from quality model")
            if trial.get("used_for_feasible_pareto") is not feasible:
                errors.append(f"{trial_id} has incorrect Pareto admission")
            if feasible:
                feasible_ids.append(trial_id)
        elif status == "runtime_failed":
            failure_ids.append(trial_id)
            censored = trial.get("execution_status") == "censored"
            expected_feedback = {}
            if censored:
                from agents.tuning.dqbo.src.services.engine.timeout_estimation import timeout_record_fields

                expected_feedback = timeout_record_fields(trial.get("censoring"))
                for field in ("estimated_objective_vector", "performance_source"):
                    if trial.get(field) != expected_feedback.get(field):
                        errors.append(f"{trial_id} has inconsistent timeout estimation {field}")
                if any(trial.get(field) is not None for field in ("quality", "quality_feasible", "objective_vector", "model_target")):
                    errors.append(f"{trial_id} censored feedback claims complete measurements")
                if trial.get("used_for_feasible_pareto") is not False:
                    errors.append(f"{trial_id} censored feedback entered the feasible front")
            if trial.get("used_for_performance_surrogates") is not expected_feedback.get("used_for_performance_surrogates", False):
                errors.append(f"{trial_id} has invalid performance training admission")
            if trial.get("used_for_quality_surrogate") is not False:
                errors.append(f"{trial_id} runtime failure entered quality model")
            if trial.get("used_for_hidden_constraint") is not (not censored):
                errors.append(f"{trial_id} has invalid P_run training admission")
        else:
            errors.append(f"{trial_id} has unsupported status {status!r}")
        selection = trial.get("selection")
        gate_options = dict(context.get("optimizer_config") or {})
        gate_start = max((
            int(change.get("after_trial_count", 0))
            for change in history.get("bo_algorithm_changes", [])
            if isinstance(change, Mapping)
            and isinstance(change.get("quality_probability_gate"), Mapping)
            and change["quality_probability_gate"].get("enabled") is True
        ), default=0)
        if (gate_options.get("quality_probability_gate_enabled") is True
                and trial.get("trial_role") != "repair" and round_index > gate_start):
            receipt = selection.get("quality_probability_gate") if isinstance(selection, Mapping) else None
            threshold = float(gate_options.get("quality_probability_threshold", 0.25))
            start_after = int(gate_options.get("quality_probability_start_after_rounds", 20))
            if (not isinstance(receipt, Mapping) or receipt.get("enabled") is not True
                    or receipt.get("threshold") != threshold):
                errors.append(f"{trial_id} lacks the configured quality probability gate receipt")
            elif round_index <= start_after:
                if (receipt.get("applied") is not False or receipt.get("p_quality") is not None
                        or receipt.get("reason") != "round_warmup"
                        or receipt.get("outcome") != "round_warmup_bypass"
                        or receipt.get("current_round") != round_index
                        or receipt.get("start_after_rounds") != start_after):
                    errors.append(f"{trial_id} applied the quality probability constraint before its start round")
            elif receipt.get("applied") is True:
                probability = receipt.get("p_quality")
                checks = receipt.get("checks") or []
                valid_probability = (isinstance(probability, (int, float))
                                     and not isinstance(probability, bool)
                                     and math.isfinite(probability) and 0 <= probability <= 1)
                valid_acceptance = (valid_probability and probability > threshold
                    and receipt.get("outcome") == "accepted"
                    and isinstance(checks, list) and bool(checks)
                    and isinstance(checks[-1], Mapping)
                    and checks[-1].get("accepted") is True
                    and checks[-1].get("p_quality") == probability
                    and checks[-1].get("candidate") == trial.get("candidate"))
                max_attempts = int(gate_options.get("quality_probability_max_attempts", 32))
                valid_fallback = (valid_probability and probability <= threshold
                    and receipt.get("outcome") == "attempt_limit_fallback"
                    and receipt.get("reason") == "max_attempts_reached"
                    and receipt.get("exhaustion_policy") == "execute_first_recommendation_after_max_attempts"
                    and receipt.get("max_attempts") == max_attempts
                    and receipt.get("retry_count") == max_attempts - 1
                    and receipt.get("fallback_selected_check_index") == 0
                    and isinstance(checks, list) and len(checks) == max_attempts
                    and all(isinstance(check, Mapping) and check.get("accepted") is False
                            and check.get("reason") in {"at_or_below_threshold", "duplicate_recommendation"}
                            and (check.get("p_quality") is None if check.get("reason") == "duplicate_recommendation"
                                 else isinstance(check.get("p_quality"), (int, float))
                                 and not isinstance(check.get("p_quality"), bool)
                                 and math.isfinite(check["p_quality"])
                                 and 0 <= check["p_quality"] <= threshold)
                            for check in checks)
                    and checks[0].get("reason") == "at_or_below_threshold"
                    and checks[0].get("p_quality") == probability
                    and checks[0].get("candidate") == trial.get("candidate"))
                if not (valid_acceptance or valid_fallback):
                    errors.append(f"{trial_id} executed without passing the quality probability threshold")
                preceding_ids = {"baseline", *(str(row.get("trial_id")) for row in trials[:round_index - 1])}
                if any(str(identifier) not in preceding_ids for identifier in receipt.get("training_trial_ids", [])):
                    errors.append(f"{trial_id} quality classifier used a future or unknown observation")
            elif (receipt.get("applied") is not False or receipt.get("p_quality") is not None
                    or receipt.get("outcome") != "cold_start_bypass"
                    or receipt.get("reason") not in {"cold_start", "insufficient_label_classes"}):
                errors.append(f"{trial_id} lacks a valid quality probability cold-start bypass")
        temporary_space = (
            selection.get("temporary_search_space")
            if isinstance(selection, Mapping)
            else None
        )
        direct_candidate = (
            selection.get("direct_complete_candidate")
            if isinstance(selection, Mapping)
            else None
        )
        direct_candidate_active = (
            isinstance(direct_candidate, Mapping)
            and direct_candidate.get("active") is True
        )
        if isinstance(temporary_space, Mapping):
            guidance_active = temporary_space.get("active") is True
            guidance_source = temporary_space.get("source")
            candidate_filter = temporary_space.get("candidate_filter")
            if guidance_active:
                if not isinstance(candidate_filter, Mapping) or not candidate_filter:
                    errors.append(
                        f"{trial_id} active temporary search space lacks a filter"
                    )
                if guidance_source is None:
                    warnings.append(
                        f"{trial_id} predates guidance-source provenance; its tool is unknown"
                    )
                elif guidance_source not in GUIDANCE_SOURCES | {"unspecified_guidance"}:
                    errors.append(
                        f"{trial_id} has unsupported guidance source {guidance_source!r}"
                    )
                if (
                    guidance_source == "adjust_cold_start_search_space"
                    and cold_start_round_limit is not None
                    and round_index > cold_start_round_limit
                ):
                    errors.append(
                        f"{trial_id} applied cold-start guidance outside its round window"
                    )
            elif guidance_source is not None and not direct_candidate_active:
                errors.append(
                    f"{trial_id} records a guidance source without an active temporary search space"
                )
        if direct_candidate_active:
            direct_source = direct_candidate.get("source")
            agent_candidate = direct_candidate.get("agent_candidate")
            candidate_after_preflight = direct_candidate.get(
                "candidate_after_preflight"
            )
            guidance_decision = (
                selection.get("guidance_decision")
                if isinstance(selection, Mapping)
                else None
            )
            is_repair = trial.get("trial_role") == "repair"
            replay_context = (
                guidance_decision.get("context")
                if isinstance(guidance_decision, Mapping)
                else None
            )
            replay_limit = (
                replay_context.get("fixed_replay_round_limit")
                if isinstance(replay_context, Mapping)
                else None
            )
            repair_cold_start_total = (
                (context.get("repair_bo") or {}).get("cold_start_total_evaluations")
                if isinstance(context, Mapping)
                else None
            )
            is_fixed_replay = (
                trial.get("trial_role") == "cold_start"
                and isinstance(replay_limit, int)
                and not isinstance(replay_limit, bool)
                and isinstance(repair_cold_start_total, int)
                and 1 <= round_index <= replay_limit < repair_cold_start_total
                and trial.get("bo_iteration") == round_index
                and isinstance(replay_context.get("configuration_source"), str)
                and bool(replay_context["configuration_source"].strip())
                and replay_context.get("source_trial_id") == trial_id
                and replay_context.get("old_measurements_reused") is False
                and trial.get("executed_in_current_run") is True
            )
            expected_direct_source = (
                "repair_bo" if is_repair else "fixed_cold_start_replay"
                if is_fixed_replay else "repair_cold_start"
                if (trial.get('repair_cold_start') is not None
                    or (guidance_decision or {}).get('context', {}).get('repair_cold_start') is not None)
                else "adjust_cold_start_search_space"
            )
            expected_method = (
                "repair_complete_candidate"
                if is_repair
                else "cold_start_agent_complete_candidate"
            )
            expected_authority = (
                "repair_bo_reasoner"
                if is_repair
                else "independent_cold_start_agent"
            )
            if direct_source != expected_direct_source:
                errors.append(f"{trial_id} direct candidate has wrong provenance")
            if selection.get("method") != expected_method:
                errors.append(f"{trial_id} direct candidate has the wrong selection method")
            if direct_candidate.get("selected_by") != expected_authority:
                errors.append(f"{trial_id} direct candidate has the wrong decision authority")
            if direct_candidate.get("native_bo_suggest_called") is not False:
                errors.append(
                    f"{trial_id} direct candidate was reselected by native BO"
                )
            if direct_candidate.get("mandatory_preflight_repair") is not True:
                errors.append(
                    f"{trial_id} direct candidate skipped mandatory preflight"
                )
            if not isinstance(agent_candidate, Mapping) or not agent_candidate:
                errors.append(f"{trial_id} direct candidate lacks the Agent input")
            if not isinstance(candidate_after_preflight, Mapping) or dict(
                candidate_after_preflight
            ) != dict(trial.get("candidate") or {}):
                errors.append(
                    f"{trial_id} direct candidate preflight receipt does not match execution"
                )
            if (
                not isinstance(guidance_decision, Mapping)
                or guidance_decision.get("complete_candidate_applied") is not True
            ):
                errors.append(
                    f"{trial_id} direct candidate lacks its applied guidance receipt"
                )
            if (
                not is_repair
                and not is_fixed_replay
                and
                ((direct_source == 'repair_cold_start' and round_index not in context.get('repair_cold_start', {}).get('trial_slots', []))
                 or (direct_source != 'repair_cold_start' and cold_start_round_limit is not None and round_index > cold_start_round_limit))
            ):
                errors.append(
                    f"{trial_id} applied cold-start candidate outside its round window"
                )
        if isinstance(selection, Mapping) and selection.get("method") in {
            "constrained_ehvi_times_p_run",
            "feasible_front_ehvi_times_p_run",
            "original_ehvi_times_quality_feasibility",
            "joint_mc_constrained_ehvi",
            "performance_only_mc_ehvi",
        }:
            try:
                ehvi = selection.get("ehvi", selection.get("alpha_cEHVI"))
                if selection.get("method") in {
                    "joint_mc_constrained_ehvi",
                    "performance_only_mc_ehvi",
                }:
                    expected_score = float(
                        selection[
                            "constrained_ehvi"
                            if selection.get("method")
                            == "joint_mc_constrained_ehvi"
                            else "ehvi"
                        ]
                    )
                    reported_samples = int(selection["monte_carlo_samples"])
                    expected_samples = int(
                        dict(context.get("optimizer_config") or {}).get(
                            "monte_carlo_samples", 10
                        )
                    )
                    if (
                        context.get("legacy_contract_available") is True
                        and reported_samples != expected_samples
                    ):
                        errors.append(
                            f"{trial_id} constrained EHVI did not use "
                            f"{expected_samples} samples"
                        )
                    if selection.get("quality_feasible_mc_samples") is not None or selection.get("method") == "joint_mc_constrained_ehvi":
                        feasible_samples = int(selection["quality_feasible_mc_samples"])
                        if not 0 <= feasible_samples <= reported_samples:
                            errors.append(
                                f"{trial_id} has invalid feasible quality-sample count"
                            )
                elif (
                    selection.get("method") == "original_ehvi_times_quality_feasibility"
                ):
                    expected_score = float(ehvi) * float(selection["p_feasible"])
                else:
                    expected_score = float(ehvi) * float(selection["p_run"])
                if not math.isclose(
                    expected_score,
                    float(selection["score"]),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{trial_id} selection score has the wrong multiplier"
                    )
                if selection.get("p_feasible") is not None or selection.get("method") != "performance_only_mc_ehvi":
                    probability = float(selection["p_feasible"])
                    if not 0.0 <= probability <= 1.0:
                        errors.append(f"{trial_id} P_feasible is outside [0,1]")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{trial_id} has invalid EHVI acquisition: {exc}")

    if repair_enabled:
        scope_payload = repair_contract.get("scope")
        scope_payload = scope_payload if isinstance(scope_payload, Mapping) else {}
        repair_fields = {str(item) for item in scope_payload.get("fields", [])}
        derived_fields = {str(item) for item in scope_payload.get("derived_fields", [])}
        repair_space = None
        try:
            repair_space = FixedSearchSpace.from_json(context["source_domain_spec"])
        except (KeyError, OSError, ValueError, TypeError) as exc:
            errors.append(f"cannot verify repair conditional parameter activity: {exc}")
        by_trial_id = {
            str(item.get("trial_id")): item
            for item in trials
            if isinstance(item, Mapping)
        }
        for trial in trials:
            if not isinstance(trial, Mapping) or trial.get("trial_role") != "repair":
                continue
            trial_id = str(trial.get("trial_id"))
            parent_id = str(trial.get("parent_trial_id") or "")
            parent = by_trial_id.get(parent_id)
            if parent is None:
                errors.append(f"{trial_id} repair lineage has no parent Trial")
                continue
            candidate = trial.get("candidate")
            anchor_id = str(trial.get("repair_anchor_trial_id") or parent_id)
            anchor = by_trial_id.get(anchor_id, parent)
            anchor_candidate = anchor.get("candidate") if anchor is not None else None
            if not isinstance(candidate, Mapping) or not isinstance(anchor_candidate, Mapping):
                errors.append(f"{trial_id} repair lineage lacks executable configurations")
                continue
            trial_allowed_fields = trial.get("repair_allowed_fields")
            allowed_fields = (
                {str(item) for item in trial_allowed_fields}
                if isinstance(trial_allowed_fields, list) and trial_allowed_fields
                else repair_fields
            )
            changed_frozen = _repair_frozen_field_changes(
                anchor=anchor_candidate, candidate=candidate,
                fields=allowed_fields, derived_fields=derived_fields,
                space=repair_space,
            )
            if changed_frozen:
                errors.append(
                    f"{trial_id} changed frozen repair fields: {changed_frozen}"
                )
            if not trial.get("repair_signature"):
                errors.append(f"{trial_id} lacks repair_signature")
        for episode in history.get("repair_episodes", []):
            if not isinstance(episode, Mapping):
                continue
            for round_record in episode.get("rounds", []):
                if not isinstance(round_record, Mapping):
                    continue
                signatures = [
                    str(item.get("repair_signature") or "")
                    for item in round_record.get("decisions", [])
                    if isinstance(item, Mapping)
                ]
                if len(signatures) == 2 and len(set(signatures)) != 2:
                    errors.append(
                        f"{episode.get('episode_id')} selected duplicate A/B signatures"
                    )

    # 第 4 步：从质量达标完整集重算可行 Pareto，拒绝低质量点进入前沿。
    if history and result:
        expected_pareto_ids = [
            item["trial_id"]
            for item in feasible_pareto_records(history)
            if item.get("source") != "baseline"
        ]
        actual_pareto_ids = [
            str(item.get("trial_id"))
            for item in result.get("feasible_pareto_front", [])
            if isinstance(item, Mapping) and item.get("trial_id") != "baseline"
        ]
        if expected_pareto_ids != actual_pareto_ids:
            errors.append("feasible Pareto front does not match D_feasible")
        counts = result.get("dataset_counts")
        expected_counts = {
            "complete": len(complete_ids),
            "quality_feasible": len(feasible_ids),
            "runtime_failed": len(failure_ids),
        }
        if repair_enabled:
            repair_trials = [
                item
                for item in trials
                if isinstance(item, Mapping) and item.get("trial_role") == "repair"
            ]
            expected_counts.update(
                {
                    "native_bo": len(trials) - len(repair_trials),
                    "repair": len(repair_trials),
                    "repair_quality_converted": sum(
                        1
                        for item in repair_trials
                        if item.get("quality_feasible") is True
                    ),
                }
            )
        if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
            errors.append("dataset_counts do not match trial history")

    # 第 6 步：输出顶层 ReAct 可消费的确定性审计 observation。
    report = AuditReport(
        passed=not errors,
        audit_type="candidate_optimizer_mechanical_audit",
        errors=errors,
        warnings=warnings,
        retryable=bool(errors),
        facts={
            "result_path": str(path),
            "rounds_requested": rounds_requested,
            "complete_count": len(complete_ids),
            "quality_feasible_count": len(feasible_ids),
            "runtime_failure_count": len(failure_ids),
            "surrogate_models": result.get("surrogate_models"),
            "quality_is_objective": False,
            "existing_bo_imported": False,
        },
    )
    return report.to_dict()
