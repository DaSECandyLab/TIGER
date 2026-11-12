"""Run-scoped counterfactual repair for quality-infeasible BO trials.

RepairBO treats a fast but semantically infeasible native BO observation as a
local anchor. R1 reuses successful representations; R2 proposes an alternative
using refreshed evidence unless R1 succeeds; R3 targets an unsuccessful measured
branch. Each stage includes bounded pre-execution audit and revision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .repair_changes import propose_changes
from .repair_reflection import compact_reflections, prior_reflections, reflect_episode

from agents.tuning.dqbo.src.services.failure_taxonomy import summarize_runtime_failure

from agents.tuning.dqbo.src.services.workload_facts import (
    extract_dynamic_workload_features,
)

from agents.tuning.candidate_optimizer.src.services.direct_inference import StructuredInferenceClient
from .repair_review import DualPathRepairReviewer
from .repair_stages import repair_stage, REPAIR_STAGE_NAMES
from agents.tuning.dqbo.src.services.engine.history import (
    complete_records,
    feasible_pareto_records,
    is_timeout_record,
    result_from_history,
    write_json_atomic,
)
from .repair_threshold import assess_trigger
from agents.tuning.dqbo.src.services.engine.pareto import hypervolume_improvement
from agents.tuning.dqbo.src.services.engine.space import FixedSearchSpace, ParameterDomain
from agents.tuning.candidate_optimizer.src.services.prompt_format import (
    PROMPT_FORMAT_SCHEMA,
    render_system_prompt,
    render_tagged_sections as _render_tagged_sections,
)

REPAIR_EPISODE_SCHEMA = "sembench.repair-bo.episode.v1"
REPAIR_EXPERIENCE_SCHEMA = "sembench.repair-bo.experience.v1"
REPAIR_EXPERIENCE_NAME = "repair_experience.json"
REPAIR_BO_V1_FIELDS = ("quantization", "kv_cache_dtype", "dtype")
PERFORMANCE_RECOVERY_SCOPE_VERSION = "repair-scope.performance-recovery.v1"
FULL_CONFIGURATION_SCOPE_VERSION = "repair-scope.full-range-llm.v1"
REPAIR_WARMUP_GUARD = {
    "version": "dp2-warmup-headroom.v2",
    "data_parallel_size": 2,
    "min_warmup_requests": 2800,
    "min_gpu_memory_utilization": 0.94,
    "target_gpu_memory_utilization": 0.92,
    "note": (
        "Repair 执行例外：DP=2 且 min(max_num_seqs,max_num_batched_tokens)>=2800、"
        "gpu_memory_utilization>=0.94 时，在参数域和机械校验允许的情况下降至0.92。"
        "为 sampler 预热留临时显存；会减少 KV 预算，不是可运行性保证。"
        "LLM 仍按本轮 scope 推荐，系统单独记录此调整；实测效果归因于调整后的完整配置。"
    ),
}


def _apply_repair_warmup_guard(
    candidate: Mapping[str, Any], *, space: FixedSearchSpace,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reserve transient memory in repair only; preserve the domain and all other fields."""
    original = dict(candidate)
    canonical = space.bo_to_canonical(original)

    def value(name: str) -> Any:
        return canonical.get(name, original.get(name))

    numbers = [value(name) for name in (
        "data_parallel_size", "max_num_seqs", "max_num_batched_tokens",
        "gpu_memory_utilization",
    )]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in numbers):
        return original, None
    dp, seqs, tokens, utilization = numbers
    if (dp != REPAIR_WARMUP_GUARD["data_parallel_size"]
            or min(seqs, tokens) < REPAIR_WARMUP_GUARD["min_warmup_requests"]
            or utilization < REPAIR_WARMUP_GUARD["min_gpu_memory_utilization"]):
        return original, None
    target = REPAIR_WARMUP_GUARD["target_gpu_memory_utilization"]
    receipt = {
        "version": REPAIR_WARMUP_GUARD["version"], "applied": False,
        "data_parallel_size": dp, "warmup_requests": min(seqs, tokens),
        "before": utilization, "after": utilization, "requested_after": target,
        "reason": "large_dp2_sampler_warmup_needs_transient_memory",
    }
    parameter = space.by_name.get("gpu_memory_utilization")
    if parameter is None or not parameter.tuned or not parameter.contains(target):
        return original, {**receipt, "skip_reason": "target_outside_tunable_domain"}
    proposed = dict(original)
    aliases = {parameter.name, parameter.bo_name}
    for name in aliases:
        if name in proposed:
            proposed[name] = target
    try:
        validation = dict(validator(proposed))
    except Exception as exc:
        return original, {**receipt, "skip_reason": "validator_exception", "detail": str(exc)[:240]}
    executable = validation.get("candidate")
    if validation.get("valid") is not True or not isinstance(executable, Mapping):
        return original, {**receipt, "skip_reason": "mechanical_validation_failed"}
    changed = {key for key in set(original) | set(executable) if original.get(key) != executable.get(key)}
    if not changed <= aliases or space.bo_to_canonical(executable).get(parameter.name) != target:
        return original, {**receipt, "skip_reason": "mechanical_projection_changed_other_fields"}
    return dict(executable), {**receipt, "applied": True, "after": target, "changed_fields": sorted(changed)}
DEFAULT_DERIVED_FIELDS = frozenset(
    {
        "model",
        "model_path",
        "served_model_name",
        "load_format",
        "quantization_param_path",
    }
)


def _repair_frozen_field_changes(
    *, anchor: Mapping[str, Any], candidate: Mapping[str, Any],
    fields: Sequence[str], derived_fields: Sequence[str],
    space: FixedSearchSpace | None,
) -> list[str]:
    """Allow only exact inactive resets; active frozen fields remain immutable."""
    allowed = set(fields) | set(derived_fields)
    if space is not None:
        allowed.update(space.by_name[name].bo_name for name in tuple(allowed)
                       if name in space.by_name)
    inactive = space.inactive_parameter_values(candidate) if space is not None else {}
    changed = {
        name for name in set(anchor) | set(candidate)
        if name not in allowed and anchor.get(name) != candidate.get(name)
        and not (name in inactive and candidate.get(name) == inactive[name])
    }
    # bo_to_canonical normalizes inactive values, so also inspect the actual
    # executable mapping rather than letting that conversion hide bad resets.
    changed.update(name for name, value in inactive.items()
                   if name in candidate and candidate[name] != value)
    return sorted(changed)


def render_tagged_sections(sections: Sequence[tuple[str, Any]]) -> str:
    """Keep the shared tagged layout, with compact JSON only for RepairBO."""

    return _render_tagged_sections(sections, compact=True)


def _representation_parameter_guide(domains: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """Explain representation choices without adding values or feasibility bans."""

    descriptions = {
        "quantization": ("权重量化方法与执行内核；本执行器会联动选择对应模型 checkpoint。", {
            "none": "关闭显式权重量化；本实验加载原始模型，16-bit 权重通常比 Int4 权重占用更多显存，留给 KV cache 和运行时临时张量的空间更少。",
            "awq_marlin": "AWQ 权重量化配合 Marlin 内核；位宽由 checkpoint 决定，本实验为 Int4，主要减少权重存储。",
            "gptq_marlin": "GPTQ 权重量化配合 Marlin 内核；位宽由 checkpoint 决定，本实验为 Int4，量化方法与 AWQ 不同，质量和速度需实测。",
            "awq": "AWQ 权重量化；位宽和兼容性取决于 checkpoint 与执行内核。",
            "gptq": "GPTQ 权重量化；位宽和兼容性取决于 checkpoint 与执行内核。",
        }),
        "dtype": ("模型浮点计算及非量化张量的数据类型；不会把 Int4 权重自动改成 16-bit 权重。", {
            "half": "FP16 浮点格式，16-bit，每元素名义存储为 2 字节。",
            "bfloat16": "BF16 浮点格式，16-bit，每元素名义存储为 2 字节。",
        }),
        "kv_cache_dtype": ("注意力 Key/Value 缓存的存储格式；不改变权重量化，不能据此推断全部激活或 sampler 临时张量的显存。", {
            "float16": "FP16 KV，16-bit，每元素 2 字节；未压缩格式需满足当前 attention backend 的模型 dtype 对齐约束。",
            "bfloat16": "BF16 KV，16-bit，每元素 2 字节；未压缩格式需满足当前 attention backend 的模型 dtype 对齐约束。",
            "fp8_e4m3": "FP8 KV，8-bit，4 位指数/3 位尾数；缓存数据名义大小约为 16-bit 的一半，另有 scale 等开销。",
            "fp8_e5m2": "FP8 KV，8-bit，5 位指数/2 位尾数；缓存数据名义大小约为 16-bit 的一半，另有 scale 等开销。",
            "int8_per_token_head": "INT8 KV，按 token、KV head 细粒度缩放；有 scale 存储及量化/反量化开销，需匹配支持的内核。",
            "fp8_per_token_head": "FP8 KV，按 token、KV head 细粒度缩放；缩放粒度与普通 FP8 不同，有 scale 及内核开销。",
            "fp8": "FP8 KV 别名；实际格式取决于平台，本 CUDA 环境对应 E4M3。",
            "auto": "KV 类型随模型 dtype 或模型默认配置解析，以实际生效配置为准。",
        }),
    }
    descriptions.update({
        "cacheblend_mode": ("CacheBlend 开关；开启时才启用 LMCache 和 blend 子参数，遵守 prefix caching 与 KV dtype 的机械约束。", {}),
        "lmcache_chunk_size": ("LMCache KV 分块 token 数；只在 CacheBlend 开启时有效。", {}),
        "cacheblend_blend_min_tokens": ("触发 blend 的最少 token 数；只在 CacheBlend 开启时有效。", {}),
        "cacheblend_blend_check_layers": ("用于检查重算 token 的模型层索引列表；只在 CacheBlend 开启时有效。", {}),
        "cacheblend_blend_recompute_ratios": ("blend 重算比例列表；与检查层配对，质量及性能需实测。", {}),
        "enable_prefix_caching": ("原生 prefix cache 开关；当前默认不允许与 CacheBlend 同时开启。", {}),
    })
    return {
        "parameters": {
            name: {
                "meaning": meaning,
                "values": {
                    str(value): values.get(str(value), "以当前 Range 和机械兼容性约束为准。")
                    for value in domains[name]
                },
            }
            for name, (meaning, values) in descriptions.items() if name in domains
        },
        "interpretation": (
            "这些是机制说明，不是新增候选、质量排序或运行失败禁用规则。机械校验通过不保证显存足够或质量达标。"
            "R1/R2 阶段 LLM 只推荐可修参数组；系统可按预热显存检查单独下调显存利用率。必须结合当前并发、token 上限、显存预算、max_model_len 和 CUDA graph 开销评估。"
            "DP 是模型副本，DP=2 不等于把权重分摊到两张卡；TP 才在模型内部切分。"
            "降低 KV 位宽不能保证解决权重、预热或 sampler OOM；任务质量是否达标以实测结果为准。"
        ),
    }


class _PromptEvidence:
    """Store each measured trial once; all prompt groups reference its real ID."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def reference(self, record: Mapping[str, Any]) -> dict[str, Any]:
        point = {
            "quality": {
                str(item["metric"]): item.get("value")
                for item in _measured_quality_metrics(record)
                if item.get("metric")
            },
            "performance": _measured_performance_metrics(record),
            "configuration": dict(record.get("candidate") or {}),
        }
        if record.get("status") == "runtime_failed":
            point["quality"] = {}
            point["performance"] = {}
        if record.get("repair_warmup_guard"):
            point["repair_warmup_guard"] = dict(record["repair_warmup_guard"])
        trial_id = record.get("trial_id")
        if not trial_id:
            # Do not invent a citable Trial ID for incomplete historical context.
            return point
        key = str(trial_id)
        previous = self.records.setdefault(key, {name: {} for name in point})
        for name, values in point.items():
            previous.setdefault(name, {}).update(values)
        ref: dict[str, Any] = {"trial_id": key}
        if record.get("configuration_similarity") is not None:
            ref["configuration_similarity"] = record["configuration_similarity"]
        return ref

    def references(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.reference(record) for record in records]

    def performance_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        # Pareto/landmark membership is useful; duplicated points, distances and
        # maximization-coordinate copies are not needed in an LLM prompt.
        result: dict[str, Any] = {
            "current_measured_feasible_pareto_trials": self.references(
                context.get("current_measured_feasible_pareto_trials", [])
            ),
            "performance_landmarks": {
                name: self.reference(record)
                for name, record in context.get("performance_landmarks", {}).items()
                if isinstance(record, Mapping)
            },
        }
        nearest = context.get("nearest_feasible_pareto_trial")
        if isinstance(nearest, Mapping):
            result["nearest_feasible_pareto_trial"] = self.reference(nearest)
        return result


def _prompt_critique(critique: Mapping[str, Any] | None) -> dict[str, Any]:
    """Measured values live in the evidence catalog, not repeated in the audit."""

    return {
        key: critique[key]
        for key in ("summary", "misused_evidence", "next_round_instructions")
        if critique and key in critique
    }


def _runtime_failure_context(record: Mapping[str, Any]) -> dict[str, str]:
    """Bound the error evidence; never send a full traceback to the reasoner."""

    saved = record.get("runtime_failure")
    source = saved if isinstance(saved, Mapping) else record
    lines = str(source.get("failure_reason") or "").splitlines()
    reason = next((line.strip() for line in reversed(lines) if line.strip()), "")
    return {
        "failure_kind": str(source.get("failure_kind") or "unknown")[:80],
        "failure_reason": summarize_runtime_failure(reason, fallback=reason or "Unknown runtime failure"),
    }


def _prompt_evidence(parent: Mapping[str, Any], cases: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _PromptEvidence()
    references = {
        "parent": evidence.reference(parent),
        "success": evidence.references(cases.get("quality_success_cases", [])),
        "failure": evidence.references(cases.get("quality_failure_cases", [])),
        "round_1": [
            {
                **evidence.reference(record),
                **{
                    key: record[key]
                    for key in ("status", "repair_strategy", "repair_success")
                    if record.get(key) is not None
                },
            }
            for record in cases.get("episode_round_1_results", [])
        ],
        "performance": evidence.performance_context(cases.get("performance_context", {})),
    }
    runtime_failures = {
        str(record["trial_id"]): record
        for record in [
            *cases.get("runtime_failure_cases", []),
            parent, *cases.get("episode_round_1_results", []),
        ]
        if record.get("status") == "runtime_failed" and record.get("trial_id")
    }
    references["runtime_failures"] = [
        {**evidence.reference(record), **_runtime_failure_context(record)}
        for record in runtime_failures.values()
    ]
    return {**references, "records": evidence.records,
            **({'historical_quality_memory': cases['historical_quality_memory']}
               if cases.get('historical_quality_memory') else {})}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _load_object(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {target}")
    return payload


def _stable_key(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(dict(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _signature_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("repair signature values must be finite")
        return float(f"{value:.12g}")
    return _json_value(value)


@dataclass(frozen=True)
class RepairScope:
    """An ordered, versioned repair parameter group compiled from Range."""

    fields: tuple[str, ...]
    domains: tuple[tuple[Any, ...], ...]
    scope_version: str
    compatibility_constraints: tuple[Mapping[str, Any], ...] = ()
    runtime_constraints: tuple[Mapping[str, Any], ...] = ()
    derived_fields: frozenset[str] = DEFAULT_DERIVED_FIELDS

    @classmethod
    def from_space(
        cls,
        space: FixedSearchSpace,
        *,
        fields: Sequence[str],
        scope_version: str,
        runtime_constraints: Sequence[Mapping[str, Any]] = (),
        derived_fields: Sequence[str] = tuple(DEFAULT_DERIVED_FIELDS),
    ) -> "RepairScope":
        normalized_fields = tuple(str(item).strip() for item in fields)
        if not normalized_fields or len(set(normalized_fields)) != len(
            normalized_fields
        ):
            raise ValueError("repair fields must be unique and non-empty")
        domains: list[tuple[Any, ...]] = []
        for field in normalized_fields:
            parameter = space.by_name.get(field)
            if parameter is None:
                raise ValueError(f"repair field is absent from fixed Range: {field}")
            values = parameter.values
            if values is None:
                raise ValueError(
                    f"RepairBO v1 requires a pre-enumerable domain for {field}"
                )
            domains.append(tuple(_signature_value(value) for value in values))
        return cls(
            fields=normalized_fields,
            domains=tuple(domains),
            scope_version=str(scope_version),
            compatibility_constraints=tuple(space.joint_constraints),
            runtime_constraints=tuple(dict(item) for item in runtime_constraints),
            derived_fields=frozenset(str(item) for item in derived_fields) | (
                frozenset({"kv_offloading_backend", "chunked_prefill", "enable_chunked_prefill", "enable_prefix_caching", "max_num_batched_tokens", "long_prefill_token_threshold", "prefix_caching_hash_algo"}) if "cacheblend_mode" in normalized_fields else frozenset()) | (
                frozenset({"enforce_eager"}) if "cacheblend_mode" in normalized_fields and any(
                    c.get("id") == "cacheblend_compatibility" and c.get("require_enforce_eager")
                    for c in space.joint_constraints) else frozenset()),
        )

    def values(
        self, candidate: Mapping[str, Any], space: FixedSearchSpace
    ) -> dict[str, Any]:
        canonical = space.bo_to_canonical(candidate)
        return {field: _signature_value(canonical[field]) for field in self.fields}

    def signature(self, candidate: Mapping[str, Any], space: FixedSearchSpace) -> str:
        ordered = [
            [field, value] for field, value in self.values(candidate, space).items()
        ]
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))

    def enumerate_raw(self) -> list[dict[str, Any]]:
        return [
            dict(zip(self.fields, values, strict=True))
            for values in product(*self.domains)
        ]

    def contract(self) -> dict[str, Any]:
        return {
            "scope_version": self.scope_version,
            "execution_guard": dict(REPAIR_WARMUP_GUARD),
            "fields": list(self.fields),
            "domains": {
                field: list(values)
                for field, values in zip(self.fields, self.domains, strict=True)
            },
            "compatibility_constraints": [
                dict(item) for item in self.compatibility_constraints
            ],
            "runtime_constraints": [dict(item) for item in self.runtime_constraints],
            "derived_fields": sorted(self.derived_fields),
            "raw_signature_count": int(
                math.prod(len(values) for values in self.domains)
            ),
        }

    def materialize_fibre(
        self,
        *,
        parent: Mapping[str, Any],
        space: FixedSearchSpace,
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Enumerate the exact valid ``(z0, r)`` fibre and rejection receipts."""

        parent_dict = dict(parent)
        parent_signature = self.signature(parent_dict, space)
        valid: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        canonical_requests: set[str] = set()
        for values in self.enumerate_raw():
            proposed = dict(parent_dict)
            for field, value in values.items():
                parameter = space.by_name[field]
                proposed[field] = value
                if parameter.bo_name in proposed or parameter.bo_name != field:
                    proposed[parameter.bo_name] = value
            if "cacheblend_mode" in self.fields:
                # Inactive blend children share one physical configuration. Avoid
                # revalidating their Cartesian duplicates in the extended scope.
                canonical = space.normalize(proposed)
                key = self.signature(canonical, space)
                if key in canonical_requests:
                    continue
                canonical_requests.add(key)
                # The mechanical validator must first apply switch-dependent
                # execution constraints (prefix/chunked prefill and token budget).
            try:
                receipt = dict(validator(proposed))
            except Exception as exc:
                rejected.append(
                    {
                        "requested_values": values,
                        "reason": "validator_exception",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            repaired = receipt.get("candidate")
            if receipt.get("valid") is not True or not isinstance(repaired, Mapping):
                rejected.append(
                    {
                        "requested_values": values,
                        "reason": "incompatible",
                        "validation_errors": list(
                            receipt.get("validation_errors") or []
                        ),
                    }
                )
                continue
            repaired_dict = dict(repaired)
            changed_frozen = _repair_frozen_field_changes(
                anchor=parent_dict, candidate=repaired_dict, fields=self.fields,
                derived_fields=self.derived_fields, space=space,
            )
            if changed_frozen:
                rejected.append(
                    {
                        "requested_values": values,
                        "reason": "frozen_parameter_changed",
                        "fields": changed_frozen,
                    }
                )
                continue
            signature = self.signature(repaired_dict, space)
            if signature == parent_signature:
                continue
            valid.setdefault(
                signature,
                {
                    "signature": signature,
                    "values": self.values(repaired_dict, space),
                    "candidate": repaired_dict,
                    "validator_receipt": receipt,
                },
            )
        return list(valid.values()), rejected


@dataclass(frozen=True)
class RepairSelectionScope:
    """Prompt and validation contract for a non-enumerated local repair space."""

    fields: tuple[str, ...]
    domains: tuple[Mapping[str, Any], ...]
    scope_version: str
    compatibility_constraints: tuple[Mapping[str, Any], ...] = ()
    runtime_constraints: tuple[Mapping[str, Any], ...] = ()
    derived_fields: frozenset[str] = DEFAULT_DERIVED_FIELDS
    candidate_generation: str = (
        "full_non_representation_range_sampling_then_performance_bo_ranking"
    )

    @classmethod
    def from_space(
        cls,
        space: FixedSearchSpace,
        *,
        fields: Sequence[str],
        scope_version: str = PERFORMANCE_RECOVERY_SCOPE_VERSION,
        runtime_constraints: Sequence[Mapping[str, Any]] = (),
        candidate_generation: str = (
            "full_non_representation_range_sampling_then_performance_bo_ranking"
        ),
    ) -> "RepairSelectionScope":
        normalized_fields = tuple(str(item).strip() for item in fields)
        if not normalized_fields or len(set(normalized_fields)) != len(
            normalized_fields
        ):
            raise ValueError("selection-scope fields must be unique and non-empty")
        domains: list[Mapping[str, Any]] = []
        for field in normalized_fields:
            parameter = space.by_name.get(field)
            if parameter is None or not parameter.tuned or parameter.domain is None:
                raise ValueError(
                    f"performance-recovery field is not tunable in Range: {field}"
                )
            domains.append(dict(parameter.domain))
        return cls(
            fields=normalized_fields,
            domains=tuple(domains),
            scope_version=str(scope_version),
            compatibility_constraints=tuple(space.joint_constraints),
            runtime_constraints=tuple(dict(item) for item in runtime_constraints),
            candidate_generation=str(candidate_generation),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "scope_version": self.scope_version,
            "fields": list(self.fields),
            "domains": {
                field: dict(domain)
                for field, domain in zip(self.fields, self.domains, strict=True)
            },
            "compatibility_constraints": [
                dict(item) for item in self.compatibility_constraints
            ],
            "runtime_constraints": [dict(item) for item in self.runtime_constraints],
            "derived_fields": sorted(self.derived_fields),
            "raw_signature_count": None,
            "candidate_generation": self.candidate_generation,
        }


def repair_scope_runtime_constraints(
    runtime_scope: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Project measured backend dependencies into the RepairBO prompt contract."""

    scope = dict(runtime_scope or {})
    if not bool(scope.get("uncompressed_kv_model_dtype_alignment")):
        return ()
    backend = str(scope.get("resolved_attention_backend") or "unknown")
    return (
        {
            "id": "attention_backend_uncompressed_kv_dtype_alignment",
            "kind": "measured_runtime_hard_constraint",
            "parameters": ["dtype", "kv_cache_dtype"],
            "resolved_attention_backend": backend,
            "activation_condition": (
                f"resolved_attention_backend == {backend}"
            ),
            "applies_when": "kv_cache_dtype is explicit float16 or bfloat16",
            "rule": (
                "effective model dtype=half requires kv_cache_dtype=float16; "
                "effective model dtype=bfloat16 requires kv_cache_dtype=bfloat16"
            ),
            "non_applicability": (
                "quantized KV cache formats remain independently selectable"
            ),
            "repair_behavior": (
                "align uncompressed kv_cache_dtype to the effective model dtype "
                "inside the frozen Range"
            ),
            "evidence_source": scope.get("source"),
            "evidence_path": scope.get("evidence_path"),
            "evidence_line": scope.get("evidence_line"),
        },
    )


def _raw_has_parameter(
    candidate: Mapping[str, Any], parameter: ParameterDomain
) -> bool:
    return parameter.name in candidate or parameter.bo_name in candidate


def grouped_configuration_similarity(
    target: Mapping[str, Any],
    other: Mapping[str, Any],
    *,
    space: FixedSearchSpace,
    repair_fields: Sequence[str],
) -> dict[str, Any]:
    """Group-balanced Gower similarity over the frozen ``z`` coordinates."""

    target_norm = space.normalize(target)
    other_norm = space.normalize(other)
    target_active = space.active_parameter_names(target_norm)
    other_active = space.active_parameter_names(other_norm)
    excluded = set(repair_fields)
    grouped: dict[str, list[float]] = {}
    missing_flags: list[dict[str, Any]] = []
    for parameter in space.parameters:
        if parameter.name in excluded:
            continue
        target_present = _raw_has_parameter(target, parameter)
        other_present = _raw_has_parameter(other, parameter)
        if not target_present or not other_present:
            missing_flags.append(
                {
                    "field": parameter.name,
                    "target_missing": not target_present,
                    "other_missing": not other_present,
                }
            )
            continue
        left_active = parameter.name in target_active
        right_active = parameter.name in other_active
        if not left_active and not right_active:
            continue
        if left_active != right_active:
            similarity = 0.0
        elif (
            parameter.decision == "fixed"
            or not parameter.domain
            or parameter.values is not None
            or parameter.kind in {
                "categorical",
                "boolean",
            }
        ):
            similarity = float(
                target_norm[parameter.name] == other_norm[parameter.name]
            )
        else:
            domain = parameter.domain or {}
            lower = float(domain["lower"])
            upper = float(domain["upper"])
            if str(domain.get("scale", "linear")) == "log":
                left = math.log(float(target_norm[parameter.name]))
                right = math.log(float(other_norm[parameter.name]))
                lower, upper = math.log(lower), math.log(upper)
            else:
                left = float(target_norm[parameter.name])
                right = float(other_norm[parameter.name])
            width = upper - lower
            similarity = (
                float(left == right)
                if abs(width) <= 1e-12
                else max(0.0, 1.0 - abs(left - right) / width)
            )
        grouped.setdefault(parameter.group_id, []).append(float(similarity))
    group_scores = {
        group: sum(values) / len(values) for group, values in sorted(grouped.items())
    }
    total = sum(group_scores.values()) / len(group_scores) if group_scores else 1.0
    return {
        "similarity": float(total),
        "group_similarity": group_scores,
        "missing_flags": missing_flags,
    }


def dynamic_workload_similarity(
    target: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, Any]:
    names = sorted(set(target) | set(other))
    scores: list[float] = []
    missing: list[dict[str, Any]] = []
    for name in names:
        if name not in target or name not in other:
            missing.append(
                {
                    "feature": name,
                    "target_missing": name not in target,
                    "other_missing": name not in other,
                }
            )
            continue
        left, right = target[name], other[name]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            scale = max(abs(float(left)), abs(float(right)), 1e-12)
            scores.append(max(0.0, 1.0 - abs(float(left) - float(right)) / scale))
        else:
            scores.append(float(left == right))
    return {
        "similarity": (sum(scores) / len(scores) if scores else None),
        "missing_flags": missing,
        "compared_feature_count": len(scores),
    }


def _record_dynamic_features(record: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = record.get("workload_snapshot")
    if isinstance(snapshot, Mapping):
        dynamic = snapshot.get("dynamic")
        if isinstance(dynamic, Mapping):
            return dict(dynamic)
    direct = extract_dynamic_workload_features(record)
    if direct:
        return direct
    result_path = record.get("result_path")
    if isinstance(result_path, str) and Path(result_path).expanduser().is_file():
        try:
            return extract_dynamic_workload_features(_load_object(result_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _edit_distance(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]
) -> int:
    return sum(left.get(field) != right.get(field) for field in fields)


def _measured_performance_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    metrics = record.get("measured_performance_metrics", record.get("metrics"))
    result: dict[str, float] = {}
    if isinstance(metrics, Mapping):
        for name in (
            "query_throughput_qps",
            "mean_query_execution_latency_ms",
            "p95_query_execution_latency_ms",
        ):
            value = metrics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[name] = float(value)
    objectives = record.get("objective_vector")
    if isinstance(objectives, Sequence) and len(objectives) >= 3:
        names_and_signs = (
            ("query_throughput_qps", 1.0),
            ("mean_query_execution_latency_ms", -1.0),
            ("p95_query_execution_latency_ms", -1.0),
        )
        for index, (name, sign) in enumerate(names_and_signs):
            value = objectives[index]
            if (
                name not in result
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                result[name] = sign * float(value)
    return result


def _measured_quality_metrics(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: item.get(key)
            for key in (
                "metric",
                "value",
                "operator",
                "threshold",
                "normalized_margin",
                "passed",
            )
        }
        for item in record.get(
            "measured_quality_metrics", record.get("quality_margins", [])
        )
        if isinstance(item, Mapping)
    ]


def _quality_gate_context(history: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the exact measured quality gate, including its baseline basis."""

    constraint = dict(history.get("quality_constraint") or {})
    baseline = history.get("baseline_observation")
    baseline_metrics: dict[str, Any] = {}
    if isinstance(baseline, Mapping):
        for item in baseline.get("quality_margins", []):
            if isinstance(item, Mapping) and item.get("metric"):
                baseline_metrics[str(item["metric"])] = item.get("value")

    thresholds: list[dict[str, Any]] = []
    per_query = constraint.get("query_metric_thresholds")
    if isinstance(per_query, Mapping):
        for query, metrics in sorted(per_query.items()):
            if not isinstance(metrics, Mapping):
                continue
            for metric, rule in sorted(metrics.items()):
                if not isinstance(rule, Mapping):
                    continue
                for operator, threshold in sorted(rule.items()):
                    metric_name = f"{query}.{metric}"
                    baseline_value = baseline_metrics.get(metric_name)
                    retention = None
                    if (
                        isinstance(threshold, (int, float))
                        and isinstance(baseline_value, (int, float))
                        and float(baseline_value) != 0.0
                    ):
                        retention = float(threshold) / float(baseline_value)
                    thresholds.append(
                        {
                            "query": str(query),
                            "metric": str(metric),
                            "operator": ">=" if operator == "min" else "<=",
                            "threshold": threshold,
                            "baseline_value": baseline_value,
                            "threshold_over_baseline": retention,
                            "inclusive": bool(constraint.get("inclusive", True)),
                        }
                    )
    optimizer = history.get("optimizer_config")
    quality_probability_affects_acquisition = bool(
        optimizer.get("quality_probability_affects_acquisition", False)
        if isinstance(optimizer, Mapping)
        else False
    )
    return {
        "metric": constraint.get("metric"),
        "threshold_source": constraint.get("threshold_source"),
        "pass_rule": constraint.get("rule"),
        "inclusive": bool(constraint.get("inclusive", True)),
        "per_query_gates": thresholds,
        "aggregate_thresholds": constraint.get("aggregate_thresholds") or {},
        "threshold_formulas": constraint.get("threshold_formulas") or {},
        "decision_rule": (
            "every listed gate must pass on measured results; performance cannot "
            "compensate for a failed quality gate"
        ),
        "acquisition_rule": (
            "quality pass probability is multiplied into BO acquisition"
            if quality_probability_affects_acquisition
            else "quality pass probability is not multiplied into BO acquisition"
        ),
    }


def retrieve_repair_cases(
    *,
    history: Mapping[str, Any],
    parent: Mapping[str, Any],
    scope: RepairScope,
    space: FixedSearchSpace,
    limit: int,
    repair_round: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Keep nearest quality cases and scope first-round runtime evidence to DP2.

    ``limit`` is retained for caller compatibility. Quality cases cover every
    signature; R3 runtime evidence retains failed Trials in execution order.
    """

    del limit
    parent_candidate = dict(parent.get("candidate") or {})
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in complete_records(history, include_baseline=True):
        if record.get("status") == "runtime_failed" or is_timeout_record(record):
            continue
        if str(record.get("trial_id")) == str(parent.get("trial_id")):
            continue
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        signature = scope.signature(candidate, space)
        config_similarity = grouped_configuration_similarity(
            parent_candidate,
            candidate,
            space=space,
            repair_fields=scope.fields,
        )
        values = scope.values(candidate, space)
        feasible = record.get("quality_feasible") is True
        category = (
            "quality_success_cases" if feasible else "quality_failure_cases"
        )
        case = {
            "trial_id": record.get("trial_id"),
            "candidate": dict(candidate),
            "repair_signature": signature,
            "repair_values": values,
            "configuration_similarity": config_similarity["similarity"],
            "measured_performance_metrics": _measured_performance_metrics(record),
            "quality_feasible": feasible,
            "measured_quality_metrics": _measured_quality_metrics(record),
        }
        buckets.setdefault((signature, category), []).append(case)

    result = {"quality_success_cases": [], "quality_failure_cases": []}
    for (signature, category), cases in sorted(buckets.items()):
        del signature
        selected = max(
            cases,
            key=lambda item: (
                float(item["configuration_similarity"]),
                str(item.get("trial_id") or ""),
            ),
        )
        result[category].append(selected)
    for key in result:
        result[key] = sorted(
            result[key],
            key=lambda item: (
                -float(item["configuration_similarity"]),
                str(item["repair_signature"]),
            ),
        )
    result["runtime_failure_cases"] = []
    for record in history.get("trials", []):
        if (not isinstance(record, Mapping)
            or record.get("status") != "runtime_failed"
            or is_timeout_record(record)
            or record.get("repair_evidence_eligible") is False
            or not isinstance(record.get("candidate"), Mapping)):
            continue
        candidate = record["candidate"]
        similarity = grouped_configuration_similarity(
            parent_candidate, candidate, space=space, repair_fields=scope.fields,
        )
        if repair_round == 1 and (
            parent_candidate.get("data_parallel_size") != 2
            or candidate.get("data_parallel_size") != 2
        ):
            continue
        result["runtime_failure_cases"].append({
            "trial_id": record.get("trial_id"),
            "status": "runtime_failed",
            "candidate": dict(record["candidate"]),
            "quality_feasible": None,
            "measured_quality_metrics": [],
            "measured_performance_metrics": {},
            "runtime_failure": _runtime_failure_context(record),
            "configuration_similarity": similarity["similarity"],
        })
    if history.get('repair_cold_start_memory'):
        from .repair_memory_ranking import rank_memory
        result['historical_quality_memory'] = rank_memory(
            history['repair_cold_start_memory'], parent_candidate, space)
    return result


def _compact_trial_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep measured evidence useful to RepairBO without copying full receipts."""

    return {
        "trial_id": record.get("trial_id"),
        "status": record.get("status"),
        "candidate": dict(record.get("candidate") or {}),
        "measured_performance_metrics": _measured_performance_metrics(record),
        "quality_feasible": record.get("quality_feasible"),
        "measured_quality_metrics": _measured_quality_metrics(record),
        "failure_kind": record.get("failure_kind"),
        "runtime_failure": (
            _runtime_failure_context(record) if record.get("status") == "runtime_failed" else {}
        ),
        "repair_strategy": record.get("repair_strategy"),
        "repair_actual_hv_gain": record.get("repair_actual_hv_gain"),
        "repair_success": record.get("repair_success"),
        "repair_warmup_guard": record.get("repair_warmup_guard"),
    }


def _trial_ids_in_context(value: Any) -> set[str]:
    trial_ids: set[str] = set()
    if isinstance(value, Mapping):
        trial_id = value.get("trial_id")
        if trial_id:
            trial_ids.add(str(trial_id))
        for item in value.values():
            trial_ids.update(_trial_ids_in_context(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            trial_ids.update(_trial_ids_in_context(item))
    return trial_ids


def _round_2_mode(results: Sequence[Mapping[str, Any]]) -> str:
    """Route runtime failures like measured quality failures, as requested."""

    return (
        "performance_recovery"
        if any(
            item.get("status") == "complete"
            and item.get("quality_feasible") is True
            for item in results
        )
        else "quality_repair"
    )


def _normalized_dominance_shortfall(
    objectives: Sequence[float],
    front: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    if not front:
        return 0.0
    point = np.asarray(objectives, dtype=float)
    pareto = np.asarray(front, dtype=float)
    ref = np.asarray(reference, dtype=float)
    upper = np.max(pareto, axis=0)
    scale = np.maximum(upper - ref, np.finfo(float).eps)
    return float(
        min(np.sum(np.maximum(candidate - point, 0.0) / scale) for candidate in pareto)
    )


def _performance_recovery_anchor(
    results: Sequence[Mapping[str, Any]],
    *,
    frozen_front: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> dict[str, Any]:
    feasible = [
        dict(item)
        for item in results
        if item.get("status") == "complete"
        and item.get("quality_feasible") is True
        and isinstance(item.get("objective_vector"), Sequence)
    ]
    if not feasible:
        raise ValueError("performance recovery requires a quality-feasible anchor")
    return min(
        feasible,
        key=lambda item: (
            _normalized_dominance_shortfall(
                item["objective_vector"], frozen_front, reference
            ),
            -float(item.get("quality_margin") or 0.0),
            -float(item["objective_vector"][0]),
            -float(item["objective_vector"][1]),
            -float(item["objective_vector"][2]),
            str(item.get("trial_id") or ""),
        ),
    )


def _performance_context(
    history: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    frozen_front: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> dict[str, Any]:
    complete = [
        item
        for item in complete_records(history, include_baseline=True)
        if isinstance(item.get("objective_vector"), Sequence)
    ]
    feasible = [item for item in complete if item.get("quality_feasible") is True]
    landmarks: dict[str, Any] = {}
    for index, name in enumerate(("best_qps", "best_mean_latency", "best_p95_latency")):
        if feasible:
            landmarks[name] = _compact_trial_context(
                max(feasible, key=lambda item: float(item["objective_vector"][index]))
            )
    pareto_records = feasible_pareto_records(history)
    anchor_values = anchor.get("objective_vector")
    anchor_objectives = (
        np.asarray(anchor_values, dtype=float)
        if isinstance(anchor_values, Sequence) and len(anchor_values) == 3
        else None
    )
    nearest: Mapping[str, Any] | None = None
    nearest_gap = math.inf
    for item in pareto_records:
        if anchor_objectives is None:
            break
        objectives = np.asarray(item["objective_vector"], dtype=float)
        gap = _normalized_dominance_shortfall(
            anchor_objectives.tolist(), [objectives.tolist()], reference
        )
        if gap < nearest_gap:
            nearest_gap = gap
            nearest = item
    return {
        "objective_coordinates": [
            "qps",
            "negative_mean_latency",
            "negative_p95_latency",
        ],
        "frozen_feasible_pareto_front": [list(item) for item in frozen_front],
        "current_measured_feasible_pareto_trials": [
            _compact_trial_context(item) for item in pareto_records
        ],
        "performance_landmarks": landmarks,
        "nearest_feasible_pareto_trial": (
            _compact_trial_context(nearest) if nearest is not None else None
        ),
        "anchor_normalized_dominance_shortfall": (
            _normalized_dominance_shortfall(
                anchor_objectives.tolist(), frozen_front, reference
            )
            if anchor_objectives is not None else None
        ),
        "evidence_rule": (
            "all performance landmarks and Pareto points in this section are measured; "
            "no BO candidate prediction is included"
        ),
    }


def _performance_recovery_fields(
    space: FixedSearchSpace, representation_fields: Sequence[str]
) -> tuple[str, ...]:
    excluded = set(representation_fields)
    if "cacheblend_mode" in excluded:
        # The CacheBlend extension permits repairing the switch only. Its
        # children remain native BO knobs, including in performance recovery.
        excluded.update({"lmcache_chunk_size", "cacheblend_blend_min_tokens",
                         "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios",
                         "cacheblend_allow_prefix_caching"})
    return tuple(
        parameter.name
        for parameter in space.tuned_parameters
        if parameter.name not in excluded
    )


def _repair_signature(
    candidate: Mapping[str, Any],
    *,
    fields: Sequence[str],
    space: FixedSearchSpace,
) -> str:
    canonical = space.bo_to_canonical(candidate)
    return json.dumps(
        [[field, _signature_value(canonical[field])] for field in fields],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def generate_performance_recovery_candidates(
    *,
    history: Mapping[str, Any],
    anchor: Mapping[str, Any],
    space: FixedSearchSpace,
    representation_fields: Sequence[str],
    runtime_constraints: Sequence[Mapping[str, Any]],
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    random_seed: int,
    pool_size: int,
    local_only: bool = False,
) -> tuple[list[dict[str, Any]], RepairSelectionScope, list[dict[str, Any]]]:
    """Sample non-representation fields, optionally limiting edits to two."""

    if pool_size < 1:
        raise ValueError("performance recovery requires a positive candidate pool")
    fields = _performance_recovery_fields(space, representation_fields)
    if not fields:
        raise ValueError("Range exposes no non-representation performance fields")
    selection_scope = RepairSelectionScope.from_space(
        space,
        fields=fields,
        runtime_constraints=runtime_constraints,
        candidate_generation=(
            "local_non_representation_max_two_fields_with_performance_scoring"
            if local_only else
            "full_non_representation_range_sampling_then_performance_bo_ranking"
        ),
    )
    anchor_candidate = dict(anchor.get("candidate") or {})
    anchor_canonical = space.bo_to_canonical(anchor_candidate)
    executed_keys = _executed_candidate_keys(history)
    allowed = set(fields)
    derived = set(selection_scope.derived_fields)
    candidates: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    def add(raw: Mapping[str, Any], *, source: str, source_trial_id: str | None) -> None:
        try:
            receipt = dict(validator(dict(raw)))
        except Exception as exc:
            rejected.append(
                {
                    "source": source,
                    "source_trial_id": source_trial_id,
                    "reason": "validator_exception",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        repaired = receipt.get("candidate")
        if receipt.get("valid") is not True or not isinstance(repaired, Mapping):
            rejected.append(
                {
                    "source": source,
                    "source_trial_id": source_trial_id,
                    "reason": "incompatible",
                    "validation_errors": list(receipt.get("validation_errors") or []),
                }
            )
            return
        repaired_dict = dict(repaired)
        repaired_canonical = space.bo_to_canonical(repaired_dict)
        if any(
            repaired_canonical[field] != anchor_canonical[field]
            for field in representation_fields
        ):
            rejected.append(
                {
                    "source": source,
                    "source_trial_id": source_trial_id,
                    "reason": "quality_anchor_representation_changed",
                }
            )
            return
        changed = [
            field
            for field in fields
            if repaired_canonical[field] != anchor_canonical[field]
        ]
        changed_outside_scope = [
            field
            for field in set(repaired_canonical) | set(anchor_canonical)
            if field not in allowed
            and field not in set(representation_fields)
            and field not in derived
            and repaired_canonical.get(field) != anchor_canonical.get(field)
        ]
        if (
            not changed
            or (local_only and len(changed) > 2)
            or changed_outside_scope
            or _stable_key(repaired_dict) in executed_keys
        ):
            return
        signature = _repair_signature(repaired_dict, fields=fields, space=space)
        candidates.setdefault(
            signature,
            {
                "signature": signature,
                "values": {
                    field: _signature_value(repaired_canonical[field])
                    for field in fields
                },
                "candidate": repaired_dict,
                "validator_receipt": receipt,
                "candidate_source": source,
                "source_trial_ids": [source_trial_id] if source_trial_id else [],
                "changed_fields": changed,
                "changed_field_count": len(changed),
            },
        )

    # Transfer every differing non-representation field from measured
    # Pareto/component-best configurations; there is no edit-count ceiling.
    measured = [
        item
        for item in complete_records(history, include_baseline=True)
        if isinstance(item.get("candidate"), Mapping)
        and isinstance(item.get("objective_vector"), Sequence)
    ]
    landmark_records: list[Mapping[str, Any]] = list(feasible_pareto_records(history))
    for index in range(3):
        if measured:
            landmark_records.append(
                max(measured, key=lambda item: float(item["objective_vector"][index]))
            )
    rng = np.random.default_rng(random_seed)
    for record in landmark_records:
        source_canonical = space.bo_to_canonical(record["candidate"])
        differing = [
            field for field in fields if source_canonical[field] != anchor_canonical[field]
        ]
        if not differing:
            continue
        groups = [[field] for field in differing] if local_only else [differing]
        for group in groups:
            raw = dict(anchor_candidate)
            for field in group:
                parameter = space.by_name[field]
                value = source_canonical[field]
                raw[field] = value
                if parameter.bo_name != field:
                    raw[parameter.bo_name] = value
            add(raw, source="measured_performance_landmark_transfer",
                source_trial_id=str(record.get("trial_id") or "") or None)

    # Every non-representation field may move in the same candidate.  The pool
    # size bounds scoring cost; it does not cap the number of parameter edits.
    parameters = [space.by_name[field] for field in fields]
    attempts = 0
    attempt_limit = max(2000, pool_size * 200)
    while len(candidates) < pool_size and attempts < attempt_limit:
        attempts += 1
        change_count = int(rng.integers(1, min(2, len(fields)) + 1 if local_only else len(fields) + 1))
        selected = list(rng.choice(parameters, size=change_count, replace=False))
        raw = dict(anchor_candidate)
        for parameter in selected:
            value = parameter.sample(rng)
            for _ in range(8):
                if value != anchor_canonical[parameter.name]:
                    break
                value = parameter.sample(rng)
            if local_only and parameter.values is None:
                center = anchor_canonical[parameter.name]
                if isinstance(center, (int, float)) and center > 0 and not center / 2 <= value <= center * 2:
                    continue
            raw[parameter.name] = value
            if parameter.bo_name != parameter.name:
                raw[parameter.bo_name] = value
        add(raw, source="local_non_representation_sampling" if local_only else "full_non_representation_range_sampling", source_trial_id=None)
    return list(candidates.values()), selection_scope, rejected


def _reference_point(history: Mapping[str, Any], margin: float) -> np.ndarray:
    records = complete_records(history, include_baseline=True)
    points = np.asarray(
        [item["objective_vector"] for item in records if item.get("objective_vector")],
        dtype=float,
    )
    if points.size == 0:
        raise ValueError(
            "RepairBO requires at least one complete performance observation"
        )
    minima = np.min(points, axis=0)
    spans = np.maximum(np.ptp(points, axis=0), np.maximum(np.abs(minima), 1.0))
    return minima - float(margin) * spans - 1e-12


def score_repair_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    space: FixedSearchSpace,
    random_seed: int,
    monte_carlo_samples: int,
    reference_margin: float,
    prediction_provider: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Score repair candidates with the current native BO posterior."""

    if not candidates:
        return []
    complete = complete_records(history, include_baseline=True)
    usable = [
        item
        for item in complete
        if isinstance(item.get("candidate"), Mapping)
        and isinstance(item.get("objective_vector"), Sequence)
    ]
    encoded_candidates = space.encode([item["candidate"] for item in candidates])
    if not usable:
        return [
            {
                **dict(item),
                "predicted_objectives": None,
                "objective_uncertainty": None,
                "ehvi": 0.0,
                "p_run": 1.0,
                "score": 0.0,
            }
            for item in candidates
        ]
    if prediction_provider is None:
        raise ValueError("Current native BO prediction provider is required; no separate repair GP fallback")
    predictions = prediction_provider(history=history, candidates=[c["candidate"] for c in candidates])
    front = np.asarray(
        [item["objective_vector"] for item in feasible_pareto_records(history)],
        dtype=float,
    )
    reference = _reference_point(history, reference_margin)
    objective_mean = np.asarray([p["predicted_objectives"] for p in predictions])
    objective_std = np.asarray([p["objective_uncertainty"] for p in predictions])
    rng = np.random.default_rng(random_seed)
    objective_samples = rng.normal(
        loc=objective_mean[None, :, :],
        scale=objective_std[None, :, :],
        size=(monte_carlo_samples, encoded_candidates.shape[0], 3),
    )
    ehvi = np.zeros(encoded_candidates.shape[0], dtype=float)
    for candidate_index in range(encoded_candidates.shape[0]):
        ehvi[candidate_index] = (
            sum(
                hypervolume_improvement(front, sample, reference)
                for sample in objective_samples[:, candidate_index, :]
            )
            / monte_carlo_samples
        )
    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        scored.append(
            {
                **dict(candidate),
                **predictions[index],
                "predicted_objectives": objective_mean[index].tolist(),
                "objective_uncertainty": objective_std[index].tolist(),
                "ehvi": float(ehvi[index]),
                "p_run": None,
                "score": float(ehvi[index]),
            }
        )
    return scored


def _response_payload(response: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("structured_output", "output", "result"):
        value = response.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return dict(response)


def _candidate_schema(
    candidate_ids: Sequence[str],
    scope: RepairScope | RepairSelectionScope,
    *,
    require_target_values: bool = False,
) -> dict[str, Any]:
    scope_contract = scope.contract()
    repair_fields = scope.fields

    def field_schema(field: str) -> dict[str, Any]:
        domain = scope_contract["domains"][field]
        if isinstance(domain, Mapping):
            values = domain.get("values")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                return {"enum": list(values)}
            kind = str(domain.get("kind") or "")
            schema: dict[str, Any] = {
                "type": "integer" if kind == "integer" else "number"
            }
            if isinstance(domain.get("lower"), (int, float)):
                schema["minimum"] = domain["lower"]
            if isinstance(domain.get("upper"), (int, float)):
                schema["maximum"] = domain["upper"]
            return schema
        return {"enum": list(domain)}

    repair_properties = {
        str(field): field_schema(str(field)) for field in repair_fields
    }
    evidence_properties: dict[str, Any] = {
        "cited_trial_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "expected_repaired_metrics": {"type": "object"},
        "performance_risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 1},
    }
    evidence_required = [
        "cited_trial_ids",
        "expected_repaired_metrics",
        "performance_risks",
        "confidence",
        "reason",
    ]
    if require_target_values:
        return {
            "type": "object",
            "properties": repair_properties,
            "required": [str(field) for field in repair_fields],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": list(candidate_ids)},
            "target_signature": {"type": "string"},
            "changes": {
                "type": "object",
                "properties": repair_properties,
                "minProperties": 1,
                "additionalProperties": False,
            },
            **evidence_properties,
        },
        "required": [
            "candidate_id",
            "target_signature",
            "changes",
            *evidence_required,
        ],
        "additionalProperties": False,
    }


def _parameter_target_schema(parameter: Any) -> dict[str, Any]:
    if not parameter.tuned:
        return {"enum": [parameter.fixed_value]}
    values = parameter.values
    if values is not None:
        return {"enum": list(values)}
    domain = dict(parameter.domain or {})
    schema: dict[str, Any] = {
        "type": "integer" if parameter.kind == "integer" else "number"
    }
    if isinstance(domain.get("lower"), (int, float)):
        schema["minimum"] = domain["lower"]
    if isinstance(domain.get("upper"), (int, float)):
        schema["maximum"] = domain["upper"]
    return schema


def _complete_configuration_schema(space: FixedSearchSpace) -> dict[str, Any]:
    fields = [parameter.name for parameter in space.parameters]
    return {
        "type": "object",
        "properties": {
            "target_values": {
                "type": "object",
                "properties": {
                    parameter.name: _parameter_target_schema(parameter)
                    for parameter in space.parameters
                },
                "required": fields,
                "additionalProperties": False,
            },
            "cited_trial_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "expected_repaired_metrics": {"type": "object"},
            "performance_risks": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "target_values",
            "cited_trial_ids",
            "expected_repaired_metrics",
            "performance_risks",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }


def _complete_range_contract(
    space: FixedSearchSpace,
    runtime_constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source": space.source_path,
        "parameters": [
            {
                "name": parameter.name,
                "bo_name": parameter.bo_name,
                "decision": parameter.decision,
                "domain": dict(parameter.domain) if parameter.domain else None,
                "fixed_value": parameter.fixed_value,
                "reference_value": parameter.reference_value,
                "activation_conditions": list(parameter.activation_conditions),
                "group_id": parameter.group_id,
            }
            for parameter in space.parameters
        ],
        "joint_constraints": [dict(item) for item in space.joint_constraints],
        "runtime_constraints": [dict(item) for item in runtime_constraints],
        "output_rule": (
            "return every Range parameter by canonical name; fixed parameters must retain "
            "their fixed value and inactive conditional parameters must use their reference value"
        ),
    }


def _rank_key(item: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(item.get("score") or 0.0),
        float(item.get("ehvi") or 0.0),
        float(item.get("p_run") or 0.0),
        str(item.get("signature") or ""),
    )


_BO_PREDICTION_FIELDS = frozenset(
    {
        "predicted_objectives",
        "objective_uncertainty",
        "objective_variance",
        "prediction_source",
        "complete_training_rows",
        "training_trial_ids",
        "ehvi",
        "p_run",
        "score",
    }
)


def _first_round_evidence_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    """Remove every BO prediction before first-round LLM selection."""

    return {
        key: value
        for key, value in dict(item).items()
        if key not in _BO_PREDICTION_FIELDS
    }


def _first_round_parent_context(parent: Mapping[str, Any]) -> dict[str, Any]:
    """Expose compact measured evidence, excluding raw receipts and BO diagnostics."""

    return _compact_trial_context(parent)


def _strategy_uses_success_reuse(strategy: str) -> bool:
    return strategy in {
        "A_evidence_reuse",
        "B_to_A_evidence_reuse",
    }


def _strategy_uses_novel_synthesis(strategy: str) -> bool:
    return strategy in {
        "B_novel_synthesis",
        "A_to_B_novel_synthesis",
        "B_evidence_tradeoff",
    }


def _first_round_case_context(
    cases: Mapping[str, Any], *, success_reuse: bool
) -> dict[str, Any]:
    """A hides quality failures; both strategies retain runtime-failure evidence."""

    selected = dict(cases)
    if success_reuse:
        selected["quality_failure_cases"] = []
    return selected


class RepairReasoner:
    """Bounded structured LLM selector with deterministic validation fallback."""

    def __init__(self, client: StructuredInferenceClient | None) -> None:
        self.client = client
        self.last_failure: dict[str, Any] | None = None

    def choose(
        self,
        *,
        strategy: str,
        parent: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        cases: Mapping[str, Any],
        scope: RepairScope | RepairSelectionScope,
        critique: Mapping[str, Any] | None = None,
        use_bo_predictions: bool = True,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError(f"{strategy} has no eligible candidates")
        self.last_failure = None
        success_reuse = _strategy_uses_success_reuse(strategy)
        novel_synthesis = _strategy_uses_novel_synthesis(strategy)
        first_round_strategy = success_reuse or novel_synthesis
        round_2_quality_repair = strategy == "round_2_quality_repair"
        trajectory_synthesis = novel_synthesis or round_2_quality_repair
        direct_target_values = first_round_strategy or round_2_quality_repair
        if direct_target_values:
            use_bo_predictions = False
        prompt_parent = (
            _first_round_parent_context(parent)
            if not use_bo_predictions
            else dict(parent)
        )
        ranked = (
            sorted((dict(item) for item in candidates), key=_rank_key, reverse=True)
            if use_bo_predictions
            else [_first_round_evidence_candidate(item) for item in candidates]
        )
        exposed: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(ranked):
            candidate_id = f"candidate_{index + 1:03d}"
            exposed_item = (
                dict(item["values"])
                if direct_target_values
                else {
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "candidate"
                    },
                    "candidate_id": candidate_id,
                }
            )
            exposed.append(exposed_item)
            by_id[candidate_id] = item

        attempts: list[dict[str, Any]] = []
        schema = _candidate_schema(
            list(by_id),
            scope,
            require_target_values=direct_target_values,
        )
        if direct_target_values:
            # Independent field domains also admit triples outside this branch.
            # Constrain the complete tuple, including the exclusion of measured A.
            schema["enum"] = [dict(item["values"]) for item in ranked]
        parent_candidate = dict(parent.get("candidate") or {})
        parent_values = {
            field: _signature_value(parent_candidate.get(field))
            for field in scope.fields
        }
        allowed_citations = _trial_ids_in_context(cases)
        allowed_citations.add(str(parent.get("trial_id") or ""))
        success_citations = {
            str(item.get("trial_id"))
            for item in cases.get("quality_success_cases", [])
            if isinstance(item, Mapping) and item.get("trial_id")
        }
        failure_citations = {
            str(item.get("trial_id"))
            for item in cases.get("quality_failure_cases", [])
            if isinstance(item, Mapping) and item.get("trial_id")
        }
        if self.client is not None:
            evidence = _prompt_evidence(prompt_parent, cases)
            allowed_citations = _trial_ids_in_context(evidence)
            system = render_tagged_sections(
                (
                    (
                        "当前角色",
                        {
                            "name": f"RepairBO {strategy} reasoner",
                            "responsibility": (
                                "从已提供且已通过机械校验的候选中选择一个修复方案，"
                                "兼顾语义质量可行性与 Pareto 性能。"
                            ),
                        },
                    ),
                    (
                        "角色边界",
                        {
                            "rules": [
                                "只能使用当前 prompt 明确提供的证据。",
                                "不得改动可修参数范围以外的冻结参数。",
                                "不得自行发明候选、repair signature 或 Trial ID。",
                                "只返回满足 API response schema 的 JSON 对象。",
                            ]
                        },
                    ),
                )
            )
            scope_contract = scope.contract()
            repair_mode = str(cases.get("repair_mode") or "representation_repair")
            objective = (
                "质量锚点已经通过全部实测硬门槛。选择且仅选择一个候选，"
                "在不改变该锚点模型表示可修参数组的前提下，优先改善当前可行 Pareto 性能。"
                if repair_mode == "performance_recovery"
                else (
                    "R3阶段只修复当前 scope.fields 列出的参数。结合R1/R2 阶段"
                    "两个点的实测结果、检索到的质量成功/失败 Trial 和 workload 特征，"
                    "直接给出你认为最可能通过全部质量硬门槛的可修参数组目标值；"
                    "R1/R2 阶段运行失败同样视为未能证明质量可行。历史成功案例是软证据，"
                    "不得强制复用某个已有成功 signature。首个提案不使用 BO 预测；"
                    "反馈修订时可参考评审中的性能预测及不确定性，预测不能代替实测。"
                )
                if repair_mode == "quality_repair"
                else (
                    "这是R1/R2阶段成功经验复用策略。只依据检索到的、与当前 Trial 最相似"
                    "的质量成功案例，从候选中选择一个已有成功 signature 进行复用。"
                    "必须依据与所选 signature 对应的成功 Trial；首个提案不参考历史质量失败案例，"
                    "但 R2 回退到复用策略时仍须参考本 Episode 的 R1 实测结果；"
                    "首个提案依据实测证据；修订时应参考评审反馈中的性能代价和不确定性。"
                    if success_reuse
                    else (
                        "这是 R2 替代方案策略。历史成功、失败和未探索三组合均可考虑；R1 已实测时 R2 必须与 R1 不同。结合与当前 Trial 最"
                        "相似的质量成功案例和质量失败案例，归纳可修参数组与各项质量门槛的"
                        "关系，再从合法 signature 候选中选择你认为最可能通过全部质量"
                        "硬门槛且实际性能较好的一个。判断只能使用已提供的实测案例、"
                        "当前 Trial 和 workload 信息；修订时可使用评审反馈，预测不等于实测。"
                        if novel_synthesis
                        else (
                            "选择且仅选择一个 eligible candidate，并严格遵守当前策略"
                            "提供的实测证据和硬约束。"
                        )
                    )
                )
            )
            prompt = render_tagged_sections(
                (
                    (
                        "当前任务",
                        {
                            "stage": f"repair_bo_{strategy.lower()}",
                            "strategy": strategy,
                            "repair_mode": repair_mode,
                            "objective": objective,
                        },
                    ),
                    ("待修复Trial", evidence["parent"]),
                    ("R1/R2 阶段执行顺序", "先提案和评审 R1；R1 在有限修订后仍被评审拒绝则跳过实测，把未执行的提案和评审意见传给 R2。否则实测 R1，再将其结果（含运行失败摘要）交给 R2。R2 仍以原父 Trial 为配置锚点，LLM 保持非可修参数组冻结（系统的预热显存检查例外）；R1 已实测时 R2 选择与 R1 不同的组合。R1实测质量达标且刷新冻结可行前沿时跳过R2；否则继续R2。R2 完成后按实际测得的结果判断是否进入R3阶段。"),
                    ("Repair预热显存检查", REPAIR_WARMUP_GUARD),
                    ("实测记录", evidence["records"]),
                    ("运行失败轨迹", evidence["runtime_failures"]),
                    ("运行失败规避规则", "运行失败的质量和性能未测得，空指标不表示通过。结合完整失败配置、失败类型和错误摘要规避风险；禁止重复已执行的完整配置。R1/R2阶段历史运行失败在父配置与失败配置均为 dp=2 时提供，不设配置相似度或字段完整性准入门槛；相似度仅供参考，缺失配置不能视为相同。当前 Episode 的 R1 结果固定提供给 R2。运行失败只是有适用边界的软风险证据，不得仅凭相同可修参数组或历史 OOM 排除整个组合。必须对比 max_num_seqs、max_num_batched_tokens、gpu_memory_utilization、并行布局和 enforce_eager 等实际差异，说明失败机制在当前配置是否仍成立；证据不足时保留候选并注明不确定性，不能断言它必然失败。"),
                    ("实测记录说明", "按 Trial ID 引用；每条只含质量实测值、性能实测值和完整配置。质量门槛统一见质量硬门槛，缺失测量不代表通过。"),
                    ("本轮修复目标分支", cases.get("repair_target", {})),
                    (
                        "本Episode前序阶段实测结果",
                        evidence["round_1"],
                    ),
                    ("质量硬门槛", cases.get("quality_gate", {})),
                    (
                        "当前实测Pareto与性能参照",
                        evidence["performance"],
                    ),
                    ("Workload特征", cases.get("workload_features", {})),
                    ("prior_repair_reflections（条件经验，仅供排序，不是硬约束）", cases.get("prior_repair_reflections", [])),
                    ("三参数及取值简介" if tuple(scope.fields) == REPAIR_BO_V1_FIELDS else "可修参数组及取值简介", _representation_parameter_guide(scope_contract["domains"])),
                    (
                        "搜索空间与机械约束",
                        {
                            "full_range_reference": cases.get(
                                "full_search_space", {}
                            ),
                            "active_repair_scope": scope_contract,
                            "frozen_parameter_rule": (
                                "除 fields 和允许联动的 derived_fields 外，"
                                "父 Trial 的所有参数保持不变；系统可按 Repair预热显存检查单独下调 gpu_memory_utilization，并记录原因。"
                            ),
                            "compatibility_constraints": scope_contract[
                                "compatibility_constraints"
                            ],
                            "runtime_constraints": scope_contract[
                                "runtime_constraints"
                            ],
                            "candidate_validation_guarantee": (
                                "eligible_candidates 均已由机械 validator 按上述约束校验；"
                                "这些约束仍是不可违反的硬约束。"
                            ),
                        },
                    ),
                    (
                        "质量证据",
                        {
                            "success_cases": evidence["success"],
                            "failure_cases": evidence["failure"],
                            "interpretation": (
                                "failure cases 是按相似度加权的软反例，不是禁用名单。"
                            ),
                        },
                    ),
                    (
                        "候选与性能预测" if use_bo_predictions else "合法候选",
                        exposed,
                    ),
                    ("Repair审计反馈", _prompt_critique(critique)),
                    ("执行前评审反馈", cases.get("proposal_feedback", {})),
                    ("R1分支未执行的评审信息", cases.get("skipped_a_review", {})),
                    ("R1跳过时的解释", "若 R1 因修订后仍未通过评审而跳过，参考其原提案、修订提案和评审意见，但这些均不是实测，不能据此宣称运行失败、质量失败或性能下降。R2 仍以原父点为锚点；未执行的 R1 提案不构成机械禁用名单。"),
                    ("配对修复的实测代价", cases.get("paired_repair_effects", [])),
                    (
                        "结构化输出机械规范",
                        {
                            "prompt_schema": PROMPT_FORMAT_SCHEMA,
                            "response_schema": schema,
                            "validation_order": (
                                [
                                    "顶层恰好是scope.fields 中全部必填 canonical 参数",
                                    "禁止包装层、解释、证据或其他字段",
                                    "每个值属于当前 Range 枚举",
                                    "兼容性、依赖与运行时硬约束",
                                    "全部目标值与当前分支合法候选逐字段完全相等",
                                    "合并父 Trial 后确认仅 scope 和系统预热显存检查允许的参数变化，且实际完整配置未运行过",
                                ]
                                if direct_target_values
                                else [
                                    "JSON Schema 类型、必填字段、additionalProperties=false",
                                    "每个值属于当前 Range 枚举或数值上下界",
                                    "兼容性、依赖与运行时硬约束",
                                    "候选标识、Trial 引用、去重与冻结参数检查",
                                ]
                            ),
                            "invalid_output_policy": (
                                "最多反馈并重试一次；连续两次无效则 fail closed，"
                                "不运行任何兜底配置。"
                            ),
                            "contract": (
                                "直接返回一个 schema-valid JSON 对象；顶层必须且只能列出 "
                                "scope.fields 中的全部目标值，并与"
                                "合法候选之一完全一致。系统校验后会将这全部目标值应用到父 Trial "
                                "的完整配置；除系统预热显存检查外其余参数保持冻结。不要返回 candidate_id、"
                                "target_values 包装层、target_signature、证据、解释、"
                                "置信度或其他任何字段。"
                                if direct_target_values
                                else (
                                    "返回且仅返回一个 schema-valid JSON 对象；candidate_id 与 "
                                    "target_signature 必须逐字复制所选候选，changes 只能包含"
                                    "实际发生变化的可修参数。"
                                )
                            ),
                        },
                    ),
                )
            )
            for attempt_index in range(2):
                try:
                    response = self.client.complete_structured(
                        system=system, user=prompt, response_schema=schema,
                    )
                    payload = _response_payload(response)
                    candidate_id = str(payload.get("candidate_id") or "")
                    if direct_target_values:
                        target_values = payload
                        chosen = next(
                            (
                                item
                                for item in ranked
                                if isinstance(target_values, Mapping)
                                and dict(target_values) == item.get("values")
                            ),
                            None,
                        )
                    else:
                        chosen = by_id.get(candidate_id)
                    expected_changes = (
                        {
                            field: chosen["values"][field]
                            for field in scope.fields
                            if chosen["values"].get(field) != parent_values.get(field)
                        }
                        if chosen is not None
                        else {}
                    )
                    cited = {str(item) for item in payload.get("cited_trial_ids", [])}
                    matching_success_citations = {
                        str(item.get("trial_id"))
                        for item in cases.get("quality_success_cases", [])
                        if isinstance(item, Mapping)
                        and item.get("trial_id")
                        and chosen is not None
                        and item.get("repair_signature") == chosen.get("signature")
                    }
                    direct_shape_valid = (
                        set(payload) == set(scope.fields)
                        if direct_target_values
                        else True
                    )
                    valid = (
                        chosen is not None
                        and direct_shape_valid
                        and (
                            direct_target_values
                            or payload.get("target_signature")
                            == chosen["signature"]
                        )
                        and (
                            direct_target_values
                            or payload.get("changes") == expected_changes
                        )
                        and (
                            direct_target_values
                            or cited.issubset(allowed_citations)
                        )
                        and (
                            not success_reuse
                            or bool(matching_success_citations)
                        )
                        and (
                            direct_target_values
                            or not trajectory_synthesis
                            or not success_citations
                            or bool(cited & success_citations)
                        )
                        and (
                            direct_target_values
                            or not trajectory_synthesis
                            or not failure_citations
                            or bool(cited & failure_citations)
                        )
                    )
                    attempts.append(
                        {
                            "attempt": attempt_index + 1,
                            "valid": valid,
                            "response": payload,
                            "transport_metadata": {
                                key: response.get(key)
                                for key in (
                                    "usage",
                                    "model",
                                    "latency_seconds",
                                    "elapsed_seconds",
                                )
                                if response.get(key) is not None
                            },
                        }
                    )
                    if valid and chosen is not None:
                        return {
                            "candidate": chosen,
                            "agent_output": payload,
                            "selection_mode": (
                                "llm_direct_triplet_configuration"
                                if direct_target_values
                                else "llm_candidate_selection"
                            ),
                            "llm_attempts": attempts,
                        }
                    prompt += "\n\n" + render_tagged_sections(
                        (
                            (
                                "输出校验反馈",
                                {
                                    "failed_attempt": attempt_index + 1,
                                    "rejected_output": payload,
                                    "allowed_triplets": exposed if direct_target_values else None,
                                    "instruction": (
                                        (
                                            "上一输出未通过协议校验。必须直接、完整复制某个"
                                            "合法候选的全部可修参数；顶层必须且只能包含 "
                                            "scope.fields 列出的参数，不得包含"
                                            "包装层、解释或证据字段。"
                                            if direct_target_values
                                            else (
                                                "上一输出未通过协议校验。必须选择已列出的 "
                                                "candidate_id，逐字复制其 target_signature，"
                                                "仅报告实际 changes，并且只引用允许的 Trial ID。"
                                            )
                                        )
                                    ),
                                    **(
                                        {}
                                        if direct_target_values
                                        else {"allowed_candidate_ids": list(by_id)}
                                    ),
                                    **(
                                        {}
                                        if direct_target_values
                                        else {
                                            "allowed_cited_trial_ids": sorted(
                                                item
                                                for item in allowed_citations
                                                if item
                                            )
                                        }
                                    ),
                                },
                            ),
                        )
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": attempt_index + 1,
                            "valid": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        if direct_target_values:
            failure = {
                "strategy": strategy,
                "selection_mode": "llm_invalid_after_retry",
                "llm_attempts": attempts,
            }
            self.last_failure = failure
            return {
                "candidate": None,
                "agent_output": None,
                **failure,
            }
        chosen = ranked[0]
        fallback_changes = {
            field: chosen["values"][field]
            for field in scope.fields
            if chosen["values"].get(field) != parent_values.get(field)
        }
        fallback_citations = [
            str(item["trial_id"])
            for key in ("quality_success_cases", "quality_failure_cases")
            for item in cases.get(key, [])
            if isinstance(item, Mapping)
            and item.get("trial_id")
            and item.get("repair_signature") == chosen["signature"]
        ]
        return {
            "candidate": chosen,
            "agent_output": {
                **(
                    {"target_values": dict(chosen["values"])}
                    if direct_target_values
                    else {
                        "candidate_id": next(
                            key for key, value in by_id.items() if value is chosen
                        ),
                        "target_signature": chosen["signature"],
                        "changes": fallback_changes,
                    }
                ),
                "cited_trial_ids": fallback_citations,
                "expected_repaired_metrics": {
                    "objective_vector": chosen.get("predicted_objectives"),
                    "quality": "must pass all measured hard thresholds",
                },
                "performance_risks": [
                    "deterministic fallback after unavailable/invalid LLM output"
                ],
                "confidence": 0.0,
                "reason": (
                    "first evidence-ordered candidate; BO predictions were hidden"
                    if not use_bo_predictions
                    else "highest deterministic performance-EHVI repair candidate"
                ),
            },
            "selection_mode": "deterministic_fallback",
            "llm_attempts": attempts,
        }

    def recommend_complete_configuration(
        self,
        *,
        strategy: str,
        parent: Mapping[str, Any],
        cases: Mapping[str, Any],
        space: FixedSearchSpace,
        runtime_constraints: Sequence[Mapping[str, Any]],
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        executed_candidate_keys: set[str],
    ) -> dict[str, Any]:
        """Let the R3 LLM directly propose every Range parameter."""

        attempts: list[dict[str, Any]] = []
        if self.client is None:
            return {
                "candidate": None,
                "agent_output": None,
                "selection_mode": "llm_unavailable",
                "llm_attempts": attempts,
            }

        schema = _complete_configuration_schema(space)
        range_contract = _complete_range_contract(space, runtime_constraints)
        allowed_citations = _trial_ids_in_context(cases)
        allowed_citations.add(str(parent.get("trial_id") or ""))
        success_citations = {
            str(item.get("trial_id"))
            for item in cases.get("quality_success_cases", [])
            if isinstance(item, Mapping) and item.get("trial_id")
        }
        failure_citations = {
            str(item.get("trial_id"))
            for item in cases.get("quality_failure_cases", [])
            if isinstance(item, Mapping) and item.get("trial_id")
        }
        expected_fields = {parameter.name for parameter in space.parameters}
        parent_candidate = dict(parent.get("candidate") or {})
        parent_canonical = space.bo_to_canonical(parent_candidate)
        system = render_tagged_sections(
            (
                (
                    "当前角色",
                    {
                        "name": "RepairBO R3 full-configuration reasoner",
                        "responsibility": (
                            "只依据已提供的实测成功/失败轨迹、R1/R2阶段结果、workload "
                            "特征和固定 Range，直接推荐一个完整参数配置。"
                        ),
                    },
                ),
                (
                    "角色边界",
                    {
                        "rules": [
                            "不得使用或推断 BO 的候选性能预测。",
                            "不得扩大 Range 或违反兼容性、运行时硬约束。",
                            "必须完整输出 Range 中的每个参数，不得省略未修改参数。",
                            "只能引用当前 prompt 中提供的 Trial ID。",
                            "只返回满足 response schema 的 JSON 对象。",
                        ]
                    },
                ),
            )
        )
        evidence = _prompt_evidence(parent, cases)
        allowed_citations = _trial_ids_in_context(evidence)
        prompt = render_tagged_sections(
            (
                (
                    "当前任务",
                    {
                        "stage": "repair_bo_round_2_full_configuration",
                        "strategy": strategy,
                        "repair_mode": cases.get("repair_mode"),
                        "objective": (
                            "综合成功和失败轨迹、R1/R2阶段实测结果、当前实测性能前沿与 "
                            "workload 特征，直接给出你认为能满足全部质量硬门槛且性能良好"
                            "的完整 Range 参数配置。质量是硬约束，性能不能补偿质量失败。"
                        ),
                    },
                ),
                ("当前锚点", evidence["parent"]),
                ("Repair预热显存检查", REPAIR_WARMUP_GUARD),
                ("实测记录", evidence["records"]),
                ("运行失败轨迹", evidence["runtime_failures"]),
                ("运行失败规避规则", "运行失败的质量和性能未测得，空指标不表示通过。结合完整失败配置、失败类型和错误摘要规避风险；禁止重复已执行的完整配置。R1/R2阶段历史运行失败在父配置与失败配置均为 dp=2 时提供，不设配置相似度或字段完整性准入门槛；相似度仅供参考，缺失配置不能视为相同。当前 Episode 的 R1 结果固定提供给 R2。运行失败只是有适用边界的软风险证据，不得仅凭相同可修参数组或历史 OOM 排除整个组合。必须对比 max_num_seqs、max_num_batched_tokens、gpu_memory_utilization、并行布局和 enforce_eager 等实际差异，说明失败机制在当前配置是否仍成立；证据不足时保留候选并注明不确定性，不能断言它必然失败。"),
                ("实测记录说明", "按 Trial ID 引用；每条只含质量实测值、性能实测值和完整配置。质量门槛统一见质量硬门槛，缺失测量不代表通过。"),
                ("本轮修复目标分支", cases.get("repair_target", {})),
                ("Repair审计反馈", _prompt_critique(cases.get("auditor_critique"))),
                (
                    "本Episode前序阶段实测结果",
                    evidence["round_1"],
                ),
                ("质量成功轨迹", evidence["success"]),
                ("质量失败轨迹", evidence["failure"]),
                ("质量硬门槛", cases.get("quality_gate", {})),
                (
                    "当前实测Pareto与性能参照",
                    evidence["performance"],
                ),
                ("Workload特征", cases.get("workload_features", {})),
                ("prior_repair_reflections（条件经验，仅供排序，不是硬约束）", cases.get("prior_repair_reflections", [])),
                ("三参数及取值简介" if "cacheblend_mode" not in space.by_name else "可修参数组及取值简介", _representation_parameter_guide({
                    name: tuple(space.by_name[name].values or ())
                    for name in (*REPAIR_BO_V1_FIELDS, "cacheblend_mode") if name in space.by_name
                })),
                ("搜索空间与机械约束", range_contract),
                (
                    "结构化输出机械规范",
                    {
                        "prompt_schema": PROMPT_FORMAT_SCHEMA,
                        "response_schema": schema,
                        "validation_order": [
                            "JSON Schema 类型、全部 Range 字段、additionalProperties=false",
                            "枚举、数值上下界、fixed value 与 conditional reference",
                            "Range joint constraints 与实测 runtime constraints",
                            "合并锚点的非 Range 执行字段",
                            "机械 validator 不得投影改变任一请求的 Range 值",
                            "完整配置不得与已执行配置重复",
                            "Trial 引用必须来自当前上下文",
                        ],
                        "invalid_output_policy": (
                            "最多反馈并重试一次；连续两次无效则 fail closed，"
                            "不运行任何兜底配置。"
                        ),
                        "contract": (
                            "target_values 必须以 canonical name 完整列出 Range 的全部参数。"
                            "系统会将其与锚点中不属于 Range 的执行参数合并，机械校验后运行"
                            "完整配置。不得返回 candidate_id 或只返回差异项。"
                        ),
                    },
                ),
            )
        )

        for attempt_index in range(2):
            validation_errors: list[str] = []
            try:
                response = self.client.complete_structured(
                    system=system,
                    user=prompt,
                    response_schema=schema,
                )
                payload = _response_payload(response)
                raw_target = payload.get("target_values")
                if not isinstance(raw_target, Mapping):
                    validation_errors.append("target_values must be an object")
                    target_values: dict[str, Any] = {}
                else:
                    target_values = dict(raw_target)
                if set(target_values) != expected_fields:
                    validation_errors.append(
                        "target_values must contain every Range parameter exactly once"
                    )
                canonical = space.normalize(target_values)
                validation_errors.extend(space.validate(target_values))

                complete_candidate = dict(parent_candidate)
                for parameter in space.parameters:
                    complete_candidate.pop(parameter.name, None)
                    if parameter.bo_name != parameter.name:
                        complete_candidate.pop(parameter.bo_name, None)
                complete_candidate.update(space.canonical_to_bo(canonical))
                receipt = dict(validator(complete_candidate))
                if receipt.get("valid") is not True:
                    validation_errors.extend(
                        str(item) for item in receipt.get("validation_errors", [])
                    )
                guard = None
                executable = receipt.get("candidate")
                if not isinstance(executable, Mapping):
                    validation_errors.append(
                        "mechanical validator did not return a complete candidate"
                    )
                    executable_candidate: dict[str, Any] = {}
                else:
                    executable_candidate = dict(executable)
                    validated_canonical = space.bo_to_canonical(executable_candidate)
                    if validated_canonical != canonical:
                        validation_errors.append(
                            "mechanical projection changed the requested Range values"
                        )
                    if not validation_errors:
                        executable_candidate, guard = _apply_repair_warmup_guard(
                            executable_candidate, space=space, validator=validator,
                        )
                        canonical = space.bo_to_canonical(executable_candidate)
                    if _stable_key(executable_candidate) in executed_candidate_keys:
                        validation_errors.append(
                            "the resulting complete configuration was already executed"
                        )

                cited = {str(item) for item in payload.get("cited_trial_ids", [])}
                if not cited.issubset(allowed_citations):
                    validation_errors.append("cited_trial_ids contain unknown trials")
                if success_citations and not cited.intersection(success_citations):
                    validation_errors.append(
                        "at least one retrieved quality-success trial must be cited"
                    )
                if failure_citations and not cited.intersection(failure_citations):
                    validation_errors.append(
                        "at least one retrieved quality-failure trial must be cited"
                    )

                valid = not validation_errors
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": valid,
                        "validation_errors": validation_errors,
                        "response": payload,
                        "transport_metadata": {
                            key: response.get(key)
                            for key in (
                                "usage",
                                "model",
                                "latency_seconds",
                                "elapsed_seconds",
                            )
                            if response.get(key) is not None
                        },
                    }
                )
                if valid:
                    changed_fields = [
                        parameter.name
                        for parameter in space.tuned_parameters
                        if canonical.get(parameter.name)
                        != parent_canonical.get(parameter.name)
                    ]
                    return {
                        "candidate": {
                            "signature": _repair_signature(
                                executable_candidate,
                                fields=[
                                    parameter.name
                                    for parameter in space.tuned_parameters
                                ],
                                space=space,
                            ),
                            "values": canonical,
                            "candidate": executable_candidate,
                            "changed_fields": changed_fields,
                            "validator_receipt": receipt,
                            "repair_warmup_guard": guard,
                        },
                        "agent_output": payload,
                        "selection_mode": "llm_direct_full_configuration",
                        "llm_attempts": attempts,
                    }
                prompt += "\n\n" + render_tagged_sections(
                    (
                        (
                            "输出校验反馈",
                            {
                                "failed_attempt": attempt_index + 1,
                                "validation_errors": validation_errors,
                                "instruction": (
                                    "重新输出完整 target_values；必须满足 Range、硬约束、"
                                    "证据引用规则，且不能重复已经运行的完整配置。"
                                ),
                                "allowed_cited_trial_ids": sorted(
                                    item for item in allowed_citations if item
                                ),
                            },
                        ),
                    )
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "candidate": None,
            "agent_output": None,
            "selection_mode": "llm_invalid_after_retry",
            "llm_attempts": attempts,
        }


class RepairAuditor:
    def __init__(self, client: StructuredInferenceClient | None) -> None:
        self.client = client

    def audit(
        self,
        *,
        parent: Mapping[str, Any],
        decisions: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        runtime_failure_cases: Sequence[Mapping[str, Any]] = (),
        skipped_branches: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        fallback = {
            "summary": "At least one first-round branch failed strict repair success.",
            "quality_findings": [
                {
                    "trial_id": item.get("trial_id"),
                    "quality_margin": item.get("quality_margin"),
                    "quality_margins": item.get("quality_margins") or [],
                }
                for item in results
            ],
            "performance_findings": [
                {
                    "trial_id": item.get("trial_id"),
                    "objective_vector": item.get("objective_vector"),
                    "actual_hv_gain": item.get("repair_actual_hv_gain"),
                }
                for item in results
            ],
            "misused_evidence": [],
            "next_round_instructions": [
                "Target an unsuccessful branch and preserve its successful sibling, if any.",
                "Do not repeat an executed complete configuration; use measured evidence.",
            ],
        }
        if self.client is None:
            return {**fallback, "mode": "deterministic"}
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "quality_findings": {"type": "array", "items": {"type": "object"}},
                "performance_findings": {"type": "array", "items": {"type": "object"}},
                "misused_evidence": {"type": "array", "items": {"type": "string"}},
                "next_round_instructions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": list(fallback),
            "additionalProperties": False,
        }
        try:
            system = render_system_prompt(
                role="RepairBO outcome auditor",
                responsibilities=(
                    "Compare the first-round branches' measured outcomes and produce "
                    "grounded instructions targeting an unsuccessful branch in the final round."
                ),
                attention=(
                    "Reference only supplied Trial IDs and measured metrics.",
                    "Do not infer universal causal rules from one episode.",
                ),
            )
            evidence = _prompt_evidence(parent, {
                "episode_round_1_results": results,
                "runtime_failure_cases": runtime_failure_cases,
            })
            gates = {
                (item.get("metric"), item.get("operator")): {
                    key: item.get(key) for key in ("metric", "operator", "threshold")
                }
                for record in [parent, *results]
                for item in _measured_quality_metrics(record)
                if item.get("metric") and item.get("threshold") is not None
            }
            user = render_tagged_sections((
                ("当前任务", "repair_bo_audit_round_1: 复盘未成功分支，质量门槛不能由性能补偿；缺失测量不代表通过。"),
                ("待修复Trial", evidence["parent"]),
                ("Repair预热显存检查", REPAIR_WARMUP_GUARD),
                ("实测记录", evidence["records"]),
                ("运行失败轨迹", evidence["runtime_failures"]),
                ("质量硬门槛", list(gates.values())),
                ("Repair实测结果", evidence["round_1"]),
                ("未执行分支及评审意见", list(skipped_branches)),
                ("Repair决策", [
                    {key: decision.get(key) for key in ("strategy", "strategy_mode")}
                    for decision in decisions
                ]),
            ))
            return {
                **_response_payload(
                    self.client.complete_structured(
                        system=system, user=user, response_schema=schema
                    )
                ),
                "mode": "llm",
            }
        except Exception as exc:
            return {
                **fallback,
                "mode": "deterministic_fallback",
                "error": f"{type(exc).__name__}: {exc}",
            }


def _seen_signatures(
    history: Mapping[str, Any], scope: RepairScope, space: FixedSearchSpace
) -> set[str]:
    seen: set[str] = set()
    records: list[Mapping[str, Any]] = []
    baseline = history.get("baseline_observation")
    if isinstance(baseline, Mapping):
        records.append(baseline)
    records.extend(
        item for item in history.get("trials", []) if isinstance(item, Mapping)
    )
    for record in records:
        if is_timeout_record(record) or record.get("repair_evidence_eligible") is False:
            continue
        candidate = record.get("candidate")
        if isinstance(candidate, Mapping):
            seen.add(scope.signature(candidate, space))
    return seen


def _executed_candidate_keys(history: Mapping[str, Any]) -> set[str]:
    return {
        _stable_key(item["candidate"])
        for item in history.get("trials", [])
        if isinstance(item, Mapping) and isinstance(item.get("candidate"), Mapping)
    }


def _historical_success_matches(
    values: Mapping[str, Any], cases: Mapping[str, Any]
) -> bool:
    """Admit observed external successes for R1, not as a quality guarantee.

    Only the already retrieved memory is consulted. Its applicability and
    counterevidence remain available to both audits; no trial is synthesized.
    """
    memory = cases.get("historical_quality_memory") or {}
    for row in memory.get("observations", []):
        observation = row.get("observation", {})
        claim = observation.get("structured_claim", {})
        if (
            observation.get("category") != "quality_triplet_preference"
            or observation.get("disposition") != "keep"
            or claim.get("disposition") != "keep"
            or claim.get("evidence_status") != "observed_pattern"
            or not claim.get("support_trial_ids")
            or not isinstance(claim.get("statistics", {}).get("passed"), (int, float))
            or claim["statistics"]["passed"] <= 0
        ):
            continue
        triplet = claim.get("triplet", {})
        if values and all(field in triplet and triplet[field] == value
                          for field, value in values.items()):
            return True
    return False


def _strategy_pools(
    *,
    scored: Sequence[Mapping[str, Any]],
    seen_signatures: set[str],
    successful_signatures: set[str],
    boundary_anchor_values: Mapping[str, Any],
    scope: RepairScope,
    excluded_signatures: set[str],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]], str]:
    available = [
        dict(item)
        for item in scored
        if str(item["signature"]) not in excluded_signatures
    ]
    reuse = [item for item in available if item["signature"] in successful_signatures]
    unseen = [item for item in available if item["signature"] not in seen_signatures]
    if reuse:
        a_pool = sorted(reuse, key=lambda item: str(item["signature"]))
        a_mode = "evidence_reuse"
    else:
        a_pool = sorted(
            unseen or available,
            key=lambda item: (
                -_edit_distance(boundary_anchor_values, item["values"], scope.fields),
                str(item["signature"]),
            ),
            reverse=True,
        )
        a_mode = ("novel_synthesis_fallback_no_success" if unseen
                  else "evidence_tradeoff_fallback_no_success")
    # Familiar successful triples remain eligible on a new full configuration.
    # Novelty is a tie-breaker, never an admission rule for B.
    b_pool = sorted(available, key=lambda item: (
        item["signature"] in successful_signatures,
        item["signature"] not in seen_signatures,
        str(item["signature"]),
    ), reverse=True)
    b_mode = "evidence_tradeoff"
    return a_pool, a_mode, b_pool, b_mode


def _trial_by_id(history: Mapping[str, Any], trial_id: str) -> dict[str, Any] | None:
    for item in history.get("trials", []):
        if isinstance(item, Mapping) and str(item.get("trial_id")) == str(trial_id):
            return dict(item)
    return None


class RepairCoordinator:
    """Durable, budget-aware R1/R2/R3 sequential repair state machine."""

    def __init__(
        self,
        *,
        history_path: str | Path,
        result_path: str | Path,
        output_dir: str | Path,
        space: FixedSearchSpace,
        scope: RepairScope,
        evaluation_budget: int,
        cold_start_evaluations: int,
        hv_epsilon: float,
        boundary_utility_epsilon: float,
        monte_carlo_samples: int,
        reference_margin: float,
        random_seed: int,
        context_limit: int,
        adaptive_threshold_enabled: bool = False,
        performance_candidate_pool_size: int = 64,
        proposal_review_enabled: bool = True,
        proposal_max_revisions: int = 1,
        reflect_enabled: bool = False,
        prediction_provider: Callable[..., Any] | None = None,
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        execute_candidate: Callable[
            [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
        ],
        structured_client: StructuredInferenceClient | None = None,
    ) -> None:
        self.history_path = Path(history_path).expanduser().resolve()
        self.result_path = Path(result_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.space = space
        self.scope = scope
        self.evaluation_budget = int(evaluation_budget)
        self.cold_start_evaluations = int(cold_start_evaluations)
        self.adaptive_threshold_enabled = adaptive_threshold_enabled
        self.hv_epsilon = float(hv_epsilon)
        self.boundary_utility_epsilon = float(boundary_utility_epsilon)
        self.monte_carlo_samples = int(monte_carlo_samples)
        self.reference_margin = float(reference_margin)
        self.random_seed = int(random_seed)
        self.context_limit = int(context_limit)
        self.performance_candidate_pool_size = int(performance_candidate_pool_size)
        if self.performance_candidate_pool_size < 1:
            raise ValueError("performance_candidate_pool_size must be positive")
        self.validator = validator
        self.execute_candidate = execute_candidate
        self.proposal_review_enabled = bool(proposal_review_enabled)
        self.prediction_provider = prediction_provider
        self.proposal_max_revisions = int(proposal_max_revisions)
        self.reflect_enabled = bool(reflect_enabled)
        self.structured_client = structured_client
        if not 0 <= self.proposal_max_revisions <= 2:
            raise ValueError("proposal_max_revisions must be between 0 and 2")
        self.reviewer = DualPathRepairReviewer(structured_client)
        self._active_episode: dict[str, Any] | None = None
        self.reasoner = RepairReasoner(structured_client)
        self.auditor = RepairAuditor(structured_client)

    def _guard_fibre(self, fibre: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        # Apply after scope validation but before full-configuration deduplication.
        guarded = []
        for item in fibre:
            candidate, receipt = _apply_repair_warmup_guard(
                item["candidate"], space=self.space, validator=self.validator,
            )
            guarded.append({**item, "candidate": candidate, "repair_warmup_guard": receipt})
        return guarded

    def _history(self) -> dict[str, Any]:
        return _load_object(self.history_path)

    def _paired_repair_effects(self, history: Mapping[str, Any], parent: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Expose measured repair costs, without claiming causal portability."""
        pairs = []
        for trial in history.get("trials", []):
            if trial.get("trial_role") != "repair" or trial.get("status") != "complete":
                continue
            anchor = _trial_by_id(history, str(trial.get("repair_anchor_trial_id") or trial.get("parent_trial_id")))
            if not anchor or anchor.get("status") != "complete":
                continue
            before, after = _measured_performance_metrics(anchor), _measured_performance_metrics(trial)
            ratios = {key: after[key] / value for key, value in before.items()
                      if value > 0 and key in after}
            similarity = grouped_configuration_similarity(
                parent["candidate"], anchor["candidate"], space=self.space, repair_fields=self.scope.fields,
            )["similarity"]
            pairs.append({
                "parent_trial_id": anchor["trial_id"], "trial_id": trial["trial_id"],
                "before": self.scope.values(anchor["candidate"], self.space),
                "after": self.scope.values(trial["candidate"], self.space),
                "quality_feasible": trial.get("quality_feasible"),
                "performance_after_over_before": ratios,
                "configuration_similarity": similarity,
                "changed_fields": sorted(k for k in set(anchor["candidate"]) | set(trial["candidate"])
                                         if anchor["candidate"].get(k) != trial["candidate"].get(k)),
                "parameter_changes": {
                    k: {"before": anchor["candidate"].get(k), "after": trial["candidate"].get(k)}
                    for k in sorted(set(anchor["candidate"]) | set(trial["candidate"]))
                    if anchor["candidate"].get(k) != trial["candidate"].get(k)
                },
            })
        return sorted(pairs, key=lambda row: row["configuration_similarity"], reverse=True)[:12]

    def _choose_with_review(self, **kwargs: Any) -> dict[str, Any]:
        """Propose, review, revise at most N times, then execute one legal point."""
        history = self._history()
        cases = dict(kwargs["cases"])
        if self.reflect_enabled:
            cases["prior_repair_reflections"] = prior_reflections(history, repair_stage(kwargs["strategy"]))
        kwargs = {**kwargs, "cases": cases}
        if not self.proposal_review_enabled:
            return self.reasoner.choose(**kwargs)
        executed = _executed_candidate_keys(history)
        pool = [dict(c) for c in kwargs["candidates"] if _stable_key(c["candidate"]) not in executed]
        if not pool:
            return {"candidate": None, "agent_output": None, "selection_mode": "candidate_exhausted", "llm_attempts": []}
        score_error = None
        try:
            pool = score_repair_candidates(
                candidates=pool, history=history, space=self.space,
                random_seed=self.random_seed + len(history.get("trials", [])),
                monte_carlo_samples=self.monte_carlo_samples, reference_margin=self.reference_margin,
                prediction_provider=self.prediction_provider,
            )
        except Exception as exc:
            score_error = f"{type(exc).__name__}: {exc}"[:400]
        # Ranking aids the local performance proposer; first-round LLM sees only
        # measured evidence until the reviewer supplies explicitly labeled feedback.
        if kwargs["cases"].get("repair_mode") == "performance_recovery":
            pool.sort(key=_rank_key, reverse=True)
        transitions = self._paired_repair_effects(history, kwargs["parent"])
        cases["paired_repair_effects"] = (
            [t for t in transitions if t["quality_feasible"] is True]
            if _strategy_uses_success_reuse(kwargs["strategy"]) else transitions
        )
        review_cases = {**cases, **retrieve_repair_cases(
            history=history, parent=kwargs["parent"], scope=self.scope, space=self.space,
            limit=self.context_limit, repair_round=2 if kwargs["strategy"].startswith("round_2") else 1,
        )}
        evidence = _prompt_evidence(kwargs["parent"], review_cases)
        evidence["quality_gate"] = cases.get("quality_gate") or _quality_gate_context(history)
        evidence["unexecuted_a_review"] = cases.get("skipped_a_review", {})
        evidence["workload_features"] = cases.get("workload_features", {
            "static": history.get("static_workload"),
            "dynamic_at_anchor": _record_dynamic_features(kwargs["parent"]),
        })
        anchor = kwargs["parent"]
        evidence["parent_performance"] = {
            "trial_id": anchor.get("trial_id"),
            "measured": _measured_performance_metrics(anchor),
            "source": "actual_measurement",
        }
        evidence["current_pareto_front"] = [
            {"trial_id": row.get("trial_id"),
             "measured": _measured_performance_metrics(row),
             "objective_vector": list(row["objective_vector"])}
            for row in feasible_pareto_records(history)
        ]
        # Audit all observed runtime failures; retrieval similarity is evidence,
        # not a gate that hides counterexamples from either route.
        for row in history.get("trials", []):
            if is_timeout_record(row) or row.get("repair_evidence_eligible") is False:
                continue
            if row.get("status") == "complete":
                evidence["records"][row["trial_id"]] = {
                    "configuration": row["candidate"], "quality": _measured_quality_metrics(row),
                    "performance": _measured_performance_metrics(row),
                    "measured_status": "complete", "quality_feasible": row.get("quality_feasible"),
                }
            if row.get("status") == "runtime_failed":
                if not any(r.get("trial_id") == row["trial_id"] for r in evidence["runtime_failures"]):
                    evidence["runtime_failures"].append({"trial_id": row["trial_id"], **_runtime_failure_context(row)})
                evidence["records"][row["trial_id"]] = {
                    "configuration": row["candidate"], "quality": {}, "performance": {},
                    "measured_status": "runtime_failed", "quality_feasible": None,
                }

        # Do not make the reviewer infer a multi-metric gate from raw values.
        # These flags are recorded measurements, never surrogate predictions.
        for record in [history.get("baseline_observation"), *history.get("trials", [])]:
            if isinstance(record, Mapping) and str(record.get("trial_id")) in evidence["records"]:
                evidence["records"][str(record["trial_id"])].update({
                    "measured_status": record.get("status"),
                    "quality_feasible": record.get("quality_feasible") if record.get("status") == "complete" else None,
                })
        trace: dict[str, Any] = {
            "repair_stage": repair_stage(kwargs["strategy"]),
            "strategy": kwargs["strategy"], "history_trial_count": len(history.get("trials", [])),
            "max_revisions": self.proposal_max_revisions, "attempts": [],
            "score_error": score_error, "evaluation_kind": "pre_execution_review_not_measurement",
            "prior_repair_reflections": cases.get("prior_repair_reflections", []),
        }
        if self._active_episode is not None:
            self._active_episode.setdefault("proposal_cycles", []).append(trace)

        def persist() -> None:
            if self._active_episode is not None:
                self._persist_episode(self._active_episode)

        persist()
        proposals: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        final: dict[str, Any] | None = None
        final_index = None
        for revision in range(self.proposal_max_revisions + 1):
            def validate_changes(changes: Mapping[str, Any]) -> tuple[Any, list[str]]:
                parent_config = dict(kwargs["parent"]["candidate"])
                original = self.space.bo_to_canonical(parent_config)
                original_executable = {
                    parameter.name: parent_config.get(parameter.name,
                        parent_config.get(parameter.bo_name, original[parameter.name]))
                    for parameter in self.space.parameters
                }
                errors = []
                for field, value in changes.items():
                    if field not in kwargs["scope"].fields:
                        errors.append(f"changes.{field}: 不在允许修改范围 {list(kwargs['scope'].fields)}，该字段冻结。")
                    elif not self.space.by_name[field].contains(value):
                        errors.append(f"changes.{field}={value!r}: 超出参数域 {self.space.by_name[field].domain}。")
                # Some providers return target values alongside the actual delta.
                # Validate every supplied field before dropping harmless no-ops;
                # frozen or out-of-domain fields must still fail validation.
                changes = {field: value for field, value in changes.items()
                           if original_executable.get(field) != value}
                if not changes:
                    errors.append("changes: 没有相对锚点的实际改动。")
                if kwargs["strategy"] == "round_2_performance_recovery" and len(changes) > 2:
                    errors.append("performance_recovery最多修改两个非表示参数。")
                if errors:
                    return None, errors
                proposed = dict(parent_config)
                for field, value in changes.items():
                    parameter = self.space.by_name[field]
                    proposed[field] = value
                    if parameter.bo_name != field or parameter.bo_name in proposed:
                        proposed[parameter.bo_name] = value
                receipt = dict(self.validator(proposed))
                executable = receipt.get("candidate")
                if receipt.get("valid") is not True or not isinstance(executable, Mapping):
                    return None, [str(x) for x in receipt.get("validation_errors", [])] or ["完整配置机械校验失败。"]
                canonical = self.space.bo_to_canonical(executable)
                # Compare the same Range encoding on both sides. The separate
                # frozen-field check enforces executable inactive resets (which
                # can differ from the in-domain encoding placeholders).
                expected = self.space.bo_to_canonical(proposed)
                inactive = self.space.inactive_parameter_values(proposed)
                for field in expected:
                    if canonical.get(field) != expected[field]:
                        errors.append(f"完整配置校验要求 {field}={canonical.get(field)!r}，但提案要求/冻结值为 {expected[field]!r}；请显式修订，不能事后投影。")
                for field in _repair_frozen_field_changes(
                    anchor=parent_config, candidate=executable,
                    fields=kwargs["scope"].fields,
                    derived_fields=kwargs["scope"].derived_fields, space=self.space,
                ):
                    errors.append(f"冻结字段 {field} 被机械校验改变或未归一化到失活参考值。")
                if all(canonical.get(field) == original.get(field)
                       for field in kwargs["scope"].fields):
                    errors.append("changes: 条件参数归一化后没有相对锚点的实际改动。")
                if errors:
                    return None, errors
                executable, guard = _apply_repair_warmup_guard(executable, space=self.space, validator=self.validator)
                values = {field: self.space.bo_to_canonical(executable)[field] for field in kwargs["scope"].fields}
                if _strategy_uses_success_reuse(kwargs["strategy"]) and not any(values == c["values"] for c in kwargs["candidates"]):
                    errors.append("R1经验复用要求可修字段取值匹配已有质量成功案例；当前取值没有对应成功证据。")
                for row in [history.get("baseline_observation") or {}, *history.get("trials", [])]:
                    if row.get("candidate") and _stable_key(row["candidate"]) == _stable_key(executable):
                        errors.append(f"完整配置与已执行 {row.get('trial_id')} 重复；该记录的分支为 {row.get('repair_strategy')}。")
                for row in cases.get("episode_round_1_results", []):
                    if kwargs["strategy"].startswith("B_") and row.get("candidate"):
                        a_values = {field: self.space.bo_to_canonical(row["candidate"])[field] for field in kwargs["scope"].fields}
                        if values == a_values:
                            errors.append(f"可修字段取值与本轮R1（{row.get('trial_id')}）重复：{a_values}；即使显存保护改变其他字段也不能重复R1。")
                if any(values == row[0]["candidate"]["values"] for row in proposals):
                    errors.append("该提案已在本次评审要求修订，请针对评审意见改变参数。")
                if errors:
                    return None, errors
                item = {"candidate": dict(executable), "values": values,
                        "signature": _repair_signature(executable, fields=kwargs["scope"].fields, space=self.space),
                        "validator_receipt": receipt, "repair_warmup_guard": guard,
                        "changed_fields": sorted(changes),
                        "inactive_parameter_resets": {
                            name: {"before": parent_config.get(name), "after": value}
                            for name, value in inactive.items()
                            if name in executable and parent_config.get(name) != value
                        }}
                cached = next((c for c in pool if c["candidate"] == item["candidate"]), None)
                if cached is not None:
                    item = {**cached, **item}
                return item, []

            from .repair_changes import propose_three_changes
            choice = propose_three_changes(
                score=lambda items: score_repair_candidates(
                    candidates=items, history=history, space=self.space,
                    random_seed=self.random_seed + len(history.get("trials", [])),
                    monte_carlo_samples=self.monte_carlo_samples, reference_margin=self.reference_margin,
                    prediction_provider=self.prediction_provider),
                client=self.reasoner.client, strategy=kwargs["strategy"], parent=kwargs["parent"],
                scope=kwargs["scope"], space=self.space, evidence=_prompt_evidence(kwargs["parent"], cases),
                cases={**cases, "parameter_guide": _representation_parameter_guide(self.scope.contract()["domains"]),
                       "repair_warmup_guard": REPAIR_WARMUP_GUARD},
                examples=[c["values"] for c in pool], validate=validate_changes,
            )
            self.reasoner.last_failure = None if choice.get("candidate") else dict(choice)
            item = choice.get("candidate")
            if not isinstance(item, Mapping):
                trace["attempts"].append({"revision": revision, "proposal_status": "invalid", "llm_attempts": choice.get("llm_attempts", []), "proposal_batch": choice.get("proposal_batch"), "mechanical_audit_batches": choice.get("mechanical_audit_batches"), "selection_mode": choice.get("selection_mode"), "prediction_error": choice.get("prediction_error")})
                persist()
                if not proposals:
                    trace.update(selection_reason="no_valid_proposal", execution_status="not_executed")
                    persist()
                    return {**choice, "proposal_review": trace}
                break
            # The reasoner hides prediction fields in first-round selections.
            selected = dict(item)
            if selected.get("predicted_objectives") is None:
                try:
                    selected = score_repair_candidates(
                        candidates=[selected], history=history, space=self.space,
                        random_seed=self.random_seed + len(history.get("trials", [])),
                        monte_carlo_samples=self.monte_carlo_samples, reference_margin=self.reference_margin,
                        prediction_provider=self.prediction_provider,
                    )[0]
                except Exception:
                    pass  # Unknown prediction is not a mechanical rejection.
            choice = {**choice, "candidate": selected}
            performance = {key: selected.get(key) for key in _BO_PREDICTION_FIELDS}
            performance.update({"status": "predicted" if not score_error and selected.get("predicted_objectives") is not None else "unavailable",
                                "coordinates": ["qps", "negative_mean_latency", "negative_p95_latency"],
                                "quality_prediction": None, "prediction_source": "current_native_bo_posterior",
                                "objective_variance": selected.get("objective_variance"),
                                "complete_training_rows": selected.get("complete_training_rows"), "error": score_error})
            entry = {
                "revision": revision, "proposal": selected["candidate"],
                "signature": selected["signature"], "llm_attempts": choice.get("llm_attempts", []),
                "performance_assessment": performance, "review_status": "pending",
                "proposal_batch": choice.get("proposal_batch"), "mechanical_audit_batches": choice.get("mechanical_audit_batches"),
                "selected_batch_index": choice.get("selected_batch_index"),
            }
            trace["attempts"].append(entry)
            persist()
            review = self.reviewer.evaluate(
                proposal={
                    "configuration": selected["candidate"],
                    "parameter_changes": {
                        k: {"before": kwargs["parent"]["candidate"].get(k), "after": selected["candidate"].get(k)}
                        for k in sorted(set(kwargs["parent"]["candidate"]) | set(selected["candidate"]))
                        if kwargs["parent"]["candidate"].get(k) != selected["candidate"].get(k)
                    },
                    "repair_warmup_guard": selected.get("repair_warmup_guard"),
                    "inactive_parameter_resets": selected.get("inactive_parameter_resets", {}),
                },
                evidence=evidence, performance=performance,
                scope={**kwargs["scope"].contract(),
                       "parameter_guide": _representation_parameter_guide(self.scope.contract()["domains"])},
                transitions=transitions,
            )
            entry.update(review_status=review["status"], assessment=review)
            proposals.append((choice, review, len(trace["attempts"]) - 1))
            persist()
            if review["status"] == "unavailable":
                break
            if review["status"] == "reviewed" and review["verdict"] == "accept":
                final, final_index = choice, len(trace["attempts"]) - 1
                trace["selection_reason"] = "accepted" if review["status"] == "reviewed" else "review_unavailable_keep_mechanically_valid_proposal"
                break
            pool = [c for c in pool if c["signature"] != selected["signature"]]
            if not pool:
                break
            cases["proposal_feedback"] = {
                "previous_proposal": selected["candidate"], "assessment": review,
                "performance_prediction_not_measurement": performance,
                "instruction": "根据具体评审意见从剩余合法候选提出不同方案；保持原锚点、scope、R1/R2/R3阶段分工不变。",
            }
        if final is None:
            trace.update(selection_reason="dual_audit_rejected_or_unavailable_skip_execution",
                         selected_attempt=None, selected_signature=None, execution_status="not_executed")
            self.reasoner.last_failure = None
            persist()
            return {"candidate": None, "agent_output": None, "selection_mode": "review_rejected",
                    "skip_execution": True, "llm_attempts": [], "proposal_review": trace}
        trace["selected_attempt"] = final_index
        trace["selected_signature"] = final["candidate"]["signature"]
        self.reasoner.last_failure = None
        persist()
        return {**final, "proposal_review": trace}

    def _persist_episode(self, episode: Mapping[str, Any]) -> None:
        history = self._history()
        episodes = [
            dict(item)
            for item in history.get("repair_episodes", [])
            if isinstance(item, Mapping)
            and str(item.get("episode_id")) != str(episode.get("episode_id"))
        ]
        episodes.append(_json_value(dict(episode)))
        history["repair_episodes"] = episodes
        compact_reflections(history)
        history["repair_scope"] = self.scope.contract()
        write_json_atomic(self.history_path, history)

    def _trigger_outcome(
        self,
        parent_trial_id: str,
        status: str,
        **details: Any,
    ) -> dict[str, Any]:
        outcome = {"status": status, **_json_value(details)}
        history = self._history()
        events = [
            dict(item)
            for item in history.get("repair_trigger_events", [])
            if isinstance(item, Mapping)
            and str(item.get("parent_trial_id")) != str(parent_trial_id)
        ]
        events.append(
            {
                "parent_trial_id": parent_trial_id,
                "evaluation_index": (
                    (_trial_by_id(history, parent_trial_id) or {}).get(
                        "evaluation_index"
                    )
                ),
                **outcome,
            }
        )
        history["repair_trigger_events"] = events
        write_json_atomic(self.history_path, history)
        return outcome

    def _finalize_artifacts(self) -> None:
        history = self._history()
        result = result_from_history(history)
        result["implementation"] = history.get("implementation")
        result["result_path"] = str(self.result_path)
        result["repair_experience_path"] = str(self.output_dir / REPAIR_EXPERIENCE_NAME)
        write_json_atomic(self.result_path, result)
        write_repair_experience(history, self.output_dir / REPAIR_EXPERIENCE_NAME)

    def _finish_episode(self, episode: dict[str, Any]) -> None:
        self._persist_episode(episode)
        if self.reflect_enabled and "reflection" not in episode:
            episode["reflection"] = reflect_episode(self.structured_client, episode, self._history())
            self._persist_episode(episode)
        self._finalize_artifacts()

    def _frozen_front(
        self, history: Mapping[str, Any]
    ) -> tuple[list[list[float]], list[float]]:
        front = [
            list(map(float, item["objective_vector"]))
            for item in feasible_pareto_records(history)
        ]
        return front, _reference_point(history, self.reference_margin).tolist()

    def _result_with_success(
        self,
        trial: Mapping[str, Any],
        *,
        frozen_front: Sequence[Sequence[float]],
        reference: Sequence[float],
    ) -> dict[str, Any]:
        enriched = dict(trial)
        if is_timeout_record(trial):
            return {**enriched, "repair_actual_hv_gain": None, "repair_success": None,
                    "repair_evidence_eligible": False}
        gain = 0.0
        if trial.get("status") == "complete" and trial.get("objective_vector"):
            gain = hypervolume_improvement(
                frozen_front, trial["objective_vector"], reference
            )
        enriched["repair_actual_hv_gain"] = float(gain)
        enriched["repair_success"] = bool(
            trial.get("quality_feasible") is True and gain > self.hv_epsilon
        )
        return enriched

    def _update_trial_result(self, trial_id: str, fields: Mapping[str, Any]) -> None:
        history = self._history()
        updated = False
        for index, item in enumerate(history.get("trials", [])):
            if isinstance(item, Mapping) and str(item.get("trial_id")) == str(trial_id):
                history["trials"][index] = {**dict(item), **_json_value(dict(fields))}
                updated = True
                break
        if not updated:
            raise RuntimeError(f"cannot update missing repair trial: {trial_id}")
        write_json_atomic(self.history_path, history)

    def _choose_first_round_branch(
        self,
        *,
        history: Mapping[str, Any],
        parent: Mapping[str, Any],
        scored: Sequence[Mapping[str, Any]],
        cases: Mapping[str, Any],
        excluded: set[str],
        critique: Mapping[str, Any] | None,
        branch: str,
        a_selected: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.reasoner.last_failure = None
        seen = _seen_signatures(history, self.scope, self.space)
        seen_values = [
            self.scope.values(item["candidate"], self.space)
            for item in [
                history.get("baseline_observation"),
                *history.get("trials", []),
            ]
            if isinstance(item, Mapping)
            and isinstance(item.get("candidate"), Mapping)
        ]
        scored = [
            {
                **dict(item),
                "diversity_from_seen": (
                    min(
                        _edit_distance(item["values"], values, self.scope.fields)
                        for values in seen_values
                    )
                    if seen_values
                    else len(self.scope.fields)
                ),
            }
            for item in scored
        ]
        successful = {
            str(item["repair_signature"])
            for item in cases.get("quality_success_cases", [])
        }
        successful.update(
            str(item["signature"]) for item in scored
            if _historical_success_matches(item["values"], cases)
        )
        parent_values = self.scope.values(parent["candidate"], self.space)
        boundary_cases = [
            (min(margins), item)
            for key in ("quality_success_cases", "quality_failure_cases")
            for item in cases.get(key, [])
            if isinstance(item, Mapping)
            and isinstance(item.get("repair_values"), Mapping)
            if (margins := [
                float(metric["normalized_margin"])
                for metric in item.get("measured_quality_metrics", [])
                if isinstance(metric.get("normalized_margin"), (int, float))
            ])
        ]
        boundary_anchor_values = (
            dict(
                min(
                    boundary_cases,
                    key=lambda item: abs(item[0]),
                )[1]["repair_values"]
            )
            if boundary_cases
            else parent_values
        )
        a_pool, a_mode, b_pool, b_mode = _strategy_pools(
            scored=scored,
            seen_signatures=seen,
            successful_signatures=successful,
            boundary_anchor_values=boundary_anchor_values,
            scope=self.scope,
            excluded_signatures=excluded,
        )
        evidence_parent = _first_round_parent_context(parent)
        if branch == "A":
            if not a_pool or not b_pool:
                return []
            a_uses_success_reuse = a_mode == "evidence_reuse"
            a = self._choose_with_review(
                strategy=("A_evidence_reuse" if a_uses_success_reuse else "A_to_B_novel_synthesis"),
                parent=evidence_parent,
                candidates=a_pool,
                cases=_first_round_case_context(cases, success_reuse=a_uses_success_reuse),
                scope=self.scope,
                critique=critique,
                use_bo_predictions=False,
            )
            return [{**a, "strategy": "A", "strategy_mode": a_mode}] if isinstance(a.get("candidate"), Mapping) or a.get("skip_execution") else []
        if branch != "B" or (a_selected is None and not cases.get("skipped_a_review")):
            raise ValueError("stage R2 requires measured R1 or its unexecuted review trace")
        if a_selected is not None:
            a_signature = str(a_selected["signature"])
            b_pool = [item for item in b_pool if str(item["signature"]) != a_signature]
        if not b_pool:
            return []
        b = self._choose_with_review(
            strategy="B_evidence_tradeoff",
            parent=evidence_parent,
            candidates=b_pool,
            cases=_first_round_case_context(
                cases, success_reuse=False
            ),
            scope=self.scope,
            critique=critique,
            use_bo_predictions=False,
        )
        if not isinstance(b.get("candidate"), Mapping):
            return []
        return [
            {**b, "strategy": "B", "strategy_mode": b_mode},
        ]

    def maybe_repair(self, parent_trial_id: str) -> dict[str, Any]:
        history = self._history()
        parent = _trial_by_id(history, parent_trial_id)
        if parent is None:
            raise ValueError(f"repair parent trial not found: {parent_trial_id}")
        existing = [
            item
            for item in history.get("repair_episodes", [])
            if isinstance(item, Mapping)
            and str(item.get("parent_trial_id")) == str(parent_trial_id)
        ]
        if existing:
            if self.reflect_enabled and existing[-1].get("status") != "running" and "reflection" not in existing[-1]:
                self._finish_episode(existing[-1])
            return {"status": "already_processed", "episode": dict(existing[-1])}

        trigger_reasons: list[str] = []
        if parent.get("trial_role") != "native_bo":
            trigger_reasons.append("parent_is_not_native_bo")
        if int(parent.get("evaluation_index") or 0) <= self.cold_start_evaluations:
            trigger_reasons.append("cold_start_not_finished")
        if parent.get("status") != "complete":
            trigger_reasons.append("runtime_or_measurement_failure")
        elif parent.get("quality_feasible") is not False:
            trigger_reasons.append("quality_did_not_fail")
        if trigger_reasons:
            return self._trigger_outcome(
                parent_trial_id, "not_triggered", reasons=trigger_reasons
            )

        fibre, rejected = self.scope.materialize_fibre(
            parent=parent["candidate"], space=self.space, validator=self.validator
        )
        fibre = self._guard_fibre(fibre)
        executed_keys = _executed_candidate_keys(history)
        available = [
            item
            for item in fibre
            if _stable_key(item["candidate"]) not in executed_keys
        ]
        if len(available) < 2:
            return self._trigger_outcome(
                parent_trial_id,
                "candidate_exhausted",
                legal_candidate_count=len(available),
                rejected_candidates=rejected,
            )
        if self.evaluation_budget - len(history.get("trials", [])) < 2:
            return self._trigger_outcome(
                parent_trial_id, "budget_truncated", stage="before_round_1"
            )

        frozen_front, reference = self._frozen_front(history)
        performance_hvi = hypervolume_improvement(
            frozen_front, parent["objective_vector"], reference
        )
        # Initial R1/R2 proposals see measured evidence only; the subsequent
        # review adds labeled performance predictions. Trigger priority uses
        # measured parent performance; quality remains a measured hard gate.
        first_round_candidates = [dict(item) for item in available]
        boundary_utility = float(performance_hvi)
        if performance_hvi <= self.hv_epsilon:
            return self._trigger_outcome(
                parent_trial_id,
                "not_triggered",
                reasons=["non_positive_potential_performance_hv_gain"],
                performance_hv_gain=float(performance_hvi),
                boundary_utility=boundary_utility,
            )
        if boundary_utility <= self.boundary_utility_epsilon:
            return self._trigger_outcome(
                parent_trial_id,
                "not_triggered",
                reasons=["boundary_utility_below_epsilon"],
                performance_hv_gain=float(performance_hvi),
                boundary_utility=boundary_utility,
            )

        adaptive = (assess_trigger(history, parent_trial_id, frozen_front, reference,
                                   float(performance_hvi), self.hv_epsilon)
                    if self.adaptive_threshold_enabled else {'enabled': False})
        if self.adaptive_threshold_enabled and not adaptive['passed']:
            return self._trigger_outcome(
                parent_trial_id, 'not_triggered',
                reasons=['relative_hv_gain_below_adaptive_threshold'],
                performance_hv_gain=float(performance_hvi),
                boundary_utility=boundary_utility, adaptive_threshold=adaptive)

        cases = retrieve_repair_cases(
            history=history,
            parent=parent,
            scope=self.scope,
            space=self.space,
            limit=self.context_limit,
        )
        cases["quality_gate"] = _quality_gate_context(history)
        cases["performance_context"] = _performance_context(
            history,
            anchor=parent,
            frozen_front=frozen_front,
            reference=reference,
        )
        cases["full_search_space"] = _complete_range_contract(
            self.space, self.scope.runtime_constraints
        )
        cases["workload_features"] = {
            "static": history.get("static_workload"),
            "dynamic_at_parent": _record_dynamic_features(parent),
        }
        if self.reflect_enabled:
            cases["prior_repair_reflections"] = prior_reflections(history, "R1")
        reasoning_parent = {
            **parent,
            "static_workload": history.get("static_workload"),
            "dynamic_workload": _record_dynamic_features(parent),
        }
        episode_id = f"repair_{parent_trial_id}"
        episode: dict[str, Any] = {
            "schema_version": REPAIR_EPISODE_SCHEMA,
            "episode_id": episode_id,
            "parent_trial_id": parent_trial_id,
            "status": "running",
            "round_1_execution_order": (
                "review_A_execute_or_skip_A_then_review_and_execute_B" if self.proposal_review_enabled
                else "recommend_A_execute_A_recommend_B_execute_B"
            ),
            "trigger": {
                "performance_hv_gain": float(performance_hvi),
                "boundary_utility": boundary_utility,
                "boundary_utility_formula": "performance_hv_gain_only",
                "adaptive_threshold": adaptive,
                "quality_decision": "measured_hard_thresholds_only",
                "hv_epsilon": self.hv_epsilon,
                "frozen_feasible_pareto_front": frozen_front,
                "frozen_reference_point": reference,
            },
            "repair_scope": self.scope.contract(),
            "parent_signature": self.scope.signature(parent["candidate"], self.space),
            "parent_quality_margin": parent.get("quality_margin"),
            "parent_quality_margins": parent.get("quality_margins") or [],
            "workload_snapshot": {
                "static": history.get("static_workload"),
                "dynamic": _record_dynamic_features(parent),
            },
            "case_snapshot": cases,
            "legal_candidate_count": len(fibre),
            "available_candidate_count": len(available),
            "compatibility_rejections": rejected,
            "rounds": [],
        }
        self._active_episode = episode
        episode["proposal_review_policy"] = {"enabled": self.proposal_review_enabled, "max_revisions_per_branch": self.proposal_max_revisions, "review_is_measurement": False,
                                             "audit_routes": ["runtime", "performance"],
                                             "all_branches_rejected_after_revision": "skip_execution",
                                             "prediction_source": "current_native_bo_posterior",
                                             "stage_order": ["R1", "R2", "R3"],
                                             "legacy_strategy_stage_names": REPAIR_STAGE_NAMES,
                                             "a_rejected_after_revision": "skip_execution_and_forward_to_b",
                                             "a_review_unavailable": "retry_once_then_skip_and_forward_to_b",
                                             "a_measured_quality_and_pareto_success": "skip_b_and_round_2_return_to_bo",
                                             "b_proposal_failure": "continue_with_measured_a_or_end_if_no_measurement"}
        self._persist_episode(episode)
        self._trigger_outcome(
            parent_trial_id,
            "triggered",
            episode_id=episode_id,
            adaptive_threshold=adaptive,
            performance_hv_gain=float(performance_hvi),
            boundary_utility=boundary_utility,
        )

        excluded: set[str] = set()
        critique: dict[str, Any] | None = None
        for repair_round in (1, 2):
            history = self._history()
            execution_count = 2 if repair_round == 1 else 1
            if (
                self.evaluation_budget - len(history.get("trials", []))
                < execution_count
            ):
                episode["status"] = "budget_truncated"
                episode["budget_truncated_at_round"] = repair_round
                self._finish_episode(episode)
                return {"status": episode["status"], "episode": episode}

            if repair_round == 1:
                scored_available = [
                    item
                    for item in first_round_candidates
                    if str(item["signature"]) not in excluded
                ]
                decisions = self._choose_first_round_branch(
                    history=history,
                    parent=reasoning_parent,
                    scored=scored_available,
                    cases=cases,
                    excluded=excluded,
                    critique=None,
                    branch="A",
                )
                round_scope: RepairScope | RepairSelectionScope = self.scope
                repair_mode = "representation_repair"
                repair_anchor = parent
                round_rejections: list[dict[str, Any]] = []
            else:
                round_1_results = [
                    dict(item)
                    for item in episode["rounds"][0].get("results", [])
                    if isinstance(item, Mapping) and not is_timeout_record(item)
                ]
                if not round_1_results:
                    episode["status"] = "no_measured_repair_candidate"
                    episode["terminal_outcome"] = "all_first_round_results_timed_out"
                    self._finish_episode(episode)
                    return {"status": episode["status"], "episode": episode}
                unresolved = [
                    item for item in round_1_results
                    if item.get("repair_success") is not True
                ]
                repair_mode = _round_2_mode(unresolved)
                pinned_round_1 = [
                    _compact_trial_context(item) for item in round_1_results
                ]
                if repair_mode == "performance_recovery":
                    repair_anchor = _performance_recovery_anchor(
                        unresolved,
                        frozen_front=frozen_front,
                        reference=reference,
                    )
                else:
                    # Target an unsuccessful branch, never its successful sibling.
                    # With two failures, prefer measured quality evidence over an
                    # unmeasured runtime failure, then the smaller quality deficit.
                    repair_anchor = max(
                        unresolved,
                        key=lambda item: (
                            item.get("status") == "complete",
                            float(item["quality_margin"])
                            if isinstance(item.get("quality_margin"), (int, float))
                            else -math.inf,
                            str(item.get("trial_id") or ""),
                        ),
                    )
                cases = retrieve_repair_cases(
                    history=history,
                    parent=repair_anchor,
                    scope=self.scope,
                    space=self.space,
                    limit=self.context_limit,
                    repair_round=2,
                )
                cases["performance_context"] = _performance_context(
                    history,
                    anchor=repair_anchor,
                    frozen_front=frozen_front,
                    reference=reference,
                )
                cases["quality_gate"] = _quality_gate_context(history)
                cases["full_search_space"] = _complete_range_contract(
                    self.space, self.scope.runtime_constraints
                )
                cases["repair_mode"] = repair_mode
                cases["repair_target"] = {
                    "trial_id": repair_anchor.get("trial_id"),
                    "strategy": repair_anchor.get("repair_strategy"),
                    "rule": "repair this unsuccessful branch; do not retune its successful sibling",
                }
                cases["auditor_critique"] = critique
                cases["episode_round_1_results"] = pinned_round_1
                cases["skipped_a_review"] = episode.get("round_1_a_skipped", {})
                cases["workload_features"] = {
                    "static": history.get("static_workload"),
                    "dynamic_at_anchor": _record_dynamic_features(repair_anchor),
                }
                if self.reflect_enabled:
                    cases["prior_repair_reflections"] = prior_reflections(history, "R3")
                episode["round_2_mode"] = repair_mode
                episode["round_2_anchor_trial_id"] = repair_anchor.get("trial_id")
                episode["round_2_target_strategy"] = repair_anchor.get("repair_strategy")
                episode["round_2_case_snapshot"] = cases
                reasoning_anchor = {
                    **repair_anchor,
                    "static_workload": history.get("static_workload"),
                    "dynamic_workload": _record_dynamic_features(repair_anchor),
                }
                if repair_mode == "quality_repair":
                    round_scope = self.scope
                    executed_keys = _executed_candidate_keys(history)
                    branch_fibre, branch_rejections = self.scope.materialize_fibre(
                        parent=repair_anchor["candidate"],
                        space=self.space,
                        validator=self.validator,
                    )
                    branch_fibre = self._guard_fibre(branch_fibre)
                    remaining = [
                        item
                        for item in branch_fibre
                        if str(item["signature"]) not in excluded
                        and _stable_key(item["candidate"]) not in executed_keys
                    ]
                    chosen = (
                        self._choose_with_review(
                            strategy="round_2_quality_repair",
                            parent=reasoning_anchor,
                            candidates=remaining,
                            cases=cases,
                            scope=round_scope,
                            critique=critique,
                            use_bo_predictions=False,
                        )
                        if remaining
                        else {
                            "candidate": None,
                            "agent_output": None,
                            "selection_mode": "candidate_exhausted",
                            "llm_attempts": [],
                        }
                    )
                    episode["round_2_candidate_rejections"] = branch_rejections
                elif self.proposal_review_enabled:
                    local_pool, round_scope, local_rejections = generate_performance_recovery_candidates(
                        history=history, anchor=repair_anchor, space=self.space,
                        representation_fields=self.scope.fields, runtime_constraints=self.scope.runtime_constraints,
                        validator=self.validator, random_seed=self.random_seed + len(history.get("trials", [])),
                        pool_size=self.performance_candidate_pool_size, local_only=True,
                    )
                    local_pool = [c for c in self._guard_fibre(local_pool)
                                  if _stable_key(c["candidate"]) not in _executed_candidate_keys(history)]
                    for item in local_pool:
                        item["values"] = {field: self.space.bo_to_canonical(item["candidate"])[field] for field in round_scope.fields}
                        item["signature"] = _repair_signature(item["candidate"], fields=round_scope.fields, space=self.space)
                    chosen = self._choose_with_review(
                        strategy="round_2_performance_recovery", parent=reasoning_anchor,
                        candidates=local_pool, cases=cases, scope=round_scope, critique=critique,
                        use_bo_predictions=True,
                    ) if local_pool else {"candidate": None, "agent_output": None, "selection_mode": "candidate_exhausted", "llm_attempts": []}
                    episode["round_2_candidate_rejections"] = local_rejections
                    episode["round_2_candidate_generation"] = "local_non_representation_max_two_fields_with_performance_scoring"
                else:
                    round_scope = RepairSelectionScope.from_space(
                        self.space,
                        fields=[
                            parameter.name
                            for parameter in self.space.tuned_parameters
                        ],
                        scope_version=FULL_CONFIGURATION_SCOPE_VERSION,
                        runtime_constraints=self.scope.runtime_constraints,
                        candidate_generation=(
                            "llm_direct_complete_range_configuration"
                        ),
                    )
                    chosen = self.reasoner.recommend_complete_configuration(
                        strategy="round_2_performance_recovery",
                        parent=reasoning_anchor,
                        cases=cases,
                        space=self.space,
                        runtime_constraints=self.scope.runtime_constraints,
                        validator=self.validator,
                        executed_candidate_keys=_executed_candidate_keys(history),
                    )
                    episode["round_2_candidate_rejections"] = []
                episode["round_2_scope"] = round_scope.contract()
                episode["round_2_llm_recommendation"] = {
                    "selection_mode": chosen.get("selection_mode"),
                    "llm_attempts": chosen.get("llm_attempts", []),
                    "agent_output": chosen.get("agent_output"),
                }
                if isinstance(chosen.get("candidate"), Mapping):
                    decisions = [
                        {
                            **chosen,
                            "strategy": "R2",
                            "strategy_mode": repair_mode,
                        }
                    ]
                else:
                    decisions = []

            if len(decisions) != 1:
                if repair_round == 2:
                    episode["status"] = "llm_recommendation_failed"
                    episode["llm_recommendation_failed_at_round"] = repair_round
                elif self.reasoner.last_failure is not None:
                    episode["status"] = "llm_recommendation_failed"
                    episode["llm_recommendation_failed_at_round"] = repair_round
                    episode["round_1_llm_failure"] = dict(
                        self.reasoner.last_failure
                    )
                else:
                    episode["status"] = "candidate_exhausted"
                    episode["candidate_exhausted_at_round"] = repair_round
                self._finish_episode(episode)
                return {"status": episode["status"], "episode": episode}

            # Persist and execute R1 before generating R2 from refreshed evidence.
            round_record: dict[str, Any] = {
                "repair_round": repair_round,
                "repair_mode": repair_mode,
                "repair_anchor_trial_id": repair_anchor.get("trial_id"),
                "repair_scope": round_scope.contract(),
                "history_trial_count_at_generation": len(history.get("trials", [])),
                "critique": critique,
                "decisions": [],
                "results": [],
                "skipped_branches": [],
            }
            episode["rounds"].append(round_record)
            for branch_index in range(execution_count):
                if repair_round == 1 and branch_index == 1:
                    history = self._history()
                    cases = {
                        **cases,
                        **retrieve_repair_cases(
                            history=history, parent=parent, scope=self.scope,
                            space=self.space, limit=self.context_limit, repair_round=1,
                        ),
                        "episode_round_1_results": [
                            _compact_trial_context(item) for item in round_record["results"]
                            if not is_timeout_record(item)
                        ],
                        "skipped_a_review": episode.get("round_1_a_skipped", {}),
                    }
                    episode["round_1_b_case_snapshot"] = _json_value(cases)
                    self._persist_episode(episode)
                    b_decisions = self._choose_first_round_branch(
                        history=history, parent=reasoning_parent,
                        scored=first_round_candidates, cases=cases, excluded=excluded,
                        critique=None, branch="B", a_selected=decisions[0]["candidate"],
                    )
                    if not b_decisions:
                        episode["round_1_b_selection_failure"] = self.reasoner.last_failure or {"reason": "candidate_exhausted"}
                        round_record["skipped_branches"].append({
                            "strategy": "B", "status": "not_executed",
                            "reason": "no_valid_b_proposal", "is_measurement": False,
                            "selection_failure": episode["round_1_b_selection_failure"],
                        })
                        self._persist_episode(episode)
                        # Retain A's measured evidence and allow the auditor/R2
                        # to recover instead of aborting the whole episode.
                        break
                    decisions.extend(b_decisions)
                decision = decisions[branch_index]
                if decision.get("skip_execution"):
                    skipped = {
                        "strategy": decision["strategy"], "status": "not_executed",
                        "reason": decision["proposal_review"]["selection_reason"],
                        "is_measurement": False, "proposal_review": decision["proposal_review"],
                    }
                    round_record["skipped_branches"].append(skipped)
                    episode["round_1_a_skipped"] = skipped
                    self._persist_episode(episode)
                    continue
                selected = dict(decision["candidate"])
                signature = str(selected["signature"])
                round_record["decisions"].append(
                    {
                        "strategy": decision["strategy"],
                        "strategy_mode": decision["strategy_mode"],
                        "repair_mode": repair_mode,
                        "repair_anchor_trial_id": repair_anchor.get("trial_id"),
                        "repair_signature": signature,
                        "repair_values": selected["values"],
                        "repair_warmup_guard": selected.get("repair_warmup_guard"),
                        "proposal_review": decision.get("proposal_review"),
                        "history_trial_count_at_generation": len(history.get("trials", [])),
                        "changed_fields": selected.get("changed_fields"),
                        "inactive_parameter_resets": selected.get("inactive_parameter_resets", {}),
                        "predictions": {
                            key: selected.get(key)
                            for key in (
                                "predicted_objectives",
                                "objective_uncertainty",
                                "objective_variance",
                                "prediction_source",
                                "complete_training_rows",
                                "ehvi",
                                "p_run",
                                "score",
                            )
                        },
                        "agent_output": decision["agent_output"],
                        "selection_mode": decision["selection_mode"],
                        "llm_attempts": decision["llm_attempts"],
                    }
                )
                excluded.add(signature)
                self._persist_episode(episode)
                metadata = {
                    "trial_role": "repair",
                    "episode_id": episode_id,
                    "parent_trial_id": parent_trial_id,
                    "repair_anchor_trial_id": repair_anchor.get("trial_id"),
                    "repair_strategy": decision["strategy"],
                    "repair_stage": repair_stage(decision["strategy"]),
                    "repair_strategy_mode": decision["strategy_mode"],
                    "repair_mode": repair_mode,
                    "repair_round": repair_round,
                    "repair_signature": selected["signature"],
                    "repair_scope_version": round_scope.scope_version,
                    "proposal_review": decision.get("proposal_review"),
                    "repair_allowed_fields": sorted(
                        {
                            alias
                            for field in round_scope.fields
                            for alias in (
                                field,
                                self.space.by_name[field].bo_name,
                            )
                        }
                    ),
                    "workload_snapshot": {
                        "dynamic": _record_dynamic_features(repair_anchor),
                    },
                    "repair_prediction": {
                        key: selected.get(key)
                        for key in (
                            "predicted_objectives",
                            "objective_uncertainty",
                            "objective_variance",
                            "prediction_source",
                            "complete_training_rows",
                            "ehvi",
                            "p_run",
                            "score",
                        )
                    },
                }
                guard = selected.get("repair_warmup_guard")
                if guard:
                    metadata["repair_warmup_guard"] = guard
                    if guard.get("applied"):
                        metadata["repair_allowed_fields"] = sorted(
                            set(metadata["repair_allowed_fields"]) | set(guard["changed_fields"])
                        )
                execution = dict(
                    self.execute_candidate(selected["candidate"], metadata)
                )
                trial = execution.get("trial")
                if not isinstance(trial, Mapping):
                    raise RuntimeError("repair execution did not return a Trial record")
                enriched = self._result_with_success(
                    trial, frozen_front=frozen_front, reference=reference
                )
                round_record["results"].append(enriched)
                if enriched["repair_success"]:
                    episode.setdefault("successful_trial_ids", []).append(
                        enriched["trial_id"]
                    )
                self._update_trial_result(
                    str(enriched["trial_id"]),
                    {
                        "repair_actual_hv_gain": enriched["repair_actual_hv_gain"],
                        "repair_success": enriched["repair_success"],
                        **({"repair_evidence_eligible": False} if is_timeout_record(enriched) else {}),
                    },
                )
                self._persist_episode(episode)

                if repair_round == 1 and decision["strategy"] == "A" and enriched["repair_success"]:
                    # Stop before generating B: only actual quality-feasible
                    # improvement against the episode's frozen front qualifies.
                    round_record["skipped_branches"].append({
                        "strategy": "B", "status": "not_executed",
                        "reason": "a_quality_feasible_pareto_improved",
                        "is_measurement": False,
                        "successful_a_trial_id": enriched["trial_id"],
                    })
                    episode["status"] = "succeeded"
                    episode["terminal_outcome"] = "a_quality_feasible_pareto_improved"
                    self._finish_episode(episode)
                    return {"status": episode["status"], "episode": episode}

            round_succeeded = any(
                item.get("repair_success") is True
                for item in round_record["results"]
            )
            if round_succeeded:
                episode["status"] = "succeeded"
                self._finish_episode(episode)
                return {"status": episode["status"], "episode": episode}
            if repair_round == 1:
                if not round_record["results"]:
                    # Both branches may be unexecuted. There is no measured
                    # anchor for R2, and neither proposal is a failed Trial.
                    episode["status"] = "no_measured_repair_candidate"
                    episode["terminal_outcome"] = "all_first_round_branches_unexecuted"
                    self._finish_episode(episode)
                    return {"status": episode["status"], "episode": episode}
                if all(is_timeout_record(row) for row in round_record["results"]):
                    episode["status"] = "no_measured_repair_candidate"
                    episode["terminal_outcome"] = "all_first_round_results_timed_out"
                    self._finish_episode(episode)
                    return {"status": episode["status"], "episode": episode}
                critique = self.auditor.audit(
                    parent=reasoning_parent,
                    decisions=round_record["decisions"],
                    results=[row for row in round_record["results"] if not is_timeout_record(row)],
                    runtime_failure_cases=cases.get("runtime_failure_cases", []),
                    skipped_branches=round_record["skipped_branches"],
                )
                episode["auditor_critique"] = critique

        all_results = [
            item
            for round_record in episode.get("rounds", [])
            if isinstance(round_record, Mapping)
            for item in round_record.get("results", [])
            if isinstance(item, Mapping) and not is_timeout_record(item)
        ]
        episode["status"] = (
            "partially_succeeded" if episode.get("successful_trial_ids")
            else "failed_after_round_2"
        )
        episode["terminal_outcome"] = (
            "one_branch_succeeded"
            if episode.get("successful_trial_ids")
            else "quality_recovered_only"
            if any(item.get("quality_feasible") is True for item in all_results)
            else "quality_not_recovered"
        )
        self._finish_episode(episode)
        return {"status": episode["status"], "episode": episode}


def write_repair_experience(
    history: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Write an auditable association-only summary of measured repairs."""

    experiences: list[dict[str, Any]] = []
    scope_payload = history.get("repair_scope")
    scope_payload = scope_payload if isinstance(scope_payload, Mapping) else {}
    scope_fields = {str(item) for item in scope_payload.get("fields", [])}
    derived_fields = {str(item) for item in scope_payload.get("derived_fields", [])}
    for episode in history.get("repair_episodes", []):
        if not isinstance(episode, Mapping):
            continue
        parent = _trial_by_id(history, str(episode.get("parent_trial_id") or ""))
        observed: list[dict[str, Any]] = []
        parent_candidate = (
            dict(parent.get("candidate") or {}) if parent is not None else {}
        )
        for round_record in episode.get("rounds", []):
            if not isinstance(round_record, Mapping):
                continue
            for result in round_record.get("results", []):
                if not isinstance(result, Mapping) or is_timeout_record(result):
                    continue
                anchor = _trial_by_id(
                    history,
                    str(
                        result.get("repair_anchor_trial_id")
                        or episode.get("parent_trial_id")
                        or ""
                    ),
                )
                comparison = anchor or parent
                comparison_candidate = (
                    dict(comparison.get("candidate") or {})
                    if comparison is not None
                    else parent_candidate
                )
                comparison_margin_by_metric = {
                    str(item.get("metric")): item
                    for item in (
                        comparison.get("quality_margins", [])
                        if comparison is not None
                        else episode.get("parent_quality_margins", [])
                    )
                    if isinstance(item, Mapping) and item.get("metric")
                }
                result_scope_fields = {
                    str(item)
                    for item in (
                        result.get("repair_allowed_fields") or scope_fields
                    )
                }
                result_candidate = dict(result.get("candidate") or {})
                result_margin_by_metric = {
                    str(item.get("metric")): item
                    for item in result.get("quality_margins", [])
                    if isinstance(item, Mapping) and item.get("metric")
                }
                quality_changes = []
                for metric in sorted(
                    set(comparison_margin_by_metric) | set(result_margin_by_metric)
                ):
                    before = comparison_margin_by_metric.get(metric, {})
                    after = result_margin_by_metric.get(metric, {})
                    before_margin = before.get("normalized_margin")
                    after_margin = after.get("normalized_margin")
                    delta = (
                        float(after_margin) - float(before_margin)
                        if isinstance(before_margin, (int, float))
                        and isinstance(after_margin, (int, float))
                        else None
                    )
                    quality_changes.append(
                        {
                            "metric": metric,
                            "before": dict(before),
                            "after": dict(after),
                            "margin_delta": delta,
                            "observed_direction": (
                                "improved"
                                if delta is not None and delta > 0.0
                                else "worsened"
                                if delta is not None and delta < 0.0
                                else "unchanged_or_unavailable"
                            ),
                        }
                    )
                parent_objectives = (
                    comparison.get("objective_vector")
                    if comparison is not None
                    else None
                )
                result_objectives = result.get("objective_vector")
                performance_delta = (
                    [
                        float(after) - float(before)
                        for before, after in zip(
                            parent_objectives, result_objectives, strict=True
                        )
                    ]
                    if isinstance(parent_objectives, Sequence)
                    and isinstance(result_objectives, Sequence)
                    and len(parent_objectives) == len(result_objectives) == 3
                    else None
                )
                decision = next(
                    (
                        item
                        for item in round_record.get("decisions", [])
                        if isinstance(item, Mapping)
                        and item.get("repair_signature")
                        == result.get("repair_signature")
                    ),
                    {},
                )
                agent_output = decision.get("agent_output")
                agent_output = (
                    agent_output if isinstance(agent_output, Mapping) else {}
                )
                observed.append(
                    {
                        "trial_id": result.get("trial_id"),
                        "status": result.get("status"),
                        "failure_reason": (
                            _runtime_failure_context(result)["failure_reason"]
                            if result.get("status") == "runtime_failed" else None
                        ),
                        "repair_signature": result.get("repair_signature"),
                        "repair_strategy": result.get("repair_strategy"),
                        "repair_stage": result.get("repair_stage") or repair_stage(result.get("repair_strategy")),
                        "repair_mode": result.get("repair_mode"),
                        "repair_round": result.get("repair_round"),
                        "repair_anchor_trial_id": result.get(
                            "repair_anchor_trial_id"
                        ),
                        "repair_allowed_fields": sorted(result_scope_fields),
                        "repair_warmup_guard": result.get("repair_warmup_guard"),
                        "proposal_review": result.get("proposal_review"),
                        "quality_margin": result.get("quality_margin"),
                        "quality_margins": result.get("quality_margins") or [],
                        "quality_metric_changes": quality_changes,
                        "objective_vector": result.get("objective_vector"),
                        "performance_delta_in_maximization_coordinates": performance_delta,
                        "actual_hv_gain": result.get("repair_actual_hv_gain"),
                        "repair_success": result.get("repair_success"),
                        "workload_snapshot": result.get("workload_snapshot"),
                        "original_repair_combination": {
                            field: comparison_candidate.get(field)
                            for field in result_scope_fields
                        },
                        "repaired_combination": {
                            field: result_candidate.get(field)
                            for field in result_scope_fields
                        },
                        "cited_trial_ids": list(
                            agent_output.get("cited_trial_ids") or []
                        ),
                        "agent_reported_confidence": agent_output.get("confidence"),
                    }
                )
        cited_trial_ids = sorted(
            {
                str(trial_id)
                for item in observed
                for trial_id in item.get("cited_trial_ids", [])
                if str(trial_id)
            }
        )
        experiences.append(
            {
                "episode_id": episode.get("episode_id"),
                "status": episode.get("status"),
                "parent_trial_id": episode.get("parent_trial_id"),
                "parent_signature": episode.get("parent_signature"),
                "parent_quality_margin": episode.get("parent_quality_margin"),
                "parent_quality_margins": episode.get("parent_quality_margins") or [],
                "parent_objective_vector": (
                    parent.get("objective_vector") if parent else None
                ),
                "workload_static_conditions": history.get("static_workload"),
                "workload_contract": history.get("workload_contract"),
                "episode_workload_snapshot": episode.get("workload_snapshot"),
                "applicability_boundary": {
                    "repair_scope_fields": sorted(scope_fields),
                    "round_1_scope_fields": sorted(scope_fields),
                    "round_2_scope": episode.get("round_2_scope"),
                    "frozen_z": {
                        name: value
                        for name, value in parent_candidate.items()
                        if name not in scope_fields and name not in derived_fields
                    },
                    "scope_version": scope_payload.get("scope_version"),
                    "cross_run_generalization_validated": False,
                },
                "observed_repairs": observed,
                "reflection": episode.get("reflection"),
                "unexecuted_branch_reviews": [
                    skipped for rr in episode.get("rounds", [])
                    for skipped in rr.get("skipped_branches", [])
                ],
                "supporting_trial_ids": [
                    item["trial_id"] for item in observed if item.get("trial_id")
                ]
                + cited_trial_ids,
                "counterexample_trial_ids": [
                    item["trial_id"]
                    for item in observed
                    if item.get("trial_id") and item.get("repair_success") is not True
                ],
                "claim_scope": (
                    "Observed within this run only; association, not a universal causal rule."
                ),
                "confidence": {
                    "kind": "measured_episode_evidence",
                    "strict_success_count": sum(
                        item.get("repair_success") is True for item in observed
                    ),
                    "measured_repair_count": len(observed),
                    "agent_reported_values_are_advisory": True,
                },
            }
        )
    payload = {
        "schema_version": REPAIR_EXPERIENCE_SCHEMA,
        "repair_scope": history.get("repair_scope"),
        "auto_injected_across_runs": False,
        "experience_count": len(experiences),
        "experiences": experiences,
    }
    write_json_atomic(output_path, payload)
    return payload


__all__ = [
    "REPAIR_EPISODE_SCHEMA",
    "REPAIR_EXPERIENCE_SCHEMA",
    "REPAIR_EXPERIENCE_NAME",
    "REPAIR_BO_V1_FIELDS",
    "PERFORMANCE_RECOVERY_SCOPE_VERSION",
    "RepairAuditor",
    "RepairCoordinator",
    "RepairReasoner",
    "RepairSelectionScope",
    "RepairScope",
    "generate_performance_recovery_candidates",
    "dynamic_workload_similarity",
    "grouped_configuration_similarity",
    "retrieve_repair_cases",
    "score_repair_candidates",
    "repair_scope_runtime_constraints",
    "write_repair_experience",
]
