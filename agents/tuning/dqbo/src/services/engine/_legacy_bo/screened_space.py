"""把参数终筛结果转换为可执行的 BO 搜索空间。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ..flock.config import canonicalize_kv_cache_dtype, canonicalize_model_dtype


# 参数筛选使用 vLLM 官方名称；旧 BO 内部把 tensor_parallel_size 简写为 tp。
SCREENING_NAME_TO_INTERNAL = {"tensor_parallel_size": "tp", "chunked_prefill": "enable_chunked_prefill"}
INTERNAL_NAME_TO_SCREENING = {value: key for key, value in SCREENING_NAME_TO_INTERNAL.items()}

SUPPORTED_SCREENED_PARAMETERS = frozenset(
    {
        "cacheblend_mode", "lmcache_chunk_size", "cacheblend_blend_min_tokens",
        "cacheblend_allow_prefix_caching", "distributed_executor_backend",
        "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios",
        "disable_custom_all_reduce", "enable_chunked_prefill",
        "dtype",
        "attention_backend",
        "quantization",
        "model_impl",
        "tp",
        "decode_context_parallel_size",
        "cp_kv_cache_interleave_size",
        "data_parallel_size",
        "data_parallel_backend",
        "disable_nccl_for_dp_synchronization",
        "block_size",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "max_model_len",
        "enable_prefix_caching",
        "prefix_caching_hash_algo",
        "max_num_batched_tokens",
        "max_num_seqs",
        "long_prefill_token_threshold",
        "scheduler_reserve_full_isl",
        "async_scheduling",
        "enforce_eager",
        "disable_cascade_attn",
        "enable_flashinfer_autotune",
        "speculative_method",
        "num_speculative_tokens",
    }
)

BOOLEAN_PARAMETERS = frozenset(
    {
        "cacheblend_allow_prefix_caching",
        "disable_custom_all_reduce", "enable_chunked_prefill",
        "disable_nccl_for_dp_synchronization",
        "enable_prefix_caching",
        "scheduler_reserve_full_isl",
        "async_scheduling",
        "enforce_eager",
        "disable_cascade_attn",
        "enable_flashinfer_autotune",
    }
)
INTEGER_PARAMETERS = frozenset(
    {
        "lmcache_chunk_size", "cacheblend_blend_min_tokens",
        "tp",
        "decode_context_parallel_size",
        "cp_kv_cache_interleave_size",
        "data_parallel_size",
        "block_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "long_prefill_token_threshold",
        "num_speculative_tokens",
    }
)


def canonical_screened_name(name: str) -> str:
    """把终筛参数名转换为 BO 内部规范名。"""
    normalized = str(name).strip().lower().replace("-", "_")
    return SCREENING_NAME_TO_INTERNAL.get(normalized, normalized)


def _parse_bool(value: Any) -> bool:
    """严格解析 LLM 产物中的布尔候选，避免字符串 ``False`` 被当成真。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean candidate: {value!r}")


def _coerce_candidate(name: str, value: Any) -> Any:
    """将 LLM 输出的字符串/空值转成 vLLM 与 HEBO 使用的真实类型。"""
    if name == "quantization":
        return "none" if value is None else str(value).strip().lower()
    if name == "dtype":
        canonical = canonicalize_model_dtype(value)
        return "half" if canonical == "half" else canonical
    if name == "kv_cache_dtype":
        canonical = canonicalize_kv_cache_dtype(value)
        return "auto" if canonical is None else canonical
    if name in BOOLEAN_PARAMETERS:
        return _parse_bool(value)
    if name in INTEGER_PARAMETERS:
        return int(value)
    if name == "gpu_memory_utilization":
        # BO 历来用整数百分比建模，执行 trial 时再除以 100。
        return int(round(float(value) * 100))
    return str(value).strip().lower()


def _deduplicate(values: list[Any]) -> list[Any]:
    """保持原顺序去重，便于默认值与终筛证据对应。"""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def load_screened_parameter_results(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """合并“参数选择”和“参数搜索域”两个独立模块的产物。

    终筛文件只决定参数名；第二模块的 ``bo_domain_spec.json`` 才能提供候选
    值或固定值。这样可机械防止第一阶段的文档示例值再次直接进入 BO。
    """
    cached = getattr(args, "_screened_parameter_specs", None)
    if cached is not None:
        return cached

    raw_path = getattr(args, "screened_parameter_results", None)
    if not raw_path:
        if getattr(args, "parameter_domain_spec", None):
            raise ValueError(
                "--parameter-domain-spec requires --screened-parameter-results"
            )
        setattr(args, "_screened_parameter_specs", {})
        return {}
    raw_domain_path = getattr(args, "parameter_domain_spec", None)
    if not raw_domain_path:
        raise ValueError(
            "--screened-parameter-results selects names only; provide an independently "
            "audited --parameter-domain-spec. Screening candidate examples are not a "
            "valid BO search space."
        )
    path = Path(raw_path).expanduser().resolve()
    domain_path = Path(raw_domain_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    domain_payload = json.loads(domain_path.read_text(encoding="utf-8"))
    raw_parameters = payload.get("parameters")
    raw_domain_parameters = domain_payload.get("parameters")
    if domain_payload.get("screening_source") == "simple":
        raise ValueError("Legacy Range domains are not supported in tiger; regenerate inputs with agents.tuning.dqbo")
    if not isinstance(raw_parameters, list):
        raise ValueError(f"screened parameter file has no parameters list: {path}")
    if not isinstance(raw_domain_parameters, list):
        raise ValueError(f"domain spec has no parameters list: {domain_path}")
    if domain_payload.get("screening_source") == "fixed_config":
        from agents.tuning.dqbo.ranges import validate_fixed_domain
        validate_fixed_domain(domain_payload)
    elif bool(domain_payload.get("manual_search_ranges_used", False)):
        raise ValueError("BO domain spec must not use manual search ranges")
    if bool(domain_payload.get("screening_candidate_values_used", False)):
        raise ValueError("BO domain spec must not reuse screening candidate values")

    domain_by_name: dict[str, dict[str, Any]] = {}
    for raw_domain in raw_domain_parameters:
        if not isinstance(raw_domain, dict):
            raise TypeError("each domain parameter must be an object")
        domain_name = str(raw_domain.get("name", "")).strip()
        if not domain_name or domain_name in domain_by_name:
            raise ValueError(f"invalid or duplicate domain parameter: {domain_name!r}")
        domain_by_name[domain_name] = raw_domain

    specs: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []
    selected_names: list[str] = []
    for raw in raw_parameters:
        if not isinstance(raw, dict) or not bool(raw.get("enters_bo", False)):
            continue
        original_name = str(raw.get("name", ""))
        selected_names.append(original_name)
        name = canonical_screened_name(original_name)
        if name not in SUPPORTED_SCREENED_PARAMETERS:
            unsupported.append(original_name)
            continue
        domain_record = domain_by_name.get(original_name)
        if domain_record is None:
            raise ValueError(
                f"selected parameter {original_name!r} is absent from domain spec"
            )
        domain_decision = str(domain_record.get("decision", ""))
        if domain_decision not in {"tune", "fixed", "conditional", "composite"}:
            raise ValueError(
                f"domain parameter {original_name!r} has invalid decision "
                f"{domain_decision!r}"
            )
        if domain_decision == "fixed":
            raw_values = [domain_record.get("fixed_value")]
            normalized_domain = None
        else:
            raw_domain = domain_record.get("domain")
            if not isinstance(raw_domain, dict):
                raise ValueError(
                    f"domain parameter {original_name!r} has no domain object"
                )
            if "values" in raw_domain:
                if not isinstance(raw_domain.get("values"), list):
                    raise ValueError(
                        f"domain parameter {original_name!r} has invalid domain.values"
                    )
                raw_values = list(raw_domain["values"])
                normalized_domain = {
                    "kind": str(raw_domain.get("kind", "categorical")),
                    "values": [
                        _coerce_candidate(name, value) for value in raw_values
                    ],
                }
            else:
                if "lower" not in raw_domain or "upper" not in raw_domain:
                    raise ValueError(
                        f"numeric domain parameter {original_name!r} needs lower and upper"
                    )
                lower = _coerce_candidate(name, raw_domain["lower"])
                upper = _coerce_candidate(name, raw_domain["upper"])
                if lower >= upper:
                    raise ValueError(
                        f"numeric domain parameter {original_name!r} requires lower < upper"
                    )
                normalized_domain = {
                    "kind": str(raw_domain.get("kind", "")),
                    "scale": str(raw_domain.get("scale", "")),
                    "granularity": str(raw_domain.get("granularity", "")),
                    "lower": lower,
                    "upper": upper,
                }
                if "base" in raw_domain:
                    normalized_domain["base"] = raw_domain["base"]
                if "step" in raw_domain:
                    normalized_domain["step"] = raw_domain["step"]
                raw_values = [lower, upper]
        values = [_coerce_candidate(name, value) for value in raw_values]
        values = _deduplicate(values)
        if not values:
            raise ValueError(f"domain parameter {original_name!r} has no values")
        if normalized_domain is not None and "values" in normalized_domain:
            normalized_domain["values"] = list(values)
        raw_reference = domain_record.get("reference_value")
        reference_value = (
            None
            if raw_reference is None
            else _coerce_candidate(name, raw_reference)
        )
        if domain_decision == "fixed":
            reference_value = values[0]
        specs[name] = {
            "name": name,
            "screening_name": original_name,
            "group_id": raw.get("group_id"),
            "screening_decision": raw.get("decision"),
            "domain_decision": domain_decision,
            "activation_conditions": list(
                domain_record.get("activation_conditions")
                or raw.get("activation_conditions")
                or []
            ),
            "depends_on": list(
                domain_record.get("depends_on") or raw.get("depends_on") or []
            ),
            "composite_group": domain_record.get("composite_group"),
            "domain": normalized_domain,
            "candidates": values,
            "reference_value": reference_value,
            "fixed": domain_decision == "fixed",
        }
    if unsupported:
        raise ValueError(
            "screened parameter results contain unsupported BO knobs: "
            + ", ".join(sorted(unsupported))
        )
    unexpected_domain_names = set(domain_by_name) - set(selected_names)
    if unexpected_domain_names:
        raise ValueError(
            "domain spec contains parameters not selected by the first module: "
            + ", ".join(sorted(unexpected_domain_names))
        )
    setattr(args, "_screened_parameter_specs", specs)
    setattr(args, "_screened_parameter_results_path", str(path))
    setattr(args, "_parameter_domain_spec_path", str(domain_path))
    return specs


def screened_parameter_names(args: argparse.Namespace) -> frozenset[str]:
    """返回第二模块决定实际参与 BO 的非固定参数名。"""

    return frozenset(
        name
        for name, spec in load_screened_parameter_results(args).items()
        if not bool(spec.get("fixed", False))
    )


def screened_reference_value(
    args: argparse.Namespace,
    name: str,
    fallback: Any = None,
) -> Any:
    """读取第二模块保护的实测 reference 或已验证固定值。"""

    spec = load_screened_parameter_results(args).get(name)
    if spec is None or spec.get("reference_value") is None:
        return fallback
    return spec["reference_value"]


def screened_candidates(
    args: argparse.Namespace,
    name: str,
    fallback: list[Any] | None = None,
) -> list[Any]:
    """读取某个参数的终筛候选；普通 BO 模式可回落到原搜索范围。"""
    spec = load_screened_parameter_results(args).get(name)
    if spec is None:
        return list(fallback or [])
    return list(spec["candidates"])


def _model_config(args: argparse.Namespace) -> dict[str, Any]:
    """读取基础模型配置，用于在启动前证明 DCP 候选是否可达。"""
    config_path = Path(args.model_path) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _ray_available_for_vllm(args: argparse.Namespace) -> bool:
    """使用 vLLM 自己的解释器检查 Ray 是否真正可导入。"""

    vllm_bin = Path(args.vllm_bin).expanduser().absolute()
    vllm_python = next(
        (
            candidate
            for candidate in (
                vllm_bin.parent / "python",
                vllm_bin.parent / "python3",
            )
            if candidate.exists()
        ),
        None,
    )
    if vllm_python is None:
        return False
    probe = (
        "import importlib.util,sys;"
        "sys.exit(0 if importlib.util.find_spec('ray') is not None else 1)"
    )
    try:
        result = subprocess.run(
            [str(vllm_python), "-c", probe],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def executable_screened_candidates(
    args: argparse.Namespace,
    name: str,
    *,
    gpu_nums: int,
) -> list[Any]:
    """按模型、硬件和已安装依赖裁掉静态可证明不可执行的候选。"""
    values = screened_candidates(args, name)
    removed: list[dict[str, Any]] = []
    executable = list(values)

    if name == "decode_context_parallel_size":
        config = _model_config(args)
        kv_heads = int(config.get("num_key_value_heads") or config.get("num_attention_heads") or 0)
        tp_values = screened_candidates(args, "tp", [1])
        kept: list[int] = []
        for value in executable:
            dcp = int(value)
            reachable = any(
                int(tp) % dcp == 0
                and (dcp == 1 or kv_heads == 0 or int(tp) > kv_heads)
                for tp in tp_values
            )
            if reachable:
                kept.append(dcp)
            else:
                removed.append(
                    {
                        "value": dcp,
                        "reason": (
                            "Qwen2 GQA requires tensor_parallel_size > "
                            f"num_key_value_heads ({kv_heads}) when DCP > 1"
                        ),
                    }
                )
        executable = kept

    if name == "data_parallel_backend" and "ray" in executable and not _ray_available_for_vllm(args):
        executable = [value for value in executable if value != "ray"]
        removed.append({"value": "ray", "reason": "Ray is not installed in the vLLM environment"})

    if name in {"tp", "data_parallel_size"}:
        executable = [int(value) for value in executable if 1 <= int(value) <= gpu_nums]

    if not executable:
        raise ValueError(f"parameter {name!r} has no executable candidates after preflight")

    audit = getattr(args, "_screened_space_audit", {})
    audit[name] = {
        "screened_candidates": values,
        "executable_candidates": executable,
        "removed_candidates": removed,
    }
    setattr(args, "_screened_space_audit", audit)
    return executable


def executable_screened_space_parameter(
    args: argparse.Namespace,
    name: str,
    *,
    gpu_nums: int,
) -> dict[str, Any] | None:
    """把审计后的语义域编译为一个 HEBO 参数定义。

    枚举域继续执行依赖和硬件预检；数值范围则保留顺序与尺度，避免被降级为
    无序 ``cat``。固定参数不产生 BO 维度，因此返回 ``None``。
    """

    spec = load_screened_parameter_results(args).get(name)
    if spec is None:
        raise KeyError(f"unknown screened parameter {name!r}")
    if bool(spec.get("fixed", False)):
        return None
    if name == "cp_kv_cache_interleave_size":
        # This child knob only exists on an active DCP branch.  If Range has
        # mechanically fixed DCP to 1, representing the child as a BO axis
        # would create many coordinates that all execute as the same config.
        dcp_values = screened_candidates(args, "decode_context_parallel_size", [1])
        if all(int(value) <= 1 for value in dcp_values):
            audit = getattr(args, "_screened_space_audit", {})
            audit[name] = {
                "screened_domain": dict(spec.get("domain") or {}),
                "compiled_hebo_parameter": None,
                "removed_candidates": [],
                "inactive_reason": (
                    "decode_context_parallel_size is fixed to 1; "
                    "cp_kv_cache_interleave_size is canonicalized to 1"
                ),
            }
            setattr(args, "_screened_space_audit", audit)
            return None
    domain = spec.get("domain")
    if not isinstance(domain, dict):
        raise ValueError(f"parameter {name!r} has no executable domain")

    if "values" in domain:
        values = executable_screened_candidates(args, name, gpu_nums=gpu_nums)
        if name in BOOLEAN_PARAMETERS and set(values) == {False, True}:
            return {"name": name, "type": "bool"}
        return {"name": name, "type": "cat", "categories": values}

    kind = str(domain.get("kind", ""))
    scale = str(domain.get("scale", ""))
    granularity = str(domain.get("granularity", ""))
    result: dict[str, Any] = {
        "name": name,
        "lb": domain["lower"],
        "ub": domain["upper"],
    }
    if kind == "integer" and scale == "log" and granularity == "any_integer":
        result.update({"type": "pow_int", "base": domain["base"]})
    elif (
        kind == "integer"
        and scale == "log"
        and granularity == "powers_of_base"
    ):
        result.update({"type": "int_exponent", "base": domain["base"]})
    elif kind == "integer" and scale == "linear" and granularity == "any_integer":
        result["type"] = "int"
    elif kind == "integer" and scale == "linear" and granularity == "step":
        result.update({"type": "step_int", "step": domain["step"]})
    elif kind == "float" and scale == "linear" and granularity == "any_float":
        result["type"] = "num"
    elif kind == "float" and scale == "log" and granularity == "any_float":
        result.update({"type": "pow", "base": domain["base"]})
    else:
        raise ValueError(
            f"parameter {name!r} has unsupported numeric domain "
            f"kind={kind!r}, scale={scale!r}, granularity={granularity!r}"
        )

    audit = getattr(args, "_screened_space_audit", {})
    audit[name] = {
        "screened_domain": dict(domain),
        "compiled_hebo_parameter": dict(result),
        "removed_candidates": [],
    }
    setattr(args, "_screened_space_audit", audit)
    return result


def preferred_default(name: str, candidates: list[Any]) -> Any:
    """从候选集中选择稳定且能作为真实 reference trial 的默认值。"""
    preferences = {
        "quantization": ["none"],
        "dtype": ["bfloat16", "half"],
        "model_impl": ["vllm", "auto"],
        "tp": [2, 1],
        "decode_context_parallel_size": [1],
        "cp_kv_cache_interleave_size": [1],
        "data_parallel_size": [1, 2],
        "data_parallel_backend": ["mp"],
        "disable_nccl_for_dp_synchronization": [False, True],
        "block_size": [16, 32],
        "gpu_memory_utilization": [90, 85, 80],
        "kv_cache_dtype": ["bfloat16", "auto", "float16"],
        "max_model_len": [2048, 4096],
        "enable_prefix_caching": [False, True],
        "prefix_caching_hash_algo": ["sha256"],
        "max_num_batched_tokens": [512, 256],
        "max_num_seqs": [32, 16],
        "long_prefill_token_threshold": [0, 16, 32],
        "scheduler_reserve_full_isl": [True, False],
        "async_scheduling": [False, True],
        "enforce_eager": [False, True],
        "disable_cascade_attn": [True, False],
        "enable_flashinfer_autotune": [False, True],
    }
    for preferred in preferences.get(name, []):
        if preferred in candidates:
            return preferred
    return candidates[0]


def screened_space_manifest(args: argparse.Namespace, design_names: list[str]) -> dict[str, Any]:
    """生成可追溯清单，区分 21 个输入旋钮与真正参与建模的自由维度。"""
    specs = load_screened_parameter_results(args)
    return {
        "schema_version": 1,
        "source": getattr(args, "_screened_parameter_results_path", None),
        "parameter_domain_spec": getattr(args, "_parameter_domain_spec_path", None),
        "selected_parameter_count": len(specs),
        "selected_parameters": [spec["screening_name"] for spec in specs.values()],
        "fixed_parameters": {
            spec["screening_name"]: spec["candidates"][0]
            for spec in specs.values()
            if spec.get("fixed")
        },
        "domain_decisions": {
            spec["screening_name"]: spec["domain_decision"]
            for spec in specs.values()
        },
        "bo_dimension_count": len(design_names),
        "bo_dimensions": list(design_names),
        "preflight": getattr(args, "_screened_space_audit", {}),
        "notes": [
            "tensor_parallel_size and data_parallel_size are coupled to consume the two-GPU budget",
            "quantization and dtype remain separate BO dimensions; incompatible pairs are repaired before execution",
            "conditional child values are normalized to their parent-disabled defaults",
            "parameter names come from screening; all values come from the independent domain spec",
        ],
    }
