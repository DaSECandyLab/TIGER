"""Flock 贝叶斯优化的候选默认值和归一化。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..flock.config import (
    canonicalize_kv_cache_dtype,
    canonicalize_model_dtype,
    is_actual_kv_cache_dtype,
    is_actual_model_dtype,
    resolve_effective_dtype,
    resolve_effective_kv_cache_dtype,
)
from .screened_space import (
    executable_screened_candidates,
    load_screened_parameter_results,
    preferred_default,
    screened_candidates,
    screened_parameter_names,
    screened_reference_value,
)

DEFAULT_TUNED_PARAMS = frozenset({"tp", "data_parallel_size", "quantization", "max_num_seqs", "max_num_batched_tokens", "block_size", "enable_prefix_caching", "disable_custom_all_reduce"})
LEGACY_TUNE_FLAG_TO_PARAMS = {"tune_gpu_memory_utilization": {"gpu_memory_utilization"}, "tune_max_model_len": {"max_model_len"}, "tune_enforce_eager": {"enforce_eager"}, "tune_cacheblend_mode": {"cacheblend_mode"}, "tune_kv_cache_dtype": {"kv_cache_dtype"}, "tune_speculative_decoding": {"speculative_method", "num_speculative_tokens"}}
CACHEBLEND_DEPENDENT_PARAMS = frozenset({"lmcache_chunk_size", "cacheblend_blend_min_tokens", "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios"})
CACHEBLEND_EFFECTIVE_DEFAULT_MODE = "auto"
CACHEBLEND_MODES = frozenset({"auto", "off", "on"})
SAFE_REFERENCE_GPU_MEMORY_UTILIZATION_PCT = 85
SAFE_REFERENCE_DEFAULT_TP = 2
SAFE_REFERENCE_DEFAULT_MAX_NUM_SEQS = 64
SAFE_REFERENCE_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
SAFE_REFERENCE_DEFAULT_CPU_OFFLOAD_GB = 0.0
SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND = "native"
SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_SIZE = 0.0
COMPRESSED_KV_CACHE_DTYPES = frozenset(
    {"fp8_e4m3", "fp8_e5m2", "int8_per_token_head", "fp8_per_token_head"}
)
MIN_COMPRESSED_KV_BLOCK_SIZE = 16
UNSUPPORTED_SEARCH_KV_CACHE_DTYPES = frozenset({"nvfp4"})


def completed_bo_iterations(state: Any) -> int:
    """Count BO suggestions only; the separately measured reference is not a round."""

    return len(state.rec_history_list)

def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


# 把可为空的 bool-like 值规整成 bool 或 None。
def optional_bool_value(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


# 把可为空的数值转成 int，保留 None/NaN。
def optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


# 把整数列表配置统一成逗号分隔字符串，便于搜索空间和历史值比较。
def canonicalize_int_list_spec(value: Any) -> str:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return ",".join(str(int(part)) for part in parts)
    if value is None:
        return ""
    return ",".join(str(int(part)) for part in value)


# 把浮点列表配置统一成逗号分隔字符串，便于搜索空间和历史值比较。
def canonicalize_float_list_spec(value: Any) -> str:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return ",".join(f"{float(part):g}" for part in parts)
    if value is None:
        return ""
    return ",".join(f"{float(part):g}" for part in value)


# 规范化 KV offloading size，缺失值按 0 处理。
def normalized_offload_size(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return max(0.0, float(value))


# 把 BO 中的空 backend 规整成 trial CLI / vLLM 实际使用的默认值。
def normalize_kv_offloading_backend(value: Any) -> str:
    if value is None or pd.isna(value):
        return SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND
    return str(value)


# 判断某些压缩 KV dtype 是否要求 block_size 至少为安全下限。
def kv_cache_dtype_requires_larger_block_size(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value) in COMPRESSED_KV_CACHE_DTYPES


# Explicit model and KV-cache storage dtypes are independent vLLM options.  Keep
# these two helpers as no-op compatibility shims for old callers: only ``auto``
# inherits the model dtype, while an explicit float16/bfloat16 KV cache remains
# legal even when it differs from the model activation dtype.
def kv_cache_dtype_is_compatible_with_model_dtype(
    model_dtype: Any,
    kv_cache_dtype: Any,
) -> bool:
    return True


# Backward-compatible no-op; explicit storage dtype is not an illegal
# model/KV-cache combination and must not be silently rewritten.
def repair_incompatible_kv_cache_dtype(
    model_dtype: Any,
    kv_cache_dtype: Any,
) -> Any:
    return kv_cache_dtype


# 根据是否调 quantization 返回当前可能出现的量化候选。
def quantization_candidates_for_search(args: argparse.Namespace) -> list[str]:
    if tune_param_enabled(args, "quantization"):
        return ["none", "awq_marlin", "gptq_marlin"]
    return [str(args.quantization or "none")]


# 解析固定 dtype 在指定 quantization 下最终会传给 vLLM 的真实 dtype。
def effective_fixed_dtype(args: argparse.Namespace, quantization: Any) -> str | None:
    raw_dtype = args.dtype if args.dtype is not None else "auto"
    return resolve_effective_dtype(
        raw_dtype,
        quantization=quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )


# 生成 dtype 搜索空间的有效候选，并去掉重复或不可用值。
def effective_dtype_categories(args: argparse.Namespace) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for quantization in quantization_candidates_for_search(args):
        for raw_value in args.dtype_categories:
            canonical = canonicalize_model_dtype(raw_value)
            if canonical is None or not is_actual_model_dtype(canonical):
                continue
            resolved = resolve_effective_dtype(
                canonical,
                quantization=quantization,
                model_path=args.model_path,
                awq_model_path=args.awq_model_path,
                gptq_model_path=args.gptq_model_path,
            )
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            categories.append(resolved)
    return categories


# 推导 CacheBlend 路径要求的 KV cache dtype。
def required_cacheblend_kv_cache_dtype(
    *,
    dtype: Any,
    quantization: Any,
    args: argparse.Namespace,
) -> str | None:
    return resolve_effective_kv_cache_dtype(
        "auto",
        dtype=dtype,
        quantization=quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )


# 生成 KV cache dtype 搜索候选，并结合 dtype/quantization/block size 做过滤。
def effective_kv_cache_dtype_categories(args: argparse.Namespace) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for quantization in quantization_candidates_for_search(args):
        if tune_param_enabled(args, "dtype"):
            dtype_candidates = effective_dtype_categories(args)
        else:
            fixed_dtype_value = effective_fixed_dtype(args, quantization)
            dtype_candidates = [fixed_dtype_value] if fixed_dtype_value is not None else []
        if not dtype_candidates:
            dtype_candidates = [None]
        for effective_dtype in dtype_candidates:
            for raw_value in args.kv_cache_dtype_categories:
                canonical = canonicalize_kv_cache_dtype(raw_value)
                if (
                    canonical is None
                    or not is_actual_kv_cache_dtype(canonical)
                    or canonical in UNSUPPORTED_SEARCH_KV_CACHE_DTYPES
                ):
                    continue
                if is_cacheblend_enabled(args.cacheblend_mode):
                    resolved = required_cacheblend_kv_cache_dtype(
                        dtype=effective_dtype,
                        quantization=quantization,
                        args=args,
                    )
                else:
                    resolved = resolve_effective_kv_cache_dtype(
                        canonical,
                        dtype=effective_dtype,
                        quantization=quantization,
                        model_path=args.model_path,
                        awq_model_path=args.awq_model_path,
                        gptq_model_path=args.gptq_model_path,
                    )
                if resolved is None or resolved in seen:
                    continue
                seen.add(resolved)
                categories.append(resolved)
    if not tune_param_enabled(args, "block_size"):
        fixed_block_size = int(getattr(args, "block_size", None) or getattr(args, "default_block_size", 16))
        if fixed_block_size < MIN_COMPRESSED_KV_BLOCK_SIZE:
            filtered = [
                value
                for value in categories
                if not kv_cache_dtype_requires_larger_block_size(value)
            ]
            if filtered:
                return filtered
    return categories


# CacheBlend 模式处理
#
# `auto` 保留原有 workload，不覆盖 SQL/query。
# `off` 和 `on` 分别创建不启用/启用 CacheBlend 的显式覆盖。
#
# 将 CacheBlend mode 规整成实际执行语义。
def normalize_cacheblend_mode(value: Any) -> str:
    """返回 BO/trial 使用的 CacheBlend 模式。"""
    mode = str(value or CACHEBLEND_EFFECTIVE_DEFAULT_MODE).strip().lower()
    if mode not in CACHEBLEND_MODES:
        raise ValueError(
            f"unsupported cacheblend_mode={value!r}; expected auto, off, or on"
        )
    return mode


# 判断 CacheBlend 是否实际处于开启状态。
def is_cacheblend_enabled(value: Any) -> bool:
    return normalize_cacheblend_mode(value) == "on"


# 计算 KV cache dtype 不参与调参时的固定取值。
def fixed_kv_cache_dtype(args: argparse.Namespace) -> str | None:
    fixed_dtype_value = effective_fixed_dtype(args, args.quantization)
    if is_cacheblend_enabled(args.cacheblend_mode):
        return required_cacheblend_kv_cache_dtype(
            dtype=fixed_dtype_value,
            quantization=args.quantization,
            args=args,
        )
    return resolve_effective_kv_cache_dtype(
        args.kv_cache_dtype if args.kv_cache_dtype is not None else "auto",
        dtype=fixed_dtype_value,
        quantization=args.quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )


# 生成 KV offloading backend 搜索候选；CacheBlend 未启用时过滤 lmcache。
def effective_kv_offloading_backend_categories(args: argparse.Namespace) -> list[str]:
    categories = [str(value) for value in args.kv_offloading_backend_categories]
    if is_cacheblend_enabled(args.cacheblend_mode) or tune_param_enabled(
        args, "cacheblend_mode"
    ):
        return categories
    return [value for value in categories if value != "lmcache"]


# 判断当前配置是否允许把 enable_prefix_caching 放进搜索空间。
def allow_prefix_caching_tuning(
    args: argparse.Namespace, requested_params: set[str] | frozenset[str]
) -> bool:
    return (
        "enable_prefix_caching" in requested_params
        and (
            not is_cacheblend_enabled(args.cacheblend_mode)
            or args.cacheblend_allow_prefix_caching
        )
    )


# 校验 CacheBlend 专属参数只在 CacheBlend 固定开启或参与搜索时才允许调。
def validate_cacheblend_specific_tuning(
    args: argparse.Namespace,
    requested_params: set[str],
) -> None:
    requested_cacheblend_specific = sorted(
        param_name
        for param_name in CACHEBLEND_DEPENDENT_PARAMS
        if param_name in requested_params
    )
    if not requested_cacheblend_specific:
        return
    if "cacheblend_mode" in requested_params or is_cacheblend_enabled(args.cacheblend_mode):
        return
    raise ValueError(
        "cacheblend-specific params require --cacheblend_mode on or "
        "--tune-params cacheblend_mode ...: "
        + ", ".join(requested_cacheblend_specific)
    )


# 合并默认调参维度、--tune_params 和旧 --tune_* flags，得到最终搜索维度集合。
def selected_tune_params(args: argparse.Namespace) -> frozenset[str]:
    cached = getattr(args, "_selected_tune_params", None)
    if cached is not None:
        return cached

    requested: set[str]
    screened_requested = screened_parameter_names(args)
    if screened_requested:
        if args.tune_params is not None:
            raise ValueError(
                "--screened-parameter-results cannot be combined with --tune-params"
            )
        requested = set(screened_requested)
    elif args.tune_params is not None:
        legacy_flags = [
            flag_name
            for flag_name in LEGACY_TUNE_FLAG_TO_PARAMS
            if bool(getattr(args, flag_name))
        ]
        if legacy_flags:
            raise ValueError(
                "--tune-params cannot be combined with legacy --tune_* flags: "
                + ", ".join(f"--{flag_name}" for flag_name in legacy_flags)
            )
        requested = set(args.tune_params)
    else:
        requested = set(DEFAULT_TUNED_PARAMS)
        for flag_name, param_names in LEGACY_TUNE_FLAG_TO_PARAMS.items():
            if bool(getattr(args, flag_name)):
                requested.update(param_names)

    speculative_tuned = bool(
        {"speculative_method", "num_speculative_tokens"} & requested
    )
    if speculative_tuned and args.speculative_config_json is not None:
        raise ValueError(
            "speculative_method/num_speculative_tokens cannot be tuned when "
            "--speculative_config_json is set"
        )

    validate_cacheblend_specific_tuning(args, requested)

    if "enable_prefix_caching" in requested and not allow_prefix_caching_tuning(args, requested):
        requested.discard("enable_prefix_caching")

    cached = frozenset(requested)
    setattr(args, "_selected_tune_params", cached)
    return cached


# 判断某个参数是否最终进入 HEBO 搜索空间。
def tune_param_enabled(args: argparse.Namespace, param_name: str) -> bool:
    selected = selected_tune_params(args)
    if param_name == "enable_prefix_caching":
        return allow_prefix_caching_tuning(args, selected)
    return param_name in selected


# 在 int_exponent 搜索空间中选择一个落在边界内的安全默认值。
def safe_int_exponent_default(
    preferred: int,
    lb: int,
    ub: int,
) -> int:
    if lb > ub:
        raise ValueError(f"invalid int exponent bounds: lb={lb} > ub={ub}")
    candidates = []
    exponent = 0
    while 2**exponent <= ub:
        value = 2**exponent
        if lb <= value <= ub:
            candidates.append(value)
        exponent += 1
    if not candidates:
        return max(lb, min(preferred, ub))
    not_above_preferred = [value for value in candidates if value <= preferred]
    if not_above_preferred:
        return not_above_preferred[-1]
    return candidates[0]


# 返回不超过给定上限的最大 2 的幂。
def safe_power_of_two_upper_bound(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"power-of-two upper bound requires limit >= 1, got {limit}")
    exponent = int(math.floor(math.log2(limit)))
    return 2**exponent


# 根据 GPU 数、模型长度和搜索边界初始化核心默认调参值。
def initialize_default_search_values(
    args: argparse.Namespace,
    min_world_size: int,
    gpu_nums: int,
    max_sequence_length: int,
) -> None:
    safe_tp_lb = max(1, int(min_world_size))
    safe_tp_ub = max(safe_tp_lb, int(gpu_nums))
    setattr(
        args,
        "default_tp",
        int(
            safe_int_exponent_default(
                SAFE_REFERENCE_DEFAULT_TP,
                safe_tp_lb,
                safe_tp_ub,
            )
        ),
    )
    setattr(args, "default_data_parallel_size", 1)
    setattr(
        args,
        "default_max_num_seqs",
        int(
            safe_int_exponent_default(
                SAFE_REFERENCE_DEFAULT_MAX_NUM_SEQS,
                int(args.max_num_seqs_lb),
                int(args.max_num_seqs_ub),
            )
        ),
    )
    setattr(
        args,
        "default_max_num_batched_tokens",
        int(
            safe_int_exponent_default(
                SAFE_REFERENCE_DEFAULT_MAX_NUM_BATCHED_TOKENS,
                int(args.max_num_batched_tokens_lb),
                int(
                    args.max_num_batched_tokens_ub
                    if args.max_num_batched_tokens_ub is not None
                    else max(8192, max_sequence_length * 2)
                ),
            )
        ),
    )
    setattr(args, "default_block_size", 16)


def default_combination(args: argparse.Namespace) -> dict[str, Any]:
    from .search_space import default_space_value
    # 本地 vLLM/Flock 路径已经不再支持关闭 chunked prefill，因此固定开启。
    max_model_len_lb = int(args.max_model_len_lb)
    max_model_len_ub = int(args.max_model_len_ub)
    if max_model_len_lb > max_model_len_ub:
        raise ValueError(
            f"max_model_len_lb={max_model_len_lb} must be <= "
            f"max_model_len_ub={max_model_len_ub}"
        )
    max_model_len_values = [0] if max_model_len_lb <= 0 else []
    exponent = math.ceil(math.log2(max(1, max_model_len_lb)))
    while 2**exponent <= max_model_len_ub:
        max_model_len_values.append(2**exponent)
        exponent += 1
    feasible_max_model_lens = [
        value
        for value in max_model_len_values
        if value == 0
        or value <= int(getattr(args, "default_max_num_batched_tokens", 4096))
    ]
    non_zero_max_model_lens = [
        value for value in feasible_max_model_lens if value > 0
    ]
    default_max_model_len_value = (
        non_zero_max_model_lens[-1]
        if non_zero_max_model_lens
        else (feasible_max_model_lens[-1] if feasible_max_model_lens else max_model_len_values[0])
    )
    screened_specs = load_screened_parameter_results(args)
    # Under the screened workflow, a knob that did not survive screening must
    # retain the engine's actual default.  The old standalone BO heuristic of
    # capping max_model_len at the batch-token budget would silently change the
    # measured Baseline Agent configuration (Qwen uses its model default when
    # the flag is omitted).
    screened_max_model_len_fallback = (
        getattr(args, "max_model_len", None)
        if screened_specs and "max_model_len" not in screened_specs
        else default_max_model_len_value
    )

    def screened_default(name: str, fallback: Any) -> Any:
        """域设计模式优先使用实测 reference 或已验证 fixed value。"""
        if name not in screened_specs:
            return fallback
        reference = screened_reference_value(args, name)
        if reference is not None:
            return reference
        candidates = screened_candidates(args, name)
        return preferred_default(name, candidates)

    return {
        "tp": int(screened_default("tp", getattr(args, "default_tp", 1))),
        "data_parallel_size": int(
            screened_default(
                "data_parallel_size",
                getattr(args, "default_data_parallel_size", 1),
            )
        ),
        "quantization": screened_default("quantization", "none"),
        "dtype": screened_default(
            "dtype", effective_fixed_dtype(args, args.quantization)
        ),
        "model_impl": screened_default(
            "model_impl", getattr(args, "model_impl", "vllm")
        ),
        "decode_context_parallel_size": int(
            screened_default(
                "decode_context_parallel_size",
                getattr(args, "decode_context_parallel_size", 1),
            )
        ),
        "cp_kv_cache_interleave_size": int(
            screened_default(
                "cp_kv_cache_interleave_size",
                getattr(args, "cp_kv_cache_interleave_size", 1),
            )
        ),
        "data_parallel_backend": screened_default(
            "data_parallel_backend", getattr(args, "data_parallel_backend", "mp")
        ),
        "distributed_executor_backend": screened_default(
            "distributed_executor_backend", getattr(args, "distributed_executor_backend", None)
        ),
        "disable_nccl_for_dp_synchronization": bool(
            screened_default(
                "disable_nccl_for_dp_synchronization",
                getattr(args, "disable_nccl_for_dp_synchronization", False),
            )
        ),
        "cpu_offload_gb": (
            float(args.cpu_offload_gb)
            if args.cpu_offload_gb is not None
            else SAFE_REFERENCE_DEFAULT_CPU_OFFLOAD_GB
        ),
        "kv_cache_dtype": fixed_kv_cache_dtype(args),
        "kv_offloading_backend": (
            SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND
            if args.kv_offloading_backend is None
            else str(args.kv_offloading_backend)
        ),
        "kv_offloading_size": normalized_offload_size(
            args.kv_offloading_size
            if args.kv_offloading_size is not None
            else SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_SIZE
        ),
        "cacheblend_mode": normalize_cacheblend_mode(args.cacheblend_mode),
        "cacheblend_allow_prefix_caching": bool(screened_default(
            "cacheblend_allow_prefix_caching", getattr(args, "cacheblend_allow_prefix_caching", False)
        )),
        "lmcache_chunk_size": int(args.lmcache_chunk_size),
        "cacheblend_blend_min_tokens": int(args.cacheblend_blend_min_tokens),
        "cacheblend_blend_check_layers": canonicalize_int_list_spec(
            args.cacheblend_blend_check_layers
        ),
        "cacheblend_blend_recompute_ratios": (
            canonicalize_float_list_spec(args.cacheblend_blend_recompute_ratios)
        ),
        "speculative_method": str(args.speculative_method or "none"),
        "num_speculative_tokens": (
            None
            if str(args.speculative_method or "none") == "none"
            else (
                int(args.num_speculative_tokens)
                if args.num_speculative_tokens is not None
                else int(args.num_speculative_tokens_lb)
            )
        ),
        "max_num_seqs": int(
            screened_default(
                "max_num_seqs", getattr(args, "default_max_num_seqs", 128)
            )
        ),
        "max_num_batched_tokens": int(
            screened_default(
                "max_num_batched_tokens",
                getattr(args, "default_max_num_batched_tokens", 4096),
            )
        ),
        "block_size": int(
            screened_default("block_size", getattr(args, "default_block_size", 16))
        ),
        "long_prefill_token_threshold": int(
            getattr(args, "long_prefill_token_threshold", 0)
        ),
        "max_model_len": optional_int(
            screened_default("max_model_len", screened_max_model_len_fallback)
        ),
        "gpu_memory_utilization": (
            float(screened_default("gpu_memory_utilization", SAFE_REFERENCE_GPU_MEMORY_UTILIZATION_PCT))
            / 100.0
            if "gpu_memory_utilization" in screened_specs
            else (
                float(args.gpu_memory_utilization)
                if args.gpu_memory_utilization is not None
                and not tune_param_enabled(args, "gpu_memory_utilization")
                else float(SAFE_REFERENCE_GPU_MEMORY_UTILIZATION_PCT / 100.0)
            )
        ),
        "enforce_eager": bool(screened_default("enforce_eager", False)),
        "enable_chunked_prefill": screened_default("enable_chunked_prefill", True),
        "scheduler_delay_factor": 0.0,
        "enable_prefix_caching": bool(
            screened_default("enable_prefix_caching", False)
        ),
        "prefix_caching_hash_algo": screened_default(
            "prefix_caching_hash_algo",
            getattr(args, "prefix_caching_hash_algo", "sha256"),
        ),
        "scheduler_reserve_full_isl": bool(
            screened_default(
                "scheduler_reserve_full_isl",
                getattr(args, "scheduler_reserve_full_isl", True),
            )
        ),
        "async_scheduling": bool(
            screened_default("async_scheduling", getattr(args, "async_scheduling", False))
        ),
        "disable_cascade_attn": bool(
            screened_default(
                "disable_cascade_attn", getattr(args, "disable_cascade_attn", True)
            )
        ),
        "enable_flashinfer_autotune": bool(
            screened_default(
                "enable_flashinfer_autotune",
                getattr(args, "enable_flashinfer_autotune", False),
            )
        ),
        "disable_custom_all_reduce": bool(
            screened_default("disable_custom_all_reduce", getattr(args, "disable_custom_all_reduce", False))
        ),
        "use_v2_block_manager": False,
    }


# 归一化候选配置，处理跨参数约束和条件依赖，保证 trial 启动参数可执行。
def normalize_conditional_combination(
    combination: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    normalized = dict(combination)
    default_values = default_combination(args)
    total_gpu_count = (
        int(args.detected_gpu_num)
        if getattr(args, "detected_gpu_num", None) is not None
        else (int(args.gpu_num) if getattr(args, "gpu_num", None) is not None else None)
    )

    normalized["cacheblend_mode"] = normalize_cacheblend_mode(
        normalized.get("cacheblend_mode", default_values["cacheblend_mode"])
    )
    cacheblend_active = is_cacheblend_enabled(normalized["cacheblend_mode"])
    if cacheblend_active:
        # 启用 cacheblend 时，若用户未显式允许，则强制关闭 prefix caching，
        # 避免两套前缀复用/传递机制同时生效导致语义或稳定性问题。
        if not args.cacheblend_allow_prefix_caching:
            normalized["enable_prefix_caching"] = False
        # 当前阶段 LMCache 与 CacheBlend 绑定：CacheBlend 开启时，候选配置显式
        # 标记为 lmcache；后续若要把二者解耦，再把这条绑定放开。
        normalized["kv_offloading_backend"] = "lmcache"
        raw_kv_cache_dtype = required_cacheblend_kv_cache_dtype(
            dtype=normalized.get("dtype", default_values["dtype"]),
            quantization=normalized.get("quantization", default_values["quantization"]),
            args=args,
        )
    else:
        # 未启用 cacheblend 时，所有仅服务于 blend / kv-transfer 路径的参数都回落到默认值，
        # 防止这些参数继续污染当前 trial 的搜索空间和启动参数。
        for param_name in CACHEBLEND_DEPENDENT_PARAMS:
            normalized[param_name] = default_values[param_name]
        # 用户当前要求是“只有启用 cacheblend 才启动 lmcache”，
        # 因此在 cacheblend 关闭时，显式清掉 lmcache backend，避免单独开启 kv offloading。
        if normalized.get("kv_offloading_backend") == "lmcache":
            normalized["kv_offloading_backend"] = None
        normalized["kv_offloading_backend"] = normalize_kv_offloading_backend(
            normalized.get("kv_offloading_backend")
        )
        raw_kv_cache_dtype = normalized.get(
            "kv_cache_dtype",
            default_values["kv_cache_dtype"],
        )

    speculative_method = str(normalized.get("speculative_method", "none") or "none")
    if speculative_method == "none":
        normalized["speculative_method"] = "none"
        normalized["num_speculative_tokens"] = None

    quantization = str(
        normalized.get("quantization", default_values["quantization"]) or "none"
    )
    raw_dtype = normalized.get("dtype", default_values["dtype"])
    normalized["dtype"] = resolve_effective_dtype(
        raw_dtype,
        quantization=quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )

    if normalized.get("enable_chunked_prefill") is None:
        normalized["enable_chunked_prefill"] = default_values["enable_chunked_prefill"]
    if cacheblend_active:
        normalized["enable_chunked_prefill"] = False
    if normalized.get("enable_prefix_caching") is None:
        normalized["enable_prefix_caching"] = default_values["enable_prefix_caching"]
    if normalized.get("enforce_eager") is None:
        normalized["enforce_eager"] = default_values["enforce_eager"]

    # 终筛新增参数先补齐默认值，后面再根据父参数关闭无效子分支。
    for name in (
        "model_impl",
        "decode_context_parallel_size",
        "cp_kv_cache_interleave_size",
        "data_parallel_backend",
        "disable_nccl_for_dp_synchronization",
        "prefix_caching_hash_algo",
        "scheduler_reserve_full_isl",
        "async_scheduling",
        "disable_cascade_attn",
        "enable_flashinfer_autotune",
    ):
        if normalized.get(name) is None:
            normalized[name] = default_values[name]

    tp = int(normalized.get("tp", default_values["tp"]))
    data_parallel_size = int(
        normalized.get(
            "data_parallel_size",
            default_values["data_parallel_size"],
        )
    )
    tp = max(tp, 1)
    data_parallel_size = max(data_parallel_size, 1)
    screened_params = screened_parameter_names(args)
    if (
        total_gpu_count is not None
        and "tp" in screened_params
        and "data_parallel_size" in screened_params
    ):
        # 只有 TP 与 DP 都被显式选入搜索域时才按 GPU 预算派生 DP。
        # 单独搜索 TP 时保留基准 DP，不引入额外的拓扑联动。
        data_parallel_size = max(1, total_gpu_count // tp)
    if total_gpu_count is not None:
        max_dp_for_tp = max(1, total_gpu_count // tp)
        if data_parallel_size > max_dp_for_tp:
            # 单机调优默认要求 tp * dp 不超过可见 GPU 数。
            # 当 BO 采到非法组合时，优先保持 tp 不变并把 dp 收敛到最近合法 2 的幂。
            data_parallel_size = safe_power_of_two_upper_bound(max_dp_for_tp)
    normalized["tp"] = tp
    normalized["data_parallel_size"] = data_parallel_size
    if data_parallel_size <= 1:
        # DP 子旋钮在 TP=2/DP=1 分支完全不生效，统一回落，避免等价配置重复。
        normalized["data_parallel_backend"] = "mp"
        normalized["disable_nccl_for_dp_synchronization"] = False

    if (
        (normalized.get("data_parallel_backend") == "ray" or normalized.get("distributed_executor_backend") == "ray")
        and os.environ.get("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0") != "1"
    ):
        # vLLM 0.20.0 的旧 RayDistributedExecutor 不声明异步调度支持；
        # Ray V2 继承 MultiprocExecutor，显式启用时才保留 async=true。
        normalized["async_scheduling"] = False

    # Qwen2-7B 是 GQA（4 个 KV heads）；TP=2 无法合法启用 DCP=2。
    dcp = int(normalized.get("decode_context_parallel_size", 1))
    if tp % dcp != 0:
        dcp = 1
    model_config_path = Path(args.model_path) / "config.json"
    if dcp > 1 and model_config_path.exists():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        kv_heads = int(
            model_config.get("num_key_value_heads")
            or model_config.get("num_attention_heads")
            or 0
        )
        if kv_heads and tp <= kv_heads:
            dcp = 1
    normalized["decode_context_parallel_size"] = dcp
    # CP KV interleaving has no effect unless DCP is active. Canonicalizing the
    # child to 1 prevents equivalent configurations from polluting BO history.
    cp_interleave = int(normalized.get("cp_kv_cache_interleave_size", 1))
    normalized["cp_kv_cache_interleave_size"] = (
        max(1, cp_interleave) if dcp > 1 else 1
    )

    if not bool(normalized.get("enable_prefix_caching")):
        normalized["prefix_caching_hash_algo"] = "sha256"

    block_size = int(normalized.get("block_size", default_values["block_size"]))
    normalized["kv_cache_dtype"] = resolve_effective_kv_cache_dtype(
        raw_kv_cache_dtype,
        dtype=normalized.get("dtype"),
        quantization=quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )
    kv_cache_dtype = normalized.get("kv_cache_dtype", default_values["kv_cache_dtype"])
    if (
        kv_cache_dtype_requires_larger_block_size(kv_cache_dtype)
        and block_size < MIN_COMPRESSED_KV_BLOCK_SIZE
    ):
        # 某些压缩型 kv cache dtype 需要更大的 block size 才能正常工作，
        # 这里统一把过小取值抬到最小合法值，避免 trial 因非法组合直接失败。
        block_size = MIN_COMPRESSED_KV_BLOCK_SIZE
    uses_kv_transfer_config = (
        args.kv_transfer_config_json is not None
        or is_cacheblend_enabled(normalized.get("cacheblend_mode", "off"))
    )
    if uses_kv_transfer_config and block_size < MIN_COMPRESSED_KV_BLOCK_SIZE:
        # 只要最终会透传 kv_transfer_config，就统一要求 block_size >= 16。
        # 这是比“压缩型 kv cache dtype”更强的一层兜底约束，用来规避
        # 规避 cacheblend / kv-transfer 路径下 block_size=8 触发的 Triton 编译失败。
        block_size = MIN_COMPRESSED_KV_BLOCK_SIZE
    normalized["block_size"] = block_size

    max_num_seqs = int(normalized.get("max_num_seqs", default_values["max_num_seqs"]))
    max_num_batched_tokens = int(
        normalized.get(
            "max_num_batched_tokens",
            default_values["max_num_batched_tokens"],
        )
    )
    max_model_len = optional_int(
        normalized.get("max_model_len", default_values["max_model_len"])
    )
    # vLLM 只要求 token budget 不小于并发序列数。启用 chunked prefill 时，
    # max_num_batched_tokens 可以合法小于 max_model_len；二者不能再被错误绑定。
    min_required_batched_tokens = max_num_seqs
    if not normalized["enable_chunked_prefill"]:
        min_required_batched_tokens = max(min_required_batched_tokens, max_model_len or args.max_model_len or 32768)
    if max_num_batched_tokens < min_required_batched_tokens:
        normalized["max_num_batched_tokens"] = min_required_batched_tokens
    else:
        normalized["max_num_batched_tokens"] = max_num_batched_tokens
    normalized["max_num_seqs"] = max_num_seqs
    normalized["max_model_len"] = max_model_len

    if "cacheblend_blend_recompute_ratios" in screened_parameter_names(args):
        actual = canonicalize_float_list_spec(normalized["cacheblend_blend_recompute_ratios"])
        for declared in screened_candidates(args, "cacheblend_blend_recompute_ratios"):
            if canonicalize_float_list_spec(declared) == actual:
                normalized["cacheblend_blend_recompute_ratios"] = declared
                break

    return normalized


# 把 HEBO 建议的一行 DataFrame 转成完整 trial 配置。
def rec_to_combination(rec: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    from .history import rec_value

    defaults = default_combination(args)
    gpu_memory_default = (
        int(round(float(args.gpu_memory_utilization) * 100))
        if args.gpu_memory_utilization is not None
        else SAFE_REFERENCE_GPU_MEMORY_UTILIZATION_PCT
    )
    raw_gpu_memory_pct = rec_value(rec, "gpu_memory_utilization", gpu_memory_default)
    gpu_memory_utilization = (
        float(raw_gpu_memory_pct) / 100.0
        if raw_gpu_memory_pct is not None and not pd.isna(raw_gpu_memory_pct)
        else defaults["gpu_memory_utilization"]
    )
    speculative_method = str(
        rec_value(
            rec,
            "speculative_method",
            defaults["speculative_method"],
        )
    )
    num_speculative_tokens = optional_int(
        rec_value(
            rec,
            "num_speculative_tokens",
            defaults["num_speculative_tokens"],
        )
    )
    if speculative_method == "none":
        num_speculative_tokens = None
    combination = {
        "distributed_executor_backend": rec_value(rec, "distributed_executor_backend", defaults["distributed_executor_backend"]),
        "cacheblend_allow_prefix_caching": bool_value(rec_value(rec, "cacheblend_allow_prefix_caching", defaults["cacheblend_allow_prefix_caching"])),
        "tp": int(rec_value(rec, "tp", defaults["tp"])),
        "data_parallel_size": int(
            rec_value(
                rec,
                "data_parallel_size",
                defaults["data_parallel_size"],
            )
        ),
        "quantization": str(rec_value(rec, "quantization", defaults["quantization"])),
        "dtype": rec_value(rec, "dtype", defaults["dtype"]),
        "model_impl": str(
            rec_value(rec, "model_impl", defaults["model_impl"])
        ),
        "decode_context_parallel_size": int(
            rec_value(
                rec,
                "decode_context_parallel_size",
                defaults["decode_context_parallel_size"],
            )
        ),
        "cp_kv_cache_interleave_size": int(
            rec_value(
                rec,
                "cp_kv_cache_interleave_size",
                defaults["cp_kv_cache_interleave_size"],
            )
        ),
        "data_parallel_backend": str(
            rec_value(
                rec, "data_parallel_backend", defaults["data_parallel_backend"]
            )
        ),
        "disable_nccl_for_dp_synchronization": bool_value(
            rec_value(
                rec,
                "disable_nccl_for_dp_synchronization",
                defaults["disable_nccl_for_dp_synchronization"],
            )
        ),
        "cpu_offload_gb": (
            float(rec_value(rec, "cpu_offload_gb", defaults["cpu_offload_gb"]))
            if rec_value(rec, "cpu_offload_gb", defaults["cpu_offload_gb"])
            is not None
            else None
        ),
        "kv_cache_dtype": rec_value(
            rec,
            "kv_cache_dtype",
            defaults["kv_cache_dtype"],
        ),
        "kv_offloading_backend": rec_value(
            rec,
            "kv_offloading_backend",
            defaults["kv_offloading_backend"],
        ),
        "kv_offloading_size": (
            normalized_offload_size(
                rec_value(rec, "kv_offloading_size", defaults["kv_offloading_size"])
            )
            if rec_value(rec, "kv_offloading_size", defaults["kv_offloading_size"])
            is not None
            else None
        ),
        "cacheblend_mode": str(
            rec_value(rec, "cacheblend_mode", defaults["cacheblend_mode"])
        ),
        "lmcache_chunk_size": int(
            rec_value(rec, "lmcache_chunk_size", defaults["lmcache_chunk_size"])
        ),
        "cacheblend_blend_min_tokens": int(
            rec_value(
                rec,
                "cacheblend_blend_min_tokens",
                defaults["cacheblend_blend_min_tokens"],
            )
        ),
        "cacheblend_blend_check_layers": canonicalize_int_list_spec(
            rec_value(
                rec,
                "cacheblend_blend_check_layers",
                defaults["cacheblend_blend_check_layers"],
            )
        ),
        "cacheblend_blend_recompute_ratios": canonicalize_float_list_spec(
            rec_value(
                rec,
                "cacheblend_blend_recompute_ratios",
                defaults["cacheblend_blend_recompute_ratios"],
            )
        ),
        "speculative_method": speculative_method,
        "num_speculative_tokens": num_speculative_tokens,
        "max_num_seqs": int(rec_value(rec, "max_num_seqs", defaults["max_num_seqs"])),
        "max_num_batched_tokens": int(
            rec_value(
                rec,
                "max_num_batched_tokens",
                defaults["max_num_batched_tokens"],
            )
        ),
        "block_size": int(rec_value(rec, "block_size", defaults["block_size"])),
        "long_prefill_token_threshold": int(
            rec_value(
                rec,
                "long_prefill_token_threshold",
                defaults["long_prefill_token_threshold"],
            )
        ),
        "max_model_len": optional_int(
            rec_value(rec, "max_model_len", defaults["max_model_len"])
        ),
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": bool_value(
            rec_value(rec, "enforce_eager", defaults["enforce_eager"])
        ),
        "enable_chunked_prefill": optional_bool_value(
            rec_value(
                rec,
                "enable_chunked_prefill",
                defaults["enable_chunked_prefill"],
            )
        ),
        "scheduler_delay_factor": float(
            int(
                rec_value(
                    rec,
                    "scheduler_delay_factor",
                    int(defaults["scheduler_delay_factor"] * 10),
                )
            )
            / 10.0
        ),
        "enable_prefix_caching": optional_bool_value(
            rec_value(
                rec,
                "enable_prefix_caching",
                defaults["enable_prefix_caching"],
            )
        ),
        "prefix_caching_hash_algo": str(
            rec_value(
                rec,
                "prefix_caching_hash_algo",
                defaults["prefix_caching_hash_algo"],
            )
        ),
        "scheduler_reserve_full_isl": bool_value(
            rec_value(
                rec,
                "scheduler_reserve_full_isl",
                defaults["scheduler_reserve_full_isl"],
            )
        ),
        "async_scheduling": bool_value(
            rec_value(rec, "async_scheduling", defaults["async_scheduling"])
        ),
        "disable_cascade_attn": bool_value(
            rec_value(
                rec, "disable_cascade_attn", defaults["disable_cascade_attn"]
            )
        ),
        "enable_flashinfer_autotune": bool_value(
            rec_value(
                rec,
                "enable_flashinfer_autotune",
                defaults["enable_flashinfer_autotune"],
            )
        ),
        "disable_custom_all_reduce": bool_value(
            rec_value(
                rec,
                "disable_custom_all_reduce",
                defaults["disable_custom_all_reduce"],
            )
        ),
        "use_v2_block_manager": bool_value(
            rec_value(rec, "use_v2_block_manager", defaults["use_v2_block_manager"])
        ),
    }
    return normalize_conditional_combination(combination, args)
