"""Flock 贝叶斯优化的 HEBO 设计空间构造。"""

from __future__ import annotations

import argparse
import math
from typing import Any

from .reference import get_ref_config
from .parameters import (
    CACHEBLEND_EFFECTIVE_DEFAULT_MODE,
    allow_prefix_caching_tuning,
    canonicalize_float_list_spec,
    canonicalize_int_list_spec,
    effective_dtype_categories,
    effective_kv_cache_dtype_categories,
    effective_kv_offloading_backend_categories,
    initialize_default_search_values,
    normalize_cacheblend_mode,
    selected_tune_params,
    tune_param_enabled,
)
from .parameters import *
from .screened_space import (
    executable_screened_space_parameter,
    load_screened_parameter_results,
    preferred_default,
    screened_candidates,
    screened_reference_value,
)

def default_space_value(name: str, args: argparse.Namespace) -> Any:
    screened_specs = load_screened_parameter_results(args)
    if name in screened_specs:
        reference = screened_reference_value(args, name)
        if reference is not None:
            return reference
        return preferred_default(name, screened_candidates(args, name))

    gpu_memory_utilization_default = (
        int(round(float(args.gpu_memory_utilization) * 100))
        if args.gpu_memory_utilization is not None
        else SAFE_REFERENCE_GPU_MEMORY_UTILIZATION_PCT
    )
    chunked_prefill_default = True

    # 1. dtype / KV / offloading 默认值要先结合当前搜索开关计算。
    dtype_categories = effective_dtype_categories(args)
    requested_dtype = (
        canonicalize_model_dtype(args.dtype)
        if args.dtype is not None
        else None
    )
    if tune_param_enabled(args, "dtype"):
        dtype_default = (
            str(requested_dtype)
            if requested_dtype in dtype_categories or not dtype_categories
            else str(dtype_categories[0])
        )
    else:
        dtype_default = effective_fixed_dtype(args, args.quantization)

    kv_cache_dtype_categories = effective_kv_cache_dtype_categories(args)
    requested_kv_cache_dtype = (
        canonicalize_kv_cache_dtype(args.kv_cache_dtype)
        if args.kv_cache_dtype is not None
        else None
    )
    if tune_param_enabled(args, "kv_cache_dtype"):
        if requested_kv_cache_dtype is not None and requested_kv_cache_dtype in kv_cache_dtype_categories:
            kv_cache_dtype_default = str(requested_kv_cache_dtype)
        elif kv_cache_dtype_categories:
            kv_cache_dtype_default = str(kv_cache_dtype_categories[0])
        else:
            kv_cache_dtype_default = str(fixed_kv_cache_dtype(args))
    else:
        kv_cache_dtype_default = fixed_kv_cache_dtype(args)

    kv_offloading_backend_categories = effective_kv_offloading_backend_categories(args)
    fixed_kv_backend = (
        SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_BACKEND
        if args.kv_offloading_backend is None
        else str(args.kv_offloading_backend)
    )
    if tune_param_enabled(args, "kv_offloading_backend"):
        if args.kv_offloading_backend is None:
            kv_offloading_backend_default = (
                str(kv_offloading_backend_categories[0])
                if kv_offloading_backend_categories
                else "native"
            )
        elif args.kv_offloading_backend in kv_offloading_backend_categories or not kv_offloading_backend_categories:
            kv_offloading_backend_default = str(args.kv_offloading_backend)
        else:
            kv_offloading_backend_default = str(kv_offloading_backend_categories[0])
    else:
        kv_offloading_backend_default = fixed_kv_backend

    # 2. CacheBlend 和 speculative 默认值要与搜索空间候选保持一致。
    cacheblend_categories = [
        normalize_cacheblend_mode(value)
        for value in args.cacheblend_mode_categories
    ]
    requested_cacheblend_mode = normalize_cacheblend_mode(args.cacheblend_mode)
    if tune_param_enabled(args, "cacheblend_mode"):
        cacheblend_mode_default = (
            requested_cacheblend_mode
            if requested_cacheblend_mode in cacheblend_categories or not cacheblend_categories
            else cacheblend_categories[0]
        )
    else:
        cacheblend_mode_default = requested_cacheblend_mode

    speculative_method = str(args.speculative_method or "none")
    if tune_param_enabled(args, "speculative_method"):
        speculative_method_default = (
            speculative_method
            if speculative_method in args.speculative_method_categories
            or not args.speculative_method_categories
            else str(args.speculative_method_categories[0])
        )
    else:
        speculative_method_default = speculative_method

    # 3. max_model_len 默认值必须能被默认 batched token budget 覆盖。
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
    max_model_len_default = (
        non_zero_max_model_lens[-1]
        if non_zero_max_model_lens
        else (feasible_max_model_lens[-1] if feasible_max_model_lens else max_model_len_values[0])
    )

    defaults = {
        "tp": int(getattr(args, "default_tp", 1)),
        "data_parallel_size": int(getattr(args, "default_data_parallel_size", 1)),
        "quantization": "none",
        "dtype": dtype_default,
        "model_impl": str(getattr(args, "model_impl", "vllm")),
        "decode_context_parallel_size": int(
            getattr(args, "decode_context_parallel_size", 1)
        ),
        "cp_kv_cache_interleave_size": int(
            getattr(args, "cp_kv_cache_interleave_size", 1)
        ),
        "data_parallel_backend": str(getattr(args, "data_parallel_backend", "mp")),
        "disable_nccl_for_dp_synchronization": bool(
            getattr(args, "disable_nccl_for_dp_synchronization", False)
        ),
        "cpu_offload_gb": (
            float(args.cpu_offload_gb)
            if args.cpu_offload_gb is not None
            else SAFE_REFERENCE_DEFAULT_CPU_OFFLOAD_GB
        ),
        "kv_cache_dtype": kv_cache_dtype_default,
        "kv_offloading_backend": kv_offloading_backend_default,
        "kv_offloading_size": (
            normalized_offload_size(
                args.kv_offloading_size
                if args.kv_offloading_size is not None
                else SAFE_REFERENCE_DEFAULT_KV_OFFLOADING_SIZE
            )
        ),
        "cacheblend_mode": cacheblend_mode_default,
        "lmcache_chunk_size": int(args.lmcache_chunk_size),
        "cacheblend_blend_min_tokens": int(args.cacheblend_blend_min_tokens),
        "cacheblend_blend_check_layers": canonicalize_int_list_spec(
            args.cacheblend_blend_check_layers
        ),
        "cacheblend_blend_recompute_ratios": (
            canonicalize_float_list_spec(args.cacheblend_blend_recompute_ratios)
        ),
        "speculative_method": speculative_method_default,
        "num_speculative_tokens": (
            None
            if speculative_method_default == "none"
            else (
                int(args.num_speculative_tokens)
                if args.num_speculative_tokens is not None
                else int(args.num_speculative_tokens_lb)
            )
        ),
        "max_num_seqs": int(getattr(args, "default_max_num_seqs", 128)),
        "max_num_batched_tokens": int(
            getattr(args, "default_max_num_batched_tokens", 4096)
        ),
        "block_size": int(getattr(args, "default_block_size", 16)),
        "long_prefill_token_threshold": int(
            getattr(args, "long_prefill_token_threshold", 0)
        ),
        "max_model_len": max_model_len_default,
        "gpu_memory_utilization": gpu_memory_utilization_default,
        "enforce_eager": False,
        "enable_chunked_prefill": chunked_prefill_default,
        "enable_prefix_caching": False,
        "prefix_caching_hash_algo": str(
            getattr(args, "prefix_caching_hash_algo", "sha256")
        ),
        "scheduler_reserve_full_isl": bool(
            getattr(args, "scheduler_reserve_full_isl", True)
        ),
        "async_scheduling": bool(getattr(args, "async_scheduling", False)),
        "disable_cascade_attn": bool(
            getattr(args, "disable_cascade_attn", True)
        ),
        "enable_flashinfer_autotune": bool(
            getattr(args, "enable_flashinfer_autotune", False)
        ),
        "disable_custom_all_reduce": False,
    }
    return defaults[name]


def build_design_space(args: argparse.Namespace, gpu_nums: int, max_sequence_length: int) -> tuple[DesignSpace, int]:
    from .._hebo.design_space.design_space import DesignSpace

    min_world_size = int(get_ref_config("min_world_size"))
    if min_world_size > gpu_nums:
        min_world_size = gpu_nums
    initialize_default_search_values(args, min_world_size, gpu_nums, max_sequence_length)
    screened_specs = load_screened_parameter_results(args)
    if screened_specs:
        para_dict: list[dict[str, Any]] = []
        for name in screened_specs:
            parameter = executable_screened_space_parameter(
                args, name, gpu_nums=gpu_nums
            )
            if parameter is None:
                continue
            # TP 与 DP 在本场景中是一个联合拓扑自由度；DP 在执行前由 TP 派生。
            if name == "data_parallel_size" and "tp" in screened_specs:
                getattr(args, "_screened_space_audit")[name]["represented_by"] = "tp"
                continue
            para_dict.append(parameter)
        if not para_dict:
            raise ValueError("screened parameter results produced an empty BO space")
        return DesignSpace().parse(para_dict), min_world_size

    tuned_params = selected_tune_params(args)
    max_num_batched_tokens_ub = args.max_num_batched_tokens_ub
    if max_num_batched_tokens_ub is None:
        max_num_batched_tokens_ub = max(8192, max_sequence_length * 2)
    include_prefix_caching_dim = allow_prefix_caching_tuning(args, tuned_params)
    para_dict: list[dict[str, Any]] = []

    # 1. 并行与量化/精度相关搜索维度。
    if tune_param_enabled(args, "tp"):
        para_dict.append(
            {
                "name": "tp",
                "type": "int_exponent",
                "lb": min_world_size,
                "ub": gpu_nums,
                "base": 2,
            }
        )
    if tune_param_enabled(args, "data_parallel_size"):
        dp_lb = max(1, int(args.data_parallel_size_lb))
        dp_ub = (
            gpu_nums
            if args.data_parallel_size_ub is None
            else min(int(args.data_parallel_size_ub), gpu_nums)
        )
        if not tune_param_enabled(args, "tp"):
            fixed_tp = int(default_space_value("tp", args))
            dp_ub = min(dp_ub, max(1, gpu_nums // fixed_tp))
        if dp_lb > dp_ub:
            raise ValueError(
                "data_parallel_size search space is empty after GPU budget "
                f"constraint: lb={dp_lb}, ub={dp_ub}, gpu_nums={gpu_nums}"
            )
        para_dict.append(
            {
                "name": "data_parallel_size",
                "type": "int_exponent",
                "lb": dp_lb,
                "ub": dp_ub,
                "base": 2,
            }
        )
    if tune_param_enabled(args, "quantization"):
        para_dict.append(
            {
                "name": "quantization",
                "type": "cat",
                "categories": ["none", "awq_marlin", "gptq_marlin"],
            }
        )
    if tune_param_enabled(args, "dtype"):
        para_dict.append(
            {
                "name": "dtype",
                "type": "cat",
                "categories": effective_dtype_categories(args),
            }
        )
    if tune_param_enabled(args, "cpu_offload_gb"):
        para_dict.append(
            {
                "name": "cpu_offload_gb",
                "type": "step_int",
                "lb": args.cpu_offload_gb_lb,
                "ub": args.cpu_offload_gb_ub,
                "step": args.cpu_offload_gb_step,
            }
        )
    if tune_param_enabled(args, "kv_cache_dtype"):
        para_dict.append(
            {
                "name": "kv_cache_dtype",
                "type": "cat",
                "categories": effective_kv_cache_dtype_categories(args),
            }
        )
    if tune_param_enabled(args, "kv_offloading_backend"):
        kv_offloading_backend_categories = effective_kv_offloading_backend_categories(args)
        para_dict.append(
            {
                "name": "kv_offloading_backend",
                "type": "cat",
                "categories": kv_offloading_backend_categories,
            }
        )
    if tune_param_enabled(args, "kv_offloading_size"):
        para_dict.append(
            {
                "name": "kv_offloading_size",
                "type": "step_int",
                "lb": args.kv_offloading_size_lb,
                "ub": args.kv_offloading_size_ub,
                "step": args.kv_offloading_size_step,
            }
        )
    # 2. CacheBlend / LMCache 相关搜索维度。
    if tune_param_enabled(args, "cacheblend_mode"):
        cacheblend_categories = [
            normalize_cacheblend_mode(value)
            for value in args.cacheblend_mode_categories
        ]
        cacheblend_categories = list(dict.fromkeys(cacheblend_categories))
        if not cacheblend_categories:
            cacheblend_categories = [CACHEBLEND_EFFECTIVE_DEFAULT_MODE]
        para_dict.append(
            {
                "name": "cacheblend_mode",
                "type": "cat",
                "categories": cacheblend_categories,
            }
        )
    if tune_param_enabled(args, "lmcache_chunk_size"):
        para_dict.append(
            {
                "name": "lmcache_chunk_size",
                "type": "int_exponent",
                "lb": args.lmcache_chunk_size_lb,
                "ub": args.lmcache_chunk_size_ub,
                "base": 2,
            }
        )
    if tune_param_enabled(args, "cacheblend_blend_min_tokens"):
        para_dict.append(
            {
                "name": "cacheblend_blend_min_tokens",
                "type": "step_int",
                "lb": args.cacheblend_blend_min_tokens_lb,
                "ub": args.cacheblend_blend_min_tokens_ub,
                "step": args.cacheblend_blend_min_tokens_step,
            }
        )
    if tune_param_enabled(args, "cacheblend_blend_check_layers"):
        para_dict.append(
            {
                "name": "cacheblend_blend_check_layers",
                "type": "cat",
                "categories": [
                    canonicalize_int_list_spec(value)
                    for value in args.cacheblend_blend_check_layers_categories
                ],
            }
        )
    if tune_param_enabled(args, "cacheblend_blend_recompute_ratios"):
        para_dict.append(
            {
                "name": "cacheblend_blend_recompute_ratios",
                "type": "cat",
                "categories": [
                    canonicalize_float_list_spec(value)
                    for value in args.cacheblend_blend_recompute_ratios_categories
                ],
            }
        )
    # 3. speculative decoding 搜索维度。
    if tune_param_enabled(args, "speculative_method"):
        para_dict.append(
            {
                "name": "speculative_method",
                "type": "cat",
                "categories": args.speculative_method_categories,
            }
        )
    if tune_param_enabled(args, "num_speculative_tokens"):
        para_dict.append(
            {
                "name": "num_speculative_tokens",
                "type": "int",
                "lb": args.num_speculative_tokens_lb,
                "ub": args.num_speculative_tokens_ub,
            }
        )
    # 4. vLLM scheduler / prefill / memory 相关核心搜索维度。
    para_dict.append(
        {
            "name": "max_num_seqs",
            "type": "int_exponent",
            "lb": (
                args.max_num_seqs_lb
                if tune_param_enabled(args, "max_num_seqs")
                else default_space_value("max_num_seqs", args)
            ),
            "ub": (
                args.max_num_seqs_ub
                if tune_param_enabled(args, "max_num_seqs")
                else default_space_value("max_num_seqs", args)
            ),
            "base": 2,
        }
    )
    para_dict.append(
        {
            "name": "max_num_batched_tokens",
            "type": "pow_int",
            "lb": (
                args.max_num_batched_tokens_lb
                if tune_param_enabled(args, "max_num_batched_tokens")
                else default_space_value("max_num_batched_tokens", args)
            ),
            "ub": (
                max_num_batched_tokens_ub
                if tune_param_enabled(args, "max_num_batched_tokens")
                else default_space_value("max_num_batched_tokens", args)
            ),
            "base": 2,
        }
    )

    if tune_param_enabled(args, "block_size"):
        para_dict.append(
            {
                "name": "block_size",
                "type": "int_exponent",
                "lb": int(args.block_size_lb),
                "ub": int(args.block_size_ub),
                "base": 2,
            }
        )
    if tune_param_enabled(args, "long_prefill_token_threshold"):
        threshold_lb = int(args.long_prefill_token_threshold_lb)
        threshold_ub = int(args.long_prefill_token_threshold_ub)
        if threshold_lb < 0 or threshold_ub < threshold_lb:
            raise ValueError(
                "invalid long_prefill_token_threshold bounds: "
                f"lb={threshold_lb}, ub={threshold_ub}"
            )
        threshold_values = [0]
        threshold_exponent = math.ceil(math.log2(max(1, threshold_lb)))
        while 2**threshold_exponent <= threshold_ub:
            threshold_values.append(2**threshold_exponent)
            threshold_exponent += 1
        para_dict.append(
            {
                "name": "long_prefill_token_threshold",
                "type": "cat",
                "categories": threshold_values,
            }
        )
    if tune_param_enabled(args, "max_model_len"):
        max_model_len_lb = int(args.max_model_len_lb)
        max_model_len_ub = int(args.max_model_len_ub)
        if max_model_len_lb > max_model_len_ub:
            raise ValueError(
                f"max_model_len_lb={max_model_len_lb} must be <= "
                f"max_model_len_ub={max_model_len_ub}"
            )
        if max_model_len_lb <= 0:
            max_model_len_values = [0]
            exponent = math.ceil(math.log2(1))
            while 2**exponent <= max_model_len_ub:
                max_model_len_values.append(2**exponent)
                exponent += 1
            para_dict.append(
                {
                    "name": "max_model_len",
                    "type": "cat",
                    "categories": max_model_len_values,
                }
            )
        else:
            para_dict.append(
                {
                    "name": "max_model_len",
                    "type": "int_exponent",
                    "lb": max_model_len_lb,
                    "ub": max_model_len_ub,
                    "base": 2,
                }
            )
    if tune_param_enabled(args, "gpu_memory_utilization"):
        para_dict.append(
            {
                "name": "gpu_memory_utilization",
                "type": "step_int",
                "lb": args.gpu_memory_utilization_lb,
                "ub": args.gpu_memory_utilization_ub,
                "step": args.gpu_memory_utilization_step,
            }
        )
    if tune_param_enabled(args, "enforce_eager"):
        para_dict.append({"name": "enforce_eager", "type": "bool"})

    prefix_caching_default = default_space_value("enable_prefix_caching", args)
    if include_prefix_caching_dim:
        para_dict.append({"name": "enable_prefix_caching", "type": "bool"})
    if tune_param_enabled(args, "disable_custom_all_reduce"):
        para_dict.append({"name": "disable_custom_all_reduce", "type": "bool"})
    return DesignSpace().parse(para_dict), min_world_size


# 返回某个搜索空间字段的默认值；用于 reference、历史补齐和固定维度。
