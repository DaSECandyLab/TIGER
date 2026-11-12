"""Flock BO 的历史观测加载和清理。"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .._hebo.design_space.design_space import DesignSpace
from .parameters import *
from .search_space import default_space_value

QUERY_OBJECTIVE_SCOPE = "query"
QUERY_OBJECTIVE_VERSION = "query_v1"
QUERY_OBJECTIVE_FIELDS = (
    "query_throughput_qps",
    "mean_query_execution_latency_ms",
    "p95_query_execution_latency_ms",
)


def has_query_objective(result: dict[str, Any], num_obj: int = 3) -> bool:
    if result.get("objective_metric_scope") != QUERY_OBJECTIVE_SCOPE:
        return False
    if result.get("objective_metric_version") != QUERY_OBJECTIVE_VERSION:
        return False
    performance_ready = all(
        value is not None and not (isinstance(value, float) and np.isnan(value))
        for value in (result.get(field) for field in QUERY_OBJECTIVE_FIELDS)
    )
    if not performance_ready:
        return False
    return True


def query_objective_vector(result: dict[str, Any], num_obj: int = 3) -> list[float]:
    """按固定顺序构造性能最小化目标，并按 ``num_obj`` 截取。"""
    if not 1 <= int(num_obj) <= 3:
        raise ValueError(f"num_obj must be in [1, 3], got {num_obj}")
    if not has_query_objective(result, num_obj):
        raise KeyError(
            "result does not contain query-level objectives "
            f"({QUERY_OBJECTIVE_SCOPE}/{QUERY_OBJECTIVE_VERSION})"
        )
    objectives = [
        -float(result["query_throughput_qps"]),
        float(result["mean_query_execution_latency_ms"]),
        float(result["p95_query_execution_latency_ms"]),
    ]
    return objectives[: int(num_obj)]

def rec_value(rec: pd.DataFrame, name: str, default: Any = None) -> Any:
    """从单行 HEBO 建议中读取指定名称的值。"""
    if name not in rec.columns:
        return default
    return rec[name].tolist()[0]

def sanitize_history_value_for_space(
    name: str,
    value: Any,
    space: DesignSpace,
    args: argparse.Namespace,
) -> Any:
    """把历史值收敛到当前搜索空间可接受的取值。

    旧 trial 结果可能包含：
    - 在当前空间里已被裁掉的类别值
    - cacheblend 关闭后被写成 None 的占位值
    - 早期记录里留下的缺失数值

    这些值如果直接喂给 HEBO，会在 `space.transform()` 阶段触发
    NaN / KeyError，进而让 BO 在 suggest() 里崩掉。
    """
    if name == "gpu_memory_utilization" and value is not None and not pd.isna(value):
        numeric_value = float(value)
        value = int(round(numeric_value * 100)) if numeric_value <= 1.0 else int(round(numeric_value))
    if name == "max_model_len":
        value = default_space_value(name, args) if value is None or pd.isna(value) else int(value)
    elif name == "num_speculative_tokens" and (value is None or pd.isna(value)):
        value = default_space_value(name, args)

    param = space.paras[name]
    fallback = default_space_value(name, args)
    fallback_missing = fallback is None or pd.isna(fallback)
    if fallback_missing:
        if param.is_categorical:
            fallback = param.categories[0]
        else:
            fallback = param.inverse_transform(np.asarray([param.opt_lb]))[0]

    value_missing = value is None or pd.isna(value)
    if value_missing:
        return fallback

    if param.is_categorical:
        if value not in param.categories:
            return fallback
        return value

    return value


# 判断历史 trial 是否匹配当前未调参数的固定默认值，不匹配则不能复用。
def historical_row_matches_defaults(
    row: dict[str, Any], args: argparse.Namespace
) -> bool:
    combination_defaults = default_combination(args)
    fixed_defaults = {
        "tp": default_space_value("tp", args),
        "data_parallel_size": default_space_value("data_parallel_size", args),
        "quantization": default_space_value("quantization", args),
        "dtype": default_space_value("dtype", args),
        "model_impl": default_space_value("model_impl", args),
        "decode_context_parallel_size": default_space_value(
            "decode_context_parallel_size", args
        ),
        "cp_kv_cache_interleave_size": default_space_value(
            "cp_kv_cache_interleave_size", args
        ),
        "data_parallel_backend": default_space_value(
            "data_parallel_backend", args
        ),
        "disable_nccl_for_dp_synchronization": default_space_value(
            "disable_nccl_for_dp_synchronization", args
        ),
        "cpu_offload_gb": default_space_value("cpu_offload_gb", args),
        "kv_cache_dtype": default_space_value("kv_cache_dtype", args),
        "kv_offloading_backend": default_space_value(
            "kv_offloading_backend", args
        ),
        "kv_offloading_size": default_space_value("kv_offloading_size", args),
        "cacheblend_mode": default_space_value("cacheblend_mode", args),
        "lmcache_chunk_size": default_space_value("lmcache_chunk_size", args),
        "cacheblend_blend_min_tokens": default_space_value(
            "cacheblend_blend_min_tokens", args
        ),
        "cacheblend_blend_check_layers": default_space_value(
            "cacheblend_blend_check_layers", args
        ),
        "cacheblend_blend_recompute_ratios": default_space_value(
            "cacheblend_blend_recompute_ratios", args
        ),
        "speculative_method": combination_defaults["speculative_method"],
        "num_speculative_tokens": combination_defaults["num_speculative_tokens"],
        "max_num_seqs": default_space_value("max_num_seqs", args),
        "max_num_batched_tokens": default_space_value("max_num_batched_tokens", args),
        "block_size": default_space_value("block_size", args),
        "long_prefill_token_threshold": default_space_value(
            "long_prefill_token_threshold", args
        ),
        "max_model_len": combination_defaults["max_model_len"],
        "gpu_memory_utilization": default_space_value("gpu_memory_utilization", args),
        "enforce_eager": default_space_value("enforce_eager", args),
        "enable_chunked_prefill": default_space_value("enable_chunked_prefill", args),
        "enable_prefix_caching": default_space_value("enable_prefix_caching", args),
        "prefix_caching_hash_algo": default_space_value(
            "prefix_caching_hash_algo", args
        ),
        "scheduler_reserve_full_isl": default_space_value(
            "scheduler_reserve_full_isl", args
        ),
        "async_scheduling": default_space_value("async_scheduling", args),
        "disable_cascade_attn": default_space_value("disable_cascade_attn", args),
        "enable_flashinfer_autotune": default_space_value(
            "enable_flashinfer_autotune", args
        ),
        "disable_custom_all_reduce": default_space_value(
            "disable_custom_all_reduce", args
        ),
    }
    for key, expected in fixed_defaults.items():
        if tune_param_enabled(args, key):
            continue
        if row.get(key) != expected:
            return False
    return True


# 过滤历史配置，只保留当前搜索空间中的参数列，并修正非法历史值。
def filter_history_to_space(
    xx: list[dict[str, Any]], space: DesignSpace, args: argparse.Namespace
) -> list[dict[str, Any]]:
    return [
        {
            key: sanitize_history_value_for_space(
                key,
                row.get(key, default_space_value(key, args)),
                space,
                args,
            )
            for key in space.para_names
        }
        for row in xx
    ]


# 从已有 vllm-*.json 结果中读取可复用的 BO 历史样本。
def read_flock_historical_data(
    res_dir_path: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    default_values = default_combination(args)
    xx = []
    yy = []
    history_files: list[str] = []
    # 1. 收集当前实验目录下所有历史 trial 结果文件。
    for root, dirs, files in os.walk(res_dir_path):
        dirs.sort()
        for name in sorted(files):
            if name.startswith("vllm"):
                history_files.append(os.path.join(root, name))
    # 2. 逐个读取历史结果，并转换成当前搜索空间的 row/objective。
    for file_path in sorted(history_files):
        with open(file_path, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        if not has_query_objective(res, args.num_obj):
            continue
        speculative_config = res.get("speculative_config") or {}
        speculative_method = str(
            speculative_config.get(
                "method", args.speculative_method or "none"
            )
        )
        speculative_tokens = speculative_config.get("num_speculative_tokens")
        if speculative_method == "none":
            speculative_tokens = None
        elif speculative_tokens is None:
            speculative_tokens = default_space_value(
                "num_speculative_tokens", args
            )
        row = {
            "tp": res["tp"],
            "data_parallel_size": int(
                res.get(
                    "data_parallel_size",
                    default_values["data_parallel_size"],
                )
            ),
            "quantization": str(res.get("quantization", "none")),
            "dtype": res.get("dtype", default_values["dtype"]),
            "model_impl": str(
                res.get("model_impl", default_values["model_impl"])
            ),
            "decode_context_parallel_size": int(
                res.get(
                    "decode_context_parallel_size",
                    default_values["decode_context_parallel_size"],
                )
            ),
            "cp_kv_cache_interleave_size": int(
                res.get(
                    "cp_kv_cache_interleave_size",
                    default_values["cp_kv_cache_interleave_size"],
                )
            ),
            "data_parallel_backend": str(
                res.get(
                    "data_parallel_backend", default_values["data_parallel_backend"]
                )
            ),
            "disable_nccl_for_dp_synchronization": bool_value(
                res.get(
                    "disable_nccl_for_dp_synchronization",
                    default_values["disable_nccl_for_dp_synchronization"],
                )
            ),
            "cpu_offload_gb": (
                float(res.get("cpu_offload_gb"))
                if res.get("cpu_offload_gb") is not None
                else default_values["cpu_offload_gb"]
            ),
            "kv_cache_dtype": res.get(
                "kv_cache_dtype", default_values["kv_cache_dtype"]
            ),
            "kv_offloading_backend": res.get(
                "kv_offloading_backend",
                default_values["kv_offloading_backend"],
            ),
            "kv_offloading_size": (
                normalized_offload_size(res.get("kv_offloading_size"))
                if res.get("kv_offloading_size") is not None
                else default_values["kv_offloading_size"]
            ),
            "cacheblend_mode": str(
                res.get("cacheblend_mode", default_values["cacheblend_mode"])
            ),
            "lmcache_chunk_size": int(
                res.get("lmcache_chunk_size", default_values["lmcache_chunk_size"])
            ),
            "cacheblend_blend_min_tokens": int(
                res.get(
                    "cacheblend_blend_min_tokens",
                    default_values["cacheblend_blend_min_tokens"],
                )
            ),
            "cacheblend_blend_check_layers": canonicalize_int_list_spec(
                res.get(
                    "cacheblend_blend_check_layers",
                    default_values["cacheblend_blend_check_layers"],
                )
            ),
            "cacheblend_blend_recompute_ratios": canonicalize_float_list_spec(
                res.get(
                    "cacheblend_blend_recompute_ratios",
                    default_values["cacheblend_blend_recompute_ratios"],
                )
            ),
            "speculative_method": speculative_method,
            "num_speculative_tokens": speculative_tokens,
            "max_num_seqs": res["max_num_seqs"],
            "max_num_batched_tokens": res["max_num_batched_tokens"],
            "block_size": res["block_size"],
            "long_prefill_token_threshold": int(
                res.get(
                    "long_prefill_token_threshold",
                    default_values["long_prefill_token_threshold"],
                )
            ),
            "max_model_len": (
                default_space_value("max_model_len", args)
                if res.get("max_model_len") is None or pd.isna(res.get("max_model_len"))
                else int(res.get("max_model_len"))
            ),
            "gpu_memory_utilization": (
                int(round(float(res.get("gpu_memory_utilization")) * 100))
                if res.get("gpu_memory_utilization") is not None
                else default_values["gpu_memory_utilization"]
            ),
            "enforce_eager": bool_value(res.get("enforce_eager", False)),
            "enable_chunked_prefill": optional_bool_value(
                res.get("enable_chunked_prefill")
            ),
            "enable_prefix_caching": optional_bool_value(
                res.get("enable_prefix_caching")
            ),
            "prefix_caching_hash_algo": str(
                res.get(
                    "prefix_caching_hash_algo",
                    default_values["prefix_caching_hash_algo"],
                )
            ),
            "scheduler_reserve_full_isl": bool_value(
                res.get(
                    "scheduler_reserve_full_isl",
                    default_values["scheduler_reserve_full_isl"],
                )
            ),
            "async_scheduling": bool_value(
                res.get("async_scheduling", default_values["async_scheduling"])
            ),
            "disable_cascade_attn": bool_value(
                res.get(
                    "disable_cascade_attn",
                    default_values["disable_cascade_attn"],
                )
            ),
            "enable_flashinfer_autotune": bool_value(
                res.get(
                    "enable_flashinfer_autotune",
                    default_values["enable_flashinfer_autotune"],
                )
            ),
            "disable_custom_all_reduce": bool_value(res["disable_custom_all_reduce"]),
            "scheduler_delay_factor": int(
                float(res.get("scheduler_delay_factor", 0.0)) * 10
            ),
            "use_v2_block_manager": bool_value(
                res.get("use_v2_block_manager", False)
            ),
        }
        row = normalize_conditional_combination(row, args)
        if not historical_row_matches_defaults(row, args):
            continue
        xx.append(row)
        yy.append(query_objective_vector(res, args.num_obj))
    return xx, yy
