"""Flock BO 的质量门禁和隐藏约束。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..flock.features.workload import extract_workload_features
from .parameters import *

SCOOT_ROOT = Path(__file__).resolve().parents[2]
HIDDEN_CONSTRAINT_CATEGORICAL_FEATURES = frozenset(
    {
        "quantization", "dtype", "model_impl", "kv_cache_dtype", "kv_offloading_backend",
        "data_parallel_backend", "distributed_executor_backend", "prefix_caching_hash_algo",
        "cacheblend_mode", "cacheblend_blend_check_layers",
        "cacheblend_blend_recompute_ratios", "speculative_method",
    }
)

STATIC_WORKLOAD_CONTEXT_KEYS = frozenset(
    {
        "terminal_count",
        "request_control_mode",
        "max_in_flight",
        "arrival_mode",
        "request_rate",
        "concurrent_query_mix",
        "terminal_query_slot_count",
        "terminal_unique_query_count",
        "query_instance_count",
        "total_llm_request_count",
        "mean_llm_requests_per_query",
        "p50_llm_requests_per_query",
        "p95_llm_requests_per_query",
        "max_llm_requests_per_query",
        "candidate_rows_p50",
        "candidate_rows_p90",
        "candidate_rows_max",
        "theoretical_global_in_flight_upper",
        "mean_wave_count",
        "max_wave_count",
        "mean_configured_request_cap",
        "max_configured_request_cap",
        "system_prompt_len_distribution",
        "system_prompt_len_p50",
        "system_prompt_len_p95",
        "system_prompt_len_max",
        "data_prompt_len_distribution",
        "data_prompt_len_p50",
        "data_prompt_len_p95",
        "data_prompt_len_max",
        "output_len_distribution",
        "output_len_p50",
        "output_len_p95",
        "output_len_max",
        "template_reuse_count",
        "template_reuse_template_count",
        "template_reuse_max_count",
        "prefix_reuse_ratio",
        "shared_prefix_len_distribution",
        "shared_prefix_len_p50",
        "shared_prefix_len_p90",
        "shared_prefix_len_max",
        "shared_cacheblend_len_distribution",
        "shared_cacheblend_len_p50",
        "shared_cacheblend_len_p90",
        "shared_cacheblend_len_max",
        "cacheblend_reuse_ratio",
        "operator_mix",
        "llm_request_count_per_query",
    }
)

RES_DIR_PREFIX = "scoot_flock"
RES_DIR = SCOOT_ROOT / "tune_res"

REPOSITORY_ROOT = Path(__file__).resolve().parents[7]

DEFAULT_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct')
DEFAULT_AWQ_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-AWQ')
DEFAULT_GPTQ_INT4_MODEL_PATH = str(REPOSITORY_ROOT / 'models/Qwen2-7B-Instruct-GPTQ-Int4')
DEFAULT_MODEL_NAME = "Qwen2-7B-Instruct"
DEFAULT_VLLM_BIN = str(REPOSITORY_ROOT / 'vllm-0.20.0/.venv/bin/vllm')
DEFAULT_SEMBENCH_DIR = str(REPOSITORY_ROOT / 'SemBench')
DEFAULT_FLOCK_EXT = (
    str(REPOSITORY_ROOT / 'flock/build/release/extension/flock/flock.duckdb_extension')
)

DEFAULT_TUNED_PARAMS = frozenset(
    {
        "tp",
        "data_parallel_size",
        "quantization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "enable_prefix_caching",
        "disable_custom_all_reduce",
    }
)
FOCUSED_30_TUNED_PARAMS = frozenset(
    {
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "max_model_len",
        "gpu_memory_utilization",
        "enable_prefix_caching",
    }
)
TUNE_PARAM_CHOICES = (
    "tp",
    "data_parallel_size",
    "quantization",
    "dtype",
    "cpu_offload_gb",
    "kv_cache_dtype",
    "kv_offloading_backend",
    "kv_offloading_size",
    "cacheblend_mode",
    "lmcache_chunk_size",
    "cacheblend_blend_min_tokens",
    "cacheblend_blend_check_layers",
    "cacheblend_blend_recompute_ratios",
    "speculative_method",
    "num_speculative_tokens",
    "max_num_seqs",
    "max_num_batched_tokens",
    "block_size",
    "max_model_len",
    "gpu_memory_utilization",
    "enforce_eager",
    "enable_prefix_caching",
    "disable_custom_all_reduce",
)
LEGACY_TUNE_FLAG_TO_PARAMS = {
    "tune_gpu_memory_utilization": {"gpu_memory_utilization"},
    "tune_max_model_len": {"max_model_len"},
    "tune_enforce_eager": {"enforce_eager"},
    "tune_cacheblend_mode": {"cacheblend_mode"},
    "tune_kv_cache_dtype": {"kv_cache_dtype"},
    "tune_speculative_decoding": {"speculative_method", "num_speculative_tokens"},
}
CACHEBLEND_DEPENDENT_PARAMS = frozenset(
    {
        "lmcache_chunk_size",
        "cacheblend_blend_min_tokens",
        "cacheblend_blend_check_layers",
        "cacheblend_blend_recompute_ratios",
    }
)
CACHEBLEND_EFFECTIVE_DEFAULT_MODE = "off"
CACHEBLEND_MODES = frozenset({"off", "on"})
QUERY_OBJECTIVE_SCOPE = "query"
QUERY_OBJECTIVE_VERSION = "query_v1"
QUERY_OBJECTIVE_FIELDS = (
    "query_throughput_qps",
    "mean_query_execution_latency_ms",
    "p95_query_execution_latency_ms",
)
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



def is_static_workload_feature_key(key: str) -> bool:
    return key in STATIC_WORKLOAD_CONTEXT_KEYS or key.startswith("operator_mix_")


# 获取 workload 静态特征上下文；第一轮前用 CLI 合成，后续用真实 trace 缓存。
def workload_feature_context(args: argparse.Namespace) -> dict[str, Any]:
    # 这里只服务于 rec_history 的记录字段：第一轮成功 trial 前只能用命令行参数
    # 合成默认上下文；拿到第一次真实画像后，会用更完整的静态画像覆盖这些字段。
    # 注意：该上下文不进入隐藏约束随机森林，也不影响 BO 的候选选择。
    cached = getattr(args, "_static_workload_feature_context", None)
    if isinstance(cached, dict):
        return cached
    terminal_sequences = None
    if args.flock_terminal_queries is not None:
        terminal_sequences = {
            str(index): [int(str(query).lstrip("Qq")) for query in query_sequence]
            for index, query_sequence in enumerate(args.flock_terminal_queries)
        }
    elif args.queries:
        shared_queries = [int(str(query).lstrip("Qq")) for query in args.queries]
        terminal_sequences = {
            str(index): shared_queries
            for index in range(int(args.flock_terminal_count))
        }

    cached = extract_workload_features(
        workload_metrics={
            "terminal_count": int(args.flock_terminal_count),
            "terminal_sql_sequences": terminal_sequences,
            "request_control_mode": args.flock_request_control_mode,
            "max_in_flight": (
                args.flock_max_in_flight
                if isinstance(args.flock_max_in_flight, int)
                else None
            ),
            "request_cap_strategy": getattr(
                args,
                "flock_max_in_flight_strategy",
                "dynamic_active_terminal_partition",
            ),
        },
        runtime_metrics={
            "flock_request_rate": str(args.flock_request_rate),
            "flock_request_arrival": str(args.flock_request_arrival),
        },
    )
    setattr(args, "_static_workload_feature_context", cached)
    return cached


# 用第一份成功 trial 的 workload features 更新静态 workload 画像缓存。
def update_static_workload_feature_context(
    args: argparse.Namespace,
    trial_features: dict[str, Any] | None,
) -> dict[str, Any]:
    context = dict(workload_feature_context(args))
    if not isinstance(trial_features, dict):
        return context
    if getattr(args, "_static_workload_profile_captured", False):
        return context

    # 静态 workload 特征只在第一次完整 profile 结果返回后写入缓存。后续 trial
    # 写 rec_history 时复用这些 query/data/prompt 画像；每轮自己的运行时和质量
    # 指标仍保留当前 trial 的值。这个缓存只影响记录，不参与调优决策。
    for key, value in trial_features.items():
        if is_static_workload_feature_key(key):
            context[key] = value
    setattr(args, "_static_workload_feature_context", context)
    setattr(args, "_static_workload_profile_captured", True)
    return context


# 组合写入 rec_history 的 workload features：静态字段复用缓存，动态字段取当前 trial。
def compose_recorded_workload_features(
    args: argparse.Namespace,
    trial_features: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(trial_features, dict):
        return None
    static_context = update_static_workload_feature_context(args, trial_features)
    recorded_features = dict(trial_features)
    for key, value in static_context.items():
        if is_static_workload_feature_key(key):
            recorded_features[key] = value
    return recorded_features


# 返回隐藏约束随机森林使用的全部特征列名。
def hidden_constraint_feature_names(args: argparse.Namespace) -> list[str]:
    return list(default_combination(args).keys())


# 将单个隐藏约束特征规整成随机森林可编码的数值或类别值。
def _hidden_constraint_feature_value(name: str, value: Any) -> Any:
    if name in HIDDEN_CONSTRAINT_CATEGORICAL_FEATURES:
        return "__none__" if value is None else str(value)
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return -1.0
    return value


# 把一个完整候选配置转换成隐藏约束模型的一行输入特征。
def combination_to_hidden_constraint_features(
    combination: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    normalized = normalize_conditional_combination(combination, args)
    feature_names = hidden_constraint_feature_names(args)
    return {
        name: _hidden_constraint_feature_value(name, normalized.get(name))
        for name in feature_names
    }


# 批量把历史候选配置转换成隐藏约束模型输入 DataFrame。
def rows_to_hidden_constraint_features(
    rows: Any,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        raw_df = rows.copy()
    else:
        raw_df = pd.DataFrame.from_records(rows)

    feature_rows = []
    for row in raw_df.to_dict(orient="records"):
        combination = rec_to_combination(pd.DataFrame.from_records([row]), args)
        feature_rows.append(combination_to_hidden_constraint_features(combination, args))

    feature_names = hidden_constraint_feature_names(args)
    return pd.DataFrame.from_records(feature_rows, columns=feature_names)


# 检查 trial 输出 JSON 是否与本轮候选配置一致，防止读到旧结果。
def combination_matches_result(combination: dict[str, Any], result: dict[str, Any]) -> bool:
    checks = {
        "tp": int(result.get("tp", -1)),
        "data_parallel_size": int(result.get("data_parallel_size", -1)),
        "quantization": str(result.get("quantization", "none")),
        "dtype": result.get("dtype"),
        "cpu_offload_gb": (
            float(result["cpu_offload_gb"])
            if result.get("cpu_offload_gb") is not None
            else None
        ),
        "kv_cache_dtype": result.get("kv_cache_dtype"),
        "kv_offloading_backend": result.get("kv_offloading_backend"),
        "kv_offloading_size": (
            normalized_offload_size(result.get("kv_offloading_size"))
            if result.get("kv_offloading_size") is not None
            else None
        ),
        "cacheblend_mode": str(result.get("cacheblend_mode", CACHEBLEND_EFFECTIVE_DEFAULT_MODE)),
        "lmcache_chunk_size": int(result.get("lmcache_chunk_size", 16)),
        "cacheblend_blend_min_tokens": int(
            result.get("cacheblend_blend_min_tokens", 16)
        ),
        "cacheblend_blend_check_layers": canonicalize_int_list_spec(
            result.get("cacheblend_blend_check_layers", [1])
        ),
        "cacheblend_blend_recompute_ratios": canonicalize_float_list_spec(
            result.get("cacheblend_blend_recompute_ratios", [0.15])
        ),
        "speculative_method": str(
            (result.get("speculative_config") or {}).get("method", "none")
        ),
        "num_speculative_tokens": optional_int(
            (result.get("speculative_config") or {}).get("num_speculative_tokens")
        ),
        "max_num_seqs": int(result.get("max_num_seqs", -1)),
        "max_num_batched_tokens": int(result.get("max_num_batched_tokens", -1)),
        "block_size": int(result.get("block_size", -1)),
        "max_model_len": optional_int(result.get("max_model_len")),
        "gpu_memory_utilization": (
            float(result["gpu_memory_utilization"])
            if result.get("gpu_memory_utilization") is not None
            else None
        ),
        "enforce_eager": bool_value(result.get("enforce_eager", False)),
        "enable_chunked_prefill": optional_bool_value(
            result.get("enable_chunked_prefill")
        ),
        "scheduler_delay_factor": float(result.get("scheduler_delay_factor", -1.0)),
        "enable_prefix_caching": optional_bool_value(
            result.get("enable_prefix_caching")
        ),
        "disable_custom_all_reduce": bool_value(result.get("disable_custom_all_reduce", False)),
        "use_v2_block_manager": bool_value(result.get("use_v2_block_manager", False)),
    }
    for key, expected in combination.items():
        actual = checks[key]
        if isinstance(expected, float):
            if abs(expected - actual) > 1e-9:
                return False
        elif expected != actual:
            return False
    return True


# 根据可选质量门禁判断本轮 trial 是否应视为失败。
def quality_gate_failed(result: dict[str, Any], args: argparse.Namespace) -> str | None:
    summary = result.get("sembench_summary") or {}
    if int(summary.get("failed_query_count") or 0) > 0:
        return f"SemBench failed_query_count={summary.get('failed_query_count')}"
    if getattr(args, "quality_thresholds_json", None) is not None:
        if summary.get("quality_gate_passed") is not True:
            evaluation = (
                result.get("flock_workload_metrics", {}).get("evaluation", {})
            )
            gate = evaluation.get("quality_gate", {})
            failures = gate.get("failed_constraints") or []
            detail = "; ".join(str(value) for value in failures[:3])
            return "per-query quality gate failed" + (f": {detail}" if detail else "")
    return None


class HiddenConstraintRandomForest:
    """用完整归一化 trial 配置拟合隐藏约束。"""

    # 初始化随机森林模型和 one-hot 后的特征列记录。
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.regressor = RandomForestRegressor()
        self.encoded_feature_columns: list[str] = []

    # 将原始候选行转换成隐藏约束模型使用的 DataFrame。
    def _to_dataframe(self, rows: Any) -> pd.DataFrame:
        return rows_to_hidden_constraint_features(rows, self.args)

    # 对类别特征做 one-hot 编码，并在预测时对齐训练时的列集合。
    def _encode_features(self, rows: Any, *, fit: bool = False) -> np.ndarray:
        df = self._to_dataframe(rows)
        encoded = pd.get_dummies(
            df,
            columns=[
                name
                for name in hidden_constraint_feature_names(self.args)
                if name in HIDDEN_CONSTRAINT_CATEGORICAL_FEATURES
            ],
            dtype=float,
        )
        if fit:
            self.encoded_feature_columns = encoded.columns.tolist()
        elif self.encoded_feature_columns:
            encoded = encoded.reindex(columns=self.encoded_feature_columns, fill_value=0.0)
        return encoded.to_numpy(dtype=float, copy=False)

    # 训练隐藏约束随机森林，标签为 trial 成功或失败。
    def fit(self, rows: Any, y: list[int]) -> HiddenConstraintRandomForest:
        self.regressor.fit(self._encode_features(rows, fit=True), y)
        return self

    # 预测候选配置通过隐藏约束的可能性。
    def predict(self, rows: Any) -> np.ndarray:
        return self.regressor.predict(self._encode_features(rows))


# 基于成功/失败历史训练隐藏约束随机森林，用于过滤高风险候选。
def fit_hidden_constraint_model(
    rec_history: list[dict[str, Any]],
    space: DesignSpace,
    args: argparse.Namespace,
) -> HiddenConstraintRandomForest | None:
    x = []
    y = []
    for data_item in rec_history:
        executed_combination = data_item.get("executed_combination")
        if executed_combination is None:
            executed_combination = rec_to_combination(
                pd.DataFrame.from_records(data_item["rec"]),
                args,
            )
        x.append(executed_combination)
        y.append(0 if data_item["obj"] is None else 1)
    if x and y and len(x) >= (len(x[0]) - 1):
        rfr = HiddenConstraintRandomForest(args)
        rfr.fit(x, y)
        return rfr
    return None


# 构造 HEBO 搜索空间；所有“哪些参数参与调优”的逻辑集中在这里。
