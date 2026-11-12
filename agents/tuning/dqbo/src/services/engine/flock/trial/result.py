"""Trial 结果聚合和 workload 特征产物写入。"""

from __future__ import annotations

import argparse
from SemBench.src.flockmtl_timing import query_execution_timing
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    resolve_effective_dtype,
    resolve_effective_kv_cache_dtype,
)
from ..features.workload import (
    categorize_workload_features,
    extract_workload_features,
)
from .arguments import (
    effective_query_ids,
    normalize_query_id,
    resolve_kv_transfer_config,
    resolve_model_path,
    resolve_speculative_config,
    resolved_chunked_prefill_setting,
    resolved_prefix_caching_setting,
)
from .metrics import (
    METRICS_PARSER_VERSION,
    PromSample,
    delta_metric_any,
    parse_lmcache_hit_summary,
    summarize_vllm_latency_metrics,
)

DEFAULT_VLLM_GENERATION_CONFIG = "vllm"
STRUCTURED_OUTPUTS_CONFIG = "{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
QUERY_OBJECTIVE_SCOPE = "query"
QUERY_OBJECTIVE_VERSION = "query_v1"


def recorded_query_ids(
    args: argparse.Namespace, workload_metrics: dict[str, Any]
) -> list[str]:
    """Include the driver's initial phase as well as terminal query sequences."""
    initial = (
        workload_metrics.get("initial_query_sequence", [])
        if args.flock_workload_driver == "tpcc"
        else []
    )
    return list(dict.fromkeys(
        normalize_query_id(query)
        for query in [*initial, *effective_query_ids(args)]
    ))


def collect_query_instance_records(
    args: argparse.Namespace,
    sembench_metrics: dict,
    workload_metrics: dict,
) -> list[dict[str, Any]]:
    """收集用于 query 级目标统计的样本记录。

    TPCC workload 使用 terminal/query-instance 级记录；
    legacy SemBench 路径退化为每个 query 一条记录。
    """
    if args.flock_workload_driver == "tpcc":
        terminals = workload_metrics.get("terminals")
        if isinstance(terminals, list):
            query_records: list[dict[str, Any]] = []
            for terminal in terminals:
                if not isinstance(terminal, dict):
                    continue
                queries = terminal.get("queries")
                if not isinstance(queries, list):
                    continue
                query_records.extend(
                    query
                    for query in queries
                    if isinstance(query, dict)
                )
            return query_records
    return [
        row for row in sembench_metrics.values() if isinstance(row, dict)
    ]


def summarize_query_objectives(
    query_records: list[dict[str, Any]],
    wall_time_s: float,
) -> dict[str, Any]:
    """从 query-instance 样本构造 query 级优化目标。"""
    successful_query_count = 0
    failed_query_count = 0
    skipped_query_count = 0
    execution_latencies_s: list[float] = []

    for query_record in query_records:
        status = str(query_record.get("status", "unknown"))
        if status == "success":
            successful_query_count += 1
            execution_time = query_record.get("execution_time")
            if execution_time is not None:
                execution_latencies_s.append(float(execution_time))
        elif status == "skipped":
            skipped_query_count += 1
        else:
            failed_query_count += 1

    mean_execution_latency_s = (
        float(sum(execution_latencies_s) / len(execution_latencies_s))
        if execution_latencies_s
        else None
    )
    p95_execution_latency_s = (
        float(np.percentile(np.asarray(execution_latencies_s, dtype=float), 95))
        if execution_latencies_s
        else None
    )

    return {
        "objective_metric_scope": QUERY_OBJECTIVE_SCOPE,
        "objective_metric_version": QUERY_OBJECTIVE_VERSION,
        "query_throughput_qps": (
            float(successful_query_count / wall_time_s) if wall_time_s > 0 else 0.0
        ),
        "mean_query_execution_latency_s": mean_execution_latency_s,
        "mean_query_execution_latency_ms": (
            mean_execution_latency_s * 1000.0
            if mean_execution_latency_s is not None
            else None
        ),
        "p95_query_execution_latency_s": p95_execution_latency_s,
        "p95_query_execution_latency_ms": (
            p95_execution_latency_s * 1000.0
            if p95_execution_latency_s is not None
            else None
        ),
        "query_latency_sample_count": len(execution_latencies_s),
        "successful_query_count": successful_query_count,
        "failed_query_count": failed_query_count,
        "skipped_query_count": skipped_query_count,
    }


def build_result(
    args: argparse.Namespace,
    sembench_metrics: dict,
    sembench_summary: dict,
    workload_metrics: dict,
    before_samples: list[PromSample],
    after_samples: list[PromSample],
    wall_time_s: float = 0.0,
    sembench_returncode: int = 0,
    logs: dict[str, str] | None = None,
    runtime_sample_summary: dict[str, float | None] | None = None,
) -> dict:
    # Enforce the execution window at aggregation, even when called directly
    # with an uncorrected process duration (not only via the trial runner).
    timing = query_execution_timing(
        workload_metrics.get("terminals", []), workload_metrics.get("started_at"),
        workload_metrics.get("completed_at"))
    if timing:
        wall_time_s = timing["execution_wall_time_s"]
    elif workload_metrics.get("timing_status") == "unavailable":
        wall_time_s = 0.0
    runtime_sample_summary = runtime_sample_summary or {}
    logs = logs or {}
    kv_transfer_config = resolve_kv_transfer_config(args)
    speculative_config = resolve_speculative_config(args)
    effective_dtype = resolve_effective_dtype(
        args.dtype,
        quantization=args.quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )
    effective_kv_cache_dtype = resolve_effective_kv_cache_dtype(
        args.kv_cache_dtype,
        dtype=effective_dtype,
        quantization=args.quantization,
        model_path=args.model_path,
        awq_model_path=args.awq_model_path,
        gptq_model_path=args.gptq_model_path,
    )
    query_objectives = summarize_query_objectives(
        collect_query_instance_records(args, sembench_metrics, workload_metrics),
        wall_time_s,
    )
    request_success_names = [
        "vllm:request_success",
        "vllm:request_success_total",
    ]
    request_stop_delta = delta_metric_any(
        before_samples,
        after_samples,
        request_success_names,
        {"finished_reason": "stop"},
    )
    request_length_delta = delta_metric_any(
        before_samples,
        after_samples,
        request_success_names,
        {"finished_reason": "length"},
    )
    request_error_delta = delta_metric_any(
        before_samples,
        after_samples,
        request_success_names,
        {"finished_reason": "error"},
    )
    request_total_delta = delta_metric_any(
        before_samples,
        after_samples,
        request_success_names,
    )
    prompt_tokens = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:prompt_tokens", "vllm:prompt_tokens_total"],
    )
    generation_tokens = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:generation_tokens", "vllm:generation_tokens_total"],
    )
    latency_metrics = summarize_vllm_latency_metrics(
        before_samples,
        after_samples,
        completed_request_count=request_total_delta,
        generation_tokens=generation_tokens,
    )
    prefix_queries = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"],
    )
    prefix_hits = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"],
    )
    external_prefix_queries = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_queries_total"],
    )
    external_prefix_hits = delta_metric_any(
        before_samples,
        after_samples,
        ["vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_hits_total"],
    )
    lmcache_hit_summary = parse_lmcache_hit_summary(Path(logs["vllm"])) if logs.get("vllm") else {}
    prefix_cache_hit_rate = (prefix_hits / prefix_queries) if prefix_queries > 0 else None
    external_prefix_cache_hit_rate = (
        external_prefix_hits / external_prefix_queries
        if external_prefix_queries > 0
        else None
    )
    lmcache_log_cache_hit_rate = lmcache_hit_summary.get("lmcache_cache_hit_rate")
    # 对 workload 特征来说，LMCache 命中率优先取 LMCache 自己日志里的 token 级
    # 命中；如果日志不可用，再退回 vLLM external-prefix counter。后者仍然是
    # KV connector/LMCache 视角，不和本地 prefix cache 混在一起。
    lmcache_cache_hit_rate = (
        lmcache_log_cache_hit_rate
        if lmcache_log_cache_hit_rate is not None
        else external_prefix_cache_hit_rate
    )

    runtime_metrics = {
        # 这里把 workload 特征提取器需要的 query 级运行指标、vLLM 请求/令牌增量
        # 和 SemBench 质量摘要；这里把这些随 trial 变化的信号统一打包。
        **query_objectives,
        "flock_request_rate": str(args.flock_request_rate),
        "flock_request_arrival": str(args.flock_request_arrival),
        "request_success_stop_delta": request_stop_delta,
        "request_success_total_delta": request_total_delta,
        **latency_metrics,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
        "lmcache_cache_hit_rate": lmcache_cache_hit_rate,
        # Flock 根据真实 request start/completion 事件重建的动态全局在途峰值；
        # 它属于运行时观测，不与静态 max_in_flight 配置上限混用。
        "actual_global_in_flight_peak": workload_metrics.get(
            "global_in_flight_peak"
        ),
        **runtime_sample_summary,
        "sembench_summary": sembench_summary,
    }
    runtime_metrics["tokenizer_path"] = resolve_model_path(args)
    workload_features = extract_workload_features(
        workload_metrics=workload_metrics,
        runtime_metrics=runtime_metrics,
    )
    # 扁平版本供 BO/隐藏约束模型使用；分组版本按建模方案给人工检查。
    workload_feature_groups = categorize_workload_features(workload_features)
    long_prefill_token_threshold = getattr(
        args, "long_prefill_token_threshold", None
    )

    result = {
        # 与 SCOOT 兼容的参数/结果字段。
        "tp": args.tp,
        # None 表示使用 vLLM 官方自动值；显式值则属于固定实验基础设施。
        "vllm_api_server_count": (
            None
            if getattr(args, "vllm_api_server_count", None) is None
            else int(args.vllm_api_server_count)
        ),
        "data_parallel_size": args.data_parallel_size,
        "model_impl": getattr(args, "model_impl", "vllm"),
        "decode_context_parallel_size": int(
            getattr(args, "decode_context_parallel_size", 1)
        ),
        "cp_kv_cache_interleave_size": int(
            getattr(args, "cp_kv_cache_interleave_size", 1)
        ),
        "data_parallel_backend": getattr(args, "data_parallel_backend", "mp"),
        "disable_nccl_for_dp_synchronization": bool(
            getattr(args, "disable_nccl_for_dp_synchronization", False)
        ),
        "quantization": args.quantization,
        "dtype": effective_dtype,
        "resolved_model_path": resolve_model_path(args),
        "generation_config": DEFAULT_VLLM_GENERATION_CONFIG,
        "structured_outputs_config": STRUCTURED_OUTPUTS_CONFIG,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "long_prefill_token_threshold": (
            None
            if long_prefill_token_threshold is None
            else int(long_prefill_token_threshold)
        ),
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": (
            float(args.gpu_memory_utilization)
            if args.gpu_memory_utilization is not None
            else None
        ),
        "enforce_eager": bool(args.enforce_eager),
        "enable_chunked_prefill": resolved_chunked_prefill_setting(args),
        "scheduler_delay_factor": float(args.scheduler_delay_factor),
        "enable_prefix_caching": resolved_prefix_caching_setting(args),
        "prefix_caching_hash_algo": getattr(
            args, "prefix_caching_hash_algo", "sha256"
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
        "disable_custom_all_reduce": bool(args.disable_custom_all_reduce),
        "use_v2_block_manager": bool(args.use_v2_block_manager),
        "cpu_offload_gb": (
            float(args.cpu_offload_gb)
            if args.cpu_offload_gb is not None
            else None
        ),
        "kv_cache_dtype": effective_kv_cache_dtype,
        "kv_cache_dtype_skip_layers": args.kv_cache_dtype_skip_layers,
        "kv_offloading_backend": args.kv_offloading_backend,
        "kv_offloading_size": (
            float(args.kv_offloading_size)
            if args.kv_offloading_size is not None
            else None
        ),
        "cacheblend_mode": args.cacheblend_mode,
        "lmcache_chunk_size": int(args.lmcache_chunk_size),
        "cacheblend_blend_special_str": args.cacheblend_blend_special_str,
        "cacheblend_blend_min_tokens": int(args.cacheblend_blend_min_tokens),
        "cacheblend_blend_check_layers": args.cacheblend_blend_check_layers,
        "cacheblend_blend_recompute_ratios": (
            args.cacheblend_blend_recompute_ratios
        ),
        "kv_transfer_config": kv_transfer_config,
        "speculative_config": speculative_config,
        "speculative_method": speculative_config.get("method") if speculative_config else None,
        "num_speculative_tokens": speculative_config.get("num_speculative_tokens") if speculative_config else None,
        **query_objectives,
        "metrics_parser_version": METRICS_PARSER_VERSION,
        "request_throughput": (
            request_total_delta / wall_time_s if wall_time_s > 0 else 0.0
        ),
        **latency_metrics,
        # 这里是 Flock/SemBench 分析用额外字段。
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "request_success_total_delta": request_total_delta,
        "request_success_stop_delta": request_stop_delta,
        "request_success_length_delta": request_length_delta,
        "request_success_error_delta": request_error_delta,
        "prefix_cache_queries_delta": prefix_queries,
        "prefix_cache_hits_delta": prefix_hits,
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
        "external_prefix_cache_queries_delta": external_prefix_queries,
        "external_prefix_cache_hits_delta": external_prefix_hits,
        "external_prefix_cache_hit_rate": external_prefix_cache_hit_rate,
        "lmcache_hit_summary": lmcache_hit_summary,
        "lmcache_log_cache_hit_rate": lmcache_log_cache_hit_rate,
        "lmcache_cache_hit_rate": lmcache_cache_hit_rate,
        "runtime_sample_summary": runtime_sample_summary,
        "wall_time_s": wall_time_s,
        "execution_timing_scope": timing.get("timing_scope", "legacy_fallback"),
        "excluded_preparation_time_s": timing.get("preparation_time_s"),
        "excluded_teardown_time_s": timing.get("teardown_time_s"),
        "workload_total_wall_time_s": workload_metrics.get("total_wall_time_s", workload_metrics.get("wall_time_s")),
        "sembench_returncode": sembench_returncode,
        "sembench_use_case": args.use_case,
        "sembench_files_override_root": getattr(
            args, "sembench_files_override_root", None
        ),
        "sembench_summary": sembench_summary,
        "sembench_metrics": sembench_metrics,
        "flock_workload_driver": args.flock_workload_driver,
        "flock_workload_metrics": workload_metrics,
        # 扁平 workload 特征保持向后兼容，便于 rec_history 和随机森林直接消费。
        "flock_workload_features": workload_features,
        # 分组 workload 特征按方案分类，并标注静态、画像、运行时和质量等提取范围。
        "flock_workload_feature_groups": workload_feature_groups,
        "flock_queries": recorded_query_ids(args, workload_metrics),
        "flock_terminal_queries": (
            [
                [normalize_query_id(query) for query in query_sequence]
                for query_sequence in args.flock_terminal_queries
            ]
            if args.flock_terminal_queries is not None
            else None
        ),
        "flock_request_rate": str(args.flock_request_rate),
        "flock_request_arrival": str(args.flock_request_arrival),
        "flock_max_in_flight": args.flock_max_in_flight,
        "flock_max_in_flight_strategy": getattr(
            args,
            "flock_max_in_flight_strategy",
            "dynamic_active_terminal_partition",
        ),
        # HTTP 连接预算是固定的 workload 基础设施配置，不是 vLLM 调优旋钮。
        "flock_http_max_connections": int(
            getattr(args, "flock_http_max_connections", 2048)
        ),
        "flock_query_concurrency": args.flock_query_concurrency,
        "flock_query_start_stagger_ms": args.flock_query_start_stagger_ms,
        "flock_terminal_count": args.flock_terminal_count,
        "flock_request_control_mode": args.flock_request_control_mode,
        "flock_vllm_batch_size": args.flock_vllm_batch_size,
        "flock_failure_policy": args.flock_failure_policy,
        # 保存 baseline Agent 的配置来源，使后续 workload feature 能反查
        # “这次测量到底由谁、基于哪些输入生成”。
        "baseline_config_path": getattr(args, "baseline_config", None),
        "baseline_config_metadata": getattr(
            args, "_baseline_config_metadata", None
        ),
        # 完整实际启动命令是复现实验的权威记录，不再依赖固定字段 schema。
        "vllm_command": list(getattr(args, "_effective_vllm_command", [])),
        "vllm_started_by_trial": bool(
            getattr(args, "_vllm_started_by_trial", False)
        ),
        "logs": logs,
    }
    return result


def _format_feature_value(value: Any) -> str:
    """把特征值格式化成适合 Markdown 表格阅读的短文本。"""
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value)


def write_workload_feature_files(
    result: dict[str, Any],
    *,
    feature_json_path: Path,
    feature_doc_path: Path,
) -> None:
    """单独落盘 workload 特征，方便人读。

    完整 result JSON 仍保留 `flock_workload_features` 和分组字段，供 BO/脚本消费；
    这里额外生成 feature-only JSON 和 Markdown，避免读者在大结果文件里翻找。
    """
    features = result.get("flock_workload_features") or {}
    groups = result.get("flock_workload_feature_groups") or {}
    feature_json_path.parent.mkdir(parents=True, exist_ok=True)
    feature_doc_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_result": result.get("trial_result_path"),
                "features": features,
                "groups": groups,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    group_titles = [
        ("benchmark_config_features", "Benchmark / Workload 配置特征"),
        ("query_features", "Query 特征"),
        ("data_features", "Data 特征"),
        ("runtime_features", "Runtime 特征"),
        ("quality_features", "Quality 特征"),
    ]
    lines = [
        "# Workload Feature Summary",
        "",
        f"- Trial: `{result.get('trial_name')}`",
        f"- Result JSON: `{result.get('trial_result_path')}`",
        f"- Workload driver: `{result.get('flock_workload_driver')}`",
        f"- Queries: `{','.join(result.get('flock_queries') or [])}`",
        "",
    ]
    for group_key, title in group_titles:
        group = groups.get(group_key)
        if not isinstance(group, dict):
            continue
        lines.extend([f"## {title}", "", "| 字段 | 值 |", "| --- | --- |"])
        for key, value in group.items():
            lines.append(f"| `{key}` | {_format_feature_value(value)} |")
        lines.append("")

    lines.extend(["## 扁平特征", "", "| 字段 | 值 |", "| --- | --- |"])
    for key in sorted(features):
        lines.append(f"| `{key}` | {_format_feature_value(features[key])} |")
    lines.append("")
    missing = groups.get("missing_or_partial_features") or {}
    if missing:
        lines.extend(["## 缺失或不适用特征", "", "| 字段 | 原因 |", "| --- | --- |"])
        for key, value in missing.items():
            lines.append(f"| `{key}` | {_format_feature_value(value)} |")
        lines.append("")
    feature_doc_path.write_text("\n".join(lines), encoding="utf-8")
