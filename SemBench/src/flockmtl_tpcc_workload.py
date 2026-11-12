#!/usr/bin/env python3
"""TPC-C style terminal workload driver for SemBench FlockMTL queries."""

from __future__ import annotations

import argparse
from query_ids import parse_query_id, query_id_sort_key
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
from flockmtl_timing import query_execution_timing
import sys
import tempfile
import time
from typing import Any

from dotenv import load_dotenv
import pandas as pd

# Add src directory to Python path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import get_evaluator
from flockmtl_cacheblend import CACHEBLEND_MODE_ENV, normalize_cacheblend_mode


DEFAULT_ACTIVE_TERMINAL_REFRESH_INTERVAL = 10
MAX_IN_FLIGHT_STRATEGY_ENV = "FLOCK_TPCC_MAX_IN_FLIGHT_STRATEGY"
DEFAULT_MAX_IN_FLIGHT_STRATEGY = "dynamic_active_terminal_partition"
GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY = "global_slot_coordination"
FIXED_EQUAL_PARTITION_MAX_IN_FLIGHT_STRATEGY = (
    "fixed_equal_terminal_partition"
)
MAX_IN_FLIGHT_STRATEGIES = (
    DEFAULT_MAX_IN_FLIGHT_STRATEGY,
    GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY,
    FIXED_EQUAL_PARTITION_MAX_IN_FLIGHT_STRATEGY,
)
HIGHER_IS_BETTER_QUALITY_METRICS = frozenset(
    {
        "precision",
        "recall",
        "spearman_correlation",
        "ranking_item_recall",
        "accuracy",
        "within1_accuracy",
    }
)
REPRESENTATIVE_QUALITY_METRICS = frozenset(
    {
        # Retrieval: set correctness and result coverage.
        "precision",
        "recall",
        # Aggregation: raw magnitude and scale-relative error.
        "absolute_error",
        "relative_error",
        # Ranking: order agreement and item coverage.
        "spearman_correlation",
        "ranking_item_recall",
        # Evaluators that expose only a task-specific accuracy.
        "accuracy",
        "within1_accuracy",
    }
)
RETRIEVAL_QUALITY_METRICS = frozenset({"precision", "recall", "f1_score"})
AGGREGATION_QUALITY_METRICS = frozenset(
    {"absolute_error", "relative_error", "mean_absolute_percentage_error"}
)
RANKING_QUALITY_METRICS = frozenset(
    {
        "spearman_correlation",
        "kendall_tau",
        "ranking_item_precision",
        "ranking_item_recall",
        "ranking_item_f1_score",
    }
)


def parse_query_sequence(query_args: list[str]) -> list[int | str]:
    """Preserve suffixed IDs such as Q3a and the order of terminal steps."""
    return [parse_query_id(arg) for arg in query_args]


def parse_terminal_query_sequences(
    terminal_query_args: list[list[str]],
) -> list[list[int]]:
    """解析按 terminal 传入的 query 序列。"""
    return [parse_query_sequence(query_args) for query_args in terminal_query_args]


def build_terminal_query_sequences(
    terminal_count: int,
    query_args: list[str] | None,
    terminal_query_args: list[list[str]] | None,
) -> list[list[int]]:
    """生成 `terminal_id -> query_id 序列` 映射。

    兼容两种输入方式：
    1. `--queries 1 2`：所有 terminal 共用同一条序列。
    2. `--terminal-queries 1 2 --terminal-queries 3 4 8`：
       每个 terminal 显式指定自己的序列。
    """
    if query_args and terminal_query_args:
        raise ValueError("queries and terminal-queries are mutually exclusive")

    if terminal_query_args:
        terminal_query_sequences = parse_terminal_query_sequences(
            terminal_query_args
        )
        if len(terminal_query_sequences) != terminal_count:
            raise ValueError(
                "terminal-queries count must match terminal-count "
                f"({len(terminal_query_sequences)} != {terminal_count})"
            )
        for terminal_id, query_ids in enumerate(terminal_query_sequences):
            if not query_ids:
                raise ValueError(
                    f"terminal-queries for terminal {terminal_id} must not be empty"
                )
        return terminal_query_sequences

    if not query_args:
        raise ValueError(
            "either queries or terminal-queries must be provided"
        )

    query_ids = parse_query_sequence(query_args)
    if not query_ids:
        raise ValueError("queries must not be empty")
    return [list(query_ids) for _ in range(terminal_count)]


def summarize_terminal_query_sequences(
    terminal_query_sequences: list[list[int]],
) -> tuple[list[int] | None, dict[str, list[int]]]:
    """整理 workload metrics 里的 query 序列字段。

    `terminal_sql_sequence` 只在所有 terminal 共用同一序列时保留，
    否则置空；`terminal_sql_sequences` 始终输出每个 terminal 的配置。
    """
    terminal_sql_sequences = {
        str(terminal_id): list(query_ids)
        for terminal_id, query_ids in enumerate(terminal_query_sequences)
    }
    if not terminal_query_sequences:
        return None, terminal_sql_sequences

    shared_query_sequence = list(terminal_query_sequences[0])
    if all(
        list(query_ids) == shared_query_sequence
        for query_ids in terminal_query_sequences[1:]
    ):
        return shared_query_sequence, terminal_sql_sequences
    return None, terminal_sql_sequences


def compute_requests_per_wave(terminal_count: int, vllm_batch_size: int) -> int:
    """按 `floor(vllm_batch_size / terminal_count)` 计算单 terminal 的请求上限。"""
    return max(1, math.floor(vllm_batch_size / max(1, terminal_count)))


def compute_equal_partition_request_cap(
    terminal_count: int, global_max_in_flight: int
) -> int:
    """按当前活跃 terminal 数等分全局预算。"""
    if terminal_count < 1:
        raise ValueError("terminal_count must be >= 1")
    if global_max_in_flight < terminal_count:
        raise ValueError(
            "global max_in_flight must be >= terminal_count for equal partition"
        )
    return global_max_in_flight // terminal_count


def make_terminal_coord_record(
    terminal_id: int,
    remaining_query_count: int,
    status: str = "active",
) -> dict[str, object]:
    """构造 terminal 协调状态。

    `remaining_query_count > 0` 表示该 terminal 后续仍可能继续向 vLLM 发请求，
    因此在 `vllm_batch_size / active_terminals` 模式下应计入分母。
    """
    return {
        "terminal_id": terminal_id,
        "remaining_query_count": max(0, int(remaining_query_count)),
        "status": status,
        "updated_at": time.time(),
    }


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """原子落盘协调状态，避免并发读到半写入文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temp_file:
        json.dump(payload, temp_file, indent=2)
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def build_flock_workload_env(
    base_env: dict[str, str],
    request_rate: str,
    request_arrival: str,
    request_control_mode: str,
    max_in_flight: int | None,
    requests_per_wave: int | None,
    global_coord_dir: str | None,
    terminal_count: int,
    vllm_batch_size: int | None,
    active_terminal_coord_dir: str | None,
    active_terminal_refresh_interval: int = DEFAULT_ACTIVE_TERMINAL_REFRESH_INTERVAL,
    max_in_flight_strategy: str = DEFAULT_MAX_IN_FLIGHT_STRATEGY,
) -> dict[str, str]:
    """构造 TPCC workload 运行时环境变量。

    这里把 terminal 级 workload 需要的请求控制参数下发给 terminal worker，
    并保证 TPCC 模式下单 terminal 仍然只串行执行一条 SQL。
    """
    env = dict(base_env)
    env["FLOCK_VLLM_REQUEST_RATE"] = str(request_rate)
    env["FLOCK_VLLM_REQUEST_ARRIVAL"] = str(request_arrival)
    env["FLOCKMTL_QUERY_CONCURRENCY"] = "1"
    env.pop("FLOCKMTL_QUERY_START_STAGGER_MS", None)
    env["FLOCK_TPCC_REQUEST_CONTROL_MODE"] = str(request_control_mode)
    env["FLOCK_TPCC_TOTAL_TERMINAL_COUNT"] = str(terminal_count)
    if request_control_mode == "vllm_batch_size_div_terminals":
        if requests_per_wave is None:
            raise ValueError(
                "requests_per_wave must be provided for "
                "vllm_batch_size_div_terminals mode"
            )
        if vllm_batch_size is None or vllm_batch_size < 1:
            raise ValueError(
                "vllm_batch_size must be provided for "
                "vllm_batch_size_div_terminals mode"
            )
        env["FLOCK_VLLM_MAX_IN_FLIGHT"] = str(requests_per_wave)
        env["FLOCK_TPCC_VLLM_BATCH_SIZE"] = str(vllm_batch_size)
        env["FLOCK_TPCC_ACTIVE_TERMINAL_REFRESH_INTERVAL"] = str(
            max(1, int(active_terminal_refresh_interval))
        )
        if active_terminal_coord_dir:
            env["FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR"] = str(
                active_terminal_coord_dir
            )
        else:
            env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR", None)
        env.pop("FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR", None)
        return env

    if max_in_flight is None:
        raise ValueError(
            "max_in_flight must be provided for max_in_flight control mode"
        )
    if max_in_flight_strategy == GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY:
        env["FLOCK_VLLM_MAX_IN_FLIGHT"] = str(max_in_flight)
        env["FLOCK_TPCC_REQUEST_CAP_STRATEGY"] = max_in_flight_strategy
        env.pop("FLOCK_TPCC_CONFIGURED_GLOBAL_MAX_IN_FLIGHT", None)
        env.pop("FLOCK_TPCC_VLLM_BATCH_SIZE", None)
        env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR", None)
        env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_REFRESH_INTERVAL", None)
        if global_coord_dir:
            env["FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR"] = str(global_coord_dir)
        else:
            env.pop("FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR", None)
        return env

    if max_in_flight_strategy not in {
        DEFAULT_MAX_IN_FLIGHT_STRATEGY,
        FIXED_EQUAL_PARTITION_MAX_IN_FLIGHT_STRATEGY,
    }:
        raise ValueError(
            f"unsupported max_in_flight strategy: {max_in_flight_strategy}"
        )
    terminal_request_cap = compute_equal_partition_request_cap(
        terminal_count=terminal_count,
        global_max_in_flight=max_in_flight,
    )
    env["FLOCK_VLLM_MAX_IN_FLIGHT"] = str(terminal_request_cap)
    env["FLOCK_TPCC_REQUEST_CAP_STRATEGY"] = max_in_flight_strategy
    env.pop("FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR", None)
    if (
        max_in_flight_strategy
        == FIXED_EQUAL_PARTITION_MAX_IN_FLIGHT_STRATEGY
    ):
        # 整轮固定按配置 terminal 数等分，不因某个 terminal 提前完成而扩容。
        env.pop("FLOCK_TPCC_CONFIGURED_GLOBAL_MAX_IN_FLIGHT", None)
        env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR", None)
        env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_REFRESH_INTERVAL", None)
        env.pop("FLOCK_TPCC_VLLM_BATCH_SIZE", None)
        return env

    env["FLOCK_TPCC_CONFIGURED_GLOBAL_MAX_IN_FLIGHT"] = str(max_in_flight)
    env["FLOCK_TPCC_ACTIVE_TERMINAL_REFRESH_INTERVAL"] = str(
        max(1, int(active_terminal_refresh_interval))
    )
    if active_terminal_coord_dir:
        env["FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR"] = str(
            active_terminal_coord_dir
        )
    else:
        env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR", None)
    env.pop("FLOCK_TPCC_VLLM_BATCH_SIZE", None)
    # 每个 terminal 在完成全部在途请求后才退出 active 集合。活跃 terminal
    # 使用同一个分母重新等分预算，份额总和始终不超过 global max_in_flight。
    return env


def build_terminal_worker_cmd(
    python_executable: str,
    worker_script: Path,
    system: str,
    use_case: str,
    query_ids: list[int],
    model: str,
    terminal_id: int,
    failure_policy: str,
    metric_output: Path,
    result_root: Path,
    query_trace_root: Path,
    scale_factor: int | None = None,
    skip_setup: bool = False,
) -> list[str]:
    """构造单个 terminal worker 的启动命令。"""
    cmd = [
        python_executable,
        str(worker_script),
        "--system",
        system,
        "--use-case",
        use_case,
        "--queries",
        *[str(query_id) for query_id in query_ids],
        "--model",
        model,
        "--terminal-id",
        str(terminal_id),
        "--failure-policy",
        failure_policy,
        "--metric-output",
        str(metric_output),
        "--result-root",
        str(result_root),
        "--query-trace-root",
        str(query_trace_root),
    ]
    if scale_factor is not None:
        cmd += ["--scale-factor", str(scale_factor)]
    if skip_setup:
        cmd.append("--skip-setup")
    return cmd


def summarize_terminal_metrics(terminals: list[dict[str, object]]) -> dict[str, object]:
    """汇总所有 terminal 的执行结果，生成 workload 级统计。"""
    total_sql = 0
    failed_sql = 0
    skipped_sql = 0
    total_execution_time = 0.0
    total_query_wall_time = 0.0
    total_wave_count = 0
    total_llm_request_count = 0
    request_mode_counts: dict[str, int] = {}
    function_counts: dict[str, int] = {}
    request_cap_source_counts: dict[str, int] = {}
    active_terminal_counts: list[int] = []
    configured_request_caps: list[int] = []
    dynamic_request_cap_refresh_count = 0

    for terminal in terminals:
        for query in terminal.get("queries", []):
            total_sql += 1
            status = query.get("status")
            if status == "failed":
                failed_sql += 1
            elif status == "skipped":
                skipped_sql += 1

            execution_time = query.get("execution_time")
            if execution_time is not None:
                total_execution_time += float(execution_time)
            query_wall_time = query.get("query_wall_time")
            if query_wall_time is not None:
                total_query_wall_time += float(query_wall_time)
            total_wave_count += int(query.get("wave_count") or 0)
            total_llm_request_count += int(query.get("total_llm_requests") or 0)

            request_mode = str(query.get("request_mode", "unknown"))
            request_mode_counts[request_mode] = (
                request_mode_counts.get(request_mode, 0) + 1
            )

            request_cap_source = query.get("request_cap_source")
            if request_cap_source is not None:
                request_cap_source_counts[str(request_cap_source)] = (
                    request_cap_source_counts.get(str(request_cap_source), 0) + 1
                )

            for active_terminal_count in (
                query.get("min_observed_active_terminal_count"),
                query.get("max_observed_active_terminal_count"),
                query.get("active_terminal_count_at_query_start"),
            ):
                if active_terminal_count is not None:
                    active_terminal_counts.append(int(active_terminal_count))

            for configured_request_cap in (
                query.get("min_observed_request_cap"),
                query.get("max_observed_request_cap"),
                query.get("configured_request_cap"),
                query.get("final_effective_request_cap"),
            ):
                if configured_request_cap is not None:
                    configured_request_caps.append(int(configured_request_cap))

            dynamic_request_cap_refresh_count += int(
                query.get("dynamic_request_cap_refresh_count") or 0
            )

            for function_name in query.get("batch_capable_functions", []):
                function_counts[function_name] = (
                    function_counts.get(function_name, 0) + 1
                )
            for function_name in query.get("non_batch_capable_functions", []):
                function_counts[function_name] = (
                    function_counts.get(function_name, 0) + 1
                )

    return {
        "query_count": total_sql,
        "failed_query_count": failed_sql,
        "skipped_query_count": skipped_sql,
        "successful_query_count": total_sql - failed_sql - skipped_sql,
        "failed_terminal_count": sum(
            1 for terminal in terminals if terminal.get("status") != "success"
        ),
        "successful_terminal_count": sum(
            1 for terminal in terminals if terminal.get("status") == "success"
        ),
        "total_execution_time": total_execution_time,
        "total_query_wall_time": total_query_wall_time,
        "total_wave_count": total_wave_count,
        "total_llm_request_count": total_llm_request_count,
        "request_mode_counts": request_mode_counts,
        "request_cap_source_counts": request_cap_source_counts,
        "max_active_terminal_count": (
            max(active_terminal_counts) if active_terminal_counts else None
        ),
        "min_active_terminal_count": (
            min(active_terminal_counts) if active_terminal_counts else None
        ),
        "max_configured_request_cap": (
            max(configured_request_caps) if configured_request_caps else None
        ),
        "min_configured_request_cap": (
            min(configured_request_caps) if configured_request_caps else None
        ),
        "dynamic_request_cap_refresh_count": dynamic_request_cap_refresh_count,
        "query_function_counts": function_counts,
    }


def load_terminal_metric(metric_path: Path) -> dict[str, object]:
    """读取单个 terminal worker 落盘的指标文件。"""
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing terminal metric output: {metric_path}")
    with metric_path.open("r", encoding="utf-8") as metric_file:
        return json.load(metric_file)


def make_json_safe(value: Any) -> Any:
    """将 workload / evaluate 结果里的 NaN、Inf 等值转换成可稳定写入 JSON 的形式。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


def load_query_results(result_output: Path) -> pd.DataFrame:
    """读取单个 query 实例的结果文件，供 evaluator 做质量评估。"""
    if not result_output.exists():
        raise FileNotFoundError(f"Missing query result output: {result_output}")
    return pd.read_pickle(result_output)


def ground_truth_row_count(ground_truth: Any) -> int:
    """MMQA returns a JSON path; other evaluators return in-memory rows."""
    if isinstance(ground_truth, (str, Path)):
        payload = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
        rows = payload["ground_truth"]
        if isinstance(rows, dict):
            return sum(len(values) for values in rows.values())
        return len(rows)
    return len(ground_truth)


def build_query_instance_key(
    terminal_id: int,
    step_index: int,
    query_id: int,
) -> str:
    """生成 query-instance 级别的稳定标识，避免不同 terminal 的同名 query 混淆。"""
    return f"terminal_{terminal_id}:step_{step_index}:Q{query_id}"


def average_metric_dicts(metric_dicts: list[dict[str, object]]) -> dict[str, float]:
    """对同类 evaluator 指标做字段级平均，用于按 query_id 聚合多个实例。"""
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for metric_dict in metric_dicts:
        for metric_name, metric_value in metric_dict.items():
            if isinstance(metric_value, bool):
                continue
            if isinstance(metric_value, (int, float)) and math.isfinite(
                float(metric_value)
            ):
                metric_sums[metric_name] = (
                    metric_sums.get(metric_name, 0.0) + float(metric_value)
                )
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
    return {
        metric_name: metric_sums[metric_name] / metric_counts[metric_name]
        for metric_name in metric_sums
        if metric_counts[metric_name] > 0
    }


def _finite_metric_values(
    metric_dicts: list[dict[str, object]], metric_name: str
) -> list[float]:
    values: list[float] = []
    for metric_dict in metric_dicts:
        value = metric_dict.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            values.append(numeric_value)
    return values


def aggregate_query_metric(
    metric_dicts: list[dict[str, object]], metric_name: str
) -> tuple[float | None, list[float] | None]:
    """Aggregate a metric, weighting classification accuracy by sample count."""

    values = _finite_metric_values(metric_dicts, metric_name)
    if len(values) != len(metric_dicts) or not values:
        return None, None
    if metric_name == "accuracy":
        weights = _finite_metric_values(metric_dicts, "classification_sample_count")
        if (
            len(weights) == len(values)
            and all(weight > 0 for weight in weights)
            and sum(weights) > 0
        ):
            return (
                sum(value * weight for value, weight in zip(values, weights))
                / sum(weights),
                weights,
            )
    return sum(values) / len(values), None


def representative_quality_metric_names(
    metric_dict: dict[str, object],
) -> frozenset[str]:
    """按 evaluator 任务族互斥选择约两个代表指标。"""
    metric_names = set(metric_dict)
    if {"accuracy", "classification_sample_count"}.issubset(metric_names):
        return frozenset({"accuracy"})
    if metric_names & RETRIEVAL_QUALITY_METRICS:
        return frozenset({"precision", "recall"})
    if metric_names & AGGREGATION_QUALITY_METRICS:
        return frozenset({"absolute_error", "relative_error"})
    if metric_names & RANKING_QUALITY_METRICS:
        return frozenset({"spearman_correlation", "ranking_item_recall"})
    if "accuracy" in metric_names:
        return frozenset({"accuracy"})
    return frozenset()


def build_metric_inventory(
    metric_dicts: list[dict[str, object]],
    expected_metric_count: int | None = None,
) -> dict[str, object]:
    """只记录原始指标覆盖情况，不跨 query 合成任何质量分数。"""
    metric_values: dict[str, list[float]] = {}
    for metric_dict in metric_dicts:
        for metric_name in representative_quality_metric_names(metric_dict):
            values = _finite_metric_values([metric_dict], metric_name)
            if values:
                metric_values.setdefault(metric_name, []).extend(values)
    metric_counts = {
        metric_name: len(values)
        for metric_name, values in metric_values.items()
        if values
    }
    recognized_metric_count = sum(
        1
        for metric_dict in metric_dicts
        if any(
            _finite_metric_values([metric_dict], metric_name)
            for metric_name in representative_quality_metric_names(metric_dict)
        )
    )
    denominator = (
        max(0, int(expected_metric_count))
        if expected_metric_count is not None
        else len(metric_dicts)
    )
    coverage = (
        min(1.0, recognized_metric_count / denominator)
        if denominator > 0
        else 0.0
    )
    return {
        "quality_metric_counts": metric_counts,
        "quality_metric_coverage": coverage,
    }


def parse_quality_thresholds(
    raw_thresholds: str | None,
) -> dict[str, dict[str, dict[str, float]]] | None:
    """解析逐 query、逐指标门禁；不允许任何跨指标综合规则。"""
    if raw_thresholds is None:
        return None
    payload = json.loads(raw_thresholds)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("quality thresholds must be a non-empty JSON object")

    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for raw_query_id, raw_metrics in payload.items():
        raw_id = str(raw_query_id).strip()
        if not raw_id.lower().startswith("q"):
            raise ValueError(f"invalid quality threshold query id: {raw_query_id}")
        query_id = f"Q{parse_query_id(raw_id)}"
        if not isinstance(raw_metrics, dict) or not raw_metrics:
            raise ValueError(f"{query_id} quality thresholds must be non-empty")
        normalized_metrics: dict[str, dict[str, float]] = {}
        for metric_name, raw_rule in raw_metrics.items():
            if metric_name not in REPRESENTATIVE_QUALITY_METRICS:
                raise ValueError(
                    f"non-representative quality metric for {query_id}: {metric_name}"
                )
            if not isinstance(raw_rule, dict) or set(raw_rule) not in ({"min"}, {"max"}):
                raise ValueError(
                    f"{query_id}.{metric_name} must contain exactly one of min/max"
                )
            operator_name = next(iter(raw_rule))
            expected_operator = (
                "min"
                if metric_name in HIGHER_IS_BETTER_QUALITY_METRICS
                else "max"
            )
            if operator_name != expected_operator:
                raise ValueError(
                    f"{query_id}.{metric_name} must use {expected_operator}"
                )
            threshold_value = raw_rule[operator_name]
            if isinstance(threshold_value, bool) or not isinstance(
                threshold_value, (int, float)
            ):
                raise ValueError(
                    f"{query_id}.{metric_name}.{operator_name} must be numeric"
                )
            threshold = float(threshold_value)
            if not math.isfinite(threshold):
                raise ValueError(
                    f"{query_id}.{metric_name}.{operator_name} must be finite"
                )
            normalized_metrics[metric_name] = {operator_name: threshold}
        normalized[query_id] = normalized_metrics
    return normalized


def evaluate_quality_thresholds(
    query_instance_metrics: list[dict[str, object]],
    thresholds: dict[str, dict[str, dict[str, float]]] | None,
) -> dict[str, object]:
    """逐 query、逐指标判断；最终结果只是所有门禁的逻辑 AND。"""
    if thresholds is None:
        return {
            "configured": False,
            "policy": "all_queries_all_configured_metrics",
            "passed": None,
            "queries": {},
            "failed_constraints": [],
        }

    query_records: dict[str, list[dict[str, object]]] = {}
    for record in query_instance_metrics:
        query_records.setdefault(f"Q{parse_query_id(record['query_id'])}", []).append(record)

    failures: list[str] = []
    query_results: dict[str, dict[str, object]] = {}
    all_query_ids = sorted(set(query_records) | set(thresholds))
    for query_id in all_query_ids:
        records = query_records.get(query_id, [])
        query_thresholds = thresholds.get(query_id)
        query_failures: list[str] = []
        metric_results: dict[str, dict[str, object]] = {}

        if not records:
            query_failures.append(f"{query_id}: query is missing")
        if query_thresholds is None:
            query_failures.append(f"{query_id}: thresholds are missing")

        successful_records = [
            record
            for record in records
            if record.get("evaluation_status") == "success"
        ]
        if len(successful_records) != len(records):
            query_failures.append(
                f"{query_id}: not every instance evaluated successfully"
            )

        configured_metric_names = set(query_thresholds or {})
        # Gate only the explicitly configured metrics.  This is intentionally
        # broader than the compact inventory above: for example Movie Q1-Q8
        # expose logical-sample ``accuracy`` alongside their legacy
        # retrieval/aggregation diagnostics, and an experiment may choose
        # accuracy as its sole gate metric.
        observed_metric_names = {
            metric_name
            for record in successful_records
            for metric_name in configured_metric_names
            if metric_name in dict(record.get("metrics") or {})
        }
        # Evaluators may expose useful diagnostic metrics that are intentionally
        # not part of this experiment's gate.  Only explicitly configured
        # metrics are constraints; unconfigured observations remain in the
        # result payload without affecting pass/fail.
        for metric_name in sorted(configured_metric_names - observed_metric_names):
            query_failures.append(f"{query_id}.{metric_name}: metric is missing")

        for metric_name in sorted(observed_metric_names & configured_metric_names):
            rule = dict((query_thresholds or {})[metric_name])
            operator_name, threshold = next(iter(rule.items()))
            metric_dicts = [
                dict(record.get("metrics") or {}) for record in successful_records
            ]
            values = _finite_metric_values(metric_dicts, metric_name)
            query_metric_value, weights = aggregate_query_metric(
                metric_dicts, metric_name
            )
            if operator_name == "min":
                passed = (
                    query_metric_value is not None
                    and query_metric_value >= threshold
                )
                operator = ">="
            else:
                passed = (
                    query_metric_value is not None
                    and abs(query_metric_value) <= threshold
                )
                operator = "<="
            if not passed:
                query_failures.append(
                    f"{query_id}.{metric_name}: query mean is not {operator} {threshold}"
                )
            metric_results[metric_name] = {
                "operator": operator,
                "threshold": threshold,
                "value": query_metric_value,
                "values": values,
                "weights": weights,
                "aggregation": (
                    "classification_sample_count_weighted_mean"
                    if weights is not None
                    else "arithmetic_mean"
                ),
                "passed": passed,
            }

        failures.extend(query_failures)
        query_results[query_id] = {
            "passed": not query_failures,
            "instance_count": len(records),
            "metrics": metric_results,
            "failures": query_failures,
        }

    return {
        "configured": True,
        "policy": "all_queries_all_configured_metrics",
        "passed": not failures,
        "queries": query_results,
        "failed_constraints": failures,
    }


def evaluate_query_instance(
    evaluator: object,
    query_record: dict[str, object],
    terminal_id: int,
) -> dict[str, object]:
    """对单个 terminal 的单次 query 执行 evaluator，并返回结构化评估结果。"""
    query_id = parse_query_id(query_record["query_id"])
    step_index = int(query_record["step_index"])
    query_instance_key = build_query_instance_key(terminal_id, step_index, query_id)
    query_status = str(query_record.get("status", "unknown"))
    result_output = query_record.get("result_output")
    base_record: dict[str, object] = {
        "query_instance_id": query_instance_key,
        "terminal_id": terminal_id,
        "step_index": step_index,
        "query_id": query_id,
        "query_status": query_status,
        "result_output": result_output,
    }

    # 只有真正成功且有结果文件的 query 才进入 evaluator；
    # failed / skipped query 仍保留一条 evaluation 记录，便于汇总时看见缺口。
    if query_status != "success":
        return {
            **base_record,
            "evaluation_status": "skipped",
            "evaluation_metric_type": None,
            "metrics": {},
            "error": f"Query status is {query_status}, evaluation skipped.",
        }
    if not result_output:
        return {
            **base_record,
            "evaluation_status": "skipped",
            "evaluation_metric_type": None,
            "metrics": {},
            "error": "Missing result_output, evaluation skipped.",
        }

    try:
        system_results = load_query_results(Path(str(result_output)))
        ground_truth = evaluator._get_ground_truth(query_id)  # noqa: SLF001
        metric = evaluator._evaluate_single_query(  # noqa: SLF001
            query_id,
            system_results,
            ground_truth,
        )
        additional_quality_hook = getattr(
            evaluator, "_additional_quality_metrics", None
        )
        additional_quality = (
            additional_quality_hook(query_id, system_results, ground_truth)
            if callable(additional_quality_hook)
            else {}
        )
        metric_dict = make_json_safe(
            {**dataclasses.asdict(metric), **dict(additional_quality or {})}
        )
        return {
            **base_record,
            "evaluation_status": "success",
            "evaluation_metric_type": type(metric).__name__,
            "metrics": metric_dict,
            "error": None,
            "result_row_count": len(system_results),
            "ground_truth_row_count": ground_truth_row_count(ground_truth),
            **metric_dict,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base_record,
            "evaluation_status": "error",
            "evaluation_metric_type": None,
            "metrics": {},
            "error": str(exc),
        }


def summarize_query_evaluations(
    query_instance_metrics: list[dict[str, object]],
    quality_thresholds: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, object]:
    """汇总 query-instance 级评估结果，产出 instance 级和 workload 级质量指标。

    `queries` 现在以 `terminal_x:step_y:Qz` 作为 key，重复出现的 Q1/Q2 会被视为
    不同 query instance；`queries_by_id` 仅作为按 query_id 聚合的辅助视图保留。
    """
    successful_instances = [
        record
        for record in query_instance_metrics
        if record.get("evaluation_status") == "success"
    ]
    skipped_instances = [
        record
        for record in query_instance_metrics
        if record.get("evaluation_status") == "skipped"
    ]
    failed_instances = [
        record
        for record in query_instance_metrics
        if record.get("evaluation_status") == "error"
    ]
    successful_metric_dicts = [
        dict(record.get("metrics") or {}) for record in successful_instances
    ]

    query_groups: dict[int, list[dict[str, object]]] = {}
    for record in query_instance_metrics:
        query_groups.setdefault(parse_query_id(record["query_id"]), []).append(record)

    query_instance_summaries: dict[str, dict[str, object]] = {}
    for record in query_instance_metrics:
        query_instance_id = str(record["query_instance_id"])
        query_instance_summaries[query_instance_id] = {
            "query_instance_id": query_instance_id,
            "query_id": parse_query_id(record["query_id"]),
            "terminal_id": int(record["terminal_id"]),
            "step_index": int(record["step_index"]),
            "status": record.get("evaluation_status"),
            "query_status": record.get("query_status"),
            "metric_type": record.get("evaluation_metric_type"),
            "result_output": record.get("result_output"),
            "error": record.get("error"),
            **dict(record.get("metrics") or {}),
        }

    query_id_summaries: dict[str, dict[str, object]] = {}
    for query_id in sorted(query_groups, key=query_id_sort_key):
        query_records = query_groups[query_id]
        query_successes = [
            record
            for record in query_records
            if record.get("evaluation_status") == "success"
        ]
        query_failures = [
            record
            for record in query_records
            if record.get("evaluation_status") == "error"
        ]
        query_skips = [
            record
            for record in query_records
            if record.get("evaluation_status") == "skipped"
        ]
        metric_types = sorted(
            {
                str(record["evaluation_metric_type"])
                for record in query_successes
                if record.get("evaluation_metric_type") is not None
            }
        )
        aggregated_metrics = average_metric_dicts(
            [dict(record.get("metrics") or {}) for record in query_successes]
        )
        query_metric_dicts = [
            dict(record.get("metrics") or {}) for record in query_successes
        ]
        weighted_accuracy, accuracy_weights = aggregate_query_metric(
            query_metric_dicts, "accuracy"
        )
        if weighted_accuracy is not None:
            aggregated_metrics["accuracy"] = weighted_accuracy
        if accuracy_weights is not None:
            aggregated_metrics["accuracy_aggregation"] = (
                "classification_sample_count_weighted_mean"
            )
        if query_successes and not query_failures and not query_skips:
            query_status = "success"
        elif query_successes:
            query_status = "partial"
        else:
            query_status = "failed"
        query_id_summaries[f"Q{query_id}"] = {
            "query_id": query_id,
            "status": query_status,
            "instance_count": len(query_records),
            "evaluated_instance_count": len(query_successes),
            "evaluation_error_count": len(query_failures),
            "skipped_instance_count": len(query_skips),
            "metric_type": (
                metric_types[0]
                if len(metric_types) == 1
                else ("mixed" if metric_types else None)
            ),
            "query_instance_ids": [
                str(record["query_instance_id"]) for record in query_records
            ],
            **aggregated_metrics,
        }

    if successful_instances and not failed_instances and not skipped_instances:
        evaluation_status = "success"
    elif successful_instances:
        evaluation_status = "partial"
    elif failed_instances:
        evaluation_status = "failed"
    else:
        evaluation_status = "skipped"

    metric_type_counts: dict[str, int] = {}
    for record in successful_instances:
        metric_type = record.get("evaluation_metric_type")
        if metric_type is not None:
            metric_type_counts[str(metric_type)] = (
                metric_type_counts.get(str(metric_type), 0) + 1
            )

    summary = {
        "evaluated_query_instance_count": len(successful_instances),
        "skipped_evaluation_count": len(skipped_instances),
        "evaluation_error_count": len(failed_instances),
        **build_metric_inventory(
            successful_metric_dicts,
            expected_metric_count=len(query_instance_metrics),
        ),
    }
    quality_gate = evaluate_quality_thresholds(
        query_instance_metrics,
        quality_thresholds,
    )
    summary["quality_gate_configured"] = quality_gate["configured"]
    summary["quality_gate_passed"] = quality_gate["passed"]

    return {
        "status": evaluation_status,
        "query_instance_count": len(query_instance_metrics),
        "summary": summary,
        "metric_type_counts": metric_type_counts,
        "queries": query_instance_summaries,
        "queries_by_id": query_id_summaries,
        "query_instances": query_instance_metrics,
        "quality_gate": quality_gate,
    }


def evaluate_terminal_queries(
    terminals: list[dict[str, object]],
    use_case: str,
    scale_factor: int | None,
    quality_thresholds: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, object]:
    """对 TPCC workload 中所有 query 实例执行 evaluator，并把结果回填到 terminal 记录。"""
    evaluator_class = get_evaluator(use_case)
    if evaluator_class is None:
        quality_gate = evaluate_quality_thresholds([], quality_thresholds)
        return {
            "status": "error",
            "error": f"Failed to load evaluator for use case: {use_case}",
            "query_instance_count": 0,
            "summary": {
                "evaluated_query_instance_count": 0,
                "skipped_evaluation_count": 0,
                "evaluation_error_count": 0,
                "quality_metric_counts": {},
                "quality_metric_coverage": 0.0,
                "quality_gate_configured": quality_gate["configured"],
                "quality_gate_passed": quality_gate["passed"],
            },
            "metric_type_counts": {},
            "queries": {},
            "query_instances": [],
            "quality_gate": quality_gate,
        }

    evaluator = evaluator_class(use_case, scale_factor)
    query_instance_metrics: list[dict[str, object]] = []
    for terminal in terminals:
        terminal_id = int(terminal["terminal_id"])
        if use_case == "cacheblend_rag":
            evaluator.terminal_id = terminal_id
        for query_record in terminal.get("queries", []):
            # 成功 query 在所有 terminal 完成、性能计时窗口闭合后统一 evaluate。
            # 旧产物或 failed/skipped query 可能已经携带 evaluation，直接复用。
            existing_evaluation = query_record.get("evaluation")
            if isinstance(existing_evaluation, dict):
                evaluation_record = dict(existing_evaluation)
            else:
                evaluation_record = evaluate_query_instance(
                    evaluator=evaluator,
                    query_record=query_record,
                    terminal_id=terminal_id,
                )
                query_record["evaluation"] = evaluation_record
                query_record["evaluation_status"] = evaluation_record[
                    "evaluation_status"
                ]
                query_record["evaluation_metric_type"] = evaluation_record[
                    "evaluation_metric_type"
                ]
                query_record["evaluation_metrics"] = dict(
                    evaluation_record.get("metrics") or {}
                )
                query_record["evaluation_error"] = evaluation_record.get("error")
            query_instance_metrics.append(evaluation_record)

    return summarize_query_evaluations(
        query_instance_metrics,
        quality_thresholds=quality_thresholds,
    )


def build_global_in_flight_timeline(
    terminals: list[dict[str, object]],
) -> dict[str, object]:
    """根据各 query trace 重建全局 in-flight 时间线。

    `wave_id` 在这里仅作为观测维度保留；新的 TPCC 调度允许不同 `wave_id`
    的请求时间区间重叠，因此时间线必须完全按真实开始/结束时间排序。
    """
    events: list[dict[str, object]] = []
    for terminal in terminals:
        terminal_id = int(terminal["terminal_id"])
        for query in terminal.get("queries", []):
            query_id = parse_query_id(query["query_id"])
            step_index = int(query["step_index"])
            for wave in query.get("waves", []):
                wave_id = int(wave["wave_id"])
                for request in wave.get("requests", []):
                    request_index = int(request["request_index"])
                    common_fields = {
                        "terminal_id": terminal_id,
                        "query_id": query_id,
                        "step_index": step_index,
                        "wave_id": wave_id,
                        "request_index": request_index,
                    }
                    started_at = request.get("started_at")
                    if started_at is not None:
                        events.append(
                            {
                                **common_fields,
                                "timestamp_s": float(started_at),
                                "event": "request_started",
                                "delta": 1,
                            }
                        )
                    completed_at = request.get("completed_at")
                    if completed_at is not None:
                        events.append(
                            {
                                **common_fields,
                                "timestamp_s": float(completed_at),
                                "event": "request_completed",
                                "delta": -1,
                            }
                        )

    events.sort(
        key=lambda event: (
            float(event["timestamp_s"]),
            0 if int(event["delta"]) < 0 else 1,
            int(event["terminal_id"]),
            query_id_sort_key(event["query_id"]),
            int(event["wave_id"]),
            int(event["request_index"]),
        )
    )

    in_flight = 0
    peak_in_flight = 0
    timeline: list[dict[str, object]] = []
    for event in events:
        in_flight += int(event["delta"])
        peak_in_flight = max(peak_in_flight, in_flight)
        timeline.append(
            {
                "timestamp_s": event["timestamp_s"],
                "event": event["event"],
                "in_flight": in_flight,
                "terminal_id": event["terminal_id"],
                "query_id": event["query_id"],
                "step_index": event["step_index"],
                "wave_id": event["wave_id"],
                "request_index": event["request_index"],
            }
        )

    return {
        "timeline": timeline,
        "peak_in_flight": peak_in_flight,
        "event_count": len(timeline),
    }


def main() -> int:
    """运行 TPCC-style terminal workload。

    执行逻辑：
    1. 根据配置生成 terminal 数量和查询序列。
    2. 为 batch-capable query 注入请求数量控制参数。
    3. 并发启动多个 terminal worker。
    4. 汇总 terminal 指标，并输出全局 in-flight 观测结果。
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    parser = argparse.ArgumentParser(
        description="Run a TPCC-style FlockMTL terminal workload"
    )
    parser.add_argument("--system", default="flockmtl")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--queries", nargs="+")
    parser.add_argument("--initial-queries", nargs="+",
                        help="Run these queries once before starting the parallel terminals; include their time and quality.")
    parser.add_argument(
        "--terminal-queries",
        action="append",
        nargs="+",
        metavar="QUERY",
        help=(
            "Per-terminal query sequence. Repeat once per terminal, for example: "
            "--terminal-queries 1 2 --terminal-queries 3 4 8"
        ),
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--scale-factor", type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--terminal-count", type=int, required=True)
    parser.add_argument(
        "--request-control-mode",
        choices=["max_in_flight", "vllm_batch_size_div_terminals"],
        required=True,
    )
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument(
        "--max-in-flight-strategy",
        choices=MAX_IN_FLIGHT_STRATEGIES,
        default=os.environ.get(
            MAX_IN_FLIGHT_STRATEGY_ENV,
            DEFAULT_MAX_IN_FLIGHT_STRATEGY,
        ),
        help=(
            "Choose dynamic active-terminal partitioning or the original "
            "cross-terminal global slot coordinator."
        ),
    )
    parser.add_argument("--vllm-batch-size", type=int, default=None)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--request-arrival", default="poisson")
    parser.add_argument(
        "--failure-policy",
        choices=["fail_fast", "continue"],
        default="fail_fast",
    )
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--quality-thresholds-json",
        default=None,
        help=(
            "JSON object of per-query, per-metric gates. Example: "
            "{\"Q1\":{\"precision\":{\"min\":0.98},"
            "\"recall\":{\"min\":0.75}}}. Each representative metric "
            "for every query must be configured."
        ),
    )
    parser.add_argument(
        "--cacheblend-mode",
        choices=["auto", "on", "off"],
        default=os.environ.get(CACHEBLEND_MODE_ENV, "auto"),
        help=(
            "Control FlockMTL SQL-side cacheblend markers for direct TPCC "
            "workload runs."
        ),
    )
    args = parser.parse_args()
    args.cacheblend_mode = normalize_cacheblend_mode(args.cacheblend_mode)
    os.environ[CACHEBLEND_MODE_ENV] = args.cacheblend_mode
    quality_thresholds = parse_quality_thresholds(args.quality_thresholds_json)

    if args.terminal_count < 1:
        raise ValueError("terminal-count must be >= 1")

    terminal_query_sequences = build_terminal_query_sequences(
        terminal_count=args.terminal_count,
        query_args=args.queries,
        terminal_query_args=args.terminal_queries,
    )
    shared_query_sequence, terminal_sql_sequences = (
        summarize_terminal_query_sequences(terminal_query_sequences)
    )

    requests_per_wave: int | None = None
    global_coord_dir: str | None = None
    active_terminal_coord_dir: str | None = None
    if args.request_control_mode == "vllm_batch_size_div_terminals":
        if args.vllm_batch_size is None or args.vllm_batch_size < 1:
            raise ValueError(
                "vllm-batch-size must be >= 1 for "
                "vllm_batch_size_div_terminals mode"
            )
        requests_per_wave = compute_requests_per_wave(
            args.terminal_count,
            args.vllm_batch_size,
        )
    else:
        if args.max_in_flight is None or args.max_in_flight < 1:
            raise ValueError(
                "max-in-flight must be >= 1 for max_in_flight mode"
            )

    metrics_output = Path(args.metrics_output)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    logs_dir = artifact_root / "terminal_logs"
    metrics_dir = artifact_root / "terminal_metrics"
    results_dir = artifact_root / "terminal_results"
    query_trace_dir = artifact_root / "query_traces"
    observability_dir = artifact_root / "observability"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    query_trace_dir.mkdir(parents=True, exist_ok=True)
    observability_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    workload_started_at = time.time()
    worker_script = Path(__file__).with_name("flockmtl_terminal_worker.py")

    with tempfile.TemporaryDirectory(prefix="flockmtl_tpcc_coord_") as coord_dir:
        use_global_slot_coordination = (
            args.request_control_mode == "max_in_flight"
            and args.max_in_flight_strategy
            == GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY
        )
        if use_global_slot_coordination and args.terminal_count > 1:
            global_coord_dir = coord_dir
        if args.request_control_mode == "vllm_batch_size_div_terminals" or (
            args.request_control_mode == "max_in_flight"
            and args.max_in_flight_strategy
            == DEFAULT_MAX_IN_FLIGHT_STRATEGY
        ):
            active_terminal_dir = Path(coord_dir) / "active_terminals"
            active_terminal_dir.mkdir(parents=True, exist_ok=True)
            active_terminal_coord_dir = str(active_terminal_dir)
            for terminal_id in range(args.terminal_count):
                state_path = active_terminal_dir / f"terminal_{terminal_id}.json"
                write_json_atomic(
                    state_path,
                    make_terminal_coord_record(
                        terminal_id=terminal_id,
                        remaining_query_count=len(
                            terminal_query_sequences[terminal_id]
                        ),
                    ),
                )

        env = build_flock_workload_env(
            base_env=base_env,
            request_rate=args.request_rate,
            request_arrival=args.request_arrival,
            request_control_mode=args.request_control_mode,
            max_in_flight=args.max_in_flight,
            requests_per_wave=requests_per_wave,
            global_coord_dir=global_coord_dir,
            terminal_count=args.terminal_count,
            vllm_batch_size=args.vllm_batch_size,
            active_terminal_coord_dir=active_terminal_coord_dir,
            active_terminal_refresh_interval=DEFAULT_ACTIVE_TERMINAL_REFRESH_INTERVAL,
            max_in_flight_strategy=args.max_in_flight_strategy,
        )
        # Preserve Python-side diagnostics even if a native extension aborts
        # the worker before its buffered stdout can be flushed.
        env["PYTHONUNBUFFERED"] = "1"

        def run_terminal(terminal_id: int, initial: bool = False) -> tuple[int, int, dict[str, object], str]:
            """启动一个 terminal worker，并回收它的日志与指标产物。"""
            log_path = logs_dir / f"terminal_{terminal_id}.log"
            metric_path = metrics_dir / f"terminal_{terminal_id}.json"
            terminal_result_root = results_dir / f"terminal_{terminal_id}"
            terminal_query_trace_root = query_trace_dir / f"terminal_{terminal_id}"
            selected_queries = terminal_query_sequences[terminal_id]
            if initial:
                selected_queries = parse_query_sequence(args.initial_queries)
                log_path = logs_dir / "initial.log"
                metric_path = metrics_dir / "initial.json"
                terminal_result_root = results_dir / "initial"
                terminal_query_trace_root = query_trace_dir / "initial"
            cmd = build_terminal_worker_cmd(
                python_executable=sys.executable,
                worker_script=worker_script,
                system=args.system,
                use_case=args.use_case,
                query_ids=selected_queries,
                model=args.model,
                terminal_id=terminal_id,
                failure_policy=args.failure_policy,
                metric_output=metric_path,
                result_root=terminal_result_root,
                query_trace_root=terminal_query_trace_root,
                scale_factor=args.scale_factor,
                skip_setup=args.skip_setup,
            )

            # 每个 terminal 都是一个独立进程：terminal 内串行、terminal 间并行。
            terminal_env = dict(env)
            barrier_dir = Path(coord_dir) / ("initial_start" if initial else "query_start")
            barrier_dir.mkdir(parents=True, exist_ok=True)
            terminal_env["FLOCK_TPCC_START_BARRIER_DIR"] = str(barrier_dir)
            terminal_env["FLOCK_TPCC_START_BARRIER_COUNT"] = str(1 if initial else args.terminal_count)
            if initial:
                # This phase has only one worker; no other terminals are active yet.
                terminal_env.pop("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR", None)
                terminal_env["FLOCK_TPCC_TOTAL_TERMINAL_COUNT"] = "1"
            if args.use_case == "cacheblend_rag" and env.get("CACHEBLEND_RAG_PARTITION_COUNT"):
                if int(env["CACHEBLEND_RAG_PARTITION_COUNT"]) != args.terminal_count:
                    raise ValueError("HotpotQA partition count must match terminal-count")
                terminal_env["CACHEBLEND_RAG_TERMINAL_ID"] = str(terminal_id)
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=terminal_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                (barrier_dir / "abort").touch()
            log_path.write_text(proc.stdout or "", encoding="utf-8")
            if metric_path.exists():
                metric = load_terminal_metric(metric_path)
            else:
                completed_at = time.time()
                metric = {
                    "terminal_id": terminal_id,
                    "failure_policy": args.failure_policy,
                    "status": "failed",
                    "query_ids": selected_queries,
                    "started_at": workload_started_at,
                    "completed_at": completed_at,
                    "wall_time_s": completed_at - workload_started_at,
                    "queries": [],
                    "query_count": 0,
                    "error": (
                        "terminal worker exited before writing its first metric; "
                        f"returncode={proc.returncode}"
                    ),
                }
            metric["worker_returncode"] = proc.returncode
            metric["worker_signal"] = (
                -proc.returncode if proc.returncode < 0 else None
            )
            if proc.returncode != 0:
                metric["status"] = "failed"
                if not metric.get("error"):
                    metric["error"] = (
                        f"terminal worker exited with returncode={proc.returncode}"
                    )
            return terminal_id, proc.returncode, metric, str(log_path)

        terminal_records: list[dict[str, object]] = []
        terminal_logs: dict[str, str] = {}
        return_code = 0
        initial_metric = None
        if args.initial_queries:
            _, initial_returncode, initial_metric, initial_log = run_terminal(0, initial=True)
            terminal_logs["initial"] = initial_log
            if initial_returncode != 0:
                return_code = initial_returncode
        with ThreadPoolExecutor(max_workers=args.terminal_count) as executor:
            futures = {
                executor.submit(run_terminal, terminal_id): terminal_id
                for terminal_id in range(args.terminal_count)
            }
            for future in as_completed(futures):
                terminal_id, terminal_returncode, metric, log_path = future.result()
                terminal_logs[str(terminal_id)] = log_path
                terminal_records.append(metric)
                if terminal_returncode != 0:
                    return_code = terminal_returncode

        if initial_metric is not None:
            first = next(record for record in terminal_records if int(record["terminal_id"]) == 0)
            offset = len(initial_metric["queries"])
            for q in first["queries"]:
                q["step_index"] += offset
            first["queries"] = initial_metric["queries"] + first["queries"]
            first["query_sequence"] = [parse_query_id(q["query_id"]) for q in first["queries"]]
            first["query_count"] = len(first["queries"])
            first["started_at"] = initial_metric["started_at"]
            first["wall_time_s"] = first["completed_at"] - first["started_at"]
            if initial_metric["status"] != "success":
                first["status"] = "failed"

    terminal_records.sort(key=lambda record: int(record["terminal_id"]))
    workload_completed_at = time.time()
    execution_timing = query_execution_timing(terminal_records, workload_started_at, workload_completed_at)
    summary = summarize_terminal_metrics(terminal_records)
    quality_evaluation_started_at = time.time()
    evaluation_metrics = evaluate_terminal_queries(
        terminals=terminal_records,
        use_case=args.use_case,
        scale_factor=args.scale_factor,
        quality_thresholds=quality_thresholds,
    )
    quality_evaluation_completed_at = time.time()
    # 将 query-instance 级 evaluate 汇总结果并入 summary，
    # 这样 TPCC workload 的结果结构与既有 SemBench summary 更接近。
    summary.update(dict(evaluation_metrics.get("summary") or {}))
    summary["evaluation_status"] = evaluation_metrics.get("status")
    global_in_flight_metrics = build_global_in_flight_timeline(terminal_records)
    global_in_flight_timeline_output = (
        observability_dir / "global_in_flight_timeline.json"
    )
    with global_in_flight_timeline_output.open("w", encoding="utf-8") as timeline_file:
        json.dump(global_in_flight_metrics["timeline"], timeline_file, indent=2)
    workload_metrics = {
        "status": "failed" if return_code != 0 else "success",
        "terminal_count": args.terminal_count,
        "terminal_sql_sequence": shared_query_sequence,
        "terminal_sql_sequences": terminal_sql_sequences,
        "initial_query_sequence": parse_query_sequence(args.initial_queries) if args.initial_queries else [],
        "request_control_mode": args.request_control_mode,
        "max_in_flight": args.max_in_flight,
        "effective_per_terminal_max_in_flight": (
            args.max_in_flight
            if args.request_control_mode == "max_in_flight"
            and args.max_in_flight_strategy
            == GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY
            else compute_equal_partition_request_cap(
                args.terminal_count,
                args.max_in_flight,
            )
            if args.request_control_mode == "max_in_flight"
            and args.max_in_flight is not None
            else None
        ),
        "request_cap_strategy": (
            args.max_in_flight_strategy
            if args.request_control_mode == "max_in_flight"
            else "dynamic_active_terminal_partition"
        ),
        "vllm_batch_size": args.vllm_batch_size,
        "requests_per_wave": requests_per_wave,
        "request_cap_formula": (
            "floor(vllm_batch_size / active_terminal_count)"
            if args.request_control_mode == "vllm_batch_size_div_terminals"
            else "global slot claims"
            if args.max_in_flight_strategy
            == GLOBAL_SLOT_MAX_IN_FLIGHT_STRATEGY
            else "floor(max_in_flight / configured_terminal_count)"
            if args.max_in_flight_strategy
            == FIXED_EQUAL_PARTITION_MAX_IN_FLIGHT_STRATEGY
            else "floor(max_in_flight / active_terminal_count)"
        ),
        "active_terminal_refresh_interval": (
            DEFAULT_ACTIVE_TERMINAL_REFRESH_INTERVAL
            if args.request_control_mode == "vllm_batch_size_div_terminals"
            or (
                args.request_control_mode == "max_in_flight"
                and args.max_in_flight_strategy
                == DEFAULT_MAX_IN_FLIGHT_STRATEGY
            )
            else None
        ),
        "failure_policy": args.failure_policy,
        "batch_capable_function_list": [
            "llm_filter",
            "llm_complete",
            "llm_embedding",
        ],
        "non_batch_capable_function_list": [
            "llm_reduce",
            "llm_rerank",
            "llm_rank",
            "llm_first",
            "llm_last",
        ],
        "started_at": workload_started_at,
        "completed_at": workload_completed_at,
        "total_wall_time_s": workload_completed_at - workload_started_at,
        "wall_time_s": execution_timing.get("execution_wall_time_s", 0.0),
        **execution_timing,
        "timing_status": "measured" if execution_timing else "unavailable",

        "quality_evaluation_started_at": quality_evaluation_started_at,
        "quality_evaluation_completed_at": quality_evaluation_completed_at,
        "quality_evaluation_wall_time_s": (
            quality_evaluation_completed_at - quality_evaluation_started_at
        ),
        "quality_evaluation_timing": "after_all_terminals_complete",
        "summary": summary,
        "evaluation": evaluation_metrics,
        "global_in_flight_peak": global_in_flight_metrics["peak_in_flight"],
        "global_in_flight_event_count": global_in_flight_metrics["event_count"],
        "global_in_flight_timeline_output": str(global_in_flight_timeline_output),
        "terminals": terminal_records,
        "terminal_logs": terminal_logs,
        "artifact_root": str(artifact_root),
    }

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as metric_file:
        json.dump(make_json_safe(workload_metrics), metric_file, indent=2)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
