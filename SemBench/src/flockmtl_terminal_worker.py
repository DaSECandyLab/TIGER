#!/usr/bin/env python3
"""Run one terminal's serial FlockMTL SQL sequence in an isolated worker."""

from __future__ import annotations

import argparse
from query_ids import parse_query_id
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from dotenv import load_dotenv
import pandas as pd

# Add src directory to Python path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import get_runner_class
from flockmtl_cacheblend import CACHEBLEND_MODE_ENV, normalize_cacheblend_mode


BATCH_CAPABLE_FUNCTIONS = ("llm_filter", "llm_complete", "llm_embedding")
NON_BATCH_CAPABLE_FUNCTIONS = (
    "llm_reduce",
    "llm_rerank",
    "llm_rank",
    "llm_first",
    "llm_last",
)
BATCH_CONTROL_ENV_KEYS = (
    "FLOCK_VLLM_MAX_IN_FLIGHT",
    "FLOCK_VLLM_GLOBAL_IN_FLIGHT_COORD_DIR",
    "FLOCK_VLLM_REQUEST_RATE",
    "FLOCK_VLLM_REQUEST_ARRIVAL",
)
QUERY_TRACE_ENV_KEYS = (
    "FLOCK_TPCC_ENABLE_WAVE_MODE",
    "FLOCK_TPCC_QUERY_TRACE_OUTPUT",
    "FLOCK_TPCC_TERMINAL_ID",
    "FLOCK_TPCC_QUERY_ID",
    "FLOCK_TPCC_STEP_INDEX",
    "FLOCK_TPCC_FUNCTION_TYPE",
    "FLOCK_TPCC_ACTIVE_TERMINAL_COUNT",
    "FLOCK_TPCC_REQUEST_CAP_SOURCE",
)
MANAGED_QUERY_ENV_KEYS = BATCH_CONTROL_ENV_KEYS + QUERY_TRACE_ENV_KEYS


def parse_query_sequence(query_args: list[str]) -> list[int | str]:
    """Preserve suffixed IDs such as Q3a and the order of terminal steps."""
    return [parse_query_id(arg) for arg in query_args]


def classify_query_mode(query_text: str) -> dict[str, object]:
    """识别当前 SQL 属于 batch-capable 还是 non-batch-capable。

    规则上优先保守：只要命中 aggregate/native LLM 函数，就保留原生路径；
    只有纯 `llm_filter` / `llm_complete` / `llm_embedding` 这类独立请求型函数
    才走 TPCC batch 路径。
    """
    lowered = query_text.lower()
    batch_functions = [
        function_name
        for function_name in BATCH_CAPABLE_FUNCTIONS
        if re.search(rf"\b{re.escape(function_name)}\s*\(", lowered)
    ]
    non_batch_functions = [
        function_name
        for function_name in NON_BATCH_CAPABLE_FUNCTIONS
        if re.search(rf"\b{re.escape(function_name)}\s*\(", lowered)
    ]

    if non_batch_functions:
        request_mode = "non-batch-capable"
    elif batch_functions:
        request_mode = "batch-capable"
    else:
        request_mode = "non-batch-capable"

    return {
        "request_mode": request_mode,
        "batch_capable_functions": batch_functions,
        "non_batch_capable_functions": non_batch_functions,
    }


def append_skipped_queries(
    query_records: list[dict[str, object]],
    query_ids: list[int],
    start_index: int,
    terminal_id: int,
) -> None:
    """在 fail-fast 触发后，为后续未执行的 SQL 追加 skipped 记录。"""
    for step_index, query_id in enumerate(query_ids[start_index:], start=start_index):
        query_record = {
            "step_index": step_index,
            "query_id": query_id,
            "status": "skipped",
            "request_mode": "not_started",
            "batch_capable_functions": [],
            "non_batch_capable_functions": [],
            "execution_time": None,
            "query_wall_time": None,
            "time_to_first_vllm_request": None,
            "first_vllm_request_path": None,
            "observed_vllm_request_count": None,
            "row_count": 0,
            "started_at": None,
            "completed_at": None,
            "error": "Skipped because a prior SQL failed under fail_fast.",
            "result_output": None,
            "wave_count": 0,
            "total_llm_requests": 0,
            "waves": [],
            "query_trace_output": None,
            "request_scheduling_mode": None,
            "configured_request_cap": None,
            "final_effective_request_cap": None,
            "min_observed_request_cap": None,
            "max_observed_request_cap": None,
            "active_terminal_count_at_query_start": None,
            "min_observed_active_terminal_count": None,
            "max_observed_active_terminal_count": None,
            "request_cap_source": None,
            "dynamic_request_cap_refresh_count": 0,
        }
        attach_query_evaluation(
            query_record,
            build_skipped_evaluation_record(
                terminal_id=terminal_id,
                step_index=step_index,
                query_id=query_id,
                query_status="skipped",
                result_output=None,
                error="Query skipped because a prior SQL failed under fail_fast.",
            ),
        )
        query_records.append(query_record)


def write_terminal_metric(metric_output: Path, metric: dict[str, object]) -> None:
    """将单个 terminal 的执行指标写入 JSON 文件。"""
    metric_output.parent.mkdir(parents=True, exist_ok=True)
    with metric_output.open("w", encoding="utf-8") as metric_file:
        json.dump(metric, metric_file, indent=2)


def make_json_safe(value: object) -> object:
    """把 evaluator 产物里的 NaN/Inf 转成 JSON 可稳定表达的值。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


def build_query_instance_key(terminal_id: int, step_index: int, query_id: int) -> str:
    """用 terminal + step + query id 唯一标识一次 query 执行。

    这样同一个 Q1 在不同 terminal 或同一 terminal 的不同 step 中出现时，
    会被视为不同 query instance，而不会被质量指标汇总逻辑合并。
    """
    return f"terminal_{terminal_id}:step_{step_index}:Q{query_id}"


def build_skipped_evaluation_record(
    *,
    terminal_id: int,
    step_index: int,
    query_id: int,
    query_status: str,
    result_output: str | None,
    error: str,
) -> dict[str, object]:
    """为 skipped/failed query 构造一条质量评估记录。"""
    return {
        "query_instance_id": build_query_instance_key(
            terminal_id, step_index, query_id
        ),
        "terminal_id": terminal_id,
        "step_index": step_index,
        "query_id": query_id,
        "query_status": query_status,
        "result_output": result_output,
        "evaluation_status": "skipped",
        "evaluation_metric_type": None,
        "metrics": {},
        "error": error,
    }


def evaluate_query_result(
    evaluator: object | None,
    *,
    terminal_id: int,
    step_index: int,
    query_id: int,
    query_status: str,
    result_output: Path | None,
    system_results: object | None,
) -> dict[str, object]:
    """在单个 query 完成后立即计算质量指标。

    这一步发生在 terminal worker 内部，因此 metric 文件会随着每个 query 完成
    逐步更新，而不必等整个 workload 的所有 terminal 都结束。
    """
    result_output_text = str(result_output) if result_output is not None else None
    if query_status != "success":
        return build_skipped_evaluation_record(
            terminal_id=terminal_id,
            step_index=step_index,
            query_id=query_id,
            query_status=query_status,
            result_output=result_output_text,
            error=f"Query status is {query_status}, evaluation skipped.",
        )
    if evaluator is None:
        return build_skipped_evaluation_record(
            terminal_id=terminal_id,
            step_index=step_index,
            query_id=query_id,
            query_status=query_status,
            result_output=result_output_text,
            error="Evaluator is unavailable, evaluation skipped.",
        )
    if system_results is None:
        return build_skipped_evaluation_record(
            terminal_id=terminal_id,
            step_index=step_index,
            query_id=query_id,
            query_status=query_status,
            result_output=result_output_text,
            error="Missing query result dataframe, evaluation skipped.",
        )

    try:
        ground_truth = evaluator._get_ground_truth(query_id)  # noqa: SLF001
        metric = evaluator._evaluate_single_query(  # noqa: SLF001
            query_id,
            system_results,
            ground_truth,
        )
        metric_dict = make_json_safe(dataclasses.asdict(metric))
        return {
            "query_instance_id": build_query_instance_key(
                terminal_id, step_index, query_id
            ),
            "terminal_id": terminal_id,
            "step_index": step_index,
            "query_id": query_id,
            "query_status": query_status,
            "result_output": result_output_text,
            "evaluation_status": "success",
            "evaluation_metric_type": type(metric).__name__,
            "metrics": metric_dict,
            "error": None,
            "result_row_count": len(system_results),
            "ground_truth_row_count": len(ground_truth),
            **metric_dict,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "query_instance_id": build_query_instance_key(
                terminal_id, step_index, query_id
            ),
            "terminal_id": terminal_id,
            "step_index": step_index,
            "query_id": query_id,
            "query_status": query_status,
            "result_output": result_output_text,
            "evaluation_status": "error",
            "evaluation_metric_type": None,
            "metrics": {},
            "error": str(exc),
        }


def attach_query_evaluation(
    query_record: dict[str, object],
    evaluation_record: dict[str, object],
) -> None:
    """把单 query 质量指标回填到 query record，方便后续汇总和人工排查。"""
    query_record["evaluation"] = evaluation_record
    query_record["evaluation_status"] = evaluation_record["evaluation_status"]
    query_record["evaluation_metric_type"] = evaluation_record[
        "evaluation_metric_type"
    ]
    query_record["evaluation_metrics"] = dict(
        evaluation_record.get("metrics") or {}
    )
    query_record["evaluation_error"] = evaluation_record.get("error")


def build_terminal_metric(
    *,
    terminal_id: int,
    failure_policy: str,
    query_ids: list[int],
    terminal_started_at: float,
    query_records: list[dict[str, object]],
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    """构造 terminal metric。

    terminal worker 每完成一个 query 都会写一次 `status=running` 的快照；
    结束时再写最终状态。这样外部读者可以在 workload 尚未结束时看到已完成
    query 的质量指标。
    """
    now = time.time()
    metric: dict[str, object] = {
        "terminal_id": terminal_id,
        "status": status,
        "failure_policy": failure_policy,
        "query_sequence": query_ids,
        "started_at": terminal_started_at,
        "completed_at": None if status == "running" else now,
        "wall_time_s": now - terminal_started_at,
        "queries": query_records,
    }
    if error is not None:
        metric["error"] = error
    return metric


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """原子写状态文件，避免并发读到半写入内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temp_file:
        json.dump(payload, temp_file, indent=2)
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def terminal_coord_file(coord_dir: str, terminal_id: int) -> Path:
    return Path(coord_dir) / f"terminal_{terminal_id}.json"


def update_terminal_coord_state(
    workload_env: dict[str, str],
    terminal_id: int,
    remaining_query_count: int,
    status: str,
) -> None:
    """更新当前 terminal 是否仍有剩余请求。

    active-terminal 分母语义是“还有请求的 terminal 数量”，因此 fail-fast
    后续被跳过的 query 不再计入 remaining_query_count。
    """
    coord_dir = workload_env.get("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR")
    if not coord_dir:
        return
    payload = {
        "terminal_id": terminal_id,
        "remaining_query_count": max(0, int(remaining_query_count)),
        "status": status,
        "updated_at": time.time(),
    }
    write_json_atomic(terminal_coord_file(coord_dir, terminal_id), payload)


def count_active_terminals(workload_env: dict[str, str]) -> int | None:
    """统计当前仍有剩余请求的 terminal 数。"""
    coord_dir = workload_env.get("FLOCK_TPCC_ACTIVE_TERMINAL_COORD_DIR")
    if not coord_dir:
        return None

    active_count = 0
    for state_path in sorted(Path(coord_dir).glob("terminal_*.json")):
        try:
            with state_path.open("r", encoding="utf-8") as state_file:
                state = json.load(state_file)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if int(state.get("remaining_query_count") or 0) > 0:
            active_count += 1

    if active_count > 0:
        return active_count

    raw_total = workload_env.get("FLOCK_TPCC_TOTAL_TERMINAL_COUNT")
    if raw_total is None:
        return None
    try:
        return max(1, int(raw_total))
    except ValueError:
        return 1


def resolve_dynamic_request_cap(
    workload_env: dict[str, str],
) -> tuple[int | None, int | None]:
    """按当前 active terminal 数量计算 query-local cap。"""
    request_control_mode = workload_env.get("FLOCK_TPCC_REQUEST_CONTROL_MODE")
    if request_control_mode == "vllm_batch_size_div_terminals":
        raw_partition_budget = workload_env.get("FLOCK_TPCC_VLLM_BATCH_SIZE")
    elif request_control_mode == "max_in_flight":
        raw_partition_budget = workload_env.get(
            "FLOCK_TPCC_CONFIGURED_GLOBAL_MAX_IN_FLIGHT"
        )
    else:
        return None, None

    if raw_partition_budget is None:
        return None, None
    try:
        partition_budget = int(raw_partition_budget)
    except ValueError:
        return None, None
    if partition_budget < 1:
        return None, None

    active_terminal_count = count_active_terminals(workload_env)
    if active_terminal_count is None:
        return None, None
    request_cap = max(1, partition_budget // max(1, active_terminal_count))
    return request_cap, active_terminal_count


def build_query_execution_env(
    workload_env: dict[str, str],
    query_mode: dict[str, object],
    terminal_id: int,
    query_id: int,
    step_index: int,
    query_trace_output: Path | None,
) -> dict[str, str]:
    """按 query 类型构造本次 SQL 需要的环境变量。

    所有包含 LLM 函数的 query 都会继承共享的请求控制参数：
    - 请求数量上限
    - 可选的全局 slot 协调目录
    - request-rate / arrival

    这样 `llm_rerank` / `llm_rank` 一类 native query 发出的 provider 请求，
    也会参与 TPCC workload 的 shared max-in-flight 计数。

    只有 batch-capable query 会额外收到 TPCC batch-mode trace / wave 元数据。
    non-batch-capable query 仍然保持 Flock 原生执行路径，只共享计数约束。
    """
    desired_env: dict[str, str] = {}
    batch_capable_functions = list(query_mode.get("batch_capable_functions") or [])
    non_batch_capable_functions = list(
        query_mode.get("non_batch_capable_functions") or []
    )
    if not batch_capable_functions and not non_batch_capable_functions:
        return desired_env

    request_cap, active_terminal_count = resolve_dynamic_request_cap(
        workload_env
    )
    if request_cap is not None:
        desired_env["FLOCK_VLLM_MAX_IN_FLIGHT"] = str(request_cap)
    for env_key in BATCH_CONTROL_ENV_KEYS:
        env_value = workload_env.get(env_key)
        if env_value is not None and env_key not in desired_env:
            desired_env[env_key] = env_value

    if str(query_mode["request_mode"]) != "batch-capable":
        return desired_env

    if request_cap is not None:
        desired_env["FLOCK_TPCC_ACTIVE_TERMINAL_COUNT"] = str(
            active_terminal_count
        )
        desired_env["FLOCK_TPCC_REQUEST_CAP_SOURCE"] = (
            "global_max_in_flight_div_active_terminals"
            if workload_env.get("FLOCK_TPCC_REQUEST_CONTROL_MODE")
            == "max_in_flight"
            else "vllm_batch_size_div_active_terminals"
        )
    elif desired_env.get("FLOCK_VLLM_MAX_IN_FLIGHT") is not None:
        desired_env["FLOCK_TPCC_REQUEST_CAP_SOURCE"] = (
            "static_or_global_max_in_flight"
        )

    desired_env["FLOCK_TPCC_ENABLE_WAVE_MODE"] = "1"
    desired_env["FLOCK_TPCC_TERMINAL_ID"] = str(terminal_id)
    desired_env["FLOCK_TPCC_QUERY_ID"] = str(query_id)
    desired_env["FLOCK_TPCC_STEP_INDEX"] = str(step_index)
    desired_env["FLOCK_TPCC_FUNCTION_TYPE"] = ",".join(
        str(name) for name in batch_capable_functions
    )
    if query_trace_output is not None:
        query_trace_output.parent.mkdir(parents=True, exist_ok=True)
        desired_env["FLOCK_TPCC_QUERY_TRACE_OUTPUT"] = str(query_trace_output)
    return desired_env


def swap_managed_query_env(desired_env: dict[str, str]) -> dict[str, str | None]:
    """临时切换 TPCC workload 管理的环境变量，并返回切换前快照。"""
    previous_env = {
        env_key: os.environ.get(env_key) for env_key in MANAGED_QUERY_ENV_KEYS
    }
    for env_key in MANAGED_QUERY_ENV_KEYS:
        if env_key in desired_env:
            os.environ[env_key] = desired_env[env_key]
        else:
            os.environ.pop(env_key, None)
    return previous_env


def restore_managed_query_env(previous_env: dict[str, str | None]) -> None:
    """恢复一次 SQL 执行前的环境变量状态。"""
    for env_key, env_value in previous_env.items():
        if env_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = env_value


def load_query_trace(query_trace_output: Path | None) -> dict[str, object] | None:
    """Read a best-effort trace without making observability correctness fatal.

    DuckDB may evaluate one LLM expression in multiple native chunks. Those
    handlers currently share one trace path, so concurrent truncate/write
    operations can leave a valid first JSON object followed by trailing bytes.
    The SQL result remains valid in that case. Preserve the first complete
    object when possible, or return a diagnostic-only mapping when not.
    """
    if query_trace_output is None or not query_trace_output.exists():
        return None
    try:
        trace_text = query_trace_output.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "trace_load_error": f"{type(exc).__name__}: {exc}",
            "trace_output_bytes": None,
        }
    try:
        trace = json.loads(trace_text)
    except json.JSONDecodeError as exc:
        try:
            trace, parsed_end = json.JSONDecoder().raw_decode(trace_text)
        except json.JSONDecodeError:
            return {
                "trace_load_error": f"{type(exc).__name__}: {exc}",
                "trace_output_bytes": len(trace_text.encode("utf-8")),
            }
        if not isinstance(trace, dict):
            return {
                "trace_load_error": f"{type(exc).__name__}: {exc}",
                "trace_output_bytes": len(trace_text.encode("utf-8")),
            }
        trace = dict(trace)
        trace["trace_load_error"] = f"{type(exc).__name__}: {exc}"
        trace["trace_trailing_character_count"] = len(trace_text) - parsed_end
    if not isinstance(trace, dict):
        return {
            "trace_load_error": "query trace root is not a JSON object",
            "trace_output_bytes": len(trace_text.encode("utf-8")),
        }
    return trace


def resolve_logical_llm_request_count(
    query_id: int,
    system_results: object,
    query_trace: dict[str, object] | None,
) -> tuple[int, int, str]:
    """Resolve full SQL input cardinality when a chunk-local trace is partial.

    Flock may invoke an LLM function in multiple DuckDB data chunks while all
    chunks share one trace path. The last writer then describes only one chunk.
    Packed classification queries therefore emit their full sample count, and
    packed Q9 emits one result row per request. Other queries keep using the
    trace count.
    """

    trace_count = int((query_trace or {}).get("total_llm_requests") or 0)
    columns = getattr(system_results, "columns", ())
    if "__quality_samples" in columns:
        from evaluator.tabular_quality import decision_samples

        return len(decision_samples(system_results)), trace_count, "logical_quality_samples"
    if "classification_sample_count" in columns and len(system_results) > 0:
        numeric = pd.to_numeric(
            system_results["classification_sample_count"], errors="coerce"
        ).dropna()
        if not numeric.empty:
            return int(numeric.iloc[0]), trace_count, "classification_sample_count"
    if (
        query_id == 9
        and "reviewId" in columns
        and "reviewScore" in columns
        and "repeat_id" in columns
    ):
        return len(system_results), trace_count, "packed_q9_result_rows"
    return trace_count, trace_count, "query_trace"


def main() -> int:
    """运行一个 terminal 的串行 SQL 序列。

    执行逻辑：
    1. 每个 terminal 独占一个临时 DuckDB/`HOME` 环境，避免互相污染。
    2. terminal 内严格按 query 顺序串行执行。
    3. 对 batch-capable query 注入 TPCC workload 环境变量，使其在 provider 层
       走“带数量上限的滚动补发”路径。
    4. 对 non-batch-capable query 不注入这些变量，保持原生执行路径。
    5. 记录每条 SQL 的结果、trace 和失败传播状态。
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    parser = argparse.ArgumentParser(
        description="Run one TPCC-style FlockMTL terminal worker"
    )
    parser.add_argument("--system", default="flockmtl")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--scale-factor", type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--terminal-id", type=int, required=True)
    parser.add_argument(
        "--failure-policy",
        choices=["fail_fast", "continue"],
        default="fail_fast",
    )
    parser.add_argument("--metric-output", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--query-trace-root", default=None)
    parser.add_argument(
        "--cacheblend-mode",
        choices=["auto", "on", "off"],
        default=os.environ.get(CACHEBLEND_MODE_ENV, "auto"),
        help="Control FlockMTL SQL-side cacheblend markers.",
    )
    args = parser.parse_args()
    args.cacheblend_mode = normalize_cacheblend_mode(args.cacheblend_mode)
    os.environ[CACHEBLEND_MODE_ENV] = args.cacheblend_mode

    query_ids = parse_query_sequence(args.queries)
    metric_output = Path(args.metric_output)
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    query_trace_root = (
        Path(args.query_trace_root) if args.query_trace_root is not None else None
    )

    os.environ["FLOCKMTL_QUERY_CONCURRENCY"] = "1"
    workload_env = os.environ.copy()

    from flockmtl_phase_timing import timed_phase
    terminal_started_at = time.time()
    query_records: list[dict[str, object]] = []

    try:
        with timed_phase("runner_import", terminal_id=args.terminal_id):
            runner_class = get_runner_class(args.system, args.use_case)
        if not runner_class:
            raise RuntimeError(
                f"Failed to import runner for {args.system}/{args.use_case}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f"flockmtl_terminal_{args.terminal_id}_",
            dir=os.environ.get("FLOCKMTL_TERMINAL_TMPDIR") or None,
        ) as temp_dir:
            # 每个 terminal 使用自己的临时 HOME 和 DuckDB 文件，确保串行状态独立。
            home_dir = Path(temp_dir)
            (home_dir / ".duckdb" / "flock_storage").mkdir(
                parents=True, exist_ok=True
            )
            temp_db = home_dir / f"{args.use_case}_terminal_{args.terminal_id}.duckdb"
            os.environ["HOME"] = str(home_dir)
            os.environ["FLOCKMTL_DB_FILE"] = str(temp_db)

            with timed_phase("runner_initialization", terminal_id=args.terminal_id):
                runner = runner_class(
                    use_case=args.use_case,
                    scale_factor=args.scale_factor,
                    skip_setup=args.skip_setup,
                    model_name=args.model,
                )
            ready_at = time.time()
            release_at = None
            if os.environ.get("FLOCK_TPCC_START_BARRIER_DIR"):
                from flockmtl_start_barrier import wait_for_query_start
                ready_at, release_at = wait_for_query_start(
                    os.environ["FLOCK_TPCC_START_BARRIER_DIR"], args.terminal_id,
                    int(os.environ["FLOCK_TPCC_START_BARRIER_COUNT"]))
                print("[query_start_barrier] " + json.dumps({"terminal_id": args.terminal_id,
                      "ready_at": ready_at, "released_at": release_at}), flush=True)
            terminal_failed = False
            try:
                for step_index, query_id in enumerate(query_ids):
                    step_started_at = time.time()
                    query_text = runner._get_query_text(query_id)  # noqa: SLF001
                    query_mode = classify_query_mode(query_text)
                    query_trace_output = None
                    if (
                        query_trace_root is not None
                        and str(query_mode["request_mode"]) == "batch-capable"
                    ):
                        query_trace_output = (
                            query_trace_root / f"step_{step_index:02d}_Q{query_id}.json"
                        )

                    desired_env = build_query_execution_env(
                        workload_env=workload_env,
                        query_mode=query_mode,
                        terminal_id=args.terminal_id,
                        query_id=query_id,
                        step_index=step_index,
                        query_trace_output=query_trace_output,
                    )
                    # 环境变量只在当前 SQL 执行期间生效，执行后立即恢复。
                    previous_env = swap_managed_query_env(desired_env)
                    try:
                        from flockmtl_screening_progress import track_screening_query
                        with track_screening_query(args.terminal_id, step_index) as progress:
                            with timed_phase("runner_execute_query", terminal_id=args.terminal_id, step=step_index, query_id=query_id):
                                metric = runner.execute_query(query_id)
                            progress['success'] = metric.status == 'success'
                    finally:
                        restore_managed_query_env(previous_env)
                    step_completed_at = time.time()
                    postprocess_started = time.perf_counter()
                    query_trace = load_query_trace(query_trace_output)
                    (
                        logical_llm_request_count,
                        trace_llm_request_count,
                        llm_request_count_source,
                    ) = resolve_logical_llm_request_count(
                        query_id,
                        metric.results,
                        query_trace,
                    )

                    result_output = (
                        result_root / f"step_{step_index:02d}_Q{query_id}.pkl"
                    )
                    metric.results.to_pickle(result_output)

                    remaining_query_count = max(
                        0, len(query_ids) - (step_index + 1)
                    )
                    update_terminal_coord_state(
                        workload_env=workload_env,
                        terminal_id=args.terminal_id,
                        remaining_query_count=(
                            0
                            if metric.status != "success" and args.failure_policy == "fail_fast"
                            else remaining_query_count
                        ),
                        status=(
                            "completed"
                            if (metric.status != "success" and args.failure_policy == "fail_fast")
                            or remaining_query_count == 0
                            else "active"
                        ),
                    )

                    if os.environ.get("FLOCK_PHASE_TIMING") == "1":
                        print("[phase_timing] " + json.dumps({"phase": "query_postprocess", "terminal_id": args.terminal_id, "step": step_index, "elapsed_s": time.perf_counter() - postprocess_started}), flush=True)
                    query_record = {
                        "sql_execution_interval": {"start": progress["start"], "end": progress["end"]},
                        "terminal_ready_at": ready_at,
                        "execution_released_at": release_at,
                        "step_index": step_index,
                        "query_id": query_id,
                        "status": metric.status,
                        "request_mode": query_mode["request_mode"],
                        "batch_capable_functions": query_mode[
                            "batch_capable_functions"
                        ],
                        "non_batch_capable_functions": query_mode[
                            "non_batch_capable_functions"
                        ],
                        "execution_time": metric.execution_time,
                        "query_wall_time": metric.query_wall_time,
                        "time_to_first_vllm_request": (
                            metric.time_to_first_vllm_request
                        ),
                        "first_vllm_request_path": (
                            metric.first_vllm_request_path
                        ),
                        "observed_vllm_request_count": (
                            metric.observed_vllm_request_count
                        ),
                        "row_count": len(metric.results),
                        "started_at": step_started_at,
                        "completed_at": step_completed_at,
                        "error": metric.error,
                        "result_output": str(result_output),
                        "wave_count": (
                            int(query_trace["wave_count"])
                            if query_trace is not None
                            and query_trace.get("wave_count") is not None
                            else 0
                        ),
                        "total_llm_requests": (
                            logical_llm_request_count
                        ),
                        "trace_llm_request_count": trace_llm_request_count,
                        "llm_request_count_source": llm_request_count_source,
                        "waves": query_trace.get("waves", [])
                        if query_trace is not None
                        else [],
                        "query_trace_output": (
                            str(query_trace_output)
                            if query_trace_output is not None
                            else None
                        ),
                        "query_trace_load_error": (
                            query_trace.get("trace_load_error")
                            if query_trace is not None
                            else None
                        ),
                        "request_scheduling_mode": (
                            query_trace.get("request_scheduling_mode")
                            if query_trace is not None
                            else None
                        ),
                        "configured_request_cap": (
                            int(query_trace["configured_request_cap"])
                            if query_trace is not None
                            and query_trace.get("configured_request_cap")
                            is not None
                            else (
                                int(desired_env["FLOCK_VLLM_MAX_IN_FLIGHT"])
                                if desired_env.get("FLOCK_VLLM_MAX_IN_FLIGHT")
                                is not None
                                else None
                            )
                        ),
                        "final_effective_request_cap": (
                            int(query_trace["final_effective_request_cap"])
                            if query_trace is not None
                            and query_trace.get("final_effective_request_cap")
                            is not None
                            else None
                        ),
                        "min_observed_request_cap": (
                            int(query_trace["min_observed_request_cap"])
                            if query_trace is not None
                            and query_trace.get("min_observed_request_cap")
                            is not None
                            else None
                        ),
                        "max_observed_request_cap": (
                            int(query_trace["max_observed_request_cap"])
                            if query_trace is not None
                            and query_trace.get("max_observed_request_cap")
                            is not None
                            else None
                        ),
                        "active_terminal_count_at_query_start": (
                            int(query_trace["active_terminal_count_at_query_start"])
                            if query_trace is not None
                            and query_trace.get("active_terminal_count_at_query_start")
                            is not None
                            else (
                                int(desired_env["FLOCK_TPCC_ACTIVE_TERMINAL_COUNT"])
                                if desired_env.get(
                                    "FLOCK_TPCC_ACTIVE_TERMINAL_COUNT"
                                )
                                is not None
                                else None
                            )
                        ),
                        "min_observed_active_terminal_count": (
                            int(query_trace["min_observed_active_terminal_count"])
                            if query_trace is not None
                            and query_trace.get("min_observed_active_terminal_count")
                            is not None
                            else None
                        ),
                        "max_observed_active_terminal_count": (
                            int(query_trace["max_observed_active_terminal_count"])
                            if query_trace is not None
                            and query_trace.get("max_observed_active_terminal_count")
                            is not None
                            else None
                        ),
                        "request_cap_source": desired_env.get(
                            "FLOCK_TPCC_REQUEST_CAP_SOURCE"
                        ),
                        "dynamic_request_cap_refresh_count": (
                            int(query_trace["dynamic_request_cap_refresh_count"])
                            if query_trace is not None
                            and query_trace.get("dynamic_request_cap_refresh_count")
                            is not None
                            else 0
                        ),
                    }
                    # 成功 query 的质量评估由 workload 父进程在所有 terminal
                    # 完成、性能计时窗口闭合后统一执行。这里不能同步 evaluate，
                    # 否则大结果集的质量计算会阻塞 terminal 发起下一条 query，
                    # 从而污染 query throughput。失败/跳过记录仍由下方逻辑附带
                    # skipped evaluation，确保错误信息不会丢失。
                    query_records.append(query_record)
                    # 每完成一个 query 就刷新 terminal metric；质量字段稍后回填。
                    write_terminal_metric(
                        metric_output,
                        build_terminal_metric(
                            terminal_id=args.terminal_id,
                            failure_policy=args.failure_policy,
                            query_ids=query_ids,
                            terminal_started_at=terminal_started_at,
                            query_records=query_records,
                            status="running",
                        ),
                    )

                    if metric.status != "success":
                        terminal_failed = True
                        if args.failure_policy == "fail_fast":
                            update_terminal_coord_state(
                                workload_env=workload_env,
                                terminal_id=args.terminal_id,
                                remaining_query_count=0,
                                status="failed",
                            )
                            append_skipped_queries(
                                query_records,
                                query_ids,
                                step_index + 1,
                                args.terminal_id,
                            )
                            write_terminal_metric(
                                metric_output,
                                build_terminal_metric(
                                    terminal_id=args.terminal_id,
                                    failure_policy=args.failure_policy,
                                    query_ids=query_ids,
                                    terminal_started_at=terminal_started_at,
                                    query_records=query_records,
                                    status="failed",
                                ),
                            )
                            break
            finally:
                close_runner = getattr(runner, "close", None)
                if callable(close_runner):
                    close_runner()

            terminal_completed_at = time.time()
            terminal_metric = build_terminal_metric(
                terminal_id=args.terminal_id,
                failure_policy=args.failure_policy,
                query_ids=query_ids,
                terminal_started_at=terminal_started_at,
                query_records=query_records,
                status="failed" if terminal_failed else "success",
            )
            terminal_metric["completed_at"] = terminal_completed_at
            terminal_metric["wall_time_s"] = (
                terminal_completed_at - terminal_started_at
            )
            write_terminal_metric(metric_output, terminal_metric)
            return 1 if terminal_failed else 0

    except Exception as exc:
        update_terminal_coord_state(
            workload_env=workload_env,
            terminal_id=args.terminal_id,
            remaining_query_count=0,
            status="failed",
        )
        terminal_completed_at = time.time()
        failed_metric = build_terminal_metric(
            terminal_id=args.terminal_id,
            failure_policy=args.failure_policy,
            query_ids=query_ids,
            terminal_started_at=terminal_started_at,
            query_records=query_records,
            status="failed",
            error=str(exc),
        )
        failed_metric["completed_at"] = terminal_completed_at
        failed_metric["wall_time_s"] = terminal_completed_at - terminal_started_at
        write_terminal_metric(metric_output, failed_metric)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
