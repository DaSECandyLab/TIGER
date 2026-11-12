"""Flock trial 的 SemBench 和 TPCC workload 执行。"""

from __future__ import annotations

import argparse
import os
import re
from .workload_monitor import run_monitored_workload
import subprocess
import tempfile
from pathlib import Path

from .arguments import (
    cacheblend_enabled,
    effective_query_ids,
    normalize_query_id,
    validate_tpcc_terminal_query_args,
)
def _remove_cacheblend_fields(sql_text: str) -> str:
    updated = re.sub(
        r"^[ \t]*'cacheblend'\s*:\s*true\s*,\s*\n?",
        "",
        sql_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    updated = re.sub(
        r"^[ \t]*'blend_special_str'\s*:\s*'[^']*'\s*,\s*\n?",
        "",
        updated,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return updated


def _ensure_prompt_has_blend_markers(prompt_text: str, blend_special_str: str) -> str:
    if blend_special_str in prompt_text:
        return prompt_text
    return re.sub(
        r"(\{\{[^}]+\}\})",
        lambda match: f"{blend_special_str}{match.group(1)}",
        prompt_text,
    )


def rewrite_query_for_cacheblend(
    sql_text: str,
    enable_cacheblend: bool,
    blend_special_str: str,
) -> str:
    updated = _remove_cacheblend_fields(sql_text)
    if not enable_cacheblend:
        return updated
    if "context_columns" not in updated:
        raise ValueError(
            "cacheblend rewrite requires the SQL to define context_columns"
        )
    prompt_match = re.search(
        r"('prompt'\s*:\s*')(?P<prompt>(?:''|[^'])*)(')",
        updated,
        flags=re.DOTALL,
    )
    if prompt_match is not None:
        prompt_text = prompt_match.group("prompt")
        updated = (
            updated[: prompt_match.start("prompt")]
            + _ensure_prompt_has_blend_markers(prompt_text, blend_special_str)
            + updated[prompt_match.end("prompt") :]
        )
    insertion = (
        f"        'cacheblend': true,\n"
        f"        'blend_special_str': '{blend_special_str}',\n"
    )
    updated = re.sub(
        r"('prompt'\s*:\s*'(?:''|[^'])*'\s*,)",
        lambda match: match.group(1) + "\n" + insertion.rstrip("\n"),
        updated,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    return updated


def prepare_sembench_query_override(
    args: argparse.Namespace,
) -> tuple[tempfile.TemporaryDirectory[str] | None, str | None]:
    experiment_root_value = getattr(args, "sembench_files_override_root", None)
    if args.cacheblend_mode == "auto" and not experiment_root_value:
        return None, None
    source_query_dir = Path(args.sembench_dir) / "files" / args.use_case / "query" / "flockmtl"
    if not source_query_dir.exists():
        raise FileNotFoundError(f"Missing SemBench query directory: {source_query_dir}")
    experiment_query_dir = (
        Path(experiment_root_value).expanduser().resolve()
        / args.use_case
        / "query"
        / "flockmtl"
        if experiment_root_value
        else None
    )
    if experiment_query_dir is not None and not experiment_query_dir.is_dir():
        raise FileNotFoundError(
            f"Missing experiment query override directory: {experiment_query_dir}"
        )
    temp_dir = tempfile.TemporaryDirectory(prefix="sembench_query_override_")
    override_files_root = Path(temp_dir.name) / "files"
    target_query_dir = override_files_root / args.use_case / "query" / "flockmtl"
    target_query_dir.mkdir(parents=True, exist_ok=True)
    for query in effective_query_ids(args):
        query_suffix = normalize_query_id(query)
        query_patterns = (
            list(experiment_query_dir.glob(f"[qQ]{query_suffix}.*"))
            if experiment_query_dir is not None
            else []
        )
        if not query_patterns:
            query_patterns = list(source_query_dir.glob(f"[qQ]{query_suffix}.*"))
        if len(query_patterns) != 1:
            raise FileNotFoundError(
                f"Expected exactly one query file for Q{query_suffix} in the experiment override or {source_query_dir}, found {query_patterns}"
            )
        source_path = query_patterns[0]
        rewritten_sql = rewrite_query_for_cacheblend(
            source_path.read_text(encoding="utf-8"),
            enable_cacheblend=cacheblend_enabled(args),
            blend_special_str=args.cacheblend_blend_special_str,
        )
        (target_query_dir / source_path.name).write_text(
            rewritten_sql,
            encoding="utf-8",
        )
    return temp_dir, str(override_files_root)
def run_sembench(
    args: argparse.Namespace,
    log_path: Path,
    query_override_root: str | None,
) -> subprocess.CompletedProcess[str]:
    """运行原有 SemBench 路径。

    这条路径仍然直接调用 `src/run.py`，适合和 TPCC workload driver 做 A/B 对比。
    """
    sembench_dir = Path(args.sembench_dir)
    python = sembench_dir / ".venvs" / "sembench" / "bin" / "python"
    cmd = [
        str(python),
        "src/run.py",
        "--systems",
        "flockmtl",
        "--use-cases",
        args.use_case,
        "--queries",
        *[str(query) for query in args.queries],
        "--model",
        args.served_model_name,
        "--scale-factor",
        str(args.scale_factor),
        "--skip-setup",
    ]
    env = os.environ.copy()
    env.update(
        {
            "FLOCK_BOOL_SCHEMA_MODE": "string_enum",
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1",
            "FLOCK_OPENAI_API_KEY": "dummy",
            "FLOCK_OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1",
            "FLOCK_EXT": args.flock_ext,
            "FLOCK_VLLM_REQUEST_RATE": str(args.flock_request_rate),
            "FLOCK_VLLM_REQUEST_ARRIVAL": str(args.flock_request_arrival),
            "FLOCK_VLLM_MAX_IN_FLIGHT": str(args.flock_max_in_flight),
            # 逻辑 in-flight 上限可以很大，但不应同时制造同等数量的 TCP 连接；
            # libcurl 会让超过该连接预算的请求在客户端等待可复用连接。
            "FLOCK_VLLM_MAX_HTTP_CONNECTIONS": str(
                args.flock_http_max_connections
            ),
            "FLOCKMTL_QUERY_CONCURRENCY": str(args.flock_query_concurrency),
            "FLOCKMTL_QUERY_START_STAGGER_MS": str(
                args.flock_query_start_stagger_ms
            ),
            "SEMBENCH_CACHEBLEND_MODE": args.cacheblend_mode,
        }
    )
    if query_override_root is not None:
        env["SEMBENCH_FILES_OVERRIDE_ROOT"] = query_override_root
    else:
        env.pop("SEMBENCH_FILES_OVERRIDE_ROOT", None)
    env.pop("FLOCK_MAX_TOKENS", None)
    return run_monitored_workload(
        cmd, cwd=args.sembench_dir, env=env, log_path=log_path,
        monitor=getattr(args, "_inference_service_monitor", None),
    )


def run_tpcc_workload(
    args: argparse.Namespace,
    log_path: Path,
    workload_metrics_path: Path,
    workload_artifact_dir: Path,
    query_override_root: str | None,
) -> subprocess.CompletedProcess[str]:
    """运行 TPCC-style workload driver。

    这里不再直接让 SemBench 一次性跑完整组 query，而是切到
    `flockmtl_tpcc_workload.py`：
    - terminal 间并发
    - terminal 内串行
    - batch-capable query 走数量受控的 rolling refill
    - non-batch-capable query 保持原生路径
    """
    if args.flock_terminal_count < 1:
        raise ValueError("flock terminal count must be >= 1")
    validate_tpcc_terminal_query_args(args)

    if (
        args.flock_request_control_mode == "max_in_flight"
        and not isinstance(args.flock_max_in_flight, int)
    ):
        raise ValueError(
            "TPCC workload mode with request_control_mode=max_in_flight "
            "requires an explicit integer --flock-max-in-flight"
        )

    if (
        args.flock_request_control_mode == "vllm_batch_size_div_terminals"
        and (args.flock_vllm_batch_size is None or args.flock_vllm_batch_size < 1)
    ):
        raise ValueError(
            "TPCC workload mode with request_control_mode="
            "vllm_batch_size_div_terminals requires --flock-vllm-batch-size >= 1"
        )

    sembench_dir = Path(args.sembench_dir)
    python = sembench_dir / ".venvs" / "sembench" / "bin" / "python"
    worker_script = sembench_dir / "src" / "flockmtl_tpcc_workload.py"
    cmd = [
        str(python),
        str(worker_script),
        "--system",
        "flockmtl",
        "--use-case",
        args.use_case,
        "--model",
        args.served_model_name,
        "--scale-factor",
        str(args.scale_factor),
        "--skip-setup",
        "--terminal-count",
        str(args.flock_terminal_count),
        "--request-control-mode",
        args.flock_request_control_mode,
        "--request-rate",
        str(args.flock_request_rate),
        "--request-arrival",
        str(args.flock_request_arrival),
        "--failure-policy",
        args.flock_failure_policy,
        "--metrics-output",
        str(workload_metrics_path),
        "--artifact-root",
        str(workload_artifact_dir),
    ]
    if args.flock_terminal_queries is not None:
        for query_sequence in args.flock_terminal_queries:
            cmd += [
                "--terminal-queries",
                *[str(query) for query in query_sequence],
            ]
    else:
        cmd += [
            "--queries",
            *[str(query) for query in args.queries],
        ]
    if args.flock_request_control_mode == "max_in_flight":
        cmd += [
            "--max-in-flight",
            str(args.flock_max_in_flight),
            "--max-in-flight-strategy",
            str(
                getattr(
                    args,
                    "flock_max_in_flight_strategy",
                    "dynamic_active_terminal_partition",
                )
            ),
        ]
    else:
        cmd += ["--vllm-batch-size", str(args.flock_vllm_batch_size)]
    if getattr(args, "quality_thresholds_json", None) is not None:
        cmd += [
            "--quality-thresholds-json",
            str(args.quality_thresholds_json),
        ]

    env = os.environ.copy()
    env.update(
        {
            "FLOCK_BOOL_SCHEMA_MODE": "string_enum",
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1",
            "FLOCK_OPENAI_API_KEY": "dummy",
            "FLOCK_OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1",
            "FLOCK_EXT": args.flock_ext,
            # TPCC terminal 子进程继承该全局预算，并按 terminal 总数均分，
            # 避免多个进程各自使用完整连接额度后再次击穿 listen queue。
            "FLOCK_VLLM_MAX_HTTP_CONNECTIONS": str(
                args.flock_http_max_connections
            ),
            # 该标记仅用于 trial 结果侧标识当前采用的是 TPCC workload driver。
            "FLOCK_TPCC_WORKLOAD_DRIVER": "1",
            "SEMBENCH_CACHEBLEND_MODE": args.cacheblend_mode,
        }
    )
    if query_override_root is not None:
        env["SEMBENCH_FILES_OVERRIDE_ROOT"] = query_override_root
    else:
        env.pop("SEMBENCH_FILES_OVERRIDE_ROOT", None)
    env.pop("FLOCK_MAX_TOKENS", None)
    return run_monitored_workload(
        cmd, cwd=args.sembench_dir, env=env, log_path=log_path,
        monitor=getattr(args, "_inference_service_monitor", None),
    )
