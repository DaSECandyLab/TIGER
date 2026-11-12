"""单次 Flock trial 的编排入口。"""

from __future__ import annotations

import argparse
from SemBench.src.flockmtl_timing import query_execution_timing
import functools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Keep diagnostics on stderr so CLI results on stdout remain machine-readable.
print = functools.partial(print, file=sys.stderr)

from .workload_monitor import ServiceMonitor

from .arguments import (
    apply_generated_baseline_config,
    apply_tuning_reference_defaults,
    lmcache_enabled,
    resolve_flock_max_in_flight_arg,
    validate_tpcc_terminal_query_args,
)
from .configuration import apply_candidate_request
from .metrics import (
    RuntimeFeatureSampler,
    load_sembench_metrics,
    load_tpcc_sembench_metrics,
    load_tpcc_workload_metrics,
    parse_prometheus,
    summarize_sembench,
)
from .environment import collect_runtime_environment
from .result import build_result, write_workload_feature_files
from .vllm import (
    build_vllm_environment,
    build_vllm_command,
    http_get,
    read_tcp_listener_counters,
    tcp_counter_delta,
    terminate_process_group,
    terminate_port,
    wait_for_vllm,
)
from .workload import (
    prepare_sembench_query_override,
    run_sembench,
    run_tpcc_workload,
)


def resolve_objective_wall_time(
    *,
    flock_workload_driver: str,
    process_wall_time_s: float,
    workload_metrics: dict[str, object],
) -> tuple[float, str]:
    """Choose the active workload window used by throughput objectives."""
    if flock_workload_driver == "tpcc":
        if workload_metrics.get("timing_status") == "unavailable":
            return 0.0, "flock_workload_metrics.query_window_unavailable"
        workload_wall_time = workload_metrics.get("wall_time_s")
        if isinstance(workload_wall_time, (int, float)) and workload_wall_time > 0:
            timing = query_execution_timing(workload_metrics.get("terminals", []))
            if timing:
                return (float(timing["execution_wall_time_s"]),
                        "flock_workload_metrics." + timing["timing_scope"])
            return (
                float(workload_wall_time),
                "flock_workload_metrics.wall_time_s_before_quality_evaluation",
            )
    return float(process_wall_time_s), "trial_process_wall_time_s"


def apply_trial_configuration(args: argparse.Namespace) -> None:
    """应用本轮显式配置，Baseline 场景不注入历史参考默认值。"""

    # Agent 配置只应用初筛候选；未入选字段保留 trial/vLLM 的真实默认值。
    has_generated_baseline = bool(getattr(args, "baseline_config", None))
    preserve_engine_defaults = bool(
        getattr(args, "preserve_engine_defaults", False)
    )
    apply_generated_baseline_config(args)
    if not has_generated_baseline and not preserve_engine_defaults:
        apply_tuning_reference_defaults(args)
    # Baseline 只提供锚点和未搜索字段；固定搜索空间内的本轮候选最后覆盖。
    apply_candidate_request(args)


def run_trial(args: argparse.Namespace) -> int:
    """运行已配置的 trial，并持久化其结果和特征产物。"""
    apply_trial_configuration(args)
    args.flock_max_in_flight = resolve_flock_max_in_flight_arg(args)
    validate_tpcc_terminal_query_args(args)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    trial_name = args.trial_name or time.strftime("flock_%Y%m%dT%H%M%SZ", time.gmtime())
    logs_dir, workload_dir = result_dir / "logs", result_dir / "workloads"
    artifact_root, feature_dir = result_dir / "artifacts", result_dir / "features"
    for directory in (logs_dir, workload_dir, artifact_root, feature_dir):
        directory.mkdir(parents=True, exist_ok=True)
    vllm_log = logs_dir / f"{trial_name}.vllm.log"
    sembench_log = logs_dir / f"{trial_name}.sembench.log"
    metrics_before_log = logs_dir / f"{trial_name}.metrics.before.prom"
    metrics_after_log = logs_dir / f"{trial_name}.metrics.after.prom"
    workload_metrics_path = workload_dir / f"{trial_name}.workload.json"
    workload_artifact_dir = artifact_root / trial_name
    workload_artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime_environment_path = workload_artifact_dir / "runtime_environment.json"
    runtime_environment = collect_runtime_environment(args)
    runtime_environment_path.write_text(
        json.dumps(runtime_environment, indent=2), encoding="utf-8"
    )
    result_path = result_dir / f"vllm-{trial_name}.json"
    feature_json_path = feature_dir / f"{trial_name}.features.json"
    feature_doc_path = feature_dir / f"{trial_name}.features.md"
    base_url = f"http://{args.metrics_host}:{args.port}"
    # 在启动前保存实际构造的完整命令；未知但可执行的初筛候选也会出现在这里。
    command = build_vllm_command(args)
    setattr(args, "_effective_vllm_command", list(command))
    setattr(args, "_vllm_started_by_trial", not args.no_start_vllm)
    vllm_proc: subprocess.Popen[str] | None = None
    vllm_pgid: int | None = None
    query_override_tempdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        query_override_tempdir, query_override_root = prepare_sembench_query_override(args)
        if args.no_start_vllm:
            wait_for_vllm(base_url, 30)
        else:
            # API 端口由命令行控制；不要把同名 shell 变量误传给 vLLM 内部
            # TCPStore 端口选择逻辑，否则 DP engine 并发启动时会争抢端口。
            vllm_env = build_vllm_environment(os.environ)
            if lmcache_enabled(args):
                vllm_env["LMCACHE_CHUNK_SIZE"] = str(args.lmcache_chunk_size)
            # 保持历史启动语义：每个 trial 只启动一次；配置或环境错误直接暴露，
            # 不通过自动重试改变 trial 的执行结果。
            terminate_port(args.port)
            with vllm_log.open("w", encoding="utf-8") as log_file:
                log_file.write("$ " + " ".join(command) + "\n")
                log_file.flush()
                vllm_proc = subprocess.Popen(
                    command,
                    env=vllm_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                vllm_pgid = vllm_proc.pid
            wait_for_vllm(base_url, args.startup_timeout, proc=vllm_proc)
        setattr(args, "_inference_service_monitor", ServiceMonitor(
            vllm_proc, vllm_log if vllm_proc is not None else None))
        before_metrics_text = http_get(f"{base_url}/metrics", timeout=10)
        metrics_before_log.write_text(before_metrics_text, encoding="utf-8")
        # Screening captures counters before terminating a slow trial. Keep the
        # endpoint and pre-workload snapshot with that trial's progress artifacts.
        screening_progress = os.environ.get('FLOCK_SCREENING_PROGRESS_DIR')
        if screening_progress:
            try:
                telemetry = Path(screening_progress) / 'telemetry'
                telemetry.mkdir(parents=True, exist_ok=True)
                (telemetry / 'before.prom').write_text(before_metrics_text, encoding='utf-8')
                temporary = telemetry / 'source.tmp'
                temporary.write_text(json.dumps({'metrics_url': f'{base_url}/metrics'}))
                temporary.replace(telemetry / 'source.json')
            except OSError:
                pass  # Missing diagnostic data must not fail the workload.
        before_samples = parse_prometheus(before_metrics_text)
        tcp_listener_before = read_tcp_listener_counters()
        sampler = RuntimeFeatureSampler(f"{base_url}/metrics")
        started = time.time()
        sampler.start()
        try:
            if args.flock_workload_driver == "tpcc":
                sembench_proc = run_tpcc_workload(
                    args, sembench_log, workload_metrics_path, workload_artifact_dir,
                    query_override_root,
                )
            else:
                sembench_proc = run_sembench(args, sembench_log, query_override_root)
        finally:
            runtime_sample_summary = sampler.stop()
        process_wall_time_s = time.time() - started
        after_metrics_text = http_get(f"{base_url}/metrics", timeout=10)
        metrics_after_log.write_text(after_metrics_text, encoding="utf-8")
        after_samples = parse_prometheus(after_metrics_text)
        tcp_listener_after = read_tcp_listener_counters()
        if args.flock_workload_driver == "tpcc":
            workload_metrics = load_tpcc_workload_metrics(workload_metrics_path)
            sembench_metrics = load_tpcc_sembench_metrics(workload_metrics)
            sembench_summary = workload_metrics.get("summary", {})
        else:
            workload_metrics = {}
            sembench_metrics = load_sembench_metrics(args)
            sembench_summary = summarize_sembench(sembench_metrics)
        objective_wall_time_s, objective_wall_time_source = (
            resolve_objective_wall_time(
                flock_workload_driver=args.flock_workload_driver,
                process_wall_time_s=process_wall_time_s,
                workload_metrics=workload_metrics,
            )
        )
        result = build_result(
            args=args, sembench_metrics=sembench_metrics, sembench_summary=sembench_summary,
            workload_metrics=workload_metrics, before_samples=before_samples,
            after_samples=after_samples,
            wall_time_s=objective_wall_time_s,
            sembench_returncode=sembench_proc.returncode,
            logs={
                "vllm": str(vllm_log),
                "sembench": str(sembench_log),
                "workload": (
                    str(workload_metrics_path)
                    if args.flock_workload_driver == "tpcc"
                    else None
                ),
                "metrics_before": str(metrics_before_log),
                "metrics_after": str(metrics_after_log),
            },
            runtime_sample_summary=runtime_sample_summary,
        )
        result.update(trial_name=trial_name, trial_result_path=str(result_path),
                      workload_feature_json_path=str(feature_json_path),
                      workload_feature_doc_path=str(feature_doc_path))
        result["trial_process_wall_time_s"] = process_wall_time_s
        result["objective_wall_time_source"] = objective_wall_time_source
        timing = query_execution_timing(
            workload_metrics.get("terminals", []), workload_metrics.get("started_at"),
            workload_metrics.get("completed_at"))
        result["workload_wall_time_including_preparation_s"] = workload_metrics.get(
            "total_wall_time_s", workload_metrics.get("wall_time_s"))
        result["excluded_preparation_time_s"] = timing.get("preparation_time_s")
        result["excluded_teardown_time_s"] = timing.get("teardown_time_s")
        result["execution_timing_scope"] = timing.get("timing_scope", "legacy_fallback")
        result["quality_evaluation_wall_time_s"] = workload_metrics.get(
            "quality_evaluation_wall_time_s"
        )
        result["tcp_listener_counters"] = {
            "scope": "node_wide_during_workload",
            "before": tcp_listener_before,
            "after": tcp_listener_after,
            "delta": tcp_counter_delta(tcp_listener_before, tcp_listener_after),
        }
        # 解释器、boot ID、GPU/驱动和关键制品随 trial 一起保存；即使 workload
        # 失败，前面提前写出的独立文件也能用于复盘隐藏环境漂移。
        result["runtime_environment"] = runtime_environment
        result["runtime_environment_path"] = str(runtime_environment_path)
        from ...timeout_estimation import baseline_timeout_reference

        result["query_timeout_reference"] = baseline_timeout_reference(result)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_workload_feature_files(result, feature_json_path=feature_json_path,
                                     feature_doc_path=feature_doc_path)
        print(f"Result written to: {result_path}")
        print(f"Workload features written to: {feature_doc_path}")
        print(json.dumps(result["sembench_summary"], indent=2))
        return sembench_proc.returncode
    finally:
        if vllm_proc is not None and not args.keep_vllm:
            terminate_process_group(vllm_proc, vllm_pgid)
        if query_override_tempdir is not None:
            query_override_tempdir.cleanup()


def main() -> int:
    from .arguments import parse_args

    return run_trial(parse_args())
