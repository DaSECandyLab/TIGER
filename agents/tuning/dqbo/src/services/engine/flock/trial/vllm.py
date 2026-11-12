"""Flock trial 的 vLLM 进程生命周期和命令构造。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

from ..config import resolve_effective_dtype, resolve_effective_kv_cache_dtype
from .arguments import (
    hma_disable_required,
    normalized_cpu_offload_gb,
    normalized_kv_offloading_size,
    resolve_kv_transfer_config,
    resolve_model_path,
    resolve_speculative_config,
    resolved_chunked_prefill_setting,
    resolved_kv_offloading_backend,
    resolved_prefix_caching_setting,
)

DEFAULT_VLLM_GENERATION_CONFIG = "vllm"
STRUCTURED_OUTPUTS_CONFIG = '{"backend":"xgrammar","disable_any_whitespace":true}'
TCP_LISTENER_COUNTERS = (
    "ListenOverflows",
    "ListenDrops",
    "TCPReqQFullDoCookies",
    "TCPBacklogDrop",
)


def read_tcp_listener_counters(
    netstat_path: Path = Path("/proc/net/netstat"),
) -> dict[str, int]:
    """读取节点级 TCP listener 计数，供 trial 前后做非侵入式差分。"""

    try:
        lines = netstat_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for index in range(0, len(lines) - 1, 2):
        names = lines[index].split()
        values = lines[index + 1].split()
        if not names or names[0] != "TcpExt:" or not values or values[0] != "TcpExt:":
            continue
        parsed = dict(zip(names[1:], values[1:]))
        try:
            return {
                name: int(parsed[name])
                for name in TCP_LISTENER_COUNTERS
                if name in parsed
            }
        except ValueError:
            return {}
    return {}


def tcp_counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    """计算单个 trial 期间单调 TCP 计数器的增量。"""

    return {
        name: max(0, int(after[name]) - int(before[name]))
        for name in before.keys() & after.keys()
    }


def build_vllm_environment(
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """构造 vLLM 子进程环境，并隔离 API 端口和内部通信端口。

    ``VLLM_PORT`` 不是 ``vllm serve --port`` 的同义环境变量。vLLM 会把它
    当作内部通信端口扫描起点；DP engine 并发启动时，多个进程可能同时选中
    ``VLLM_PORT + 1``，随后在初始化 TCPStore 时触发 ``EADDRINUSE``。外部 API
    端口已经通过 ``--port`` 显式传入，因此这里必须移除继承的 ``VLLM_PORT``，
    让各 engine 使用操作系统分配的临时端口。
    """
    environment = dict(base_environment or os.environ)
    environment.pop("VLLM_PORT", None)
    # Prefix-cache block hashes must be stable across independently launched
    # Trials when cbor-based hash algorithms are in the frozen Range.
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def http_get(url: str, timeout: int = 5) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def wait_for_vllm(
    base_url: str,
    timeout_s: int,
    proc: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready with returncode={proc.returncode}"
            )
        try:
            http_get(f"{base_url}/v1/models", timeout=5)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"vLLM did not become ready within {timeout_s}s: {last_error}")


def terminate_port(port: int) -> None:
    if not is_port_open("127.0.0.1", port):
        return
    subprocess.run(
        ["bash", "-lc", f"lsof -t -i:{port} | xargs -r kill -TERM"],
        check=False,
    )
    for _ in range(10):
        if not is_port_open("127.0.0.1", port):
            return
        time.sleep(1)
    subprocess.run(
        ["bash", "-lc", f"lsof -t -i:{port} | xargs -r kill -KILL"],
        check=False,
    )


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_exited_group_leader(proc: subprocess.Popen[str] | None) -> None:
    """Reap a terminated direct child so its zombie does not keep the PGID alive."""

    if proc is None:
        return
    # Popen.poll() performs waitpid(WNOHANG) and therefore reaps the direct
    # child when it has exited. Descendants, if any, remain visible through
    # process_group_exists and are still covered by the TERM/KILL deadlines.
    proc.poll()


def terminate_process_group(
    proc: subprocess.Popen[str] | None,
    pgid: int | None,
    timeout_s: int = 30,
) -> None:
    if pgid is None:
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
    if not process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _reap_exited_group_leader(proc)
        if not process_group_exists(pgid):
            return
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    kill_deadline = time.time() + 10
    while time.time() < kill_deadline:
        _reap_exited_group_leader(proc)
        if not process_group_exists(pgid):
            return
        time.sleep(0.2)
    _reap_exited_group_leader(proc)


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    model_path = resolve_model_path(args)
    kv_transfer_config = resolve_kv_transfer_config(args)
    speculative_config = resolve_speculative_config(args)
    kv_offloading_size = normalized_kv_offloading_size(args)
    kv_offloading_backend = resolved_kv_offloading_backend(args, kv_transfer_config)
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
    cmd = [
        args.vllm_bin,
        "serve",
        model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--generation-config",
        DEFAULT_VLLM_GENERATION_CONFIG,
        "--structured-outputs-config",
        STRUCTURED_OUTPUTS_CONFIG,
    ]
    # 未显式指定时不要注入 --api-server-count，让 vLLM 使用其官方默认值。
    # 这样 DP=1/2 会分别沿用历史的 1/2 个 API server，而不是被固定值改写。
    vllm_api_server_count = getattr(args, "vllm_api_server_count", None)
    if vllm_api_server_count is not None:
        cmd += ["--api-server-count", str(vllm_api_server_count)]
    if effective_dtype is not None:
        cmd += ["--dtype", str(effective_dtype)]
    if kv_offloading_backend is not None:
        cmd += ["--kv-offloading-backend", str(kv_offloading_backend)]
    if args.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    cpu_offload_gb = normalized_cpu_offload_gb(args)
    if cpu_offload_gb > 0:
        cmd += ["--cpu-offload-gb", str(cpu_offload_gb)]
    if args.tp is not None:
        cmd += ["--tensor-parallel-size", str(args.tp)]
    if args.data_parallel_size is not None:
        cmd += ["--data-parallel-size", str(args.data_parallel_size)]
    cmd += [
        "--model-impl",
        str(getattr(args, "model_impl", "vllm")),
        "--decode-context-parallel-size",
        str(getattr(args, "decode_context_parallel_size", 1)),
        "--cp-kv-cache-interleave-size",
        str(getattr(args, "cp_kv_cache_interleave_size", 1)),
        "--data-parallel-backend",
        str(getattr(args, "data_parallel_backend", "mp")),
        (
            "--disable-nccl-for-dp-synchronization"
            if getattr(args, "disable_nccl_for_dp_synchronization", False)
            else "--no-disable-nccl-for-dp-synchronization"
        ),
    ]
    if args.pp is not None:
        cmd += ["--pipeline-parallel-size", str(args.pp)]
    if args.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.block_size is not None:
        cmd += ["--block-size", str(args.block_size)]
    long_prefill_token_threshold = getattr(
        args, "long_prefill_token_threshold", None
    )
    if long_prefill_token_threshold is not None:
        cmd += [
            "--long-prefill-token-threshold",
            str(long_prefill_token_threshold),
        ]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    chunked_prefill_setting = resolved_chunked_prefill_setting(args)
    if chunked_prefill_setting is True:
        cmd.append("--enable-chunked-prefill")
    elif chunked_prefill_setting is False:
        cmd.append("--no-enable-chunked-prefill")
    prefix_caching_setting = resolved_prefix_caching_setting(args)
    if prefix_caching_setting is True:
        cmd.append("--enable-prefix-caching")
    elif prefix_caching_setting is False:
        cmd.append("--no-enable-prefix-caching")
    cmd += [
        "--prefix-caching-hash-algo",
        str(getattr(args, "prefix_caching_hash_algo", "sha256")),
        (
            "--scheduler-reserve-full-isl"
            if getattr(args, "scheduler_reserve_full_isl", True)
            else "--no-scheduler-reserve-full-isl"
        ),
        (
            "--async-scheduling"
            if getattr(args, "async_scheduling", False)
            else "--no-async-scheduling"
        ),
        (
            "--disable-cascade-attn"
            if getattr(args, "disable_cascade_attn", True)
            else "--no-disable-cascade-attn"
        ),
        (
            "--enable-flashinfer-autotune"
            if getattr(args, "enable_flashinfer_autotune", False)
            else "--no-enable-flashinfer-autotune"
        ),
    ]
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if args.quantization != "none":
        cmd += ["--quantization", args.quantization]
    if effective_kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", effective_kv_cache_dtype]
    if args.kv_cache_dtype_skip_layers:
        cmd += ["--kv-cache-dtype-skip-layers", *args.kv_cache_dtype_skip_layers]
    if kv_offloading_size > 0:
        cmd += ["--kv-offloading-size", str(kv_offloading_size)]
    if hma_disable_required(args, kv_transfer_config):
        # 启用 Native KV offloading 会创建 OffloadingConnector，LMCache 会创建
        # 另一路 LMCache 会创建 LMCacheConnectorV1。我们观察到的 BO 失败案例里二者都和 HMA 不兼容，
        # 因此只要启用 offloading 就关闭 HMA。
        cmd.append("--disable-hybrid-kv-cache-manager")
    if kv_transfer_config is not None:
        cmd += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
    if speculative_config is not None:
        cmd += ["--speculative-config", json.dumps(speculative_config)]
    cmd.extend(args.extra_vllm_args)
    return cmd
