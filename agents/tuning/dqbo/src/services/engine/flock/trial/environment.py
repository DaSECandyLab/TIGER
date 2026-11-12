"""记录影响 trial 可复现性的机器、解释器和制品状态。"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


_SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "FLOCK_GLOBAL_COORD_DISPATCH_BURST",
    "CUDA_MODULE_LOADING",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "RAY_ADDRESS",
    "VLLM_HOST_IP",
    "VLLM_PORT",
)


def require_unified_python() -> None:
    """拒绝从仓库 `.venv` 之外启动正式 BO/trial CLI。"""
    expected_prefix = Path(__file__).resolve().parents[8] / ".venv"
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != expected_prefix.resolve():
        raise RuntimeError(
            "Flock BO/trial must use the unified repository Python: "
            f"{expected_prefix / 'bin' / 'python'}; current prefix={sys.prefix}"
        )


def _read_text(path: str) -> str | None:
    """读取短系统文件；不存在或无权读取时返回空值。"""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _path_fingerprint(value: Any) -> dict[str, Any] | None:
    """记录路径及其轻量 stat，避免对模型目录执行昂贵的全量哈希。"""
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    resolved = path.resolve(strict=False)
    fingerprint: dict[str, Any] = {
        "configured_path": str(path),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
    }
    try:
        stat = resolved.stat()
    except OSError:
        return fingerprint
    fingerprint.update(
        {
            "is_directory": resolved.is_dir(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return fingerprint


def _package_versions() -> dict[str, str | None]:
    """记录 BO Python 环境中会影响数值与调度行为的核心包版本。"""
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "torch", "scipy", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def _nvidia_inventory() -> list[dict[str, str]]:
    """采集 GPU 身份、驱动和启动时钟；失败不会阻断正式实验。"""
    fields = (
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "driver_version",
        "power.limit",
        "clocks.current.graphics",
        "clocks.current.memory",
    )
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for values in csv.reader(proc.stdout.splitlines(), skipinitialspace=True):
        if len(values) != len(fields):
            continue
        rows.append(
            {field: value.strip() for field, value in zip(fields, values)}
        )
    return rows


def collect_runtime_environment(
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """构造可随结果落盘的运行指纹。

    只记录可复现实验需要的非敏感环境变量；API key 等凭据不会进入产物。
    """
    boot_time = None
    for line in (_read_text("/proc/stat") or "").splitlines():
        if line.startswith("btime "):
            try:
                boot_time = datetime.fromtimestamp(
                    int(line.split()[1]), timezone.utc
                ).isoformat()
            except (IndexError, ValueError, OSError):
                pass
            break

    uptime_seconds = None
    uptime_text = _read_text("/proc/uptime")
    if uptime_text:
        try:
            uptime_seconds = float(uptime_text.split()[0])
        except (IndexError, ValueError):
            pass

    path_fields = (
        "vllm_bin",
        "sembench_dir",
        "flock_ext",
        "model_path",
        "awq_model_path",
        "gptq_model_path",
    )
    artifacts = {
        name: _path_fingerprint(getattr(args, name, None))
        for name in path_fields
        if args is not None and getattr(args, name, None) is not None
    }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "python": {
            "executable": sys.executable,
            "executable_resolved": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "packages": _package_versions(),
        },
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel_release": platform.release(),
            "boot_id": _read_text("/proc/sys/kernel/random/boot_id"),
            "boot_time_utc": boot_time,
            "uptime_seconds": uptime_seconds,
        },
        "environment": {
            key: os.environ.get(key) for key in _SAFE_ENVIRONMENT_KEYS
        },
        "gpu_inventory": _nvidia_inventory(),
        "artifacts": artifacts,
    }
