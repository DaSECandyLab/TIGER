"""Canonical runtime-failure taxonomy for BO execution and local Repair.

Normalize executor diagnostics into bounded failure families for local history,
quality-safe dataset routing, and repair evidence.
"""

from __future__ import annotations

import re
from typing import Any


CAPACITY_FAILURE = "capacity_failure"
PARAMETER_COMPATIBILITY_FAILURE = "parameter_compatibility_failure"
TIMEOUT_FAILURE = "timeout_failure"
ENVIRONMENT_FAILURE = "environment_failure"
MEASUREMENT_FAILURE = "measurement_failure"
UNKNOWN_RUNTIME_FAILURE = "unknown_runtime_failure"

RUNTIME_FAILURE_KINDS = frozenset(
    {
        CAPACITY_FAILURE,
        PARAMETER_COMPATIBILITY_FAILURE,
        TIMEOUT_FAILURE,
        ENVIRONMENT_FAILURE,
        MEASUREMENT_FAILURE,
        UNKNOWN_RUNTIME_FAILURE,
    }
)

_CAPACITY_PATTERNS = (
    r"out of memory",
    r"cuda oom",
    r"cublas_status_alloc_failed",
    r"no available memory for (?:the )?cache blocks",
    r"unable to allocate .*cache block",
    r"cannot allocate .*cache block",
    r"maximum number of tokens that can be stored in kv cache",
    r"larger than the maximum number of tokens that can be stored in kv cache",
    r"larger than the available kv cache memory",
    r"kv cache.*(?:insufficient|not enough|too small)",
    r"free memory .* less than the desired gpu memory utilization",
)
_PARAMETER_COMPATIBILITY_PATTERNS = (
    r"query and key must have the same dtype",
    r"num_speculative_tokens.*(?:invalid|greater than|must be)",
    r"speculative.*tokens.*(?:invalid|must be)",
    r"incompatible .*dtype",
    r"validation error",
    r"invalid (?:argument|configuration|value)",
)
_ENVIRONMENT_PATTERNS = (
    r"address already in use",
    r"gpu.*(?:already in use|contaminat)",
    r"device.*(?:already in use|busy)",
)


def classify_runtime_failure(
    diagnostic_text: str | None,
    *,
    fallback: Any = None,
) -> str:
    """Return a stable failure family without retaining diagnostic prose."""

    fallback_text = str(fallback or "").strip().lower()
    if fallback_text in RUNTIME_FAILURE_KINDS:
        return fallback_text
    if fallback_text in {"oom", "cuda_oom", "kv_cache_oom", "memory_error"}:
        return CAPACITY_FAILURE
    if fallback_text in {"timeout", "timed_out"}:
        return TIMEOUT_FAILURE
    if fallback_text in {"invalid_result", "missing_result", "incomplete_measurement"}:
        return MEASUREMENT_FAILURE

    text = str(diagnostic_text or "").lower()
    if any(re.search(pattern, text) for pattern in _CAPACITY_PATTERNS):
        return CAPACITY_FAILURE
    if any(
        re.search(pattern, text)
        for pattern in _PARAMETER_COMPATIBILITY_PATTERNS
    ):
        return PARAMETER_COMPATIBILITY_FAILURE
    if any(re.search(pattern, text) for pattern in _ENVIRONMENT_PATTERNS):
        return ENVIRONMENT_FAILURE
    # Match actual transport failures, not configuration/shutdown prose such as
    # "Shutdown initiated (timeout=0)" found in otherwise unrelated vLLM logs.
    if re.search(r"timeout was reached|\b(?:read|connect(?:ion)?|request) timed out\b"
                 r"|\b(?:TimeoutError|ReadTimeout|ConnectTimeout):"
                 r"|\bvllm did not become ready within \d+(?:\.\d+)?s\b", text, re.IGNORECASE):
        return TIMEOUT_FAILURE
    if fallback_text in {"executor_exception", "process_exit", "startup_failure"}:
        return UNKNOWN_RUNTIME_FAILURE
    return UNKNOWN_RUNTIME_FAILURE


def is_capacity_failure(value: Any) -> bool:
    return str(value or "") == CAPACITY_FAILURE


__all__ = [
    "CAPACITY_FAILURE",
    "ENVIRONMENT_FAILURE",
    "MEASUREMENT_FAILURE",
    "PARAMETER_COMPATIBILITY_FAILURE",
    "RUNTIME_FAILURE_KINDS",
    "TIMEOUT_FAILURE",
    "UNKNOWN_RUNTIME_FAILURE",
    "classify_runtime_failure",
    "is_capacity_failure",
    "summarize_runtime_failure",
]


def summarize_runtime_failure(text: str, *, fallback: str) -> str:
    """Keep a short cause instead of a wrapper exit code or full traceback."""

    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    def compact(value: str) -> str:
        return re.split(r"\s+(?:Please try|Try (?:increasing|lowering|decreasing)|See https?://)", value)[0][:240]

    if len(lines) == 1 and "Stage=KV cache initialization." in lines[0]:
        return lines[0][:240]
    # Extract the resource shortfall before a generic engine wrapper can hide it.
    for line in reversed(lines):
        capacity = re.search(
            r"max seq len \((\d+)\).*?\(([\d.]+) GiB KV cache is needed"
            r".*?available KV cache memory \(([-\d.]+) GiB\)", line,
        )
        if capacity:
            length, needed, available = capacity.groups()
            estimated = re.search(r"estimated maximum model length is (\d+)", line)
            summary = (
                f"KV cache capacity check failed: max_model_len={length}; "
                f"required={needed} GiB; available={available} GiB."
            )
            if estimated:
                summary += f" Estimated supported max_model_len={estimated.group(1)}."
            return summary[:240]

    if "No available memory for the cache blocks" in clean:
        budgets = re.findall(r"Available KV cache memory:\s*(-?[\d.]+) GiB", clean)
        lengths = re.findall(r"Using max model len (\d+)", clean)
        summary = "No available memory for the cache blocks. Stage=KV cache initialization."
        if budgets:
            summary += f" Minimum reported KV budget={min(budgets, key=float)} GiB."
        if lengths:
            summary += f" max_model_len={lengths[-1]}."
        return summary[:240]

    sampler = re.search(
        r"CUDA out of memory occurred when warming up sampler with \d+ dummy requests\.",
        clean,
    )
    # Allocation/free-memory figures must come from the same OOM diagnostic.
    oom_lines = [line for line in lines if "CUDA out of memory" in line and "Tried to allocate" in line]
    if oom_lines:
        line = oom_lines[-1]
        allocation = re.search(r"Tried to allocate ([\d.]+ [KMGT]i?B)", line)
        free = re.search(r"of which ([\d.]+ [KMGT]i?B) is free", line)
        gpu = re.search(r"GPU (\d+)", line)
        summary = sampler.group(0) if sampler else "CUDA out of memory."
        if gpu:
            summary += f" GPU={gpu.group(1)}."
        if allocation:
            summary += f" Requested allocation={allocation.group(1)}."
        if free:
            summary += f" Free memory={free.group(1)}."
        return summary[:240]

    # Prefer actionable diagnostics over the generic engine/wrapper exception.
    for marker in (
        "CUDA out of memory occurred when warming up sampler",
        "No available memory for the cache blocks",
        "CUDA out of memory",
    ):
        for line in reversed(lines):
            if marker in line:
                return compact(line[line.index(marker):])
    error_lines = [line for line in reversed(lines) if "Engine core initialization failed" not in line]
    error_lines.extend(line for line in reversed(lines) if "Engine core initialization failed" in line)
    for line in error_lines:
        match = re.search(r"\b(?:[\w.]*Error|[\w.]*Exception):\s*\S", line)
        if match and not line.startswith(('File ', 'raise ')):
            return compact(line[match.start():])
    return compact(" ".join(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", fallback).split()))
