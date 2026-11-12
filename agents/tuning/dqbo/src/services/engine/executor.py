"""Candidate Optimizer 自有 BO Trial executor。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from agents.tuning.dqbo.src.services.failure_taxonomy import (
    TIMEOUT_FAILURE,
    classify_runtime_failure,
    summarize_runtime_failure,
)


TRIAL_MODULE = (
    "agents.tuning.dqbo.src.services.engine.flock.trial.cli"
)


@dataclass(frozen=True)
class TrialExecution:
    status: str
    result: Mapping[str, Any] | None
    result_path: str | None
    failure_kind: str | None = None
    failure_reason: str | None = None
    command: tuple[str, ...] = ()
    return_code: int | None = None
    censoring: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status != "complete":
            reason = self.failure_reason or self.failure_kind or "Unknown runtime failure"
            object.__setattr__(self, "failure_reason", summarize_runtime_failure(reason, fallback=reason))


class TrialExecutor(Protocol):
    def execute(
        self,
        candidate: Mapping[str, Any],
        *,
        active_parameters: Sequence[str],
        trial_id: str,
    ) -> TrialExecution: ...


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _failure_kind(text: str) -> str:
    """Compatibility wrapper for callers/tests using the former helper."""

    return classify_runtime_failure(text)


def _workload_failure_diagnostics(result_path: Path, request_path: Path) -> list[str]:
    """Read bounded SQL failure causes from this attempt's structured result.

    The wrapper prints counts only; provider timeouts otherwise disappear behind
    its generic exit code. Never diagnose an attempt from a stale result file.
    """
    try:
        if result_path.stat().st_mtime_ns < request_path.stat().st_mtime_ns:
            return []
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, Mapping):
            return []
        workload = result.get("flock_workload_metrics")
        if not isinstance(workload, Mapping):
            return []
        errors = []
        for terminal in workload.get("terminals", []):
            if not isinstance(terminal, Mapping):
                continue
            for query in terminal.get("queries", []):
                if not isinstance(query, Mapping) or query.get("status") != "failed":
                    continue
                error = query.get("error")
                if isinstance(error, str) and error.strip():
                    cause = error.strip()[:1000]
                    if cause not in errors:
                        errors.append(cause)
                if len(errors) >= 16:
                    return errors
        return errors
    except (OSError, ValueError, TypeError):
        return []


def _terminate_trial_wrapper(
    proc: subprocess.Popen[str], *, grace_seconds: float = 45.0
) -> None:
    """Ask the owned Trial wrapper to clean its workload and vLLM groups."""

    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


class BOTrialExecutor:
    """通过 Candidate 自有 trial module 启动一次真实 Flock/SemBench Trial。"""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        baseline_config: str | Path,
        trial_argv: Sequence[str],
        python_executable: str | None = None,
        timeout_seconds: float = 3600.0,
        stop_condition: Callable[[str], Mapping[str, Any] | None] | None = None,
        trial_environment: Callable[[str], Mapping[str, str]] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.trial_dir = self.output_dir / "trials"
        self.request_dir = self.output_dir / "trial_requests"
        self.log_dir = self.output_dir / "logs"
        for directory in (self.trial_dir, self.request_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.baseline_config = str(Path(baseline_config).expanduser().resolve())
        self.trial_argv = [str(item) for item in trial_argv]
        self.python_executable = str(python_executable or sys.executable)
        self.timeout_seconds = float(timeout_seconds)
        self.stop_condition = stop_condition
        self.trial_environment = trial_environment
        if self.timeout_seconds <= 0:
            raise ValueError("trial timeout_seconds must be positive")

    def execute(
        self,
        candidate: Mapping[str, Any],
        *,
        active_parameters: Sequence[str],
        trial_id: str,
    ) -> TrialExecution:
        request_path = self.request_dir / f"{trial_id}.json"
        result_path = self.trial_dir / f"vllm-{trial_id}.json"
        log_path = self.log_dir / f"{trial_id}.executor.log"
        _write_json(
            request_path,
            {
                "schema_version": "sembench.candidate-trial-request.v1",
                "trial_id": trial_id,
                "trial_name": trial_id,
                "result_dir": str(self.trial_dir),
                "baseline_config": self.baseline_config,
                "candidate": dict(candidate),
                "active_parameters": list(active_parameters),
                "trial_argv": list(self.trial_argv),
            },
        )
        command = (
            self.python_executable,
            "-m",
            TRIAL_MODULE,
            "--request",
            str(request_path),
        )
        completed: subprocess.Popen[str]
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("$ " + " ".join(command) + "\n")
                log_file.flush()
                completed = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[6],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    env=({**os.environ, **self.trial_environment(trial_id)}
                         if self.trial_environment is not None else None),
                )
                try:
                    if self.stop_condition is None:
                        completed.wait(timeout=self.timeout_seconds)
                    else:
                        deadline = time.monotonic() + self.timeout_seconds
                        while completed.poll() is None:
                            censoring = self.stop_condition(trial_id)
                            if censoring is not None:
                                # Persist before cleanup so an interrupted controller
                                # cannot discard the measured cutoff evidence.
                                _write_json(self.trial_dir / f"{trial_id}.censoring.json", censoring)
                                _terminate_trial_wrapper(completed)
                                return TrialExecution(
                                    status="censored", result=None, result_path=None,
                                    failure_kind=TIMEOUT_FAILURE,
                                    failure_reason="Query execution reached cutoff "
                                    + str(censoring.get("cutoff_s", "unknown")) + " seconds",
                                    command=command, censoring=dict(censoring),
                                )
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(command, self.timeout_seconds)
                            try:
                                completed.wait(timeout=min(1.0, remaining))
                            except subprocess.TimeoutExpired:
                                continue
                except subprocess.TimeoutExpired:
                    _terminate_trial_wrapper(completed)
                    raise
        except subprocess.TimeoutExpired:
            return TrialExecution(
                status="runtime_failed",
                result=None,
                result_path=None,
                failure_kind=TIMEOUT_FAILURE,
                failure_reason=f"trial exceeded {self.timeout_seconds} seconds",
                command=command,
            )
        except BaseException:
            if "completed" in locals():
                _terminate_trial_wrapper(completed)
            raise
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            # The wrapper traceback only says that vLLM exited before becoming
            # ready.  The capacity diagnostic itself is written by the owned
            # Trial runner below ``trials/logs``. Preserve a bounded cause as
            # well as the normalized family for subsequent repair evidence.
            vllm_log_path = self.trial_dir / "logs" / f"{trial_id}.vllm.log"
            diagnostic_text = log_text
            workload_errors = _workload_failure_diagnostics(result_path, request_path)
            if workload_errors:
                diagnostic_text += "\n" + "\n".join(workload_errors)
            if vllm_log_path.is_file():
                try:
                    diagnostic_text += "\n" + vllm_log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    pass
            return TrialExecution(
                status="runtime_failed",
                result=None,
                result_path=str(result_path) if result_path.exists() else None,
                failure_kind=classify_runtime_failure(diagnostic_text),
                failure_reason=summarize_runtime_failure(
                    diagnostic_text,
                    fallback=(workload_errors[-1] if workload_errors else
                              f"Candidate trial exited with code {completed.returncode}"),
                ),
                command=command,
                return_code=completed.returncode,
            )
        if not result_path.is_file():
            return TrialExecution(
                status="runtime_failed",
                result=None,
                result_path=None,
                failure_kind="missing_result",
                failure_reason=f"trial did not write {result_path}",
                command=command,
                return_code=completed.returncode,
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return TrialExecution(
                status="runtime_failed",
                result=None,
                result_path=str(result_path),
                failure_kind="invalid_result",
                failure_reason=str(exc),
                command=command,
                return_code=completed.returncode,
            )
        if not isinstance(result, Mapping):
            return TrialExecution(
                status="runtime_failed",
                result=None,
                result_path=str(result_path),
                failure_kind="invalid_result",
                failure_reason="trial result must be a JSON object",
                command=command,
                return_code=completed.returncode,
            )
        return TrialExecution(
            status="complete",
            result=dict(result),
            result_path=str(result_path),
            command=command,
            return_code=completed.returncode,
        )
