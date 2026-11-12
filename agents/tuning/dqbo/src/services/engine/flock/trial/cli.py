"""Candidate Optimizer 自有单 Trial 子进程入口。"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
from typing import Any

from .arguments import parse_args
from .runner import run_trial


class TrialTerminationRequested(BaseException):
    """Internal control flow used to unwind through ``run_trial`` cleanup."""


_TERMINATION_REQUESTED = False


def _terminate_owned_processes(signum: int, _frame: object) -> None:
    """Stop same-group workload children, then let run_trial clean vLLM."""

    del signum
    global _TERMINATION_REQUESTED
    if _TERMINATION_REQUESTED:
        return
    _TERMINATION_REQUESTED = True
    # BOTrialExecutor starts this wrapper as its own session/process-group
    # leader. Avoid signalling an interactive shell group when invoked by hand.
    if os.getpgrp() == os.getpid():
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise TrialTerminationRequested()


def _arm_parent_death_signal() -> None:
    """Have Linux terminate this wrapper if its executor process disappears."""

    parent_pid = os.getppid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
            return
    except (AttributeError, OSError):
        return
    # Close the small race in which the parent died immediately before prctl.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    result = list(argv)
    try:
        index = result.index(option)
    except ValueError:
        result.extend([option, value])
    else:
        if index + 1 >= len(result):
            raise ValueError(f"{option} requires a value")
        result[index + 1] = value
    return result


def execute_request(path: str | Path) -> int:
    request_path = Path(path).expanduser().resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Candidate trial request must be an object")
    argv = [str(item) for item in payload.get("trial_argv", [])]
    argv = _replace_option(argv, "--result-dir", str(payload["result_dir"]))
    argv = _replace_option(argv, "--trial-name", str(payload["trial_name"]))
    argv = _replace_option(
        argv, "--baseline-config", str(payload["baseline_config"])
    )
    if "--preserve-engine-defaults" not in argv:
        argv.append("--preserve-engine-defaults")
    namespace = parse_args(argv)
    setattr(namespace, "_candidate_parameters", dict(payload["candidate"]))
    setattr(
        namespace,
        "_candidate_active_parameters",
        list(payload.get("active_parameters", payload["candidate"])),
    )
    return run_trial(namespace)


def main(argv: list[str] | None = None) -> int:
    previous_term = signal.signal(signal.SIGTERM, _terminate_owned_processes)
    previous_int = signal.signal(signal.SIGINT, _terminate_owned_processes)
    _arm_parent_death_signal()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        return execute_request(args.request)
    except TrialTerminationRequested:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


if __name__ == "__main__":
    raise SystemExit(main())
