"""Run prepare → native BO/local Repair → mechanical audit in this process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

from agents.audit import AuditReport

_HEARTBEAT_SECONDS = 30


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter for the Trial executor; defaults to the current interpreter.')
    return parser


def _history_snapshot(output_dir: Path) -> tuple[int, int, str | None]:
    try:
        payload = json.loads((output_dir / 'bo_history.json').read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0, None
    trials = payload.get('trials', [])
    if not isinstance(trials, list):
        return 0, 0, None
    completed = sum(isinstance(t, dict) and t.get('status') == 'complete' for t in trials)
    latest = trials[-1].get('status') if trials and isinstance(trials[-1], dict) else None
    return len(trials), completed, str(latest) if latest is not None else None


def _heartbeat(done: threading.Event, output_dir: Path, rounds: int) -> None:
    started = time.monotonic()
    while not done.wait(_HEARTBEAT_SECONDS):
        attempted, complete, latest = _history_snapshot(output_dir)
        print(f'[Candidate BO] elapsed={int(time.monotonic() - started)}s '
              f'rounds={attempted}/{rounds} complete={complete} latest={latest or "running"}',
              file=sys.stderr, flush=True)


def run_candidate_optimizer(config: Mapping[str, Any], *, python_executable: str | None = None) -> dict[str, Any]:
    """Execute the local pipeline and return its result and independent audit."""
    from agents.tuning.candidate_optimizer.src.services.workflow import (
        audit_candidate_result, execute_bo, prepare_bo_context,
    )
    parameters = dict(config)
    if python_executable is not None:
        parameters.setdefault('python_executable', python_executable)
    try:
        prepared = prepare_bo_context(**parameters)
    except Exception as exc:
        return {'status': 'audit_failed', 'audit': AuditReport(
            passed=False, audit_type='candidate_optimizer_context_audit',
            errors=[str(exc)], retryable=True,
            facts={'output_dir': parameters.get('output_dir')}).to_dict()}

    done = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(
        done, Path(parameters['output_dir']).expanduser().resolve(), int(parameters['rounds'])),
        daemon=True, name='candidate-optimizer-progress')
    heartbeat.start()
    try:
        execution = execute_bo(prepared['context_path'])
    except Exception as exc:
        return {'status': 'audit_failed', 'context': prepared,
                'execution': {'status': 'execution_failed'}, 'audit': AuditReport(
                    passed=False, audit_type='candidate_optimizer_execution_audit',
                    errors=[str(exc)], retryable=True,
                    facts={'context_path': prepared['context_path']}).to_dict()}
    finally:
        done.set()
        heartbeat.join()

    audit = audit_candidate_result(execution['result_path'], expected_context_path=prepared['context_path'])
    return {'status': 'completed' if audit['passed'] else 'audit_failed',
            'result_path': execution['result_path'], 'history_path': execution['history_path'],
            'context': prepared, 'execution': execution, 'audit': audit}


def main() -> None:
    args = _parser().parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise TypeError('Candidate config must be a JSON object')
    result = run_candidate_optimizer(config, python_executable=args.python)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'result_path': result.get('result_path'),
                      'history_path': result.get('history_path'),
                      'audit_passed': result.get('audit', {}).get('passed')}, ensure_ascii=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
