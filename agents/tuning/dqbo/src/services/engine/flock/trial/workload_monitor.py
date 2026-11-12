"""Bound workload lifetime to the inference service owned by the trial."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .vllm import terminate_process_group


class InferenceServiceFailed(RuntimeError):
    pass


class ServiceMonitor:
    def __init__(self, process=None, log_path=None):
        self.process = process
        self.log_path = Path(log_path) if log_path is not None else None
        self.offset = 0
        self.pending = ''

    def check(self):
        if self.process is not None and self.process.poll() is not None:
            raise InferenceServiceFailed(
                f'vLLM service exited during workload (code {self.process.returncode}); see vLLM log')
        if self.log_path is not None and self.log_path.exists():
            with self.log_path.open(errors='replace') as stream:
                stream.seek(self.offset)
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    text = self.pending + chunk
                    if any(marker in text for marker in (
                        'EngineCore encountered a fatal error',
                        'EngineDeadError',
                        'Engine core initialization failed',
                    )):
                        raise InferenceServiceFailed('vLLM fatal engine failure during workload; see vLLM log')
                    self.pending = text[-256:]
                self.offset = stream.tell()


def run_monitored_workload(cmd, *, cwd, env, log_path, monitor=None, poll_seconds=2.0):
    """Stream logs to disk; clean the owned workload group on every exit path."""
    if monitor is not None:
        monitor.check()
    with Path(log_path).open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                                stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:
                    if monitor is not None:
                        monitor.check()
                    continue
                if monitor is not None:
                    monitor.check()
                return subprocess.CompletedProcess(cmd, code, stdout=None)
        finally:
            # The leader can exit before terminal workers: always clean its group.
            terminate_process_group(proc, proc.pid, timeout_s=3)
