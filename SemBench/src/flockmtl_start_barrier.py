"""Cross-process readiness barrier for independent terminal workers."""
import fcntl
import json
import os
import time
from pathlib import Path


def wait_for_query_start(directory, terminal_id, terminal_count, timeout_s=600):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    ready_at = time.time()
    with (root / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (root / f'ready_{terminal_id}').write_text(str(ready_at))
        if all((root / f'ready_{i}').exists() for i in range(terminal_count)):
            if not (root / 'release.json').exists() and not (root / 'abort').exists():
                tmp = root / f'release.{os.getpid()}.tmp'
                tmp.write_text(json.dumps({'released_at': time.time()}))
                tmp.replace(root / 'release.json')
    while True:
        if (root / 'abort').exists():
            raise RuntimeError('A terminal failed before the shared query start')
        if (root / 'release.json').exists():
            return ready_at, json.loads((root / 'release.json').read_text())['released_at']
        if time.monotonic() >= deadline:
            (root / 'abort').touch()
            raise TimeoutError('Timed out waiting for all terminals to initialize')
        time.sleep(0.01)
