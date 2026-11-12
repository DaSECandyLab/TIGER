"""Opt-in query execution intervals for conservative screening time cutoffs."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


@contextmanager
def track_screening_query(terminal_id, step):
    directory = os.environ.get('FLOCK_SCREENING_PROGRESS_DIR')
    record = {'success': False}
    root = Path(directory) if directory else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    path = root / f'{os.getpid()}_{terminal_id}_{step}.json' if root is not None else None
    record.update(start=time.monotonic(), end=None, pid=os.getpid(), status='running')

    def save():
        if path is None:
            return
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record))
        temporary.replace(path)

    save()
    try:
        yield record
    finally:
        record.update(end=time.monotonic(), status='complete' if record['success'] else 'failed')
        save()
