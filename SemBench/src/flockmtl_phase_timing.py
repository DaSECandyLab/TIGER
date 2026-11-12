"""Opt-in phase timing for diagnosing workload overhead."""
import json
import os
import time
from contextlib import contextmanager

@contextmanager
def timed_phase(phase, **details):
    if os.environ.get('FLOCK_PHASE_TIMING') != '1':
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        print('[phase_timing] ' + json.dumps({
            'phase': phase, 'pid': os.getpid(), 'elapsed_s': time.perf_counter() - started,
            **details,
        }), flush=True)
