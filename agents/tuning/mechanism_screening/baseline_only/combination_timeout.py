"""Right-censor slow probes using query time, excluding startup/preparation/quality."""
import json
import math
import os
from pathlib import Path
import time
from urllib.request import urlopen


def throughput_time_estimate(censoring, baseline_wall_time_s=None):
    """Project total query wall time at the observed successful-request rate."""
    result = {'status': 'unavailable', 'kind': 'throughput_projection',
              'is_measurement': False, 'is_lower_bound': False,
              'assumption': 'Remaining requests sustain the observed average successful-request throughput; '
                            'request mix, cache behavior and scheduling may change this rate.',
              'formula': 'expected_requests / (successful_requests / elapsed_s)'}
    requests = (censoring.get('progress') or {}).get('requests') or {}
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    completed, expected, elapsed = (requests.get(k) for k in
                                    ('successful_requests', 'expected_requests', 'elapsed_s'))
    if censoring.get('kind') != 'right_censored' or requests.get('status') != 'measured':
        return result | {'reason': 'No measured request progress for a censored run'}
    if not all(positive(v) for v in (completed, expected, elapsed)) or completed >= expected:
        return result | {'reason': 'Need positive elapsed time and incomplete positive successful-request progress'}
    if requests.get('failed_requests') != 0:
        return result | {'reason': 'Failed or unknown request count prevents a successful-completion projection'}
    rate = completed / elapsed
    if not positive(rate):
        return result | {'reason': 'Non-finite or zero throughput'}
    total = expected / rate
    if not positive(rate) or not positive(total):
        return result | {'reason': 'Non-finite throughput projection'}
    result.update(status='estimated', successful_requests=completed, expected_requests=expected,
                  observed_elapsed_s=elapsed, observed_successful_requests_per_second=rate,
                  estimated_total_wall_time_s=total,
                  estimated_remaining_wall_time_s=(expected-completed)/rate,
                  time_scope='first query start to projected final request completion; excludes engine startup')
    if positive(baseline_wall_time_s):
        result['baseline_wall_time_s'] = baseline_wall_time_s
        result['estimated_slowdown_vs_baseline'] = total / baseline_wall_time_s
        result['required_speedup_to_match_baseline'] = total / baseline_wall_time_s
        result['required_runtime_reduction_fraction'] = max(0.0, 1-baseline_wall_time_s/total)
    return result


def request_counts(before_text, after_text):
    """Return terminal request deltas, preserving failures and detecting resets."""
    from agents.tuning.dqbo.src.services.engine.flock.trial.metrics import parse_prometheus
    before, after = parse_prometheus(before_text), parse_prometheus(after_text)
    aliases = ('vllm:request_success_total', 'vllm:request_success')
    def series(samples):
        name = next((n for n in aliases if any(s.name == n for s in samples)), None)
        result = {}
        for s in samples:
            if s.name != name:
                continue
            key = tuple(sorted(s.labels.items()))
            if key in result or not math.isfinite(s.value) or s.value < 0:
                raise ValueError('Invalid or duplicate request counter')
            result[key] = s.value
        return result
    start, end = series(before), series(after)
    if not end:
        raise ValueError('Request counters unavailable')
    if not start.keys() <= end.keys():
        raise ValueError('Request counter series disappeared')
    counts = {'successful_requests': 0.0, 'failed_requests': 0.0}
    for key, value in end.items():
        delta = value - start.get(key, 0)
        if delta < 0:
            raise ValueError('Request counter reset; delta unavailable')
        # Servers may expose additional finish reasons even when none occurred.
        # A zero delta needs no classification and must not hide useful progress.
        if delta == 0:
            continue
        reason = dict(key).get('finished_reason')
        if reason not in {'stop', 'length', 'error', 'abort', 'aborted'}:
            raise ValueError('Unknown request finish reason')
        counts['successful_requests' if reason in {'stop', 'length'} else 'failed_requests'] += delta
    return counts


def execution_lower_bound(records, now):
    intervals = []
    for record in records:
        if record.get('status') == 'failed':
            return None
        start, end = record.get('start'), record.get('end')
        if end is None:
            end = now
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (start, end)) or not 0 <= start <= end <= now:
            return None
        intervals.append((start, end))
    # Concurrent queries count once. Gaps, initialization, preprocessing and quality
    # evaluation are excluded; this union is a lower bound on workload query wall time.
    total, right = 0.0, 0.0
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


class QueryTimeCutoff:
    def __init__(self, directory, seconds, *, expected_sql_count=None, expected_request_count=None):
        self.directory, self.seconds = Path(directory), float(seconds)
        self.expected_sql_count = expected_sql_count
        self.expected_request_count = expected_request_count
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('Query execution cutoff must be positive and finite')

    def environment(self, trial_id):
        return {'FLOCK_SCREENING_PROGRESS_DIR': str((self.directory / trial_id).resolve())}

    def request_progress(self, trial_id, first_query_start):
        telemetry = self.directory / trial_id / 'telemetry'
        result = {'status': 'unavailable', 'successful_requests': None,
                  'failed_requests': None, 'expected_requests': self.expected_request_count,
                  'successful_request_fraction': None, 'average_successful_requests_per_second': None,
                  'rate_scope': 'since_first_query_start_to_snapshot_end; includes inter-query gaps; not instantaneous'}
        try:
            source = json.loads((telemetry / 'source.json').read_text())
            before = (telemetry / 'before.prom').read_text()
            started = time.monotonic()
            with urlopen(source['metrics_url'], timeout=2) as response:
                # Bound telemetry volume as well as request wait time.
                payload = response.read(8 * 1024 * 1024 + 1)
            ended = time.monotonic()
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError('Metrics snapshot too large')
            snapshot = payload.decode('utf-8')
            (telemetry / 'cutoff.prom').write_text(snapshot)
            counts = request_counts(before, snapshot)
            elapsed = ended - first_query_start
            if elapsed <= 0:
                raise ValueError('Invalid request rate interval')
            result.update(counts, status='measured', snapshot_start_monotonic=started,
                          snapshot_end_monotonic=ended, elapsed_s=elapsed,
                          snapshot_path=str(telemetry / 'cutoff.prom'),
                          average_successful_requests_per_second=counts['successful_requests'] / elapsed)
            expected = self.expected_request_count
            if isinstance(expected, (int, float)) and not isinstance(expected, bool) and math.isfinite(expected) and expected > 0:
                result['successful_request_fraction'] = counts['successful_requests'] / expected
            result['denominator_scope'] = 'reference workload successful request count; not submitted requests'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['reason'] = str(exc)
        return result

    def __call__(self, trial_id):
        records = []
        for path in (self.directory / trial_id).glob('*.json'):
            try:
                record = json.loads(path.read_text())
                if record.get('end') is None:
                    os.kill(int(record['pid']), 0)
                records.append(record)
            except (OSError, ValueError, KeyError, TypeError):
                return None  # Corrupt/stale/dead-worker state is not slowdown evidence.
        now = time.monotonic()
        lower = execution_lower_bound(records, now)
        if lower is None or lower < self.seconds or not any(r.get('end') is None for r in records):
            return None
        progress = {
            'successful_query_count': sum(r.get('status') == 'complete' and r.get('success') is True for r in records),
            'expected_query_count': self.expected_sql_count,
        }
        progress['requests'] = self.request_progress(trial_id, min(r['start'] for r in records))
        return {'kind': 'right_censored', 'objective': 'wall_time_s',
                'lower_bound_s': lower, 'cutoff_s': self.seconds,
                'scope': 'union_of_query_execution_intervals',
                'recorded_monotonic': now, 'progress': progress}
