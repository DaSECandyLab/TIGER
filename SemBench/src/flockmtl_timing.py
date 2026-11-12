"""Compute elapsed query windows without startup or shutdown overhead."""
import math


def query_execution_timing(terminals, started_at=None, completed_at=None):
    queries = [q for t in terminals for q in t.get('queries', [])
               if q.get('status') != 'skipped']
    if not queries:
        return {}
    for q in queries:
        start, end = q.get('started_at'), q.get('completed_at')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)) or end < start:
            return {}  # Do not silently shorten a partially recorded run.
    first = min(q['started_at'] for q in queries)
    releases = [q.get('execution_released_at') for q in queries]
    synchronized = all(isinstance(v, (int, float)) and math.isfinite(v)
                       and v <= q['started_at'] for q, v in zip(queries, releases))
    if synchronized:
        first = min(releases)
    last = max(q['completed_at'] for q in queries)
    if last <= first:
        return {}
    result = {'execution_started_at': first, 'execution_completed_at': last,
              'execution_wall_time_s': last - first,
              'timing_scope': ('barrier_release_to_last_query_completion' if synchronized
                               else 'first_query_start_to_last_query_completion')}
    # Distinct barrier releases identify sequential execution phases (e.g.
    # Q1, then three replay terminals). Keep gaps within each phase, but
    # exclude setup between phases from execution throughput.
    result['execution_span_time_s'] = last - first
    result['terminal_preparation_time_s'] = 0.0
    if synchronized:
        phases = {}
        for q in queries:
            release = q['execution_released_at']
            phases[release] = max(phases.get(release, release), q['completed_at'])
        intervals = sorted(phases.items())
        if len(intervals) > 1 and all(
                intervals[i][1] <= intervals[i + 1][0]
                for i in range(len(intervals) - 1)):
            preparation = sum(intervals[i + 1][0] - intervals[i][1]
                              for i in range(len(intervals) - 1))
            result['terminal_preparation_time_s'] = preparation
            result['execution_wall_time_s'] = sum(end - start for start, end in intervals)
            result['timing_scope'] = 'sum_of_barrier_phase_execution_windows'
            result['execution_phases'] = [
                {'released_at': start, 'completed_at': end, 'execution_time_s': end - start}
                for start, end in intervals]
    if isinstance(started_at, (int, float)) and math.isfinite(started_at) and started_at <= first:
        result['preparation_time_s'] = first - started_at
    if isinstance(completed_at, (int, float)) and math.isfinite(completed_at) and completed_at >= last:
        result['teardown_time_s'] = completed_at - last
    return result
