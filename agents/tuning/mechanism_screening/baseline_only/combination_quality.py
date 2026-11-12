"""Baseline-relative quality gates and reproducible HotpotQA answer scoring."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def field(payload, path):
    for key in path.split('.'):
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def bind_quality(config, baseline):
    if not config:
        return None
    config = deepcopy(config)
    fraction = config.get('fraction', .9)
    if not numeric(fraction) or not 0 < fraction <= 1:
        raise ValueError('Quality fraction must be in (0, 1]')
    metrics = config.get('metrics')
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError('Quality metrics must map names to result field paths')
    values = {}
    for name, path in metrics.items():
        if not isinstance(name, str) or not name or not isinstance(path, str) or not path:
            raise ValueError('Quality metric names and paths must be nonempty strings')
        value = field(baseline, path)
        if not numeric(value) or not 0 < value <= 1:
            raise ValueError(f'Baseline quality {name} must be present, finite and in (0, 1]')
        values[name] = value
    config.update(fraction=fraction, baseline_values=values,
                  thresholds={k: fraction * v for k, v in values.items()})
    return config


def assess_quality(result, config, *, complete):
    if not config:
        return {'status': 'disabled'}
    if not complete:
        return {'status': 'unknown', 'reason': 'Workload not completely successful'}
    values = {k: field(result, path) for k, path in config['metrics'].items()}
    unavailable = [k for k, v in values.items() if not numeric(v) or not 0 <= v <= 1]
    if unavailable:
        return {'status': 'unknown', 'reason': 'Missing or invalid quality metrics',
                'unavailable_metrics': unavailable}
    checks = {k: {'value': v, 'baseline': config['baseline_values'][k],
                  'threshold': config['thresholds'][k],
                  'passed': v >= config['thresholds'][k]} for k, v in values.items()}
    return {'status': 'passed' if all(c['passed'] for c in checks.values()) else 'failed',
            'checks': checks, 'min_retention_ratio': min(v / config['baseline_values'][k]
                                                       for k, v in values.items())}


def hotpotqa_config(dataset):
    root = Path(dataset).expanduser().resolve()
    paths = [root / name for name in ('manifest.json', 'warmup.jsonl', 'test.jsonl')]
    paths += sorted((root / 'query/gold_sql').glob('Q*.sql'))
    paths += [Path(__file__), ROOT / 'SemBench/src/cacheblend_hotpotqa.py']
    return {'fraction': .9, 'metrics': {'exact_match': 'screening_quality.exact_match',
                                       'f1_score': 'screening_quality.f1_score'},
            'hotpotqa_dataset': str(root),
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def enrich_hotpotqa(result, config):
    """Read only this run's raw answers; a scoring error stays unknown, never quality-failed."""
    result = deepcopy(result)
    result.pop('screening_quality', None)
    result.pop('screening_quality_error', None)
    try:
        for path, expected in config.get('source_hashes', {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Quality source changed after policy was frozen: {path}')
        workload = result.get('logs', {}).get('workload')
        if not workload:
            raise ValueError('Result has no workload artifact path')
        proc = subprocess.run([str(ROOT / 'SemBench/.venvs/sembench/bin/python'), str(Path(__file__).resolve()),
                               str(config['hotpotqa_dataset']), str(workload)],
                              capture_output=True, text=True, timeout=180, check=True)
        result['screening_quality'] = json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['screening_quality_error'] = str(exc)
    return result


def score_hotpotqa(dataset, workload_path):
    """Same scorer/gold SQL as SemBench; aggregate all requests, including initial Q1."""
    sys.path.insert(0, str(ROOT / 'SemBench/src'))
    import duckdb
    import pandas as pd
    from cacheblend_hotpotqa import load_dataset, evaluate_answers

    dataset, workload_path = Path(dataset), Path(workload_path)
    payload = json.loads(workload_path.read_text())
    if payload.get('status') != 'success':
        raise ValueError('Quality evidence requires a successful complete workload')
    expected = Counter((str(t), int(q)) for t, seq in payload['terminal_sql_sequences'].items() for q in seq)
    expected.update(('0', int(q)) for q in payload.get('initial_query_sequence', []))
    records = [(str(t['terminal_id']), q) for t in payload['terminals'] for q in t['queries']]
    if Counter((t, int(q['query_id'])) for t, q in records) != expected:
        raise ValueError('Quality query coverage does not match the workload plan')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    con = duckdb.connect(':memory:')
    totals = Counter(); count = 0; per_query = []; sources = []
    try:
        load_dataset(con, dataset)
        root = Path(payload['artifact_root']).resolve()
        for terminal, record in records:
            if record.get('status', 'success') != 'success':
                raise ValueError('Incomplete SQL cannot supply quality evidence')
            path = Path(record['result_output']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Answer artifact is outside this workload artifact root')
            prediction = pd.read_pickle(path)
            if not {'row_id', 'answer'} <= set(prediction.columns) or prediction.row_id.isna().any() or prediction.row_id.duplicated().any():
                raise ValueError('Answer artifacts require unique row_id and answer columns')
            sql = (dataset / f"query/gold_sql/Q{int(record['query_id'])}.sql").read_text()
            gold = con.execute(sql).fetchdf()
            if manifest.get('answer_encoding') == 'json_aliases':
                gold['answer'] = gold.answer.map(json.loads)
            scores = evaluate_answers(dict(zip(prediction.row_id, prediction.answer)), dict(zip(gold.row_id, gold.answer)))
            n = len(gold); count += n
            for key in ('exact_match', 'f1_score'):
                totals[key] += scores[key] * n
            per_query.append({'terminal': terminal, 'query': record['query_id'], 'requests': n, **scores})
            sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if not count or count != payload['summary']['total_llm_request_count']:
            raise ValueError('Scored requests differ from complete workload request count')
        return {k: v / count for k, v in totals.items()} | {
            'request_count': count, 'per_query': per_query, 'sources': sources,
            'scorer': 'SemBench.cacheblend_hotpotqa.evaluate_answers', 'workload_path': str(workload_path)}
    finally:
        con.close()


if __name__ == '__main__':
    print(json.dumps(score_hotpotqa(sys.argv[1], sys.argv[2]), ensure_ascii=False))
