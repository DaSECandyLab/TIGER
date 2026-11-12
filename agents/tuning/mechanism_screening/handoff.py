"""Deterministic screening-to-DQBO joint-combination contract."""
from collections.abc import Mapping
import json
from typing import Any

MECHANISMS = ('enable_prefix_caching', 'chunked_prefill', 'quantization', 'dtype',
              'kv_cache_dtype', 'cacheblend_mode', 'speculative_method')


def canonical_json_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def retained_alternatives(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get('status') != 'complete':
        raise ValueError('Simple screening must be complete before Range starts')
    rows = payload.get('retained_combinations')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Simple screening needs nonempty retained_combinations')
    result, identifiers, tuples = [], set(), set()
    dropped = set(payload.get('dropped_combination_ids', []))
    for row in rows:
        identifier = row.get('combination_id') if isinstance(row, Mapping) else None
        mechanisms = row.get('mechanisms') if isinstance(row, Mapping) else None
        if not identifier or identifier in identifiers or identifier in dropped:
            raise ValueError('Invalid, duplicate or dropped retained combination ID')
        if not isinstance(mechanisms, Mapping) or set(mechanisms) != set(MECHANISMS):
            raise ValueError('Each retained combination must resolve exactly seven mechanisms')
        for name, value in mechanisms.items():
            if name in MECHANISMS[:2]:
                if not isinstance(value, bool):
                    raise ValueError(f'{name} requires a boolean')
            elif not isinstance(value, str) or not value or value == 'auto':
                raise ValueError(f'{name} requires a concrete string')
        key = canonical_json_value(mechanisms)
        if key in tuples:
            raise ValueError('Duplicate retained mechanism tuple')
        configuration = row.get('configuration')
        if not isinstance(configuration, Mapping) or any(configuration.get(k) != v for k, v in mechanisms.items()):
            raise ValueError('Retained configuration disagrees with its mechanism tuple')
        bindings = {}
        for name, diff in row.get('mechanical_diff', {}).items():
            basis = diff.get('basis') if isinstance(diff, Mapping) else None
            if isinstance(basis, Mapping) and basis.get('rule_id') != 'domain_activation':
                if name in MECHANISMS or configuration.get(name) != diff.get('after'):
                    raise ValueError('Invalid mechanical background binding')
                bindings[name] = diff['after']
        result.append({'combination_id': identifier, 'mechanisms': dict(mechanisms),
                       'required_bindings': bindings})
        identifiers.add(identifier)
        tuples.add(key)
    return result


def combination_constraint(alternatives):
    return {'id': 'screened_mechanism_combinations', 'parameters': list(MECHANISMS),
            'alternatives': alternatives}
