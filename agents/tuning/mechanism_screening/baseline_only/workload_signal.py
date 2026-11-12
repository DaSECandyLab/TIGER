"""Equal-weight averages of signed deviations from expert feature references."""
from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping
import json
import math
from pathlib import Path

DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / 'config/feature_reference_expert_v7.json'
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lookup(features, key):
    if key in features:
        return features[key]
    value = features
    for part in key.split('.'):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def score_mechanisms(
    features: Mapping,
    reference: Mapping | None = None,
    *,
    unavailable_features: Collection[str] = (),
    static_features: Mapping | None = None,
) -> dict:
    """Score available observations; callers mark inactive/unusable fields explicitly.

    Conditional directions are expert priors. This calculation does not infer
    their causal premises, feasibility, quality, or realized performance gains.
    """
    if reference is None:
        reference = json.loads(DEFAULT_REFERENCE.read_text())
    observations = dict(features)
    if static_features is not None:
        observations['static'] = static_features
    unavailable = set(unavailable_features)
    results = {}
    for mechanism, rule in reference['mechanisms'].items():
        directions = rule['feature_directions']
        contributions, skipped = {}, {}
        for key, direction in directions.items():
            if isinstance(direction, bool) or direction not in (-1, 1):
                raise ValueError(f'Invalid direction for {mechanism}.{key}')
            mean = _number(reference['features'][key]['reference_mean'])
            if mean is None or mean <= 0:
                raise ValueError(f'Reference mean must be finite and positive: {key}')
            if key in unavailable:
                skipped[key] = 'unavailable_or_inactive'
                continue
            raw = _lookup(observations, key)
            if raw is None:
                skipped[key] = 'missing_value'
                continue
            value = _number(raw)
            if value is None or value < 0:
                skipped[key] = 'invalid_value'
                continue
            deviation = (value - mean) / mean
            if not math.isfinite(deviation):
                skipped[key] = 'nonfinite_deviation'
                continue
            contributions[key] = {
                'current_value': value,
                'reference_mean': mean,
                'direction': direction,
                'relative_deviation': deviation,
                'signed_deviation': direction * deviation,
                'observation_source': 'static_features' if key.startswith('static.')
                else 'baseline_features',
            }
        count = len(contributions)
        # Divide first so averaging large finite contributions cannot overflow.
        score = (
            math.fsum(item['signed_deviation'] / count for item in contributions.values())
            if count else None
        )
        results[mechanism] = {
            'raw_score': score,
            'score': score if count else 0.0,
            'score_source': 'observed_feature_average' if count else 'missing_signal_zero',
            'imputed_from': [],
            'positive_option': rule.get('positive_option'),
            'negative_option': rule.get('negative_option'),
            'valid_feature_count': count,
            'configured_feature_count': len(directions),
            'coverage': count / len(directions) if directions else None,
            'reason': 'no_direction_defined' if not directions else (
                'no_valid_features' if not count else None
            ),
            'contributions': contributions,
            'skipped_features': skipped,
        }
    return {
        'reference_id': reference['reference_id'],
        'source_kind': reference['source_kind'],
        'mechanisms': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--reference', default=DEFAULT_REFERENCE, type=Path)
    parser.add_argument('--static-features', type=Path)
    parser.add_argument('--unavailable-feature', action='append', default=[])
    args = parser.parse_args()
    raw = json.loads(args.features.read_text())
    features = raw.get('flock_workload_features', raw.get('features', raw))
    if not isinstance(features, Mapping):
        parser.error('Expected a workload feature object')
    static = json.loads(args.static_features.read_text()) if args.static_features else None
    if static is not None and not isinstance(static, Mapping):
        parser.error('Expected a static workload feature object')
    result = score_mechanisms(
        features,
        json.loads(args.reference.read_text()),
        unavailable_features=args.unavailable_feature,
        static_features=static,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
