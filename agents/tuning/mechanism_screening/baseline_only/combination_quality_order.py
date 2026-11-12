"""Conditional quality implications from an explicitly supplied user premise."""
from .combination_prompts import MECHANISMS

INVARIANT = {'enable_prefix_caching', 'chunked_prefill'}
ORDERED = set(MECHANISMS) - INVARIANT
PREMISE_ID = 'user.quality_default_monotonicity'


def validate_defaults(defaults, catalog):
    if not isinstance(defaults, dict) or set(defaults) != ORDERED:
        raise ValueError('Quality monotonic defaults must specify exactly the five non-cache-switch mechanisms')
    for key, value in defaults.items():
        if not isinstance(value, str) or not any(c['mechanisms'][key] == value for c in catalog.values()):
            raise ValueError(f'Unknown quality default for {key}: {value}')


def failure_closure(defaults, anchor, catalog):
    """Keep failed nondefault coordinates fixed; free defaults and invariant switches.

    Default -> any alternative is non-improving, in any context, including joint
    changes and required mechanical dependencies. No ordering is asserted between
    distinct alternatives, or when restoring a nondefault coordinate to default.
    """
    predicate = {k: [anchor[k]] for k in sorted(ORDERED) if anchor[k] != defaults[k]}
    members = {identifier for identifier, row in catalog.items()
               if all(row['mechanisms'][k] in values for k, values in predicate.items())}
    return predicate, members
