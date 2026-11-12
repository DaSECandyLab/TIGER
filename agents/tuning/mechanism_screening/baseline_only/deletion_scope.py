"""Narrow decision-first deletion boundaries; these are conditional user premises."""

PRECISION = ('quantization', 'dtype', 'kv_cache_dtype')
QUALITY_PREMISE = 'user.quality_exact_precision_tuple'
QUALITY_CLAIM = (
    'Always fix the measured quantization/dtype/kv_cache_dtype tuple, including defaults. '
    'Precision tuples are unordered because their coordinates interact. Also fix '
    'cacheblend_mode when on and speculative_method when not none. Off mechanisms '
    'may be enabled without repairing this failure; prefix caching and chunked prefill '
    'are quality invariant. This is a user premise, not additional measurements.'
)


def quality_scope(anchor, catalog):
    predicate = {k: [anchor[k]] for k in PRECISION}
    for key, inactive in (('cacheblend_mode', 'off'), ('speculative_method', 'none')):
        if anchor[key] != inactive:
            predicate[key] = [anchor[key]]
    return predicate, {k for k, row in catalog.items()
                       if all(row['mechanisms'][p] in values for p, values in predicate.items())}
