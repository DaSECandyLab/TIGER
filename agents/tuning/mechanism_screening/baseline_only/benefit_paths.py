"""Conditional mechanism knowledge, not measured gains or mandatory retention rules."""

BENEFIT_PATHS = {
    'screening.benefit.ngram_scope': {
        'mechanism': 'speculative_method=ngram',
        'source': 'Screening prior: no default resource-enabling interaction for ngram',
        'causal_path': 'No pre-injected capacity/topology benefit path. Judge ngram primarily by measured '
                       'relative performance and workload evidence for useful speculation.',
        'conditions': ['A claim of a special cross-parameter benefit needs specific contextual evidence'],
        'countereffects': ['Untried draft length, lookup ranges or batch settings are not themselves '
                           'a concrete benefit path',
                           'Unknown acceptance or validation cost does not counter observed slowdown'],
        'relative_advantage': 'Do not manufacture a retention path merely because further tuning is possible. '
                              'This is not a universal assertion that ngram can never interact or accelerate.',
    },
    'screening.benefit.weight_quantization': {
        'mechanism': 'quantization',
        'source': 'Screening mechanism prior; applicability must be assessed against supplied context',
        'causal_path': 'Fewer stored weight bytes -> memory headroom -> a model replica may fit at lower TP, '
                       'allowing more DP replicas on the same GPUs, or more concurrent sequences / larger batches. '
                       'Reduced weight traffic can also help a bandwidth-limited execution.',
        'conditions': ['Supported quantization kernels and acceptable quality',
                       'Total weights, KV, activations and workspace fit the proposed layout',
                       'Enough workload concurrency to use extra replicas or batch capacity'],
        'countereffects': ['Dequantization/kernel overhead', 'Communication and scheduling costs',
                           'The unquantized control may already fit the same layout or batch'],
        'relative_advantage': 'Explain what layout, capacity or traffic advantage the control lacks; '
                              'memory saving alone does not establish a throughput gain.',
    },
    'screening.benefit.kv_cache_dtype': {
        'mechanism': 'kv_cache_dtype',
        'source': 'Screening mechanism prior; applicability must be assessed against supplied context',
        'causal_path': 'Fewer bytes per cached token -> smaller KV footprint / traffic -> potentially more '
                       'resident sequences, larger batches or lower attention memory traffic.',
        'conditions': ['The selected format actually reduces bytes relative to the control',
                       'Backend support, quality and scale handling are adequate',
                       'KV capacity or traffic is relevant at this workload length and concurrency'],
        'countereffects': ['Conversion/scaling/kernel costs', 'Non-KV bottlenecks',
                           'Smaller KV does not reduce model weight size or by itself make TP=1 feasible'],
        'relative_advantage': 'Compare feasible batch capacity or attention traffic with the control; '
                              'low occupancy weakens capacity benefit but does not disprove bandwidth benefit.',
    },
    'screening.benefit.dtype': {
        'mechanism': 'dtype',
        'source': 'Screening mechanism prior; applicability must be assessed against supplied context',
        'causal_path': 'A genuinely narrower model format can reduce storage/traffic and enable capacity '
                       'changes. Equal-width formats may instead differ in supported kernels or execution paths.',
        'conditions': ['Compare actual byte widths and legal backend/model support',
                       'For equal-width formats, identify a specific applicable kernel difference',
                       'Respect quality constraints and materialized dtype/KV dependencies'],
        'countereffects': ['BF16 and FP16 are both 16-bit: switching between them does not halve weights '
                           'or inherently enable more DP replicas or a larger batch',
                           'No general performance ordering follows from the format name'],
        'relative_advantage': 'Name an applicable storage or kernel advantage over the control; '
                              'do not copy the quantization memory argument onto BF16 versus FP16.',
    },
}
