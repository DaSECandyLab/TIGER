"""Baseline/static signal ranking for the LLM's combination prescreen."""
from copy import deepcopy

from .workload_signal import score_mechanisms


POLICY_ID = 'screening.signed_workload_signal_policy'
REFERENCE_ID = 'screening.workload_signal_reference'
INVARIANT_SWITCHES = frozenset({'enable_prefix_caching', 'chunked_prefill'})
POLICY = {
    'source': 'Explicit user-supplied heuristic prescreening principle',
    'claim': '按机制分数绝对值降序审查，符号表示方向。LLM 结合分数、逐项 '
             'workload feature、机制知识和约束判断强弱，不设新的数值阈值。'
             '强正信号仅可把 prefix caching 和 chunked prefill 固定开启；'
             '其他机制的强正信号不直接授权固定开启。强负信号可以把机制关闭，'
             'KV cache 则回到 16bit 分支，不要求该机制保持推理行为不变。'
             '跨参数约束的分支须单独判断，不能被单项固定规则误删。'
             '初筛适度保守，默认保留；整体信号较突出且主要事实与知识支持方向时可以固定。'
             '仅在整体分数明显回落且较大正负贡献严重抵消时优先保留，不按字段数量多数固定。'
             '是否严重抵消按实际贡献幅度判断，不能因存在一项反向贡献就否决固定。'
             '多个较大的同向贡献可构成主导依据；不要求接近全表最大分数或证明收益为零。'
             '未知指标不是负面事实，也不单独否决固定；有具体反向证据或可信反转路径时保留。'
             '不预设固定数量或机制，分数为零或缺失信号不能支持固定；'
             '这些结论是初筛先验，不是实测收益。',
    'inference_behavior_invariant_switches': sorted(INVARIANT_SWITCHES),
    'strong_signal_threshold': None,
    'strength_decided_by': 'LLM using supplied features and applicable knowledge',
    'prescreening_style': 'moderately conservative; retain weaker net signals with substantial cancellation; unknown metrics alone do not veto fixing; no fixed top-k',
}


def activation_values(mechanism, values):
    """Map simplified on/off branches to actual values in the legal catalog."""
    enabled, disabled = [], []
    for value in values:
        branch = None
        if mechanism in INVARIANT_SWITCHES and isinstance(value, bool):
            branch = value
        elif mechanism == 'cacheblend_mode' and value in ('on', 'off'):
            branch = value == 'on'
        elif mechanism in ('quantization', 'speculative_method') and isinstance(value, str):
            branch = value != 'none'
        elif mechanism == 'kv_cache_dtype' and isinstance(value, str):
            if value.startswith(('fp8', 'int8')) or value == '8bit':
                branch = True
            elif value in ('auto', 'bfloat16', 'float16', 'half', 'bf16', 'fp16', '16bit'):
                branch = False
        if branch is not None:
            (enabled if branch else disabled).append(value)
    return enabled, disabled


def build_signal_context(evidence, catalog, reference):
    """Consume only admissible facts from this workflow, retaining exact citations."""
    observations, fact_refs = {}, {}
    keys = dict.fromkeys(
        key for rule in reference['mechanisms'].values() for key in rule['feature_directions'])
    for key in keys:
        paths = ([key] if key.startswith('static.') else
                 [f'baseline.flock_workload_features.{key}', f'baseline.{key}'])
        for path in paths:
            fact = evidence['facts'].get(path)
            if fact is not None and fact.get('available', True) and fact.get('value') is not None:
                observations[key] = fact['value']
                fact_refs[key] = path
                break
    result = score_mechanisms(observations, reference)
    ordered = sorted(result['mechanisms'],
                     key=lambda name: -abs(result['mechanisms'][name]['score']))
    rows = list(catalog.values()) if isinstance(catalog, dict) else list(catalog)
    ranking = []
    for rank, name in enumerate(ordered, 1):
        item = deepcopy(result['mechanisms'][name])
        for key, contribution in item['contributions'].items():
            contribution['fact_ref'] = fact_refs[key]
        values = []
        for row in rows:
            value = row['mechanisms'][name]
            if value not in values:
                values.append(value)
        enabled, disabled = activation_values(name, values)
        score = item['score']
        ranking.append(item | {
            'rank': rank, 'mechanism': name, 'absolute_score': abs(score),
            'direction': 'positive' if score > 0 else 'negative' if score < 0 else 'neutral',
            'inference_behavior_invariant': name in INVARIANT_SWITCHES,
            'enabled_values': enabled, 'disabled_values': disabled,
            'legal_values': values,
            'positive_fixing_allowed': name in INVARIANT_SWITCHES and score > 0,
            'negative_fixing_allowed': bool(disabled) and score < 0,
        })
    return {
        'reference_id': reference['reference_id'], 'reference_source_kind': reference['source_kind'],
        'policy_knowledge_ref': POLICY_ID, 'reference_knowledge_ref': REFERENCE_ID,
        'ordering': 'descending absolute score; stable mechanism order for ties',
        'strong_signal_threshold': None, 'strength_decided_by': 'LLM',
        'ranking': ranking,
    }
