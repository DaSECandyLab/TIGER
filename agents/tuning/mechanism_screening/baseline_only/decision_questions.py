"""Performance-only partial-combination questions and evidence-guided selection."""
from copy import deepcopy
from .combination_prompts import MECHANISMS, SCHEMAS, TASKS, PREDICATE, obj, arr, TEXT, EVIDENCE
from .combination_space import digest

OUTCOMES = ('faster', 'not_faster', 'inconclusive')
NULL_TEXT = {'anyOf': [TEXT, {'type': 'null'}]}
BRANCH = obj(affected_ids=arr(TEXT), expected_decision=TEXT, required_evidence=TEXT)
QUESTION = obj(predicate=PREDICATE, question=TEXT,
    decision_value=TEXT, shared_basis=TEXT, strongest_counterexample=TEXT,
    varying_dimensions=arr(obj(mechanism={'enum': list(MECHANISMS)},
                               shared_effect=TEXT, reversal_condition=TEXT)),
    comparison=obj(combination_id=TEXT, control_observation_id=TEXT, changed_parameters=arr({**TEXT, 'pattern': '^[a-z][a-z0-9_]*$'}, 1)),
    evidence=EVIDENCE, outcomes=obj(**{k: BRANCH for k in OUTCOMES}))
STANDARD = '''
# Unified screening decision standard
Separate measured facts, applicable knowledge, conditional forecasts and unresolved questions.
Initial screening MAY delete evidence-supported performance scopes without new runs. If none
is justified return rules=[] and explain the gaps; no forced deletion or deprioritization.
A candidate is a DECISION QUESTION about a partial combination, with a concrete representative
and comparator, not a mechanism popularity score. Single-mechanism questions are allowed but
not mandatory. Choose the next experiment using TWO criteria only: its ability to eliminate
clearly poor combinations or find clearly good ones, and how many combinations share the
behavior its result can meaningfully represent. Large membership alone is not commonality. Small scopes can find valuable winners.
State why varying coordinates share the causal argument and which interactions could reverse it.
Known benefit does not imply value in measuring it again. Unknown cost is not measured harm.
Actual protocol, full configuration differences, censoring and historical controls
matter. Frozen-background speed is not the family's tuned optimum. Consider DP/TP/batch and
other relevant reversal paths without making hypothetical tuning an automatic retention veto.
Performance scope is autonomous LLM judgment with reasons, not a fixed deletion whitelist.
Pre-run outcome branches are hypotheses, NEVER automatic deletion instructions. Assess actual
results, expose contradictions, and say what was learned versus promised before the run.
Structural validation checks legal scopes and citations, not truth of natural-language reasoning.
'''


def register_prompts(continuation, quality_standard):
    SCHEMAS['next_decision'] = obj(action={'enum': ['run', 'stop']},
        decision={'anyOf': [QUESTION, {'type': 'null'}]}, stop_reason=NULL_TEXT)
    TASKS['next_decision'] = """# Task: directly choose the next decision and executable configuration
Use current workload features (static input structure and dynamic baseline/runtime observations),
mechanism knowledge, experiment_options, remaining_catalog and performance history to propose ONE next experiment,
not a list of candidates followed by a separate ranking. Base the choice on exactly two criteria:
1. decision_value: explain how the result could eliminate clearly poor combinations or discover
   clearly good combinations relative to existing observations. Distinguish a material opportunity
   from a trivial fluctuation; use observed workload scale and uncertainty, not an invented fixed
   speedup threshold. Repeating known behavior without changing any judgment is not useful.
2. shared_basis and varying_dimensions: explain which common mechanism behavior the representative
   tests, how broadly the result can inform legal combinations, and which variations could reverse
   it. Raw member count is not representativeness. Prefer meaningful coverage, while allowing a
   smaller scope when it offers a compelling good configuration. No minimum-count quota.
Choose a partial-combination predicate including the representative. Single-mechanism predicates
are allowed. varying_dimensions means variation AMONG MEMBERS MATCHING THE PREDICATE,
NOT changes from the baseline/control and NOT a list of mechanisms involved in the experiment.
If all seven mechanism values are fixed by the predicate, varying_dimensions MUST be [].
Do not list fixed coordinates here; discuss their effects in decision_value/shared_basis instead.
For a genuinely partial predicate, list only coordinates having multiple values among its legal
members, with shared_effect and reversal_condition. If you claim representation of a wider family,
make the predicate describe that family and contain the chosen full representative; otherwise
state that this is a single-point question rather than calling it an entire mechanism branch.
Select one unattempted runnable configuration and an existing complete control; do not filter by
answer quality. changed_parameters is an array of BARE PARAMETER NAMES, never descriptions:
correct: ["cacheblend_mode", "chunked_prefill", "enable_prefix_caching", "enforce_eager"];
wrong: ["cacheblend_mode: off -> on"]. Include ALL names that actually differ, including
mechanical dependencies. Values belong in the supplied full configurations, not this array. The full config
comes from the catalog; do not invent extra knobs. If a new control is needed, measure it first
against an available control and revisit the combined comparison next round.
Forecast faster/not_faster affected scopes as conditional performance hypotheses only. They may
include other remaining branches; inconclusive must have affected_ids=[]. Do not forecast quality
or preapprove deletion. Cite performance facts and knowledge. Budgets and runtime comparability
are execution constraints, not extra scored criteria. No grades, weights, probabilities or
five-axis checklist.
Ground BOTH criteria in the supplied context: decision_value must connect specific workload
facts to a mechanism benefit/limitation, then explain why this comparison could expose a clearly
good or poor combination. shared_basis must explain why those workload conditions apply to the
represented scope. Cite the exact fact and knowledge IDs; merely listing citations is insufficient.
Illustrative reasoning, not a trigger rule: limited useful native prefix-cache reuse together
with high potential non-prefix segment reuse may make CacheBlend-on a promising experiment.
Check the actual input segmentation, available backend/configuration and historical observations;
consider known linked overheads and meaningful counterexamples. Distinguish low native cache hit
rate from low realized time savings, and theoretical reuse potential from measured runtime gains.
Do not subtract or directly compare prefix and segment percentages with different denominators.
This example neither guarantees a win nor makes CacheBlend mandatory. Qualitative likelihood is
allowed when grounded in evidence, but do not invent numerical probabilities or measured speedups.
action=run requires decision and null stop_reason. If no useful experiment
is justified, action=stop requires null decision and a substantive stop_reason.
"""
    for stage in ('decision_initial', 'initial_audit', 'next_decision'):
        key = 'performance_' + stage
        SCHEMAS[key] = SCHEMAS[stage]
        TASKS[key] = TASKS[stage] + continuation + STANDARD
    TASKS['regional_performance'] += continuation + STANDARD + quality_standard


class QuestionWorkflow:
    def performance_context(self, context):
        """Expose only performance evidence before the post-run assessment boundary."""
        import json
        import re
        result = deepcopy(context)
        metrics = (self.policy.get('quality') or {}).get('metrics', {})
        pattern = re.compile(r'quality|accuracy|exact_match|f1_score|\bf1\b|\bem\b|bleu|rouge|质量|准确率', re.I)
        def forbidden(value):
            return bool(pattern.search(str(value)))
        for key in list(result):
            if forbidden(key):
                result.pop(key)
        result['policy'] = {k: v for k, v in result['policy'].items()
                            if not forbidden(k) and k != 'screening_scope'}
        for kind in ('facts', 'unavailable_facts'):
            result['evidence'][kind] = {k: v for k, v in result['evidence'].get(kind, {}).items()
                if not forbidden(k) and not any(k == 'baseline.' + path or k.startswith('baseline.' + path + '.')
                                               for path in metrics.values())}
        result['evidence']['knowledge'] = {k: v for k, v in result['evidence']['knowledge'].items()
                                           if not forbidden(k) and not forbidden(json.dumps(v, ensure_ascii=False))}
        result['observations'] = {k: {field: o[field] for field in (
            'observation_id', 'combination_id', 'configuration', 'configuration_hash',
            'status', 'objective_value', 'metrics', 'censoring') if field in o}
            for k, o in result['observations'].items()}
        # Assessment prose and resolved scopes may encode quality. Keep the original
        # performance question/design and observed timing, not post-quality conclusions.
        result['history'] = []
        for h in self.state['history']:
            if h.get('stage') != 'experiment':
                continue
            item = {k: h[k] for k in ('stage', 'round', 'decision_id', 'decision', 'design', 'observation_id') if k in h}
            review = h.get('question_review')
            if h.get('outcome_class') in ('feasible_win', 'feasible_not_win') and review and not forbidden(json.dumps(review, ensure_ascii=False)):
                item['performance_learning'] = review
            result['history'].append(item)
        result.pop('assessed_comparisons', None)
        result.pop('single_mechanism_comparisons', None)
        return result

    def call(self, stage, context, validator, stem):
        if stage in ('decision_initial', 'initial_audit', 'next_decision'):
            context = self.performance_context(context)
            original_validator = validator
            def validator(raw):
                def check(value):
                    if isinstance(value, dict):
                        if 'fact_refs' in value and 'knowledge_refs' in value:
                            if not set(value['fact_refs']) <= context['evidence']['facts'].keys():
                                raise ValueError('Pre-assessment citation is not available performance evidence')
                            if not set(value['knowledge_refs']) <= context['evidence']['knowledge'].keys():
                                raise ValueError('Pre-assessment knowledge is not available performance evidence')
                        for item in value.values():
                            check(item)
                    elif isinstance(value, list):
                        for item in value:
                            check(item)
                check(raw)
                return original_validator(raw)
            stage = 'performance_' + stage
        return super().call(stage, context, validator, stem)

    def experiment_options(self):
        attempted = {o['configuration_hash'] for o in self.state['observations'].values()}
        return [self.catalog[k]
                for k in sorted(self.remaining()) if not self.catalog[k]['blockers']
                and self.catalog[k]['configuration_hash'] not in attempted]

    def normalize_decision(self, raw):
        if raw['action'] == 'stop':
            if raw['decision'] is not None or not raw['stop_reason']:
                raise ValueError('Stop requires null decision and a reason')
            return None
        if raw['action'] != 'run' or raw['decision'] is None or raw['stop_reason'] is not None:
            raise ValueError('Run requires one decision and null stop_reason')
        q = raw['decision']
        self.refs(q['evidence'])
        if not q['evidence']['fact_refs'] or not q['evidence']['knowledge_refs']:
            raise ValueError('Decision requires workload facts and mechanism knowledge')
        members = self.members(q['predicate'])
        row, control = self.validate_comparison(q['comparison'], members)
        if row['combination_id'] not in {r['combination_id'] for r in self.experiment_options()}:
            raise ValueError('Representative must be runnable, remaining and unattempted')
        if control['observation_id'] not in q['evidence']['observation_refs']:
            raise ValueError('Decision must cite its selected control observation')
        varying = {p for p in MECHANISMS if len({digest(self.catalog[k]['mechanisms'][p]) for k in members}) > 1}
        supplied = [v['mechanism'] for v in q['varying_dimensions']]
        if set(supplied) != varying or len(supplied) != len(varying):
            raise ValueError('Explain every varying dimension exactly once: ' + str(sorted(varying))
                             + f'. Predicate matches {len(members)} combinations. This means variation WITHIN that scope, not changes versus baseline. '
                             + ('Return varying_dimensions=[] for this fixed scope; put control differences in comparison.changed_parameters.' if not varying else 'Do not include fixed mechanism coordinates.'))
        for outcome, branch in q['outcomes'].items():
            allowed = set() if outcome == 'inconclusive' else self.remaining()
            if not set(branch['affected_ids']) <= allowed:
                raise ValueError('Outcome hypothesis exceeds allowed scope for ' + outcome)
        key = 'q_' + digest([sorted(members), row['configuration_hash'], control['configuration_hash'],
                              self.policy['objective'], self.policy['direction']])[:16]
        return deepcopy(q) | {'decision_id': key, 'representative': row,
            'performance_target_ids': sorted(members), 'target_member_count': len(members),
            'performance_scope_is_unproven': True,
            'configuration_difference': self.configuration_difference(row['configuration'], control['configuration'])}

    def propose_next_decision(self, stem):
        if 'decision_plan' not in self.state:
            raw = self.call('next_decision', self.context(experiment_options=self.experiment_options()),
                            self.normalize_decision, stem + '_decision')
            self.state['decision_plan'] = {'round': self.state['round'], 'response': raw,
                                           'decision': self.normalize_decision(raw)}
            self.save()
        return self.state['decision_plan']['decision']
