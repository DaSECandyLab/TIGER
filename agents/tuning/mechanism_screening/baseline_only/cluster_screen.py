"""Workload-conditioned LLM partitions with class-level performance decisions."""
from copy import deepcopy
import re

from .decision_screen import DecisionWorkflow, PERFORMANCE_RULE, INITIAL_RULE
from .decision_questions import STANDARD
from .combination_screen import Workflow
from .combination_prompts import SCHEMAS, TASKS, PREDICATE, EVIDENCE, TEXT, MECHANISMS, obj, arr
from .combination_space import digest
from .io import write

ALGORITHM = 'llm-mechanism-classes.v3.1'
CLASS = obj(predicate=PREDICATE, shared_basis=TEXT, reversal_condition=TEXT, evidence=EVIDENCE)
INITIAL_GROUP = obj(values={'type': 'array', 'minItems': 1, 'uniqueItems': True,
    'items': {'type': ['string', 'boolean']}},
    **{k: v for k, v in CLASS['properties'].items() if k != 'predicate'})
SPLIT = obj(class_id=TEXT, dimension={'enum': list(MECHANISMS)}, reason=TEXT, groups=arr(INITIAL_GROUP, 2))
SCHEMAS['performance_cluster_partition'] = obj(dimension={'enum': list(MECHANISMS)},
    groups=arr(INITIAL_GROUP, 1), rationale=TEXT)
SCHEMAS['performance_cluster_decision'] = obj(action={'enum': ['run', 'stop']},
    decision={'anyOf': [obj(class_id=TEXT, question=TEXT, decision_value=TEXT,
        representativeness=TEXT, comparison=obj(combination_id=TEXT, control_observation_id=TEXT),
        evidence=EVIDENCE), {'type': 'null'}]},
    stop_reason={'anyOf': [TEXT, {'type': 'null'}]})
CLASS_RULE = obj(**{k: v for k, v in PERFORMANCE_RULE['properties'].items() if k != 'predicate'}, class_id=TEXT)
CLASS_RULE['properties']['action'] = {'enum': ['drop', 'retain']}
INITIAL_CLASS_RULE = obj(**{k: v for k, v in INITIAL_RULE['properties'].items() if k != 'predicate'}, class_id=TEXT)
SCHEMAS['performance_cluster_initial'] = obj(refinement_reasoning=TEXT, splits=arr(SPLIT) | {'maxItems': 0},
    decisions=arr(INITIAL_CLASS_RULE), summary=TEXT)
SCHEMAS['cluster_assess'] = obj(performance_signal={'enum': ['good', 'bad', 'inconclusive']},
    refinement_reasoning=TEXT, splits=arr(SPLIT), decisions=arr(CLASS_RULE),
    **{k: v for k, v in SCHEMAS['regional_performance']['properties'].items() if k != 'decisions'})
CLUSTER_STANDARD = '''
# LLM mechanism partition contract
Classes are workload-conditioned, provisional shared PERFORMANCE hypotheses, not measured
performance equivalence. Use interactions between reuse path, precision/kernel/memory path,
and speculative decoding, including legal mechanical dependencies and downstream tuning.
Do not cluster by Hamming distance, mechanism names alone, or member count. Never invent
runtime gains. Account for class-varying coordinates that might reverse the conclusion.
Use predicates over the supplied legal catalog. The program computes exact IDs/membership.
Start with coarse classes, split when a concrete interaction changes
that behavior; neither a mandatory number of classes nor all singleton classes is desirable.
Quality must not influence class boundaries, shared_basis, reversal_condition, or evidence.
Do not mention quality, EM or F1 in refinement_reasoning, split reasons, or class descriptions.
For citations, fact_refs accepts exact evidence.facts IDs or an EXISTING non-null numeric/text
performance metric path under an observation, e.g. probe_001.metrics.wall_time_s. Cite that
observation in observation_refs too. Never invent paths; inspect observations[id].metrics.
Unmeasured/unavailable metrics cannot establish benefits or costs.
Use explicitly supplied weak-interaction expert priors when assessing shared behavior across
other parameter values. Cite them as user assumptions, not measured independence. Such a
prior can support broad transfer without separately testing every background, unless actual
counterevidence or a specific interaction limits its applicability.
Class evidence must cite workload facts and applicable mechanism knowledge. A class can remain
unresolved: membership alone never authorizes deleting unmeasured configurations.
'''
TASKS['performance_cluster_partition'] = '''# Task: partition remaining combinations with LLM mechanism reasoning
Return ONE dimension and disjoint groups of its values covering EVERY remaining legal configuration.
Output groups[].values, NOT predicates. The program constructs {dimension: values} predicates.
The FIRST partition MUST split on ONE shared mechanism dimension only, chosen by the LLM
from workload context. Every predicate must contain exactly that same single dimension.
Do not cross-product reuse, quantization, dtype and speculation in the initial partition.
For example CacheBlend=on is ONE entire coarse class and CacheBlend=off another;
none/gptq/awq subdivisions come later when performance evidence or a concrete interaction
justifies refinement. Do not label unresolved precision interactions as initial exclusions.
Each class must explain the shared performance pathway under THIS workload, why omitted/varying
coordinates can provisionally share it, and what would require splitting. Later child predicates can fix multiple interacting mechanisms. Use all remaining combinations, including execution-blocked
members (blockers limit representative selection, not membership). Do not label classes good
or bad yet. Cite the supplied exact fact, knowledge and observation IDs.
''' + CLUSTER_STANDARD + STANDARD.replace('return rules=[]', 'return the empty deletion array required by this stage schema')
TASKS['performance_cluster_initial'] = """# Task: workload-context initial screening AFTER LLM classification
The coarse mechanism_classes ALREADY exist. Examine these classes using workload features,
applicable mechanism knowledge and the baseline, before selecting any new experiment.
Screen only the EXISTING coarse classes. splits MUST be []. Do not deepen the initial tree.
If evidence supports only a subgroup of a class, retain the parent for later experimental
refinement rather than broadening that deletion. Propose drop decisions for WHOLE existing
classes only. No supported deletion means decisions=[]; unmentioned classes remain eligible.
Explain in refinement_reasoning why the existing coarse scope is supported or unresolved.
For example, substantial USEFUL native prefix reuse and its time-saving opportunity can
support examining cacheblend=off plus prefix=off as a disadvantaged class. Check that the
reported ratio reflects usable reuse and relevant workload cost, and assess competing kernel,
precision, speculation, scheduling and downstream tuning paths. A high ratio alone is not a
universal automatic deletion trigger, and the argument must not accidentally cover CacheBlend-on.
Explain benefit_limitation_evidence, cost_evidence and scope_justification for each proposed
class deletion. Missing costs are not measured harm. Initial knowledge-backed deletion is
allowed without a new ablation for every class, when the workload argument is sufficiently
clear. Quality plays no part here. Do not repeat classification from scratch.
""" + CLUSTER_STANDARD + STANDARD.replace('return rules=[]', 'return the empty deletion array required by this stage schema')
TASKS['performance_cluster_decision'] = '''# Task: choose one existing class and its next experiment
Consider ALL mechanism_classes in the supplied context. Choose using only:
(1) decision_value: potential to eliminate clearly poor or discover clearly good combinations;
(2) representativeness: how many combinations share the behavior tested and why the chosen
point's result can inform them. Explain the comparison with the strongest alternative class
briefly within these two fields; do not output a candidate scoring table.
Ground both in workload features, applicable knowledge and performance observations. Low useful
prefix reuse and high theoretical segment reuse can support a CacheBlend experiment, but are
not measured savings and percentages with different denominators are not directly comparable.
Select class_id from mechanism_classes and combination_id from its runnable_ids. Select an
existing complete control from observations. IDs are opaque: copy them exactly, never construct
or guess them. Full configurations and actual changed parameters are computed by the program.
Choose a representative testing common behavior or the strongest plausible reversal boundary,
not simply the numerical center. Evidence must cite the selected control, workload facts and
mechanism knowledge. action=run requires decision and null stop_reason; stop requires a
substantive stop_reason and null decision. No worthwhile remaining experiment is a valid stop.
''' + CLUSTER_STANDARD + STANDARD.replace('return rules=[]', 'return the empty deletion array required by this stage schema')
TASKS['cluster_assess'] = """# Task: FIRST refine the mechanism partition, THEN decide whole-class performance actions
Step 1 -- performance_signal and refinement_reasoning: judge the actual PERFORMANCE gap
against the matched control and relevant historical measurements, not baseline alone.
Use good for a meaningful advantage, bad for a meaningful disadvantage, or inconclusive
when the gap is small/uncertain or the measurement does not resolve the comparison.
The program's feasible_win/feasible_not_win is only a numeric ordering, NOT evidence of
significance. Explain gap magnitude, comparability and uncertainty concisely in
refinement_reasoning; do not invent repeatability, thresholds or statistical confidence.
For good results, prioritize OTHER competing classes: find which rival subgroups are now
clearly disadvantaged. For bad results, prioritize the TESTED class: find which subgroups
share the observed problem and which could reverse it. This is a review priority, not a
fixed deletion whitelist; justify any other affected scope using the actual evidence.
Where a supplied expert prior authorizes exclusions following an attributable slowdown,
consider that prior when judging the signal; do not require an arbitrary large-gap threshold
or repeated tests of every background. Explain why simultaneous configuration changes do
not defeat attribution. A prior alone without the required observation does not show harm.
For inconclusive results, retain unresolved classes; do not issue class drop decisions.
A useful one-layer refinement for the next experiment remains allowed.

Step 2 -- splits: first consider whether the relevant class needs splitting, before deciding
its fate. Use workload features, actual measurements, mechanism interactions, mechanical
constraints and applicable expert priors to choose the dimension most likely to CHANGE THE
RETENTION/DELETION CONCLUSION. Each split.reason must explain, together:
- the causal link from that dimension to the measured gap;
- which groups could receive different judgments, and what could reverse the conclusion;
- why this is the coarsest adequate boundary; group values sharing a judgment together.
These are the split standard, not extra experiment-acquisition scores. Do not split merely
because values differ or to maximize member count. An unresolved subgroup can remain for
later rounds. Different parents can use different axes; there is no fixed axis order.
For CacheBlend, inspect linked prefix/chunked/eager settings: an axis already constant
within a class cannot split it. A supplied limited-benefit expert prior can lower a split's
priority when it cannot change the judgment; it is not a measured bound or proof of no gain.

Step 3 -- decisions: assess WHOLE resulting leaf classes, using drop or retain.
Consider meaningful optimization paths and counterexamples to transferring the measured gap.
One poor representative does not prove the whole class poor. Nevertheless ALL children may
be dropped if the gap and applicable knowledge support that scope despite meaningful reversal
paths. No mandatory survivor and no exhaustive measurement of every combination is required.
If the parent already supports one common judgment, splits=[] and deciding the entire parent
is valid; do not manufacture a split only to delete identical children. Missing evidence means
retain for exploration, not verified good. Performance deletion remains contextual LLM judgment.

# Refinement structure: coarse to fine ACROSS rounds
Each split selects ONE dimension of a class present BEFORE this assessment, with disjoint,
exhaustive, nonempty groups. The program combines values with the parent predicate.
Child IDs are parent_id + '.1', '.2', etc. Each affected class may gain AT MOST ONE depth
per round. Newly created children MUST NOT be split again in this round. They can be refined
in a later round after another experiment. No recursive splits in one response.
Decisions may reference only final leaves, never a split parent. Unaffected classes persist.
Quality cannot motivate performance_signal, refinement or performance class deletion.
""" + TASKS['regional_performance'].replace('a quality-passing CacheBlend-on win',
    'a CacheBlend-on performance win').replace('together, including quality.',
    'together; quality is handled separately in final assess.') + CLUSTER_STANDARD



class ClusterWorkflow(DecisionWorkflow):
    algorithm = ALGORITHM
    acquisition_method = 'llm_partition_two_criterion_class_decision'

    def active_classes(self):
        remaining = self.remaining()
        return {k: v | {'member_ids': sorted(set(v['member_ids']) & remaining)}
                for k, v in self.state.get('mechanism_classes', {}).items()
                if set(v['member_ids']) & remaining}

    def context(self, *, include_quality=False, **extra):
        result = super().context(include_quality=include_quality, **extra)
        runnable = {r['combination_id'] for r in self.experiment_options()}
        result['mechanism_classes'] = [v | {'class_id': k, 'member_count': len(v['member_ids']),
            'runnable_ids': sorted(set(v['member_ids']) & runnable)} for k, v in self.active_classes().items()]
        return result

    def normalize_initial_partition(self, raw):
        classes = [{k: v for k, v in group.items() if k != 'values'} |
                   {'predicate': {raw['dimension']: group['values']}} for group in raw['groups']]
        return self.normalize_partition(classes, self.remaining())

    def normalize_partition(self, classes, expected, parent=None):
        if parent is None:
            axes = [tuple(c['predicate']) for c in classes]
            if any(len(axis) != 1 for axis in axes) or len(set(axes)) != 1:
                raise ValueError('Initial partition must be coarse: split on ONE shared mechanism dimension only')
        groups, seen = {}, set()
        for index, group in enumerate(classes, 1):
            self.refs(group['evidence'])
            if not group['evidence']['fact_refs'] or not group['evidence']['knowledge_refs']:
                raise ValueError('Each class needs workload facts and mechanism knowledge')
            members = self.members(group['predicate'])
            if not members <= expected or seen & members:
                raise ValueError(f'Classes overlap or escape parent; predicate={group["predicate"]}; '
                                 f'overlap={sorted(seen & members)}; outside={sorted(members - expected)}')
            key = f'{parent}.{index}' if parent else 'class_' + digest(sorted(members))[:12]
            groups[key] = deepcopy(group) | {'member_ids': sorted(members)}
            seen |= members
        if seen != expected:
            raise ValueError(f'Partition must cover every member; missing IDs={sorted(expected - seen)}')
        return groups

    def refined_classes(self, splits):
        groups = deepcopy(self.active_classes())
        eligible = set(groups)
        # Also guard repeated assessment transactions within the same round.
        for update in self.state.get('class_history', []):
            if update['round'] == self.state['round']:
                for previous in update.get('splits', []):
                    eligible.difference_update(
                        f"{previous['class_id']}.{i}" for i in range(1, len(previous['groups']) + 1))
        parents = set()
        for split in splits:
            key = split['class_id']
            if key in parents or key not in eligible:
                raise ValueError('Refine at most ONE layer per round; split only pre-existing leaves once, never newly created children')
            if len(split['groups']) < 2:
                raise ValueError('A split needs at least two nonempty child groups')
            parents.add(key)
            children_raw = [{k: v for k, v in group.items() if k != 'values'} |
                {'predicate': groups[key]['predicate'] | {split['dimension']: group['values']}}
                for group in split['groups']]
            children = self.normalize_partition(children_raw, set(groups[key]['member_ids']), key)
            groups.pop(key)
            groups.update(children)
        return groups

    def ensure_classes(self, stem):
        if 'mechanism_classes' not in self.state:
            raw = self.call('cluster_partition', self.context(), self.normalize_initial_partition, stem + '_classes')
            self.state['mechanism_classes'] = self.normalize_initial_partition(raw)
            self.state['class_history'] = [{'round': self.state['round'], 'partition': deepcopy(self.state['mechanism_classes']),
                                           'rationale': raw['rationale']}]
            self.save()

    def expand_class_actions(self, raw):
        self.validate_refinement_language(raw)
        groups = self.refined_classes(raw['splits'])
        rules, seen = [], set()
        for rule in raw['decisions']:
            key = rule['class_id']
            if key not in groups or key in seen:
                raise ValueError('Decision must reference each final leaf at most once: ' + key)
            seen.add(key)
            rules.append({k: v for k, v in rule.items() if k != 'class_id'} | {'predicate': groups[key]['predicate']})
        return groups, rules

    def initial_pass(self, stem):
        self.ensure_classes(stem)
        if 'initial_completed_round' in self.state:
            self.state['initial_completed_round'] = self.state['round']
            self.save()
            return
        def validate(raw):
            if raw['splits']:
                raise ValueError('Initial screening must keep coarse classes: splits must be empty')
            _, rules = self.expand_class_actions(raw)
            return self.rules(rules, self.remaining(), initial=True)
        raw = self.call('cluster_initial', self.context(), validate, stem + '_initial')
        groups, proposed_rules = self.expand_class_actions(raw)
        proposed = [{'rule_id': 'r_' + digest(r)[:12], 'rule': r} for r in proposed_rules]
        audit = {'reviews': [], 'summary': 'No proposed class exclusions'}
        if proposed:
            audit = self.call('initial_audit', self.context(proposed_initial_rules=proposed),
                lambda r: self.validate_audit(r, proposed), stem + '_initial_audit')
        accepted = {r['rule_id'] for r in audit['reviews'] if r['verdict'] == 'supported'}
        rules = self.rules([p['rule'] for p in proposed if p['rule_id'] in accepted], self.remaining(), initial=True)
        accepted_ids = {r['class_id'] for r, p in zip(raw['decisions'], proposed) if p['rule_id'] in accepted}
        self._class_update = {'classes': groups, 'refinement_reasoning': raw['refinement_reasoning'],
            'splits': raw['splits'], 'decisions': [r for r in raw['decisions'] if r['class_id'] in accepted_ids],
            'stage': 'initial_after_classification'}
        self.apply(rules)
        self.state['history'].append({'stage': 'initial', 'round': self.state['round'], 'rules': rules,
            'proposed_rules': proposed, 'audit': audit, 'summary': raw['summary'],
            'source': 'class_scoped_workload_context', 'refinement_reasoning': raw['refinement_reasoning']})
        self.state['initial_completed_round'] = self.state['round']
        self.save()

    def call(self, stage, context, validator, stem):
        if stage in ('cluster_partition', 'cluster_initial', 'cluster_decision'):
            context = self.performance_context(context)
            original = validator
            def validator(raw):
                self.check_performance_refs(raw, context)
                return original(raw)
            return Workflow.call(self, 'performance_' + stage, context, validator, stem)
        if stage == 'regional_performance':
            # Performance class reasoning never needs mixed quality knowledge.
            context = deepcopy(context)
            context['evidence']['knowledge'] = self.performance_context(context)['evidence']['knowledge']
            original = validator
            def expand(raw):
                self.validate_performance_signal(raw)
                self.validate_refinement_language(raw)
                self.check_performance_refs(raw['splits'], self.performance_context(context))
                groups = self.refined_classes(raw['splits'])
                decisions, seen = [], set()
                for rule in raw['decisions']:
                    key = rule['class_id']
                    if key not in groups or key in seen:
                        raise ValueError('Decision must reference each current/post-split class at most once: ' + key)
                    seen.add(key)
                    decisions.append({k: v for k, v in rule.items() if k != 'class_id'} |
                                     {'predicate': groups[key]['predicate']})
                return {k: v for k, v in raw.items() if k not in ('performance_signal', 'refinement_reasoning', 'splits', 'decisions')} | {'decisions': decisions}
            raw = Workflow.call(self, 'cluster_assess', context, lambda r: original(expand(r)), stem)
            # Defer mutation until the assessment transaction is applied by the parent loop.
            self._class_update = {'classes': self.refined_classes(raw['splits']),
                                  'performance_signal': raw['performance_signal'],
                                  'refinement_reasoning': raw['refinement_reasoning'],
                                  'splits': raw['splits'], 'decisions': raw['decisions']}
            return expand(raw)
        return super().call(stage, context, validator, stem)

    @staticmethod
    def validate_performance_signal(raw):
        if raw.get('performance_signal') not in ('good', 'bad', 'inconclusive'):
            raise ValueError('Assess requires performance_signal: good, bad or inconclusive')
        if raw['performance_signal'] == 'inconclusive' and any(
                rule['action'] == 'drop' for rule in raw['decisions']):
            raise ValueError('Inconclusive performance cannot justify class drop decisions')

    @staticmethod
    def measured_fact(ref, context):
        parts = ref.split('.')
        if len(parts) < 3 or parts[1] != 'metrics':
            return False
        observation = context['observations'].get(parts[0])
        if not observation or observation['status'] != 'complete':
            return False
        value = observation.get('metrics', {})
        for part in parts[2:]:
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
        return value is not None and not isinstance(value, (dict, list))

    def refs(self, evidence):
        extra = [r for r in evidence['fact_refs'] if r not in self.evidence['facts']]
        if extra:
            context = self.context(include_quality=False)
            for ref in extra:
                if not self.measured_fact(ref, context) or ref.split('.')[0] not in evidence['observation_refs']:
                    raise ValueError('Unknown/unavailable measured fact ' + ref +
                        '; use exact evidence.facts IDs or existing observation_id.metrics paths and cite observation_refs')
        return super().refs(evidence | {'fact_refs': [r for r in evidence['fact_refs'] if r not in extra]})

    @staticmethod
    def validate_refinement_language(raw):
        text = str(raw['refinement_reasoning']) + str(raw['splits'])
        if re.search(r'quality|exact_match|\bf1\b|\bem\b|质量|准确率', text, re.I):
            raise ValueError('Refinement must use performance only: remove quality/EM/F1 from refinement_reasoning and splits; final decisions assess quality separately')

    def check_performance_refs(self, value, context):
        if isinstance(value, dict):
            if 'fact_refs' in value and 'knowledge_refs' in value:
                bad_facts = [r for r in value['fact_refs'] if r not in context['evidence']['facts'] and not self.measured_fact(r, context)]
                bad_knowledge = sorted(set(value['knowledge_refs']) - context['evidence']['knowledge'].keys())
                if bad_facts or bad_knowledge:
                    raise ValueError(f'Class reasoning needs performance-only evidence. Unavailable facts: {bad_facts}; '
                        f'unavailable or quality-mixed knowledge: {bad_knowledge}. '
                        'Reconsider these specific citations using an applicable performance-only source; '
                        'do not change the evidence scope or cite quality conditions to justify splitting.')
            for item in value.values():
                self.check_performance_refs(item, context)
        elif isinstance(value, list):
            for item in value:
                self.check_performance_refs(item, context)

    def normalize_class_decision(self, raw):
        if raw['action'] == 'stop':
            if raw['decision'] is not None or not raw['stop_reason']:
                raise ValueError('Stop requires null decision and a reason')
            return None
        q = raw['decision']
        if raw['action'] != 'run' or q is None or raw['stop_reason'] is not None:
            raise ValueError('Run requires a decision and null stop_reason')
        groups = self.active_classes()
        if q['class_id'] not in groups:
            raise ValueError('Select an existing class_id: ' + str(list(groups)))
        group = groups[q['class_id']]
        key = q['comparison']['combination_id']
        allowed = set(group['member_ids']) & {r['combination_id'] for r in self.experiment_options()}
        if key not in allowed:
            raise ValueError('Representative must be in this class runnable_ids: ' + str(sorted(allowed)))
        control_id = q['comparison']['control_observation_id']
        control = self.state['observations'].get(control_id)
        if not control or control['status'] != 'complete':
            raise ValueError('Control must be an existing complete observation')
        self.refs(q['evidence'])
        if not q['evidence']['fact_refs'] or not q['evidence']['knowledge_refs'] or control_id not in q['evidence']['observation_refs']:
            raise ValueError('Cite workload facts, mechanism knowledge, and selected control')
        row = self.catalog[key]
        diff = self.configuration_difference(row['configuration'], control['configuration'])
        comparison = q['comparison'] | {'changed_parameters': sorted(diff)}
        self.validate_comparison(comparison, set(group['member_ids']))
        return deepcopy(q) | {'predicate': group['predicate'], 'shared_basis': group['shared_basis'],
            'strongest_counterexample': group['reversal_condition'], 'comparison': comparison,
            'representative': row, 'performance_target_ids': group['member_ids'],
            'target_member_count': len(group['member_ids']), 'performance_scope_is_unproven': True,
            'configuration_difference': diff,
            'decision_id': 'q_' + digest([q['class_id'], group['member_ids'], key, control_id])[:16]}

    def propose_next_decision(self, stem):
        self.ensure_classes(stem)
        if 'decision_plan' not in self.state:
            raw = self.call('cluster_decision', self.context(experiment_options=self.experiment_options()),
                            self.normalize_class_decision, stem + '_decision')
            self.state['decision_plan'] = {'round': self.state['round'], 'response': raw,
                                          'decision': self.normalize_class_decision(raw)}
            self.save()
        return self.state['decision_plan']['decision']

    def performance_review(self, outcome, pending, assessment, stem):
        self._class_update = None
        result = super().performance_review(outcome, pending, assessment, stem)
        if self._class_update is not None:
            result['class_update'] = deepcopy(self._class_update)
        return result

    def apply(self, rules):
        super().apply(rules)
        update = getattr(self, '_class_update', None)
        if update is not None:
            self.state['mechanism_classes'] = update['classes']
            self.state.setdefault('class_history', []).append({'round': self.state['round'], **deepcopy(update)})
            self._class_update = None
        if 'mechanism_classes' in self.state:
            self.state['mechanism_classes'] = self.active_classes()

    def export(self, reason):
        result = super().export(reason)
        result['algorithm'] = ALGORITHM
        result['mechanism_classes'] = self.active_classes()
        result['class_history'] = self.state.get('class_history', [])
        write(self.output / 'screening_result.json', result)
        return result
