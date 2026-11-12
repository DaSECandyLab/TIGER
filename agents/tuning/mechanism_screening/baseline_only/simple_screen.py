"""Screen each mechanism once, then select and assess one experiment at a time."""
from copy import deepcopy
from math import fsum, isfinite

from .combination_screen import Workflow, load
from .decision_screen import DecisionWorkflow
from .combination_prompts import MECHANISMS
from .combination_space import digest
from .initial_screening import mechanism_knowledge, independent_exclusion_scope
from .signal_screening import POLICY as SIGNAL_POLICY, build_signal_context
from .workload_signal import DEFAULT_REFERENCE
from .io import ROOT, write
from . import simple_prompts  # Register the simple workflow stage contracts.


class SimpleWorkflow(DecisionWorkflow):
    algorithm = 'simple-mechanism-screening.v13'
    acquisition_method = 'workload_screening_question'

    def __init__(self, interaction_knowledge=None, signal_reference=None, **kwargs):
        self.signal_reference = deepcopy(signal_reference if signal_reference is not None else load(DEFAULT_REFERENCE))
        self.interactions = deepcopy(interaction_knowledge if interaction_knowledge is not None else
            load(ROOT / 'my_storage/static_knowledge_base/vllm-0.20.0/mechanism_interactions.json'))
        kwargs['evidence'] = deepcopy(kwargs['evidence'])
        for relation in self.interactions['relations']:
            kwargs['evidence']['knowledge'][relation['id']] = deepcopy(relation)
        kwargs['policy'] = deepcopy(kwargs['policy']) | {
            'simple_algorithm': self.algorithm,
            'precision_timing_neutral_fraction_of_baseline': 0.10,
            'interaction_digest': digest(self.interactions),
            'signal_reference_digest': digest(self.signal_reference),
            'signal_policy_digest': digest(SIGNAL_POLICY),
            'input_length_reference_tokens': 1000,
            'input_length_fact': 'baseline.flock_workload_features.prompt_tokens_per_request',
        }
        super().__init__(**kwargs)

    def feasible(self, observation):
        return (observation['status'] == 'complete' and observation.get('objective_value') is not None
                and self.quality(observation)['status'] in ('passed', 'disabled'))

    def timing_band(self, left, right):
        a, b = left['configuration'], right['configuration']
        quantization_changed = a.get('quantization') != b.get('quantization')
        low_kv_changed = (a.get('kv_cache_dtype') != b.get('kv_cache_dtype') and
            any(str(c.get('kv_cache_dtype', '')).startswith(('fp8', 'int8')) for c in (a, b)))
        precision = quantization_changed or low_kv_changed
        fraction = self.policy['precision_timing_neutral_fraction_of_baseline'] if precision else .02
        reference = self.state['observations']['baseline'] if precision else left
        return {'fraction': fraction, 'reference_observation_id': reference['observation_id'],
                'absolute_threshold_s': fraction * reference['objective_value'],
                'precision_comparison': precision, 'inclusive': True}

    def validate_timing_signal(self, raw, outcome, control):
        if self.policy['objective'] != 'wall_time_s':
            return
        band = self.timing_band(control, outcome)
        if self.feasible(outcome):
            if (abs(outcome['objective_value'] - control['objective_value']) <= band['absolute_threshold_s']
                    and raw['signal'] != 'inconclusive'):
                raise ValueError('Timing contrast is within the inclusive noise band; signal must be inconclusive')
        elif raw['signal'] == 'good':
            raise ValueError('Incomplete or infeasible observations cannot establish a good timing signal')
        elif (outcome['status'] == 'censored' and raw['signal'] == 'bad' and
              outcome['censoring']['lower_bound_s'] - control['objective_value'] <= band['absolute_threshold_s']):
            raise ValueError('Censored lower bound does not exceed the inclusive noise band; signal must be inconclusive')

    def assess(self, outcome, pending):
        assessment = super().assess(outcome, pending)
        control = self.state['observations'][pending['design']['control_observation_id']]
        if (self.policy['objective'] == 'wall_time_s' and self.feasible(outcome) and
                abs(outcome['objective_value'] - control['objective_value']) <=
                self.timing_band(control, outcome)['absolute_threshold_s']):
            assessment['outcome_class'] = 'inconclusive'
        return assessment

    def input_length_reference(self):
        path = self.policy['input_length_fact']
        value = self.evidence['facts'].get(path, {}).get('value')
        threshold = self.policy['input_length_reference_tokens']
        valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                 and isfinite(value) and value >= 0)
        relation = ('unknown' if not valid else 'below' if value < threshold
                    else 'above' if value > threshold else 'at_reference')
        return {'fact_ref': path, 'value': value, 'reference_tokens': threshold,
                'relation': relation, 'ratio_to_reference': value / threshold if valid else None,
                'basis': 'mean_input_tokens_per_request',
                'rule': '明显低于 1000 才视为短输入，明显高于 1000 才视为长输入；'
                        '接近参照值时长短不明确。relation 仅表示数值关系，不是长短分类；'
                        '未设置额外比例阈值，由模型结合输入分布判断明显程度。'}

    def scope_contract(self):
        return {'performance_basis': 'workload facts, historical measurements and relevant mechanism interactions',
                'quality_deletion_boundary': 'exact precision tuple plus enabled quality mechanisms',
                'deletions_final_within_run': True,
                'handoff': 'Preserve full surviving combinations and their conditional relationships.'}

    def context(self, **extra):
        context = self.performance_context(super().context(include_quality=False))
        # Keep gate eligibility visible without exposing answer-quality metrics to
        # performance screening. Recompute for restored and reused observations.
        for identifier, observation in self.state['observations'].items():
            status = self.quality(observation)['status']
            context['observations'][identifier].update(
                quality_gate_status=status,
                quality_gate_passed=status == 'passed' if status in ('passed', 'failed') else None)
        context.update(performance_judgment={'precision_neutral_fraction_of_baseline': 0.10,
                       'other_neutral_fraction_of_control': 0.02, 'boundary_inclusive': True,
                       'precision_scope': 'actual weight quantization change or low precision KV format change, including forced companions'},
                       interaction_knowledge=self.interactions,
                       questions=self.state.get('screening_questions', []),
                       feasible_control_ids=[k for k, o in self.state['observations'].items() if self.feasible(o)])
        blend_on = [r for r in self.catalog.values() if r['mechanisms']['cacheblend_mode'] == 'on']
        context['configuration_constraints'] = []
        if blend_on and all(r['mechanisms']['enable_prefix_caching'] is False for r in blend_on):
            context['configuration_constraints'].append({
                'when': {'cacheblend_mode': ['on']},
                'requires': {'enable_prefix_caching': [False]},
                'source': 'current_legal_catalog',
                'description': '当前合法域中，CacheBlend 开启时必须关闭原生 prefix caching；'
                               '不存在同时开启二者的合法配置。提出问题和比较机制束时须包含这一伴随变化。',
            })
        context['screening_history'] = [
            {k: h[k] for k in ('stage', 'round', 'observation_id', 'signal', 'summary', 'question_review') if k in h}
            | {'decisions': [d for d in h.get('decisions', h.get('rules', []))
                             if d.get('criterion') == 'performance']}
            for h in self.state['history']]
        # Present phase-specific facts together before the larger evidence catalog.
        # Original facts remain separate from the user-specified input-length reference.
        profile_paths = {
            'input_tokens_per_request': 'baseline.flock_workload_features.prompt_tokens_per_request',
            'output_tokens_per_request': 'baseline.flock_workload_features.generation_tokens_per_request',
            'output_tokens_p50': 'baseline.flock_workload_features.output_len_p50',
            'output_tokens_p95': 'baseline.flock_workload_features.output_len_p95',
            'native_prefix_hit_rate': 'baseline.prefix_cache_hit_rate',
            'theoretical_segment_reuse': 'static.variants.on.duplicate_segment_payload_fraction',
            'segment_tokens_p50': 'static.variants.on.segment_tokens.p50',
            'kv_cache_usage_mean': 'baseline.flock_workload_features.kv_cache_usage_mean',
        }
        facts = context['evidence']['facts']
        profile = {name: {'fact_ref': path, 'value': facts[path]['value']}
                   for name, path in profile_paths.items() if path in facts}
        return {'workload_profile': profile,
                'input_length_reference': self.input_length_reference(),
                **context, **extra}

    def initial_context(self, target_mechanism=None):
        context = self.context()
        signals = build_signal_context(self.evidence, self.catalog, self.signal_reference)
        identifiers = [row[0] for row in context['remaining_catalog']['rows']]
        context['remaining_catalog'] = {
            'columns': ['combination_id', 'configuration'],
            'rows': [[identifier, deepcopy(self.catalog[identifier]['configuration'])]
                     for identifier in identifiers],
        }
        context['catalog_interpretation'] = simple_prompts.INITIAL_CATALOG_INTERPRETATION
        if target_mechanism is not None:
            signal = next(row for row in signals['ranking'] if row['mechanism'] == target_mechanism)
            # Keep this request about one mechanism, without importing the
            # standalone signal workflow's action permissions or other scores.
            fields = ('mechanism', 'score', 'raw_score', 'absolute_score', 'direction', 'score_source',
                      'valid_feature_count', 'configured_feature_count', 'coverage', 'contributions',
                      'skipped_features', 'positive_option', 'negative_option', 'legal_values',
                      'enabled_values', 'disabled_values', 'positive_fixing_allowed', 'negative_fixing_allowed')
            context['target_workload_signal'] = {key: deepcopy(signal[key]) for key in fields}
            context['target_workload_signal'].update(
                reference_id=self.signal_reference['reference_id'],
                reference_source_kind=self.signal_reference['source_kind'],
                reference_note=self.signal_reference['mechanisms'][target_mechanism].get('note'),
                score_definition='mean(direction * (current_value - reference_mean) / reference_mean) over valid features',
                strength_threshold=None)
            contributions = [item['signed_deviation'] for item in signal['contributions'].values()]
            count = len(contributions)
            positive = fsum(max(value, 0) / count for value in contributions) if count else 0.
            negative = fsum(max(-value, 0) / count for value in contributions) if count else 0.
            gross = positive + negative
            retained_fraction = min(1., max(0., abs(signal['score']) / gross)) if gross else None
            context['target_workload_signal']['cancellation_analysis'] = {
                'positive_mean_contribution': positive,
                'negative_mean_magnitude': negative,
                'gross_mean_magnitude': gross,
                'net_fraction_of_gross': retained_fraction,
                'cancellation_fraction': 1. - retained_fraction if retained_fraction is not None else None,
                'purpose': '仅展开同一得分的幅度抵消，不是新评分、阈值或筛选结论。',
            }
            context['signal_fixing_policy'] = deepcopy(SIGNAL_POLICY)
            # Numeric calibration contains no other mechanism's knowledge or
            # identity. It lets the model distinguish a notable net signal from
            # cancellation without assuming that only the maximum can be strong.
            context['score_magnitude_reference'] = {
                'absolute_scores_descending': [row['absolute_score'] for row in signals['ranking']
                                               if row['valid_feature_count'] > 0],
                'source': 'same workload and unchanged expert scoring rules; mechanism names omitted',
                'rule': '用于判断数值尺度；不按排名或固定 top-k 决定，也不要求接近最大值才算强信号。'}
            context['evidence'] = deepcopy(context['evidence'])
            context['evidence']['knowledge'] = mechanism_knowledge(context['evidence']['knowledge'], target_mechanism)
            relations = [r for r in self.interactions['relations']
                         if r['type'] == 'performance_interaction' and target_mechanism in r['mechanisms']]
            context['interaction_knowledge'] = {
                'relations': deepcopy(relations),
                'semantics': deepcopy(self.interactions.get('semantics', {})),
            }
            # Previous mechanisms' prose must not leak their knowledge into this request.
            context['history'] = []
            context['screening_history'] = []
            context['configuration_constraints'] = []
            context['mechanical_screening_policy'] = simple_prompts.INITIAL_MECHANICAL_POLICY
            if target_mechanism in simple_prompts.INITIAL_FEATURE_GUIDANCE:
                context['feature_screening_guidance'] = simple_prompts.INITIAL_FEATURE_GUIDANCE[target_mechanism]
            context['target_mechanism'] = target_mechanism
            context['target_mechanism_values'] = list(dict.fromkeys(
                self.catalog[identifier]['mechanisms'][target_mechanism] for identifier in identifiers))
        else:
            # Only the question aggregation receives the cross-mechanism scores.
            context['mechanism_score_summary'] = [
                {key: row[key] for key in ('mechanism', 'score', 'absolute_score', 'direction',
                                           'valid_feature_count', 'configured_feature_count')}
                for row in signals['ranking']]
        return context

    def initial_exclusions(self, raw, mechanism):
        if raw['exclusions'] and raw.get('signal_strength') != 'strong':
            raise ValueError('Weak or unavailable net signals must have exclusions=[]. '
                             'Reconsider the score and cancellation by magnitude; do not relabel a weak '
                             'signal as strong merely to keep a proposed deletion. Deleting a subset '
                             'such as all 8bit KV formats is still fixing and is not exempt.')
        if any(mechanism not in rule['predicate'] for rule in raw['exclusions']):
            raise ValueError(f'Each initial exclusion must constrain target_mechanism {mechanism}')
        available = mechanism_knowledge(self.evidence['knowledge'], mechanism)
        decisions = []
        for rule in raw['exclusions']:
            if not set(rule['evidence']['knowledge_refs']) <= available.keys():
                raise ValueError('Initial citations must use only the target mechanism knowledge supplied in this request')
            scope = independent_exclusion_scope(self.catalog, mechanism, rule['predicate'][mechanism])
            proposed = self.members(rule['predicate'])
            for decision in self.exclusions([rule], initial=True, scope_ids=scope):
                decision['protected_combination_ids'] = sorted(proposed - scope)
                decision['scope_basis'] = 'independent target choice; forced companions protected by legal catalog'
                decisions.append(decision)
        if set().union(*(set(d['combination_ids']) for d in decisions)) >= self.remaining():
            raise ValueError('Cannot exclude the entire remaining space')
        return decisions

    def initial_pass(self, stem):
        if not self.state.get('simple_initial_completed'):
            completed = self.state.setdefault('simple_initial_mechanisms', {})
            for mechanism in MECHANISMS:
                if mechanism in completed:
                    continue
                raw = Workflow.call(self, 'performance_simple_initial', self.initial_context(mechanism),
                                    lambda r: self.initial_exclusions(r, mechanism),
                                    f'{stem}_initial_{mechanism}')
                rules = self.initial_exclusions(raw, mechanism)
                self.apply(rules)
                self.state['history'].append({'stage': 'initial', 'round': self.state['round'],
                    'target_mechanism': mechanism, 'signal_strength': raw['signal_strength'],
                    'rules': rules, 'summary': raw['summary']})
                completed[mechanism] = raw
                self.save()
            # Mechanism order must not determine the first experiment. Reconcile
            # questions with all exclusions before ranking the surviving questions.
            context = self.initial_context()
            context['mechanism_screenings'] = deepcopy(completed)
            if any(raw['questions'] for raw in completed.values()):
                questions = Workflow.call(self, 'performance_simple_initial_questions', context,
                                          lambda r: None, stem + '_initial_questions')
                self.state['screening_questions'] = questions['questions']
                self.state['history'].append({'stage': 'initial_questions', 'round': self.state['round'],
                                              'summary': questions['summary']})
            else:
                self.state['screening_questions'] = []
            self.state['simple_initial_completed'] = True
        self.state['initial_completed_round'] = self.state['round']
        self.save()

    def normalize_decision(self, raw):
        if raw['action'] == 'stop':
            if raw['decision'] is not None or not raw['stop_reason']:
                raise ValueError('Stop needs null decision and a reason')
            return None
        q = raw['decision']
        if q is None or raw['stop_reason'] is not None:
            raise ValueError('Run needs a decision and null stop_reason')
        questions = self.state.get('screening_questions', [])
        if self.state['attempts'] == 0 and questions and q['question'] != questions[0]:
            raise ValueError('The first experiment must validate the first initial question; '
                             'copy questions[0] verbatim into decision.question and choose its representative')
        self.refs(q['evidence'])
        options = {r['combination_id']: r for r in self.experiment_options()}
        if q['combination_id'] not in options:
            raise ValueError('Choose a runnable unattempted combination from experiment_options')
        control = self.state['observations'].get(q['control_observation_id'])
        if control is None or not self.feasible(control):
            raise ValueError('Choose a complete feasible historical control')
        row = options[q['combination_id']]
        difference = self.configuration_difference(row['configuration'], control['configuration'])
        comparison = {k: q[k] for k in ('combination_id', 'control_observation_id')}
        comparison['changed_parameters'] = list(difference)
        self.validate_comparison(comparison, {row['combination_id']})
        return q | {'comparison': comparison, 'decision_id': 'q_' + digest(comparison)[:16],
                    'predicate': {k: [v] for k, v in row['mechanisms'].items()},
                    'performance_target_ids': [row['combination_id']], 'configuration_difference': difference}

    def propose_next_decision(self, stem):
        if 'decision_plan' not in self.state:
            if self.state.get('screening_stop_reason'):
                raw = {'action': 'stop', 'decision': None, 'stop_reason': self.state['screening_stop_reason']}
            else:
                raw = Workflow.call(self, 'performance_simple_select',
                    self.context(experiment_options=self.experiment_options(),
                        required_question=(self.state.get('screening_questions') or [None])[0]
                            if self.state['attempts'] == 0 else None),
                    self.normalize_decision, stem + '_decision')
            if raw['action'] == 'stop':
                self.state['screening_stop_reason'] = raw['stop_reason']
            self.state['decision_plan'] = {'round': self.state['round'], 'response': raw,
                                           'decision': self.normalize_decision(raw)}
            self.save()
        return self.state['decision_plan']['decision']

    def supported_loss(self, rule, members):
        pair = rule['comparison']
        if not set(pair.values()) <= set(rule['evidence']['observation_refs']):
            raise ValueError('Cite both historical comparison observations')
        better, worse = (self.state['observations'][pair[k]] for k in
                         ('better_observation_id', 'worse_observation_id'))
        if not self.feasible(better):
            raise ValueError('Better observation must be complete and feasible')
        if self.feasible(worse):
            value = worse['objective_value']
        elif (worse['status'] == 'censored' and self.policy['objective'] == 'wall_time_s'
              and self.policy['direction'] == 'minimize'):
            value = worse['censoring']['lower_bound_s']
        else:
            raise ValueError('Worse observation needs a feasible measurement or comparable censored lower bound')
        if (self.policy['objective'] == 'wall_time_s' and
                abs(value - better['objective_value']) <= self.timing_band(better, worse)['absolute_threshold_s']):
            raise ValueError('Timing contrast is within the inclusive noise band')
        if not self.strictly_better(better['objective_value'], value):
            raise ValueError('Comparison does not establish the claimed loss')
        a, b = better['configuration'], worse['configuration']
        changed = {k for k in MECHANISMS if a.get(k) != b.get(k)}
        signature = set(changed)
        # Only a changed triggering mechanism can account for a forced companion.
        for relation in self.interactions['relations']:
            when = relation.get('when', {})
            if (relation['type'] == 'hard_link' and when and set(when) <= changed
                    and any(all(c.get(k) in values for k, values in when.items())
                            and all(c.get(k) in values for k, values in relation.get('requires', {}).items())
                            for c in (a, b))):
                signature -= set(relation.get('requires', {})) - set(when)
        if not signature or any(any(self.catalog[i]['configuration'].get(k) != b.get(k)
                                           for k in signature) for i in members):
            raise ValueError('Exclusion must stay within the observed losing mechanism branch; narrow its predicate')
        for observation in self.state['observations'].values():
            if observation.get('combination_id') not in members or not self.feasible(observation):
                continue
            competitive = not self.strictly_better(better['objective_value'], observation['objective_value'])
            if self.policy['objective'] == 'wall_time_s':
                competitive |= abs(observation['objective_value'] - better['objective_value']) <= self.timing_band(better, observation)['absolute_threshold_s']
            if competitive:
                raise ValueError('Exclusion contradicts a competitive measured member')
        return {k: [b[k]] for k in sorted(signature)}

    def exclusions(self, rules, *, initial=False, scope_ids=None):
        decisions = []
        for rule in rules:
            self.refs(rule['evidence'])
            members = self.members(rule['predicate'], allow_already_deleted=not initial)
            if not members:
                continue  # Repeating a historical deletion cannot change the surviving space.
            if scope_ids is not None:
                members &= set(scope_ids)
                if not members:
                    continue  # A valid global predicate outside the performance scope has no effect.
            if initial:
                if not rule['evidence']['fact_refs'] or not rule['evidence']['knowledge_refs']:
                    raise ValueError('Feature exclusion needs workload facts and mechanism knowledge')
                if any(o.get('combination_id') in members and self.feasible(o)
                       for o in self.state['observations'].values()):
                    raise ValueError('Feature-only exclusion cannot discard the measured reference')
                signature = None
            else:
                signature = self.supported_loss(rule, members)
            decisions.append(rule | {'action': 'drop', 'criterion': 'performance',
                                      'combination_ids': sorted(members), 'loss_signature': signature,
                                      'basis': 'feature_heuristic' if initial else 'measured_scoped_heuristic'})
        if set().union(*(set(d['combination_ids']) for d in decisions)) >= self.remaining():
            raise ValueError('Cannot exclude the entire remaining space')
        return decisions

    def performance_review(self, outcome, pending, assessment, stem):
        point_drops = {i for r in assessment['decisions'] for i in r['combination_ids'] if r['action'] == 'drop'}
        context = self.context()
        # Quality exclusion stays a separate program decision and is never broadened here.
        context['remaining_catalog']['rows'] = [r for r in context['remaining_catalog']['rows'] if r[0] not in point_drops]
        context.update(performance_outcome=context['observations'][outcome['observation_id']],
                       performance_control=context['observations'][pending['design']['control_observation_id']],
                       selected_question=pending['decision'])
        scope_ids = {r[0] for r in context['remaining_catalog']['rows']}

        control = self.state['observations'][pending['design']['control_observation_id']]
        if self.policy['objective'] == 'wall_time_s':
            context['selected_comparison_timing_band'] = self.timing_band(control, outcome)

        def validate(raw):
            self.validate_timing_signal(raw, outcome, control)
            decisions = self.exclusions(raw['exclusions'], scope_ids=scope_ids)
            dropping = {i for d in decisions for i in d['combination_ids']}
            if point_drops | dropping >= self.remaining():
                raise ValueError('Assessment cannot exclude the entire remaining space')
            return decisions

        raw = Workflow.call(self, 'performance_simple_assess', context, validate,
                            stem + '_assessment')
        decisions = validate(raw)
        assessment['decisions'].extend(decisions)
        assessment.update(signal=raw['signal'], summary=raw['summary'],
                          questions=raw['questions'], continue_screening=raw['continue_screening'],
                          question_review={'resolution': 'partial', 'learned': raw['summary'],
                                           'remaining_gap': '; '.join(raw['questions'])},
                          performance_assessment=raw,
                          ignored_already_deleted_exclusions=[r for r in raw['exclusions']
                              if not self.members(r['predicate'], allow_already_deleted=True)],
                          newly_dropped=len(point_drops | {i for d in decisions for i in d['combination_ids']}))
        self.state['screening_questions'] = raw['questions']
        if not raw['continue_screening']:
            self.state['screening_stop_reason'] = raw['summary']
        return assessment

    def export(self, reason):
        result = super().export(reason)
        result.update(screening_questions=self.state.get('screening_questions', []),
                      screening_stop_reason=self.state.get('screening_stop_reason'),
                      scope='Workload-specific heuristic screening; deletions are final within this run.')
        write(self.output / 'screening_result.json', result)
        return result
