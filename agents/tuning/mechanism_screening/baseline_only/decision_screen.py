"""Decision-first screening: explicit regional questions and auditable acquisition.

Partial-combination questions are compared by decision value, without invented probabilities.
Quality implications and contextual performance judgments are recorded separately.
"""
from copy import deepcopy

from .combination_screen import Workflow, load
from .combination_space import digest
from .combination_prompts import MECHANISMS, SCHEMAS, TASKS, obj, arr, TEXT, EVIDENCE, RULE
from .deletion_scope import QUALITY_PREMISE as PREMISE_ID, QUALITY_CLAIM, quality_scope
from .io import write
from .benefit_paths import BENEFIT_PATHS
from .decision_questions import QuestionWorkflow, register_prompts

ALGORITHM = 'decision-first.v12'
OPTIMIZATION_REVIEW = obj(considered_axes=arr(TEXT),
    surviving_reversal={'anyOf': [TEXT, {'type': 'null'}]}, reasoning=TEXT)
PERFORMANCE_RULE = obj(**RULE['properties'], optimization_review=OPTIMIZATION_REVIEW)
INITIAL_RULE = obj(**{**RULE['properties'], 'action': {'enum': ['drop']},
    'criterion': {'enum': ['performance']},
    'benefit_limitation_evidence': TEXT, 'cost_evidence': TEXT, 'scope_justification': TEXT,
    'optimization_review': OPTIMIZATION_REVIEW})
SCHEMAS['decision_initial'] = obj(rules=arr(INITIAL_RULE), summary=TEXT)
TASKS['decision_initial'] = '''# Task: evidence-supported zero-new-run regional screening
You MAY delete legal combinations using sufficiently supported workload/context knowledge.
Emit conditional drop rules only; all unmentioned combinations remain retained for active
decision-first exploration. Do not assign heuristic deprioritization to unexplored branches.
For each rule explain: benefit_limitation_evidence (observed workload facts restricting
benefit), cost_evidence (actual measured cost or an applicable, sourced mechanism cost),
and scope_justification (why the argument holds for every member and where exceptions lie).
Cite evidence in the rule's fact_refs and knowledge_refs. UNKNOWN cost is not a positive
cost measurement; it cannot alone support exclusion. Low reuse alone does not establish
net harm. Baseline success does not establish that an alternative branch is inferior.
Explicitly test the strongest counterexample against the proposed scope and preserve
unresolved exceptions by narrowing the predicate or omitting the rule. The mere presence
of an unmeasured variant is not a mandatory exception when the same supported argument
applies. Rules based on knowledge are recorded as knowledge_prior, not measured dominance.
Example: for an off branch with evidenced negligible useful prefix reuse and an applicable
cache overhead, a scoped prefix-on exclusion can be justified; do not transfer this argument
to CacheBlend's non-prefix segment reuse. High static segment reuse is a concrete potential
benefit; unknown blend/eager/transfer overhead is insufficient to rule out that branch.
Use criterion=performance. Prefix/chunked predicates must explicitly scope cacheblend_mode.
Precision variants require the same contextual benefit/cost assessment as other mechanisms.
Empty rules are valid, but do not avoid well-supported deletions.
'''
SCHEMAS['initial_audit'] = obj(reviews=arr(obj(rule_id=TEXT,
    verdict={'enum': ['supported', 'defer']}, reason=TEXT, evidence=EVIDENCE)), summary=TEXT)
TASKS['initial_audit'] = '''# Task: independent counterexample audit of proposed initial deletions
Review every proposed_initial_rule exactly once. These are hypotheses, NOT established
facts. A supported verdict means the cited workload facts and applicable mechanism knowledge
justify the stated entire scope despite its strongest counterexample. Otherwise defer;
deferred rules do not delete or deprioritize anything and can be revisited next round.
Do not demand an ablation experiment for every knowledge-backed exclusion. A grounded
absence of usable benefit plus an applicable known cost can be sufficient without a run.
However distinguish absence of evidence from evidence of absence: unmeasured acceptance,
unknown overhead, short output alone, or a successful baseline are not enough to establish
net disadvantage. A generic alpha>cost condition is not evidence that alpha<=cost here.
Low occupancy is not proof of absent bandwidth benefit or absent speculative headroom.
Unknown costs cannot negate a high, explicitly supported reuse opportunity. Check all
precision/backend variants actually covered; preserve a possible supported counterexample.
Return supported only if you can explain why the counterexample does not defeat the scope.
No replacement rules, invented facts or probabilities. Cite exact fact and knowledge IDs.
'''
QUESTION_REVIEW = obj(resolution={'enum': ['answered', 'partial', 'unresolved']}, learned=TEXT, remaining_gap=TEXT)
SCHEMAS['regional_performance'] = obj(decisions=arr(PERFORMANCE_RULE), interpretation=TEXT,
    question_review={'anyOf': [QUESTION_REVIEW, {'type': 'null'}]},
    optimization_opportunities=arr(obj(combination_id=TEXT, parameters=arr(TEXT, 1),
        reason=TEXT, causal_path=TEXT, applicability=TEXT, relative_advantage=TEXT,
        magnitude_assessment=TEXT, evidence=EVIDENCE)), limitations=arr(TEXT))
TASKS['regional_performance'] = """# Task: contextual assessment of which combinations to delete
Use the current run, ALL relevant historical measurements, workload facts and mechanism
knowledge to autonomously decide which remaining combinations are worth retaining.
Review measured losers AND unmeasured ranges, including branches outside selected_ids.
For example, a quality-passing CacheBlend-on win, substantial segment reuse and weak
native prefix reuse may support deleting SOME CacheBlend-off combinations. Explain the
shared disadvantage and narrow the predicate around meaningful exceptions; neither an
ablation for every linked switch nor a run of every covered combination is required.
Do not automatically keep all unmeasured variants or automatically delete an entire off
branch. Assess the measured gap and applicable knowledge together, including quality.
Compare with relevant historical feasible controls, not only baseline: a candidate can
beat baseline but lose to an earlier CacheBlend result. Cite the current outcome, its
matched control, and any historical winner used to justify deletion. Never delete a
measured feasible member at least as good as the cited winning witness.
An unmeasured acceptance rate is unknown, not a positive reason to retain and not a
reason to disregard observed slowdown. Short outputs alone are not a proof of harm,
but may strengthen an actual slowdown argument. Untried internal knobs or generic
batch tuning do not establish a mechanism-specific route to reversing the gap.
Formal bounds and exhaustive tuning are not required. For each exclusion explain scope,
applicable costs/benefits, meaningful counterarguments, untested and reopening conditions.
When selected_question is supplied, question_review must be an object: explain what the
actual run answered relative to its decision_value and performance outcome hypotheses, and what
remains unknown. Do not repeat the full assessment; keep only new learning and remaining gaps.
Without selected_question, question_review may be null. This review creates no extra deletions.
Use criterion=performance; quality propagation is handled separately. Use remaining_catalog
and assessment_scope_ids. Empty decisions are valid when the combined evidence warrants
retention, but explain why the strongest proposed deletion is not justified.
"""
CONTINUATION_CONTRACT = '''
# Downstream optimization contract
The seven mechanisms identify families to hand off for later optimization, NOT their
best achievable performance. Read downstream_optimization, including axes OUTSIDE those
seven: DP/TP topology, scheduling, concurrency, batch/token budgets, memory allocation
and other explicitly declared directions. A slower measurement at frozen TP/DP is NOT
grounds for automatically deleting the mechanism family, even the family represented
by that exact measured point. Quantization may trade kernel overhead for memory headroom
that enables a different legal DP/TP layout or larger batch; evaluate this possibility
without claiming an unmeasured speedup as fact. Unknown/unset downstream ranges cannot
be treated as evidence that the corresponding optimization direction is unavailable.
For every performance drop, optimization_review should identify relevant axes and explain
which directions might reverse the gap. A possible reversal is NOT a mandatory veto:
weigh its plausibility, magnitude and supporting evidence against the measured gap and
workload knowledge, and autonomously choose retain, drop or a narrower scope. You need not
exhaust every axis, demand formal bounds, or run every potential alternative. If dropping
despite a possible reversal, state why that possibility is insufficient to retain the scope.
surviving_reversal records a remaining possibility, NOT an automatic keep instruction.
For regional_performance, list concrete surviving directions in optimization_opportunities
using existing combination IDs and meaningful parameter names. Additional axes may be
proposed from mechanism knowledge even if a downstream numeric range is not yet supplied.
Read the registered screening.benefit.* knowledge paths. They are conditional priors,
not measured gains, blanket precision protection or mandatory retention instructions.
For each retained optimization_opportunity give causal_path, applicability,
relative_advantage and magnitude_assessment. Explain the resource/kernel change, its
workload prerequisites, why it could improve RELATIVE to the comparator (which may also
be tuned), and whether it plausibly matters against the observed gap. A qualitative
assessment is enough; do not invent speedups. Missing measurements alone, a list of knobs,
or generic "larger batch may help" are insufficient positive retention reasons.
Ngram has no automatic capacity/topology benefit in this prior. Internal lookup/draft
length tuning may be relevant only with contextual support for a relative benefit;
do not force either retention or deletion solely from the mechanism name.
These are future experiments, not observed gains. Empty decisions remain valid when
supported by the overall evidence, without inventing an optimization opportunity.
'''
SCOPE_STANDARD = '''
# Scope reasoning standard (decision-first.v9)
Choose deletion scopes autonomously; the following is a reasoning standard, NOT a
whitelist or a requirement to measure every combination. Start with the smallest
supported scope and assess which unmeasured parameter changes could reverse the argument.
For example, a CacheBlend-on win can motivate reviewing corresponding off + prefix=false
combinations with matching quantization/dtype/KV/speculation, while respecting mechanical
linkages. It does NOT by itself resolve all off combinations, prefix=true alternatives,
or different precision/speculation tuples. You MAY delete those or any other legal points
when workload, knowledge and observations support their lack of downstream potential;
explain why the variations inside EACH proposed predicate do not overturn the conclusion.
Parameters both branches can tune need not have equal feasible domains or equal gains.
Do not label linked eager/prefix/chunked changes as measured costs without evidence.
Two individually beneficial mechanisms do not imply their untested conjunction is harmful;
a stronger deletion claim needs an interaction argument, not merely unequal single gains
or absence of a joint measurement. Keep semantic claims consistent with predicate values.
Do not require exhaustive tuning, formal proof, or automatic retention of all untested points.
'''
QUALITY_STANDARD = '''
# Post-run quality boundary
Quality is a HARD boundary: fix the full measured precision triple INCLUDING defaults,
plus CacheBlend when on and speculation when enabled (exact method). Use the computed
closure; never extrapolate a quality failure to a different precision triple.
'''
register_prompts(CONTINUATION_CONTRACT + SCOPE_STANDARD, QUALITY_STANDARD)

class DecisionWorkflow(QuestionWorkflow, Workflow):
    algorithm = ALGORITHM
    acquisition_method = 'direct_two_criterion_decision'
    def context(self, *, include_quality=False, **extra):
        return super().context(include_quality=include_quality, **extra) | {'downstream_optimization': self.policy.get('downstream_optimization', {
            'parameters': {}, 'status': 'not declared; do not equate the fixed execution point with a tuned optimum'}),
            'screening_target': 'mechanism combinations including their remaining parameter-optimization potential'}

    def assessment_scope(self, selected, outcome):
        if outcome['status'] == 'complete' and self.quality(outcome)['status'] in ('passed', 'disabled'):
            return self.remaining()
        return super().assessment_scope(selected, outcome)

    def performance_control(self, rule, outcome):
        # Keep the original pair auditable, but allow a stronger historical comparator.
        matched = self.state['pending']['design']['control_observation_id']
        refs = rule['evidence']['observation_refs']
        if matched not in refs:
            raise ValueError('Drop must cite the matched control as well as treatment')
        feasible = [self.state['observations'][key] for key in refs
                    if key != outcome['observation_id']
                    and self.state['observations'][key]['status'] == 'complete'
                    and self.quality(self.state['observations'][key])['status'] in ('passed', 'disabled')]
        if not feasible:
            return super().performance_control(rule, outcome)
        return min(feasible, key=lambda o: o['objective_value'] *
                   (1 if self.policy['direction'] == 'minimize' else -1))

    def scope_contract(self):
        frozen = {k: v for k, v in self.reference.items() if k not in MECHANISMS
                  and all(c['configuration'].get(k) == v for c in self.catalog.values())}
        return {'kind': 'enumerated_full_configurations_only', 'frozen_background': frozen,
                'catalog_configuration_digest': digest({k: c['configuration_hash'] for k, c in self.catalog.items()}),
                'transfer_to_other_backgrounds': False,
                'quality_deletion_boundary': 'exact precision tuple plus enabled quality mechanisms',
                'performance_scope_policy': 'LLM judgment with narrow-scope reasoning standard; no whitelist',
                'rule_predicates_are_not_global_parameter_bans': True,
                'downstream_optimization': self.policy.get('downstream_optimization', {}),
                'measured_point_is_not_optimized_mechanism_family': True,
                'mechanically_linked_fields': 'Use the actual full configuration attached to each combination_id',
                'handoff': 'The full configurations describe measured representatives. Judge mechanism-family '
                           'decisions together with downstream_optimization and optimization_review; never '
                           'transfer the measured runtime ordering itself to another DP/TP/batch setting.'}

    def rules(self, rules, scope, **kwargs):
        for rule in rules:
            if rule['action'] == 'drop' and rule.get('criterion', 'performance') == 'performance':
                review = rule.get('optimization_review')
                if not review or not review.get('reasoning'):
                    raise ValueError('Performance deletion must explain the downstream optimization tradeoff')
        result = super().rules(rules, scope, ignore_already_deleted=True, **kwargs)
        return [r | {'scope_contract': self.scope_contract()} for r in result]

    def __init__(self, **kwargs):
        kwargs['evidence'] = deepcopy(kwargs['evidence'])
        kwargs['evidence']['knowledge'].update(deepcopy(BENEFIT_PATHS))
        kwargs['evidence']['knowledge'][PREMISE_ID] = {'claim': QUALITY_CLAIM,
            'source': 'User-specified exact-tuple quality deletion boundary'}
        kwargs['policy'] = deepcopy(kwargs['policy']) | {
            'contextual_precision_screening': True,
            'protected_dimensions': [],
            'algorithm': self.algorithm,
            'quality_scope': 'exact_precision_tuple',
            'quality_stage': 'post_run_assessment_only',
            'acquisition': {'method': self.acquisition_method,
                'criteria': ['eliminate_clearly_bad_or_find_clearly_good', 'represent_shared_behavior'],
                'new_runs_per_decision': 1},
        }
        super().__init__(**kwargs)

    def quality_failure_closure(self, observation):
        identifier = observation.get('combination_id')
        if identifier not in self.catalog or self.quality(observation)['status'] != 'failed':
            return None
        predicate, members = quality_scope(self.catalog[identifier]['mechanisms'], self.catalog)
        conflicts = sorted(o['observation_id'] for o in self.state['observations'].values()
                           if o.get('combination_id') in members and self.quality(o)['status'] == 'passed')
        return {'predicate': predicate, 'combination_ids': sorted(members & self.remaining()),
                'premise_ref': PREMISE_ID, 'conflicting_observation_ids': conflicts,
                'basis': 'user_assumed_exact_precision_tuple', 'applicable': not conflicts}

    def validate_measured_drop(self, rule, members, outcome):
        basis = super().validate_measured_drop(rule, members, outcome)
        if rule.get('criterion', 'performance') == 'quality_failure':
            closure = self.quality_failure_closure(outcome)
            allowed = set(closure['combination_ids']) if closure and closure['applicable'] else {outcome['combination_id']}
            if not members <= allowed:
                raise ValueError('Quality deletion exceeds exact precision/enabled-mechanism scope: ' + str(sorted(allowed)))
            if members - {outcome['combination_id']}:
                if PREMISE_ID not in rule['evidence']['knowledge_refs']:
                    raise ValueError('Quality propagation must cite ' + PREMISE_ID)
                basis = 'user_assumed_exact_precision_tuple'
        return basis

    def validate_audit(self, raw, proposed):
        ids = [r['rule_id'] for r in raw['reviews']]
        if len(ids) != len(set(ids)) or set(ids) != {r['rule_id'] for r in proposed}:
            raise ValueError('Review every proposed initial rule exactly once')
        for review in raw['reviews']:
            self.refs(review['evidence'])
            if review['verdict'] == 'supported' and (not review['evidence']['fact_refs']
                                                     or not review['evidence']['knowledge_refs']):
                raise ValueError('Supported initial deletion requires workload and knowledge references')

    def initial_pass(self, stem):
        initial = self.call('decision_initial', self.context(),
            lambda r: self.rules(r['rules'], self.remaining(), initial=True), stem + '_initial')
        proposed = [{'rule_id': 'r_' + digest(r)[:12], 'rule': r} for r in initial['rules']]
        audit = {'reviews': [], 'summary': 'No proposed initial exclusions'}
        if proposed:
            audit = self.call('initial_audit', self.context(proposed_initial_rules=proposed),
                lambda r: self.validate_audit(r, proposed), stem + '_initial_audit')
        accepted = {r['rule_id'] for r in audit['reviews'] if r['verdict'] == 'supported'}
        rules = self.rules([p['rule'] for p in proposed if p['rule_id'] in accepted], self.remaining(), initial=True)
        self.apply(rules)
        self.state['history'].append({'stage': 'initial', 'round': self.state['round'], 'rules': rules,
            'proposed_rules': proposed, 'audit': audit, 'summary': initial['summary'],
            'source': 'workload_context_and_mechanism_knowledge'})
        self.state['initial_completed_round'] = self.state['round']
        self.save()

    def assess(self, outcome, pending):
        quality = self.quality(outcome)['status']
        decisions = []
        unresolved = ['Performance ranges await contextual assessment; unmeasured members may be excluded using measured gaps and applicable knowledge.']
        outcome_class = 'inconclusive'
        if outcome['status'] == 'complete' and quality == 'failed':
            outcome_class = 'quality_failed'
            closure = self.quality_failure_closure(outcome)
            ids = [outcome['combination_id']]
            basis = 'measured_quality_failure'
            if closure and closure['applicable']:
                ids = closure['combination_ids']
                basis = 'user_assumed_exact_precision_tuple'
            elif closure:
                unresolved.append('Exact-tuple quality prior conflicts with measured passing evidence; no range propagation.')
            passing_ids = {o.get('combination_id') for o in self.state['observations'].values()
                           if self.quality(o)['status'] == 'passed'}
            ids = sorted(set(ids) - passing_ids)
            decisions.append({'action': 'drop', 'criterion': 'quality_failure', 'combination_ids': ids,
                'basis': basis, 'observation_refs': [outcome['observation_id']],
                'knowledge_refs': [PREMISE_ID] if basis == 'user_assumed_exact_precision_tuple' else [],
                'reason': '完整质量失败；固定实测精度三元组及已开启机制，不外推到其他精度组合。',
                'reopen_if': '质量知识适用范围改变或出现合格实测反例时重新评估。'})
        elif outcome['status'] == 'complete' and quality in ('passed', 'disabled'):
            control_id = pending.get('design', {}).get('control_observation_id', 'baseline')
            control = self.state['observations'][control_id]
            outcome_class = ('feasible_win' if self.strictly_better(outcome['objective_value'], control['objective_value'])
                             else 'feasible_not_win')
            feasible = [o for o in self.state['observations'].values()
                        if o['status'] == 'complete' and self.quality(o)['status'] in ('passed', 'disabled')]
            winner = min(feasible, key=lambda o: o['objective_value'] * (1 if self.policy['direction'] == 'minimize' else -1))
            unresolved.append(f'Measured best is {winner["observation_id"]}; slower fixed-background points '
                              'remain eligible until downstream optimization potential has been assessed.')
        for rule in decisions:
            rule['combination_ids'] = sorted(set(rule['combination_ids']) & self.remaining())
            rule['scope_contract'] = self.scope_contract()
        decisions = [r for r in decisions if r['combination_ids']]
        dropping = {i for r in decisions for i in r['combination_ids']}
        if dropping >= self.remaining():
            raise ValueError('Decision propagation cannot eliminate the entire legal space')
        return {'outcome_class': outcome_class, 'decisions': decisions, 'unresolved': unresolved,
                'quality': self.quality(outcome), 'newly_dropped': len(dropping)}

    def performance_review(self, outcome, pending, assessment, stem):
        if outcome['status'] not in ('complete', 'censored'):
            assessment['question_review'] = {'resolution': 'unresolved',
                'learned': '运行未完整完成，没有可用结论。', 'remaining_gap': '原性能问题尚未解决。'}
            return assessment
        if outcome['status'] == 'complete' and self.quality(outcome)['status'] not in ('passed', 'disabled'):
            assessment['question_review'] = {'resolution': 'unresolved',
                'learned': '仅在最终评估使用质量结果；支持的质量删除已由程序记录。',
                'remaining_gap': '性能候选问题不据此视为已解决。'}
            return assessment
        point_drops = {k for r in assessment['decisions'] if r['action'] == 'drop' for k in r['combination_ids']}
        def validate(raw):
            if pending.get('decision') and raw.get('question_review') is None:
                raise ValueError('Assess must explain what the selected performance question learned')
            if outcome['status'] == 'censored' and raw.get('question_review') and raw['question_review']['resolution'] == 'answered':
                raise ValueError('Censored runtime only partially answers a performance question')
            if any(r['criterion'] != 'performance' for r in raw['decisions']):
                raise ValueError('Quality propagation is already handled separately; performance criteria only')
            for opportunity in raw['optimization_opportunities']:
                if opportunity['combination_id'] not in self.catalog:
                    raise ValueError('Unknown combination in downstream optimization opportunity')
                self.refs(opportunity['evidence'])
            reviewed = point_drops | {self.state['observations'][pending['design']['control_observation_id']]['combination_id']}
            rules = self.rules(raw['decisions'], set(pending['selected_ids']), outcome=outcome, reviewed_ids=reviewed)
            if any(r['action'] != 'drop' and set(r['combination_ids']) & point_drops for r in rules):
                raise ValueError('Regional retention cannot undo an already supported measured-point deletion')
            return rules
        raw = self.call('regional_performance', self.context(include_quality=True,
            performance_outcome=self.context(include_quality=True)['observations'][outcome['observation_id']],
            performance_control=self.context(include_quality=True)['observations'][pending['design']['control_observation_id']],
            point_assessment=assessment, selected_question=pending.get('decision'),
            selected_ids=pending['selected_ids'],
            assessment_scope_ids=sorted(self.assessment_scope(pending['selected_ids'], outcome))),
            validate, stem + '_regional_performance')
        rules = validate(raw)
        assessment['decisions'].extend(rules)
        assessment['ignored_already_deleted_rules'] = [r for r in raw['decisions']
            if r['action'] == 'drop' and not self.members(r['predicate'], allow_already_deleted=True)]
        assessment['regional_interpretation'] = raw['interpretation']
        assessment['question_review'] = raw.get('question_review')
        assessment['optimization_opportunities'] = raw['optimization_opportunities']
        assessment['unresolved'] = raw['limitations']
        assessment['newly_dropped'] = len({k for r in assessment['decisions'] if r['action'] == 'drop' for k in r['combination_ids']})
        return assessment

    def export(self, reason):
        result = super().export(reason)
        result['algorithm'] = self.algorithm
        result['scope_contract'] = self.scope_contract()
        result['acquisition_policy'] = self.policy['acquisition']
        result['decision_records'] = self.state.get('decision_records', [])
        result['scope'] = 'Prior-knowledge quality propagation and autonomous contextual performance screening '
        result['scope'] += 'including downstream optimization potential. Slower measured points do not automatically trigger deletion. '
        result['scope'] += 'Untested improvements are opportunities, not measurements. No safety theorem claimed.'
        write(self.output / 'screening_result.json', result)
        return result

    def _run(self):
        if self.state['status'] == 'complete':
            return load(self.output / 'screening_result.json')
        if self.state['status'] == 'initial':
            self.state['status'] = 'iterating'
            self.save()
        # Even a zero-execution budget permits an initial knowledge screening pass.
        if self.state.get('initial_completed_round') != self.state['round']:
            self.initial_pass(f"round_{self.state['round']:03d}")
        while self.state['round'] < self.policy['max_rounds']:
            stem = f"round_{self.state['round']:03d}"
            if 'pending' not in self.state:
                if self.state.get('initial_completed_round') != self.state['round']:
                    self.initial_pass(stem)
                if self.state['attempts'] >= self.policy['max_runs']:
                    return self.export('additional_run_budget_exhausted')
                if not self.experiment_options():
                    return self.export('no_runnable_unattempted_representatives')
                chosen = self.propose_next_decision(stem)
                write(self.output / f'{stem}_decision.json', self.state['decision_plan'])
                if chosen is None:
                    self.state.setdefault('decision_records', []).append(deepcopy(self.state['decision_plan']))
                    return self.export('no_worthwhile_experiment')
                design = dict(action='run', **chosen['comparison'], reuse_observation_id=None,
                              evidence=chosen['evidence'], reason=chosen['question'])
                selected = chosen['performance_target_ids']
                self.validate_design(design, set(selected))
                self.state['pending'] = {'decision': chosen, 'selected_ids': selected, 'design': design}
                self.save()
            pending = self.state['pending']
            if 'observation_id' in pending:
                outcome = self.state['observations'][pending['observation_id']]
                if outcome['status'] == 'started':
                    outcome.update(status='interrupted', reason='Interrupted attempt retained; never silently rerun')
                    self.save()
            else:
                outcome = self.execute(pending['design'])
                if outcome is None:
                    write(self.output / 'pending_experiment.json', pending)
                    return {'status': 'awaiting_execution', 'additional_run_attempts': self.state['attempts']}
            assessment = self.assess(outcome, pending)
            assessment = self.performance_review(outcome, pending, assessment, stem)
            write(self.output / f'{stem}_assessment.json', assessment)
            self.apply(assessment['decisions'])
            self.state['history'].append({'stage': 'experiment', **pending, **assessment,
                'decision_id': pending['decision']['decision_id'],
                'decision_predicate': pending['decision']['predicate'],
                'observation_id': outcome['observation_id'],
                'assessed_combination_ids': sorted({i for r in assessment['decisions'] for i in r['combination_ids']}
                                                 | {outcome['combination_id']})})
            self.state.setdefault('decision_records', []).append(deepcopy(self.state['decision_plan']))
            self.state.pop('decision_plan')
            self.state.pop('pending')
            self.state['round'] += 1
            self.save()
        if self.state.get('initial_completed_round') != self.state['round']:
            self.initial_pass(f"round_{self.state['round']:03d}")
        return self.export('screening_round_budget_exhausted')
