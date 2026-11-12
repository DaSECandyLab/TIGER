"""Mechanism-first screening with conditional interaction knowledge and tuning handoff."""
from copy import deepcopy
import math

from .combination_screen import Workflow, load, performance_payload
from .decision_screen import DecisionWorkflow, INITIAL_RULE
from .decision_questions import QUESTION, BRANCH
from .combination_prompts import SCHEMAS, TASKS, MECHANISMS, EVIDENCE, TEXT, PREDICATE, obj, arr
from .combination_space import digest, matches
from .io import ROOT, write
from .interaction_prompts import register_interaction_prompts

DEFAULT_INTERACTIONS = ROOT / 'my_storage/static_knowledge_base/vllm-0.20.0/mechanism_interactions.json'
REVIEW = obj(relation_id=TEXT, activation={'enum': ['active', 'weak', 'unknown']},
             reasoning=TEXT, evidence=EVIDENCE)
CONFIDENCE_FIELDS = {'deletion_confidence': {'type':'number','minimum':0,'maximum':1},
                     'confidence_reason':TEXT}
CONFIDENT_INITIAL_RULE = obj(**{k:v for k,v in INITIAL_RULE['properties'].items() if k!='reopen_if'},
                             next_evidence=TEXT, **CONFIDENCE_FIELDS)
SIGNAL = obj(predicate=PREDICATE, judgment={'enum': ['promising', 'unfavorable', 'unresolved']},
             reasoning=TEXT, evidence=EVIDENCE, **CONFIDENCE_FIELDS)
ANALYSIS = obj(mechanism={'enum': list(MECHANISMS)}, applicability=TEXT,
               interaction_reviews=arr(REVIEW), scopes=arr(SIGNAL), open_question=TEXT)
SCHEMAS['performance_interaction_initial'] = obj(mechanism_analyses=arr(ANALYSIS, 7) | {'maxItems': 7},
                                    rules=arr(CONFIDENT_INITIAL_RULE), summary=TEXT)
SCHEMAS['performance_interaction_initial_audit'] = obj(reviews=arr(obj(rule_id=TEXT,
    verdict={'enum':['supported','defer']}, reason=TEXT, evidence=EVIDENCE,
    **CONFIDENCE_FIELDS)), summary=TEXT)
question = deepcopy(QUESTION)
question['properties']['outcomes'] = obj(good=BRANCH, bad=BRANCH)
question['properties']['selection_reason'] = obj(clear_separation=TEXT, regional_conclusion=TEXT,
                                                tuning_relevance=TEXT, alternative_question=TEXT)
question['required'].append('selection_reason')
SCHEMAS['performance_interaction_decision'] = obj(action={'enum': ['run', 'stop']},
    decision={'anyOf': [question, {'type': 'null'}]}, stop_reason={'anyOf': [TEXT, {'type': 'null'}]})
SCHEMAS['performance_interaction_stop_audit'] = obj(
    verdict={'enum':['stop_supported','continue_screening']},
    reviews=arr(obj(mechanism={'enum':list(MECHANISMS)}, screening_question=TEXT,
        good_result_effect=TEXT, bad_result_effect=TEXT,
        can_change_screening={'type':'boolean'}, interpretable_test_exists={'type':'boolean'},
        reasoning=TEXT, evidence=EVIDENCE)), summary=TEXT)
SCHEMAS['performance_interaction_assess'] = deepcopy(SCHEMAS['regional_performance'])
SCHEMAS['performance_interaction_assess']['properties'].update(promising_scopes=arr(SIGNAL), interaction_reviews=arr(REVIEW))
SCHEMAS['performance_interaction_assess']['required'] += ['promising_scopes', 'interaction_reviews']
REGION_REVIEW = obj(predicate=PREDICATE, verdict={'enum': ['drop', 'promising', 'unresolved']},
    gap_explanation=TEXT, interaction_exception=TEXT, relative_reversal_path=TEXT,
    magnitude_assessment=TEXT, evidence=EVIDENCE,
    joint_interaction_review=obj(relation_ids=arr(TEXT), combined_effect=TEXT,
                               reversal_assessment=TEXT))
SCHEMAS['performance_interaction_assess']['properties']['region_reviews'] = arr(REGION_REVIEW)
SCHEMAS['performance_interaction_assess']['required'].append('region_reviews')


SCHEMAS['performance_propagation_signal'] = obj(
    mechanism={'enum': list(MECHANISMS)}, signal={'enum': ['good', 'bad', 'inconclusive']},
    attributable={'type': 'boolean'}, reasoning=TEXT, evidence=EVIDENCE)


SCHEMAS['performance_linked_groups'] = obj(groups=arr(obj(
    group_id=TEXT, verdict={'enum':['promising','drop','unresolved']},
    relation_ids=arr(TEXT), combined_effect=TEXT, reversal_assessment=TEXT,
    magnitude_assessment=TEXT, reason=TEXT, evidence=EVIDENCE,
    untested_conditions=arr(TEXT), next_evidence=TEXT, **CONFIDENCE_FIELDS)), summary=TEXT)

HISTORICAL_COMPARISON = obj(better_observation_id=TEXT, worse_observation_id=TEXT,
    loss_signature=PREDICATE, transfer_reason=TEXT)
SCHEMAS['performance_history_scope'] = obj(regions=arr(obj(region_id=TEXT,
    verdict={'enum':['promising','drop','unresolved']}, evidence=EVIDENCE,
    supporting_comparisons=arr(HISTORICAL_COMPARISON), historical_reasoning=TEXT,
    interaction_assessment=TEXT, tuning_assessment=TEXT,
    untested_conditions=arr(TEXT), next_evidence=TEXT, **CONFIDENCE_FIELDS)), summary=TEXT)

register_interaction_prompts(TASKS)


class InteractionWorkflow(DecisionWorkflow):
    algorithm = 'mechanism-interaction-screening.v4'
    acquisition_method = 'clear_regional_screening_question'

    def __init__(self, interaction_knowledge=None, **kwargs):
        self.interactions = deepcopy(interaction_knowledge if interaction_knowledge is not None
                                     else load(DEFAULT_INTERACTIONS))
        ids = [r['id'] for r in self.interactions['relations']]
        if len(ids) != len(set(ids)) or set(self.interactions['mechanisms']) != set(MECHANISMS):
            raise ValueError('Interaction catalog requires seven mechanisms and unique relations')
        kwargs['evidence'] = deepcopy(kwargs['evidence'])
        kwargs['evidence']['knowledge']['screening.feature_prescreen_policy'] = {
            'source':'explicit user screening preference',
            'claim':'Initial screening uses clear workload-feature signals as heuristic direction evidence. '
                    'When measured native prefix cache hits are high and reusable full prefix blocks '
                    'show substantial useful reuse, favor prefix caching and propose '
                    'excluding prefix-off with CacheBlend-off unless supplied evidence identifies a concrete '
                    'active exception. Do not require on/off timing or speculative bookkeeping-cost proof. '
                    'Nonzero hits, full prefix blocks, or prefill-dominated execution alone do not '
                    'establish substantial reuse. Low hit rates do not activate this exclusion prior; '
                    'retain the scope as unresolved unless independent evidence supports exclusion. '
                    'This is a feature-based screening prior, not a measured performance guarantee. '
                    'Keep the confidence threshold and isolate actual interaction exceptions.'}
        kwargs['evidence']['knowledge']['screening.interaction_catalog'] = deepcopy(self.interactions)
        for relation in self.interactions['relations']:
            kwargs['evidence']['knowledge'][relation['id']] = deepcopy(relation)
        kwargs['evidence']['knowledge']['screening.applicability_prior'] = {
            'source': 'user prior', 'claim': 'CacheBlend: long input, limited useful native prefix reuse, '
            'high theoretical segment reuse. Speculation: long output and meaningful decode share. '
            'These are conditional opportunities, not measured gains or automatic exclusions.'}
        kwargs['evidence']['knowledge']['screening.user_weak_interaction_propagation'] = {
            'source': 'explicit user instruction',
            'claim': 'A clearly attributable mechanism comparison propagates directly across noninteracting '
                     'dimensions while preserving each side-specific hard-link background. Hold obvious performance-interaction neighbor values at the measured '
                     'background. Propose losing-side exclusions and winning-side promising regions; '
                     'only changed interaction backgrounds require LLM exception assessment. Generic future '
                     'tuning is not a veto. All proposals require final all-history scope review and deletion '
                     'confidence gating. Deleted regions are not reopened. This is a screening approximation, '
                     'not proven independence.'}
        threshold = kwargs['policy'].get('deletion_confidence_threshold',.9)
        self.validate_confidence_number(threshold)
        kwargs['policy'] = deepcopy(kwargs['policy']) | {'interaction_digest': digest(self.interactions),
            'deletion_confidence_threshold':threshold}
        super().__init__(**kwargs)
        self.policy['acquisition']['criteria'] = ['clear_good_bad_separation', 'regional_conclusion', 'downstream_tuning_relevance']

    @staticmethod
    def validate_confidence_number(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Deletion confidence must be a finite number between 0 and 1')

    def validate_confidence(self, review):
        self.validate_confidence_number(review.get('deletion_confidence'))
        if not isinstance(review.get('confidence_reason'),str) or not review['confidence_reason'].strip():
            raise ValueError('Deletion confidence requires an explanation for the entire scope')

    def effective_verdict(self, review):
        self.validate_confidence(review)
        if review['verdict']=='drop' and review['deletion_confidence']<self.policy['deletion_confidence_threshold']:
            return 'unresolved'
        return review['verdict']

    def apply(self, rules):
        for rule in rules:
            if rule['action']!='drop' and any(self.state['states'][i]=='drop' for i in rule['combination_ids']):
                raise ValueError('Deleted scopes are final and cannot be restored during this screening run')
            if rule['action']=='drop' and rule.get('criterion','performance')=='performance':
                self.validate_confidence(rule)
                if rule['deletion_confidence']<self.policy['deletion_confidence_threshold']:
                    raise ValueError('Cannot apply a performance deletion below the confidence threshold')
        return super().apply(rules)

    def remaining(self):
        return super().remaining() - getattr(self, '_propagation_excluded', set())

    def context(self, **kwargs):
        return super().context(**kwargs) | {'interaction_knowledge': self.interactions,
            'mechanism_screening': self.state.get('mechanism_screening', []),
            'promising_regions': self.state.get('promising_regions', []),
            'automatic_propagation': self.state.get('pending', {}).get('propagation', {})}

    def validate_reviews(self, reviews, mechanism=None, exhaustive=False):
        allowed = {r['id'] for r in self.interactions['relations'] if r['type'] in ('hard_link', 'performance_interaction')
                   and (mechanism is None or mechanism in r['mechanisms'])}
        supplied = [r['relation_id'] for r in reviews]
        if len(supplied) != len(set(supplied)) or not set(supplied) <= allowed:
            raise ValueError('Unknown, duplicate or unrelated interaction; allowed IDs: ' + str(sorted(allowed)))
        if exhaustive and set(supplied) != allowed:
            raise ValueError('Review every listed hard/performance interaction of ' + str(mechanism) + '; missing=' + str(sorted(allowed-set(supplied))))
        for r in reviews:
            self.refs(r['evidence'])

    def validate_signals(self, signals, mechanism=None):
        for s in signals:
            self.validate_confidence(s)
            if mechanism and mechanism not in s['predicate']:
                raise ValueError('Mechanism scope must include its focal mechanism')
            self.members(s['predicate'])
            self.refs(s['evidence'])
            if s['judgment'] != 'unresolved' and not s['evidence']['fact_refs']:
                raise ValueError('Scope needs workload evidence')
            if s['judgment'] == 'unfavorable' and not s['evidence']['knowledge_refs']:
                raise ValueError('Unfavorable scope needs mechanism evidence; keep unresolved when knowledge is missing')

    def validate_initial(self, raw):
        names = [a['mechanism'] for a in raw['mechanism_analyses']]
        if len(names) != len(set(names)) or set(names) != set(MECHANISMS):
            raise ValueError('Analyze all seven mechanisms exactly once')
        errors = []
        unfavorable = set()
        for a in raw['mechanism_analyses']:
            try:
                self.validate_reviews(a['interaction_reviews'], a['mechanism'], exhaustive=True)
            except ValueError as exc:
                errors.append(a['mechanism']+': '+str(exc))
            try:
                self.validate_signals(a['scopes'], a['mechanism'])
            except ValueError as exc:
                errors.append(a['mechanism']+': '+str(exc))
                continue
            memberships = [self.members(s['predicate']) for s in a['scopes']]
            covered = set().union(*memberships)
            if sum(map(len,memberships)) != len(covered):
                errors.append('Initial scopes must be disjoint within '+a['mechanism'])
            for scope, members in zip(a['scopes'], memberships):
                if scope['judgment']=='unfavorable':
                    unfavorable.update(members)
                # A forced companion value is not the same question as choosing
                # that value with the parent mechanism disabled.
                for relation in self.interactions['relations']:
                    when=relation.get('when',{})
                    if (relation['type']!='hard_link' or not when or
                        a['mechanism'] not in relation.get('requires',{})):
                        continue
                    activation={matches(self.catalog[i]['configuration'],when) for i in members}
                    if len(activation)>1:
                        errors.append('Split '+a['mechanism']+' scopes by '+relation['id']+
                                      ' activation: assess forced companion and independent background separately')
            if covered != self.remaining():
                missing = sorted(self.remaining() - covered)
                examples = [{k:self.catalog[c]['mechanisms'][k] for k in MECHANISMS}
                            for c in missing[:2]]
                errors.append('Initial scopes must cover all legal values/backgrounds of '+a['mechanism']+
                                 '; use broad unresolved scopes when uncertain, not per-configuration enumeration'+
                                 '; missing_count='+str(len(missing))+'; missing_examples='+str(examples))
        if errors:
            raise ValueError('\n'.join(errors))
        for rule in raw['rules']:
            self.validate_confidence(rule)
            if not self.members(rule['predicate']) <= unfavorable:
                raise ValueError('Initial exclusion must be covered by explicitly unfavorable conditional scopes')
        self.rules(raw['rules'], self.remaining(), initial=True)

    def call(self, stage, context, validator, stem):
        mapping = {'decision_initial': 'performance_interaction_initial', 'next_decision': 'performance_interaction_decision',
                   'regional_performance': 'performance_interaction_assess'}
        if stage not in mapping:
            return super().call(stage, context, validator, stem)
        context = self.performance_context(context)
        for key in ('performance_outcome', 'performance_control'):
            if key in context:
                context[key] = performance_payload(context[key])
        context.pop('point_assessment', None)
        if stage == 'regional_performance':
            context.pop('mechanism_screening', None)
        def validate(raw):
            # Recursively reject citations to evidence removed by the performance boundary.
            def refs(value):
                if isinstance(value, dict):
                    if 'fact_refs' in value and 'knowledge_refs' in value:
                        if not set(value['fact_refs']) <= context['evidence']['facts'].keys() or not set(value['knowledge_refs']) <= context['evidence']['knowledge'].keys():
                            raise ValueError('Citation unavailable in performance context: facts=' + str(sorted(set(value['fact_refs']) - context['evidence']['facts'].keys())) + '; knowledge=' + str(sorted(set(value['knowledge_refs']) - context['evidence']['knowledge'].keys())))
                    for v in value.values(): refs(v)
                elif isinstance(value, list):
                    for v in value: refs(v)
            refs(raw)
            if stage == 'decision_initial': self.validate_initial(raw)
            if stage == 'regional_performance':
                self.remove_redundant_propagation(raw)
                self.validate_regions(raw, context)
                self.validate_reviews(raw['interaction_reviews'])
                focal = self.state.get('pending', {}).get('propagation', {}).get('focal_mechanism')
                if focal:
                    required = {r['id'] for r in self.interactions['relations'] if focal in r['mechanisms']}
                    if not required <= {r['relation_id'] for r in raw['interaction_reviews']}:
                        raise ValueError('Assess BOTH hard linkages and performance interactions of focal mechanism: ' + str(sorted(required)))
                self.validate_signals(raw['promising_scopes'])
                for s in raw['promising_scopes']:
                    if s['judgment'] != 'promising' or context['performance_outcome']['status'] != 'complete':
                        raise ValueError('Promising signal requires a complete measured outcome')
                    if context['performance_outcome']['observation_id'] not in s['evidence']['observation_refs']:
                        raise ValueError('Measured promising scope must cite the actual outcome')
            normalized = validator(raw)
            if stage == 'next_decision' and raw['action'] == 'stop':
                mechanisms = self.stop_review_mechanisms()
                audit_context = deepcopy(context)
                audit_context.update(proposed_stop=raw,required_mechanism_reviews=mechanisms)
                audit = Workflow.call(self,'performance_interaction_stop_audit',audit_context,
                    self.validate_stop_audit,stem+'_stop_audit_'+digest(raw)[:12])
                self.state.setdefault('stop_audits',[]).append({'round':self.state['round'],
                    'proposed_stop':deepcopy(raw),'audit':deepcopy(audit)})
                self.save()
                if audit['verdict'] != 'stop_supported':
                    questions=[{'mechanism':r['mechanism'],'question':r['screening_question'],
                                'good':r['good_result_effect'],'bad':r['bad_result_effect']}
                               for r in audit['reviews'] if r['can_change_screening'] and r['interpretable_test_exists']]
                    raise ValueError('Stop rejected by independent screening-value audit. '
                        'Low expected benefit or missing evidence is not low test value. '
                        'Select a valid representative for an unresolved question: '+str(questions))
            return normalized
        raw = Workflow.call(self, mapping[stage], context, validate, stem)
        if stage == 'decision_initial':
            self.state['mechanism_screening'] = deepcopy(raw['mechanism_analyses'])
        if stage == 'regional_performance':
            self.state.setdefault('interaction_history', []).append({
                'round': self.state['round'], 'reviews': deepcopy(raw['interaction_reviews']),
                'region_reviews': deepcopy(raw['region_reviews'])})
            self.state.setdefault('promising_regions', []).extend(
                deepcopy(s) | {'round': self.state['round'], 'basis': 'measured_contextual_judgment'}
                for s in raw['promising_scopes'])
        return raw

    def joint_relations(self, members):
        plan = self.state.get('pending', {}).get('propagation', {})
        focal = plan.get('focal_mechanism')
        if not focal:
            return set()
        relevant = {focal}
        focal_relations = [r for r in self.interactions['relations'] if focal in r['mechanisms']]
        for r in focal_relations:
            relevant.update(r['mechanisms'])
        # Include relations among the focal mechanism and all its companions. This
        # intentionally errs toward explicit review rather than assumed joint weakness.
        return {r['id'] for r in self.interactions['relations']
                if focal in r['mechanisms'] or (len(r['mechanisms']) > 1
                    and set(r['mechanisms']) <= relevant)}

    def remove_redundant_propagation(self, raw):
        """Ignore same-verdict echoes for wholly automatic regions, never exceptions."""
        plan = self.state.get('pending', {}).get('propagation', {})
        direct_good, direct_bad = set(plan.get('promising_ids', [])), set(plan.get('drop_ids', []))
        for field in ('decisions', 'promising_scopes', 'region_reviews'):
            keep = []
            for entry in raw[field]:
                ids = {i for i,r in self.catalog.items() if matches(r['mechanisms'], entry['predicate'])}
                verdict = entry.get('action', entry.get('verdict', entry.get('judgment')))
                redundant = ((verdict in ('retain','promising') and ids and ids <= direct_good)
                             or (verdict == 'drop' and ids and ids <= direct_bad))
                if not redundant:
                    keep.append(entry)
            raw[field] = keep

    def validate_regions(self, raw, context):
        required = self.remaining()
        covered = set()
        drops = set().union(*(self.members(r['predicate']) for r in raw['decisions']
                              if r['action'] == 'drop'))
        verdicts = {}
        promising = set().union(*(self.members(r['predicate']) for r in raw['promising_scopes']))
        for review in raw['region_reviews']:
            members = self.members(review['predicate'])
            self.refs(review['evidence'])
            joint = review['joint_interaction_review']
            relation_ids = joint['relation_ids']
            known = {r['id'] for r in self.interactions['relations']}
            if len(relation_ids) != len(set(relation_ids)) or not set(relation_ids) <= known:
                raise ValueError('Joint review references unknown or duplicate relations')
            needed = self.joint_relations(members)
            if not needed <= set(relation_ids):
                raise ValueError('Joint review omitted relevant linkages: ' + str(sorted(needed-set(relation_ids))))
            for member in members:
                if member in verdicts and verdicts[member] != review['verdict']:
                    raise ValueError('Conflicting full-space region verdicts for ' + member + ': ' + str(self.catalog[member]['mechanisms']) + '; previous=' + verdicts[member] + '; current=' + review['verdict'] + ', predicate=' + str(review['predicate']) + '. These are JOINT configuration regions, not independent per-mechanism labels. Scope later regions around exceptions; if CacheBlend-on is promising, unresolved predicates must not also include those on combinations.')
                verdicts[member] = review['verdict']
            if review['verdict'] == 'promising' and not members <= promising:
                raise ValueError('Promising region requires matching promising_scopes')
            if not review['evidence']['knowledge_refs'] or not review['evidence']['fact_refs']:
                raise ValueError('Competitor review needs workload and knowledge evidence')
            if review['verdict'] == 'drop' and not members <= drops:
                raise ValueError('Competitor drop requires matching decisions')
            if review['verdict'] != 'drop' and members & drops:
                raise ValueError('Competitor retention conflicts with a drop')
            covered |= members
        if required != covered:
            raise ValueError('Region review must cover ALL remaining combinations; uncovered count=' + str(len(required-covered)))
        if drops != {k for k,v in verdicts.items() if v == 'drop'} or promising != {k for k,v in verdicts.items() if v == 'promising'}:
            raise ValueError('Region labels must agree exactly with drops and promising scopes')

    def hard_background_compatible(self, config, anchor, focal):
        """Compare linkage structure, not exact values inside an allowed tuple family."""
        neighbors = set(self.interactions['mechanisms'][focal]['hard_link_neighbors'])
        changed = {k for k in neighbors if config.get(k) != anchor.get(k)}
        if not changed:
            return True
        relaxed = set()
        for relation in self.interactions['relations']:
            if relation['type'] != 'hard_link' or focal not in relation['mechanisms']:
                continue
            tuples = relation.get('allowed_tuples', [])
            if not tuples:
                continue
            fields = set().union(*(set(t) for t in tuples))
            # Other categorical effects (e.g. quantization) remain protected by
            # the performance-neighbor check before this helper is used.
            if (any(all(config.get(k) == v for k,v in t.items()) for t in tuples)
                    and any(all(anchor.get(k) == v for k,v in t.items()) for t in tuples)):
                relaxed |= fields - {focal}
        return changed <= relaxed

    def propagation_plan(self, signal, outcome, pending):
        """Compute exact regions from the user's weak-interaction screening premise."""
        focal = signal['mechanism']
        control = self.state['observations'][pending['design']['control_observation_id']]
        t, c = outcome['configuration'], control['configuration']
        empty = {'drop_ids': [], 'promising_ids': [], 'exception_ids': sorted(self.remaining()),
                 'signal': signal, 'basis': 'user_weak_interaction_propagation'}
        if (outcome['status'] not in ('complete','censored') or signal['signal'] == 'inconclusive'
                or not signal['attributable']):
            return empty
        if t.get(focal) == c.get(focal):
            raise ValueError('Focal mechanism must change in treatment/control')
        if outcome['status'] == 'censored':
            lower = (outcome.get('censoring') or {}).get('lower_bound_s')
            if (self.policy['objective'] != 'wall_time_s' or self.policy['direction'] != 'minimize'
                    or lower is None or lower <= control['objective_value'] or signal['signal'] != 'bad'):
                raise ValueError('Censored propagation requires a worse runtime lower bound and bad signal')
            better = False
        else:
            better = self.strictly_better(outcome['objective_value'], control['objective_value'])
            if outcome['objective_value'] == control['objective_value'] or (signal['signal'] == 'good') != better:
                raise ValueError('Clear signal contradicts actual numeric ordering')
        # Admit only forced companion changes, not arbitrary changes to another knob.
        permitted = {focal}
        for r in self.interactions['relations']:
            if r['type'] != 'hard_link' or focal not in r['mechanisms']:
                continue
            when = r.get('when', {})
            for config in (t, c):
                if when and all(config.get(k) in values for k, values in when.items()):
                    permitted.update(k for k, values in r.get('requires', {}).items()
                                     if config.get(k) in values)
        differences = set(self.configuration_difference(t, c))
        if differences - permitted:
            return empty | {'blocked_attribution_fields': sorted(differences - permitted)}
        # Hard-link neighbors also bound transfer. Preserve each side's observed
        # companion values; on/off need not have equal companion values.
        hard_neighbors = set(self.interactions['mechanisms'][focal]['hard_link_neighbors'])
        neighbors = set(self.interactions['mechanisms'][focal]['performance_neighbors'])
        # Pairwise weak judgments alone do not establish joint independence.
        if any(t.get(k) != c.get(k) for k in neighbors):
            return empty | {'blocked_interaction_fields': sorted(neighbors)}
        winner, loser = (outcome, control) if better else (control, outcome)
        win, lose = winner['configuration'][focal], loser['configuration'][focal]
        drops, promising = set(), set()
        for identifier in self.remaining():
            config = self.catalog[identifier]['configuration']
            if any(config.get(k) != t.get(k) for k in neighbors):
                continue
            if config[focal] == lose:
                if self.hard_background_compatible(config, loser['configuration'], focal):
                    drops.add(identifier)
            elif config[focal] == win:
                if self.hard_background_compatible(config, winner['configuration'], focal):
                    promising.add(identifier)
        # A real contrary observation overrides the weak-interaction approximation.
        counterexamples = {o.get('combination_id') for o in self.state['observations'].values()
            if o['status'] == 'complete' and self.quality(o)['status'] in ('passed','disabled')
            and not self.strictly_better(winner['objective_value'], o['objective_value'])}
        drops -= counterexamples
        return empty | {'drop_ids': sorted(drops), 'promising_ids': sorted(promising),
            'exception_ids': sorted(self.remaining() - drops - promising),
            'fixed_interaction_values': {k:t[k] for k in sorted(neighbors)},
            'hard_link_transfer': 'preserve observed companion switches; allow equivalent legal precision tuples',
            'hard_link_backgrounds': {
                'winner': {k:winner['configuration'].get(k) for k in sorted(hard_neighbors)},
                'loser': {k:loser['configuration'].get(k) for k in sorted(hard_neighbors)}},
            'winner_observation_id': winner['observation_id'], 'loser_observation_id': loser['observation_id'],
            'counterexample_ids': sorted(counterexamples & self.remaining()),
            'focal_mechanism': focal}

    def performance_review(self, outcome, pending, assessment, stem):
        if (outcome['status'] not in ('complete','censored') or
                (outcome['status'] == 'complete' and self.quality(outcome)['status'] not in ('passed','disabled'))):
            return super().performance_review(outcome, pending, assessment, stem)
        if 'propagation' not in pending:
            control_id = pending['design']['control_observation_id']
            context = self.performance_context(self.context())
            context.update(performance_outcome=context['observations'][outcome['observation_id']],
                           performance_control=context['observations'][control_id],
                           selected_question=pending.get('decision'))
            def validate(signal):
                self.refs(signal['evidence'])
                if not {outcome['observation_id'], control_id} <= set(signal['evidence']['observation_refs']):
                    raise ValueError('Signal must cite treatment and matched control')
                self.propagation_plan(signal, outcome, pending)
            signal = Workflow.call(self, 'performance_propagation_signal', context, validate, stem+'_signal')
            pending['propagation'] = self.propagation_plan(signal, outcome, pending)
            self.save()
        plan = pending['propagation']
        direct = []
        for action, key in [('drop','drop_ids'), ('retain','promising_ids')]:
            if plan[key]:
                direct.append({'action':action, 'criterion':'performance', 'combination_ids':plan[key],
                    'basis':'user_weak_interaction_propagation',
                    'reason':plan['signal']['reasoning'],
                    'observation_refs':[outcome['observation_id'], pending['design']['control_observation_id']],
                    'knowledge_refs':['screening.user_weak_interaction_propagation'],
                    'next_evidence':'See final history review; deleted scopes are not reopened.',
                    'scope_contract':self.scope_contract()})
        assessment = self.screen_linked_groups(outcome, pending, assessment, stem)
        assessment['decisions'].extend(direct)
        assessment['automatic_propagation'] = deepcopy(plan)
        assessment['newly_dropped'] = len({k for r in assessment['decisions'] if r['action']=='drop' for k in r['combination_ids']})
        return self.history_scope_review(assessment, stem)

    def validate_historical_scope(self, review, members):
        self.validate_confidence(review)
        self.refs(review['evidence'])
        if not members<=self.remaining():
            raise ValueError('Historical scope review may only update remaining candidates')
        measured_ids = {key for key, obs in self.state['observations'].items()
                        if obs.get('combination_id') in members
                        and obs['status'] in ('complete', 'censored')}
        if not measured_ids <= set(review['evidence']['observation_refs']):
            raise ValueError('Region contains historical measurements that must be reviewed and cited: '
                             + str(sorted(measured_ids)))
        if review['verdict'] != 'drop':
            return
        covered = set()
        for comparison in review['supporting_comparisons']:
            better_id, worse_id = comparison['better_observation_id'], comparison['worse_observation_id']
            if not {better_id,worse_id} <= set(review['evidence']['observation_refs']):
                raise ValueError('Historical support must cite both observations')
            better, worse = (self.state['observations'][i] for i in (better_id,worse_id))
            if better['status'] != 'complete' or self.quality(better)['status'] not in ('passed','disabled'):
                raise ValueError('Historical winner must be complete and feasible')
            if worse['status'] == 'complete':
                value = worse['objective_value']
            elif (worse['status'] == 'censored' and self.policy['objective']=='wall_time_s'
                  and self.policy['direction']=='minimize'):
                value = worse['censoring']['lower_bound_s']
            else:
                raise ValueError('Historical loser needs a measured objective or comparable lower bound')
            if not self.strictly_better(better['objective_value'],value):
                raise ValueError('Historical support contradicts numeric ordering')
            signature = comparison['loss_signature']
            if not signature or not comparison['transfer_reason'].strip():
                raise ValueError('Historical deletion needs a nonempty loss signature and transfer explanation')
            for key, values in signature.items():
                if (key not in MECHANISMS or values != [worse['configuration'].get(key)]
                        or better['configuration'].get(key)==worse['configuration'].get(key)):
                    observed = {k:[worse['configuration'].get(k)] for k in MECHANISMS
                                if better['configuration'].get(k)!=worse['configuration'].get(k)}
                    raise ValueError('Loss signature must use observed inferior changed mechanism values. '
                        f'Region {review.get("region_id")}: {worse_id} is the measured loser; '
                        f'allowed changed values are {observed}; invalid entry is {key}={values}. '
                        'loss_signature describes the measured loss, NOT the target region background. '
                        'Use the focal mechanism alone if its hard links account for companion changes; '
                        'explain transfer to different companion backgrounds in transfer_reason. '
                        'Do not copy unmeasured target values into the observed loss signature.')
            changed = {k for k in MECHANISMS
                       if better['configuration'].get(k)!=worse['configuration'].get(k)}
            explained = set(signature)
            for relation in self.interactions['relations']:
                when = relation.get('when',{})
                if (relation['type']=='hard_link' and when and set(when)<=set(signature)
                    and any(all(config.get(k) in values for k,values in when.items())
                            for config in (better['configuration'],worse['configuration']))):
                    explained.update(relation.get('requires',{}))
            if changed-explained:
                raise ValueError('Historical loss signature omits jointly changed mechanisms; '
                    'a bundle loss cannot be attributed to an isolated companion setting')
            subset = {i for i in members if matches(self.catalog[i]['mechanisms'],signature)}
            for observation in self.state['observations'].values():
                if (observation.get('combination_id') in subset and observation['status']=='complete'
                    and self.quality(observation)['status'] in ('passed','disabled')
                    and not self.strictly_better(better['objective_value'],observation['objective_value'])):
                    raise ValueError('Historical deletion contradicts a competitive measured member')
            covered |= subset
        if covered != members:
            example = self.catalog[next(iter(members-covered))]['mechanisms']
            raise ValueError('Every deletion must lie within an inferior mechanism scope supported by history. '
                f"Region {review.get('region_id','unknown')} contains unsupported values {example}. "
                'Every key in a loss_signature must match a deleted member; sharing only some keys '
                'does not count. In particular an on loss cannot support off members, including '
                'off members with prefix/chunked disabled or different quantization. With only that '
                'contrast available, use unresolved for those regions. Check ALL regions, not only this one.')
        measured = {o.get('combination_id') for o in self.state['observations'].values()
                    if o['status'] in ('complete','censored')}
        if members-measured and not review['untested_conditions']:
            raise ValueError('Historical extrapolation must disclose untested conditions')

    def measured_history_contrasts(self):
        """Expose factual completed-run contrasts; attribution remains a review decision."""
        observations = [(key, obs) for key, obs in self.state['observations'].items()
                        if obs['status'] == 'complete' and obs.get('objective_value') is not None
                        and self.quality(obs)['status'] in ('passed', 'disabled')]
        contrasts = []
        for n, (left_id, left) in enumerate(observations):
            for right_id, right in observations[n+1:]:
                changes = {key:{'left':left['configuration'].get(key),
                                'right':right['configuration'].get(key)}
                           for key in sorted(set(left['configuration']) | set(right['configuration']))
                           if left['configuration'].get(key) != right['configuration'].get(key)}
                links = [r['id'] for r in self.interactions['relations']
                         if r['type'] == 'hard_link' and r.get('when')
                         and set(r['when']) & set(changes)
                         and any(matches(obs['configuration'], r['when']) for obs in (left, right))]
                contrasts.append({'left_observation_id':left_id, 'right_observation_id':right_id,
                    'left_combination_id':left.get('combination_id'),
                    'right_combination_id':right.get('combination_id'),
                    'objective':self.policy['objective'], 'direction':self.policy['direction'],
                    'left_value':left['objective_value'], 'right_value':right['objective_value'],
                    'relative_change_from_left':((right['objective_value']-left['objective_value'])
                        / left['objective_value']) if left['objective_value'] > 0 else None,
                    'actual_changed_fields':changes, 'activated_hard_links':links,
                    'interpretation':'Observed net configuration contrast, not proof of isolated-field causality.'})
        return contrasts

    def history_scope_review(self, assessment, stem):
        proposals = deepcopy(assessment['decisions'])
        members = [i for r in proposals for i in r['combination_ids']]
        if len(members)!=len(set(members)) or set(members)!=self.remaining():
            raise ValueError('Final confidence review must cover every remaining candidate exactly once')
        regions = [{'region_id':f'hr_{n:03d}', 'proposed_decision':r,
                    'member_mechanisms':[self.catalog[i]['mechanisms'] for i in r['combination_ids']]}
                   for n,r in enumerate(proposals)]
        indexed = {r['region_id']:r for r in regions}
        for region in regions:
            ids = set(region['proposed_decision']['combination_ids'])
            region['measured_members'] = [
                {'observation_id':key, 'combination_id':obs['combination_id'],
                 'status':obs['status'], 'objective_value':obs.get('objective_value')}
                for key, obs in self.state['observations'].items()
                if obs.get('combination_id') in ids and obs['status'] in ('complete','censored')]
        context = self.performance_context(self.context())
        context.pop('remaining_catalog',None)
        context.update(proposed_regions=regions,
            measured_history_contrasts=self.measured_history_contrasts(),
            history_policy='all observations jointly; latest local comparison is provisional')
        def validate(raw):
            ids = [r['region_id'] for r in raw['regions']]
            if len(ids)!=len(set(ids)) or set(ids)!=set(indexed):
                raise ValueError('Evaluate every proposed region exactly once against the full history')
            for review in raw['regions']:
                proposal = indexed[review['region_id']]['proposed_decision']
                self.validate_historical_scope(review,
                    set(proposal['combination_ids']))
            drops = {i for r in raw['regions'] if self.effective_verdict(r)=='drop'
                     for i in indexed[r['region_id']]['proposed_decision']['combination_ids']}
            if self.remaining()<=drops:
                raise ValueError('Cannot delete the entire remaining candidate space')
        raw = Workflow.call(self,'performance_history_scope',context,validate,stem+'_history_scope')
        assessment['provisional_decisions'] = deepcopy(proposals)
        assessment['historical_scope_review'] = raw
        assessment['decisions'] = [indexed[r['region_id']]['proposed_decision'] | {
            'action':'drop' if self.effective_verdict(r)=='drop' else 'retain',
            'screening_label':self.effective_verdict(r),'proposed_verdict':r['verdict'],
            'deletion_confidence':r['deletion_confidence'],'confidence_reason':r['confidence_reason'],
            'confidence_threshold':self.policy['deletion_confidence_threshold'],
            'confidence_gate_passed':r['deletion_confidence']>=self.policy['deletion_confidence_threshold'],
            'basis':'all_history_scope_review',
            'reason':r['historical_reasoning'],'evidence':r['evidence'],
            'untested_conditions':r['untested_conditions'],'next_evidence':r['next_evidence'],
            'historical_support':r['supporting_comparisons']} for r in raw['regions']]
        assessment['newly_dropped'] = len({i for r in assessment['decisions'] if r['action']=='drop'
                                          for i in r['combination_ids'] if self.state['states'][i]!='drop'})
        assessment['regional_interpretation'] = raw['summary']
        final = {r['group_id']:r for r in assessment['decisions'] if 'group_id' in r}
        for group in assessment.get('interaction_groups',[]):
            if group['group_id'] in final:
                decision = final[group['group_id']]
                group['provisional_verdict'] = group['verdict']
                group.update(verdict=decision['screening_label'], reason=decision['reason'],
                             evidence=decision['evidence'], historical_support=decision['historical_support'],
                             deletion_confidence=decision['deletion_confidence'],
                             confidence_reason=decision['confidence_reason'])
        if 'automatic_propagation' in assessment:
            plan = assessment['automatic_propagation']
            plan['status'] = 'proposal_reviewed_by_all_history_confidence_gate'
            dropped = {i for r in assessment['decisions'] if r['action']=='drop' for i in r['combination_ids']}
            promising = {i for r in assessment['decisions'] if r['screening_label']=='promising' for i in r['combination_ids']}
            plan['effective_drop_ids'] = sorted(set(plan['drop_ids']) & dropped)
            plan['effective_promising_ids'] = sorted(set(plan['promising_ids']) & promising)
        for entry in self.state.get('interaction_group_history',[]):
            if entry['round']==self.state['round'] and 'interaction_groups' in assessment:
                entry.update(stage='final_all_history_confidence_review',groups=deepcopy(assessment['interaction_groups']))
        assessment['question_review'] = {'resolution':'partial' if any(self.effective_verdict(r)=='unresolved'
            for r in raw['regions']) else 'answered', 'learned':raw['summary'],
            'remaining_gap':'See all-history scope judgments'}
        return assessment

    def interaction_groups(self, plan, outcome, pending):
        focal = plan.get('focal_mechanism', plan['signal']['mechanism'])
        control = self.state['observations'][pending['design']['control_observation_id']]
        anchors = [control['configuration'], outcome['configuration']]
        relations = [r for r in self.interactions['relations'] if focal in r['mechanisms']]
        companions = {m for r in relations for m in r['mechanisms']}
        relations = [r for r in self.interactions['relations'] if focal in r['mechanisms']
                     or (len(r['mechanisms']) > 1 and set(r['mechanisms']) <= companions)]
        groups = {}
        for identifier in plan['exception_ids']:
            row = self.catalog[identifier]; config = row['configuration']
            anchor = next((a for a in anchors if a[focal] == config[focal]), anchors[0])
            changed = {k for k in companions - {focal} if config.get(k) != anchor.get(k)}
            affected = sorted(r['id'] for r in relations if changed & set(r['mechanisms']))
            key = 'ig_' + digest([focal, config[focal], affected])[:12]
            group = groups.setdefault(key, {'group_id':key, 'focal_mechanism':focal,
                'focal_value':config[focal], 'relation_ids':affected, 'member_ids':[],
                'anchor_configuration':anchor})
            group['member_ids'].append(identifier)
        for group in groups.values():
            related = {focal} | {m for r in relations if r['id'] in group['relation_ids'] for m in r['mechanisms']}
            tuples = {digest({k:self.catalog[i]['mechanisms'][k] for k in sorted(related)}):
                      {k:self.catalog[i]['mechanisms'][k] for k in sorted(related)} for i in group['member_ids']}
            group['joint_value_tuples'] = list(tuples.values())
            group['member_count'] = len(group['member_ids'])
        assert sum(g['member_count'] for g in groups.values()) == len(plan['exception_ids'])
        return groups

    def screen_linked_groups(self, outcome, pending, assessment, stem):
        groups = self.interaction_groups(pending['propagation'], outcome, pending)
        self.state.setdefault('interaction_group_history', [])
        if not groups:
            assessment['question_review'] = {'resolution':'answered', 'learned':pending['propagation']['signal']['reasoning'],
                                            'remaining_gap':'No interaction exceptions'}
            return assessment
        context = self.performance_context(self.context())
        # Groups, not an overlapping predicate catalog, are the unit of this call.
        context.pop('remaining_catalog', None)
        context.pop('mechanism_screening', None)
        context.update(interaction_groups=list(groups.values()),
            performance_outcome=context['observations'][outcome['observation_id']],
            performance_control=context['observations'][pending['design']['control_observation_id']],
            required_observation_refs_for_every_group=[outcome['observation_id'],
                                                       pending['design']['control_observation_id']])
        def validate(raw):
            ids = [g['group_id'] for g in raw['groups']]
            if len(ids) != len(set(ids)) or set(ids) != set(groups):
                raise ValueError('Evaluate each supplied group exactly once; expected=' + str(sorted(groups)))
            for review in raw['groups']:
                self.validate_confidence(review)
                group = groups[review['group_id']]
                if set(review['relation_ids']) != set(group['relation_ids']):
                    raise ValueError('Joint group review must include ALL and ONLY its linkage IDs')
                self.refs(review['evidence'])
                if not review['evidence']['fact_refs'] or not review['evidence']['knowledge_refs']:
                    raise ValueError('Group review needs workload and knowledge evidence')
        raw = Workflow.call(self, 'performance_linked_groups', context, validate, stem+'_linked_groups')
        records=[]
        for review in raw['groups']:
            group = groups[review['group_id']]
            records.append(group | review)
            assessment['decisions'].append({'action':'drop' if review['verdict']=='drop' else 'retain',
                'criterion':'performance','combination_ids':group['member_ids'],
                'basis':'joint_interaction_group_judgment','reason':review['reason'],
                'evidence':review['evidence'],'next_evidence':review['next_evidence'],
                'group_id':group['group_id'],'screening_label':review['verdict'],
                'scope_contract':self.scope_contract()})
        assessment['interaction_groups'] = records
        assessment['regional_interpretation'] = raw['summary']
        assessment['question_review'] = {'resolution':'partial' if any(r['verdict']=='unresolved' for r in records) else 'answered',
            'learned':raw['summary'], 'remaining_gap':'See joint interaction group judgments'}
        self.state['interaction_group_history'].append({'round':self.state['round'],'groups':deepcopy(records)})
        return assessment

    def initial_pass(self, stem):
        # Zero-run initial screening happens once; later group assessments already
        # update the full remaining space and are supplied as performance history.
        if any(h.get('stage') == 'experiment' for h in self.state['history']):
            self.state['initial_completed_round'] = self.state['round']
            self.save()
            return
        initial = self.call('decision_initial',self.context(),lambda r:None,stem+'_initial')
        proposed = [{'rule_id':'r_'+digest(r)[:12],'rule':r} for r in initial['rules']]
        audit = {'reviews':[],'summary':'No proposed exclusions'}
        def validate(raw):
            self.validate_audit(raw,proposed)
            for review in raw['reviews']:
                self.validate_confidence(review)
        if proposed:
            audit = Workflow.call(self,'performance_interaction_initial_audit',
                self.performance_context(self.context(proposed_initial_rules=proposed)),
                validate,stem+'_initial_audit')
        reviews = {r['rule_id']:r for r in audit['reviews']}
        accepted, deferred = [], []
        for entry in proposed:
            raw, review = entry['rule'], reviews[entry['rule_id']]
            confidence = min(raw['deletion_confidence'],review['deletion_confidence'])
            decision = raw | {'deletion_confidence':confidence,
                'proposed_deletion_confidence':raw['deletion_confidence'],
                'audit_deletion_confidence':review['deletion_confidence'],
                'confidence_reason':raw['confidence_reason']+'; audit: '+review['confidence_reason'],
                'confidence_threshold':self.policy['deletion_confidence_threshold']}
            if review['verdict']=='supported' and confidence>=self.policy['deletion_confidence_threshold']:
                accepted.append(decision | {'confidence_gate_passed':True})
            else:
                deferred.append(decision | {'action':'retain','screening_label':'unresolved',
                    'confidence_gate_passed':False,'rule_id':entry['rule_id']})
        rules = self.rules(accepted,self.remaining(),initial=True)
        self.apply(rules)
        self.state['history'].append({'stage':'initial','round':self.state['round'],'rules':rules,
            'proposed_rules':proposed,'audit':audit,'confidence_deferred':deferred,
            'summary':initial['summary'],'source':'workload_context_and_mechanism_knowledge',
            'mechanism_analyses':deepcopy(self.state['mechanism_screening'])})
        self.state['initial_completed_round']=self.state['round']
        self.save()

    def stop_review_mechanisms(self):
        return [m for m in MECHANISMS if len({digest(self.catalog[i]['mechanisms'][m])
                                             for i in self.remaining()})>1]

    def validate_stop_audit(self, raw):
        supplied=[r['mechanism'] for r in raw['reviews']]
        if len(supplied)!=len(set(supplied)) or set(supplied)!=set(self.stop_review_mechanisms()):
            raise ValueError('Stop audit must review every surviving varying mechanism exactly once')
        for review in raw['reviews']:
            self.refs(review['evidence'])
        useful=any(r['can_change_screening'] and r['interpretable_test_exists'] for r in raw['reviews'])
        if (raw['verdict']=='continue_screening') != useful:
            raise ValueError('Stop verdict must agree with the existence of an interpretable question that can change screening')

    def normalize_decision(self, raw):
        if raw['action'] == 'run' and set(raw['decision']['outcomes']) != {'good', 'bad'}:
            raise ValueError('Forecast exactly good and bad screening outcomes')
        raw = deepcopy(raw)
        if raw['action'] == 'run' and len(self.members(raw['decision']['predicate'])) == 1:
            # Within-scope variation is mechanically empty for a singleton.
            # Keep actual control differences in comparison.changed_parameters.
            raw['decision']['varying_dimensions'] = []
        return super().normalize_decision(raw)

    def export(self, reason):
        result = super().export(reason)
        result['mechanism_screening'] = self.state.get('mechanism_screening', [])
        result['stop_audits'] = self.state.get('stop_audits', [])
        signals = list(self.state.get('promising_regions', []))
        signals += [s | {'basis': 'knowledge_prior'} for a in self.state.get('mechanism_screening', [])
                    for s in a['scopes'] if s['judgment'] == 'promising']
        result['promising_regions'] = [s | {'remaining_ids': sorted(self.members(s['predicate'], allow_already_deleted=True))}
                                       for s in signals if self.members(s['predicate'], allow_already_deleted=True)]
        result['propagated_promising_ids'] = sorted({i for h in self.state['history']
            for i in h.get('automatic_propagation', {}).get('effective_promising_ids', []) if i in self.remaining()})
        result['automatic_propagation_history'] = [h['automatic_propagation'] for h in self.state['history'] if 'automatic_propagation' in h]
        result['interaction_group_history'] = self.state.get('interaction_group_history', [])
        result['group_promising_ids'] = sorted({i for h in self.state.get('interaction_group_history', [])
            for g in h['groups'] if g['verdict']=='promising' for i in g['member_ids'] if i in self.remaining()})
        result['interaction_history'] = self.state.get('interaction_history', [])
        result['interaction_knowledge'] = self.interactions
        latest_labels = {}
        for record in self.state['history']:
            for decision in record.get('decisions',[]):
                if 'screening_label' in decision:
                    latest_labels.update({i:decision['screening_label'] for i in decision['combination_ids']})
        for field in ('propagated_promising_ids','group_promising_ids'):
            result[field] = [i for i in result[field] if latest_labels.get(i)=='promising']
        result['historical_scope_reviews'] = [h['historical_scope_review'] for h in self.state['history']
                                               if 'historical_scope_review' in h]
        result['deletion_confidence_policy'] = {
            'threshold':self.policy['deletion_confidence_threshold'],
            'score':'LLM confidence in excluding the entire group; not a calibrated probability',
            'initial_gate':'supported audit and min(proposal, audit) >= threshold',
            'history_gate':'supported historical scope and final confidence >= threshold',
            'below_threshold':'retain as unresolved','deleted_scopes':'never reopened'}
        write(self.output / 'screening_result.json', result)
        return result
