"""Budgeted, evidence-guided screening of mechanically enumerated combinations."""
from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

from .combination_prompts import MECHANISMS, PROTECTED, SCHEMAS, VERSION, request
from .combination_space import (canonical, digest, enumerate_combinations, matches, reference_config,
                                apply_measured_runtime_constraints)
from .io import ROOT, write
from .combination_quality import bind_quality, assess_quality, hotpotqa_config, enrich_hotpotqa
from .combination_quality_order import PREMISE_ID, validate_defaults, failure_closure


def client():
    from dotenv import dotenv_values
    from agents.llm_model_policy import FIXED_SCREENING_MODEL
    from agents.tuning.mechanism_screening.client import OpenAICompatibleClient
    env = {**dotenv_values(ROOT / '.env'), **os.environ}
    return OpenAICompatibleClient(
        base_url=env.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com',
        api_key=env.get('DEEPSEEK_API_KEY'), model=FIXED_SCREENING_MODEL,
        max_tokens=16000, temperature=0, timeout=300,
    )


def load(path):
    return json.loads(Path(path).read_text())


def quality_field(key):
    lowered = key.lower()
    return any(token in lowered for token in ('quality', 'accuracy', 'bleu', 'rouge', 'exact_match')) or lowered in {'f1', 'f1_score'}


def performance_payload(value):
    if isinstance(value, dict):
        return {k: performance_payload(v) for k, v in value.items() if not quality_field(k)}
    if isinstance(value, list):
        return [performance_payload(v) for v in value]
    return value


def feature_summary(value):
    """Keep aggregate features; full input snapshots retain bins and per-query detail."""
    omitted = {'histogram', 'coverage', 'sources', 'llm_request_count_source_per_query',
               'llm_request_count_per_query'}
    if isinstance(value, dict):
        return {k: feature_summary(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, list):
        return [feature_summary(v) for v in value]
    return value


def flatten(value, prefix, kind):
    """Exact fact IDs; null values remain unavailable and cannot support a rule."""
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f'{prefix}.{key}', kind))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            result.update(flatten(child, f'{prefix}.{i}', kind))
    else:
        result[prefix] = {'value': value, 'kind': kind}
    return result


def make_evidence(baseline, static, knowledge):
    if not isinstance(knowledge, dict) or not knowledge:
        raise ValueError('Knowledge must be a nonempty object keyed by stable source/claim IDs')
    # No sibling result discovery, no implicit historical evidence, no baseline switch prerequisite.
    keys = ('wall_time_s', 'request_throughput', 'prompt_tokens', 'generation_tokens',
            'prefix_cache_hit_rate', 'flock_workload_features', 'runtime_sample_summary',
            'flock_terminal_count', 'flock_max_in_flight',
            'successful_query_count', 'request_success_total_delta')
    facts = flatten(feature_summary(performance_payload({k: baseline[k] for k in keys if k in baseline})), 'baseline', 'measured_baseline')
    facts.update(flatten(baseline.get('runtime_environment', {}).get('gpu_inventory', []),
                         'baseline.gpu_inventory', 'measured_hardware'))
    facts.update(flatten(feature_summary(performance_payload(static)), 'static', 'static_feature_not_runtime_measurement'))
    unavailable = {}
    blend_active = baseline.get('cacheblend_mode') == 'on'
    prefix_active = baseline.get('enable_prefix_caching') is not False
    for key in list(facts):
        inactive = ((not blend_active and key.startswith('baseline.')
                     and any(token in key for token in ('cacheblend', 'lmcache', 'rag_retrieved_chunk')))
                    or (not prefix_active and key.startswith('baseline.') and 'prefix_cache_hit' in key))
        if facts[key]['value'] is None or inactive:
            unavailable[key] = facts.pop(key) | {'available': False,
                'reason': 'mechanism inactive in reference; not evidence against activation' if inactive else 'not observed'}
        else:
            facts[key]['available'] = True
    return {'facts': facts, 'unavailable_facts': unavailable, 'knowledge': knowledge}


def metric(result, policy):
    """Execution completeness is separate from answer-quality feasibility."""
    if not isinstance(result, dict) or result.get('sembench_returncode') != 0:
        return None
    for key in ('request_success_error_delta', 'failed_query_count'):
        if result.get(key) != 0:
            return None
    for key, expected in policy['completion'].items():
        if result.get(key) != expected:
            return None
    value = result.get(policy['objective'])
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


class Workflow:
    def __init__(self, *, enumeration, reference, evidence, baseline, policy, output,
                 llm, executor=None, input_identity=None):
        policy = deepcopy(policy)
        if (policy.get('quality') or {}).get('hotpotqa_dataset'):
            baseline = enrich_hotpotqa(baseline, policy['quality'])
        if policy.get('quality'):
            policy['quality'] = bind_quality(policy['quality'], baseline)
            policy['screening_scope'] = 'quality_constrained_performance'
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        enumeration = apply_measured_runtime_constraints(enumeration, baseline)
        self.llm, self.executor = llm, executor
        self.runtime_constraint_scope = enumeration['runtime_constraint_scope']
        evidence = deepcopy(evidence)
        for key, fact in flatten(self.runtime_constraint_scope, 'runtime_constraint_scope',
                                 'measured_runtime_constraint').items():
            target = 'facts' if fact['value'] is not None else 'unavailable_facts'
            evidence.setdefault(target, {})[key] = fact | {'available': fact['value'] is not None}
        self.catalog = {c['combination_id']: c for c in enumeration['combinations']}
        self.reference, self.evidence, self.policy = reference, evidence, policy
        self.input_identity = input_identity or {}
        defaults = (policy.get('quality') or {}).get('monotonic_defaults')
        if defaults is not None:
            validate_defaults(defaults, self.catalog)
        if defaults is not None and policy.get('quality_scope') != 'exact_precision_tuple':
            evidence['knowledge'][PREMISE_ID] = {
                'source': 'Explicit user-supplied screening premise, not measured evidence',
                'defaults': defaults,
                'claim': 'Prefix caching and chunked prefill preserve quality. In every context, '
                         'changing any other mechanism from its declared default to a nondefault '
                         'cannot improve any quality metric, including joint changes and mechanical '
                         'dependencies. Distinct nondefault values are not ordered.',
            }
        self.identity = digest({'version': VERSION, 'enumeration': enumeration, 'reference': reference,
                                'evidence': evidence, 'baseline': baseline, 'policy': policy,
                                'inputs': input_identity})
        path = self.output / 'state.json'
        if path.exists():
            self.state = load(path)
            if self.state['identity'] != self.identity:
                raise ValueError('Cannot resume with changed inputs, prompts, policy or constraints')
        else:
            value = metric(baseline, policy)
            if value is None:
                raise ValueError('Baseline must satisfy completion and performance metric checks')
            baseline_id = next((k for k, c in self.catalog.items() if c['configuration'] == reference), None)
            self.state = {
                'identity': self.identity, 'status': 'initial', 'round': 0, 'attempts': 0,
                'states': {k: 'retain' for k in self.catalog}, 'history': [],
                'observations': {'baseline': {
                    'observation_id': 'baseline', 'kind': 'measured', 'status': 'complete',
                    'combination_id': baseline_id, 'configuration': reference,
                    'configuration_hash': digest(reference), 'objective_value': value,
                    'result': baseline,
                }},
            }
            self.save()
        write(self.output / 'legal_combinations.json', enumeration)

    def save(self):
        # Also refresh restored/reused observations, not only newly executed trials.
        for observation in self.state['observations'].values():
            quality = self.quality(observation)
            observation['quality'] = quality
            observation['quality_gate_status'] = quality['status']
            observation['quality_gate_passed'] = (
                quality['status'] == 'passed' if quality['status'] in ('passed', 'failed') else None)
        write(self.output / 'state.json', self.state)

    def remaining(self):
        return {k for k, v in self.state['states'].items() if v != 'drop'}

    def context(self, *, include_quality=True, **extra):
        if 'selected_ids' in extra:
            extra['selected_mechanisms'] = [
                {'combination_id': identifier, **self.catalog[identifier]['mechanisms']}
                for identifier in extra['selected_ids']
            ]
        columns = ['combination_id', *MECHANISMS, 'priority', 'blockers', 'mechanical_diff']
        compact = {'columns': columns, 'rows': [
            [identifier, *[self.catalog[identifier]['mechanisms'][k] for k in MECHANISMS],
             self.state['states'][identifier], self.catalog[identifier]['blockers'],
             self.catalog[identifier]['mechanical_diff']]
            for identifier in sorted(self.remaining())]}
        observations = {}
        for identifier, observation in self.state['observations'].items():
            observations[identifier] = {k: v for k, v in observation.items() if k not in {'result', 'execution'}}
            if observation.get('status') == 'censored':
                from .combination_timeout import throughput_time_estimate
                baseline_time = (self.state['observations']['baseline'].get('result') or {}).get('wall_time_s')
                if baseline_time is None and self.policy['objective'] == 'wall_time_s':
                    baseline_time = self.state['observations']['baseline'].get('objective_value')
                censoring = deepcopy(observation.get('censoring') or {})
                censoring['throughput_time_estimate'] = throughput_time_estimate(censoring, baseline_time)
                observations[identifier]['censoring'] = censoring
            result = observation.get('result') or {}
            observations[identifier]['metrics'] = feature_summary(performance_payload({
                key: result[key] for key in (
                    self.policy['objective'],
                    *self.policy['completion'], 'request_success_error_delta', 'failed_query_count',
                    'prefix_cache_hit_rate', 'runtime_sample_summary', 'flock_workload_features',
                ) if key in result
            }))
            if include_quality:
                observations[identifier]['quality'] = self.quality(observation)
        return {
            'policy': self.policy, 'reference_configuration': self.reference,
            'runtime_constraint_scope': self.runtime_constraint_scope,
            'evidence': self.evidence,
            'remaining_catalog': compact,
            'history': self.state['history'], 'observations': observations,
            'quality_failure_closures': {k: self.quality_failure_closure(o)
                                        for k, o in self.state['observations'].items()
                                        if self.quality_failure_closure(o) is not None} if include_quality else {},
            'assessed_comparisons': self.assessed_comparisons(),
            'single_mechanism_comparisons': self.single_mechanism_comparisons(),
            'remaining_run_budget': self.policy['max_runs'] - self.state['attempts'],
            **extra,
        }

    def quality(self, observation):
        if observation['status'] == 'censored':
            return {'status': 'skipped', 'reason': 'Censored observations do not undergo quality evaluation'}
        return assess_quality(observation.get('result') or {}, self.policy.get('quality'),
                              complete=observation['status'] == 'complete')

    def quality_failure_closure(self, observation):
        defaults = (self.policy.get('quality') or {}).get('monotonic_defaults')
        identifier = observation.get('combination_id')
        if defaults is None or identifier not in self.catalog or self.quality(observation)['status'] != 'failed':
            return None
        predicate, members = failure_closure(defaults, self.catalog[identifier]['mechanisms'], self.catalog)
        conflicts = sorted(o['observation_id'] for o in self.state['observations'].values()
                           if o.get('combination_id') in members and self.quality(o)['status'] == 'passed')
        return {'predicate': predicate, 'combination_ids': sorted(members & self.remaining()),
                'premise_ref': PREMISE_ID, 'conflicting_observation_ids': conflicts,
                'basis': 'user_assumed_quality_monotonicity', 'applicable': not conflicts}

    def strictly_better(self, left, right):
        return left < right if self.policy['direction'] == 'minimize' else left > right

    def assessment_scope(self, selected, outcome):
        """Global reconsideration needs a complete feasible winner or measured quality failure."""
        if not self.policy.get('quality') or outcome['status'] != 'complete':
            return set(selected)
        quality = self.quality(outcome)['status']
        control = self.state['observations'][self.state['pending']['design']['control_observation_id']]
        if quality == 'failed' or (quality == 'passed' and
                self.strictly_better(outcome['objective_value'], control['objective_value'])):
            return self.remaining()
        return set(selected)

    def performance_control(self, rule, outcome):
        return self.state['observations'][self.state['pending']['design']['control_observation_id']]

    def validate_measured_drop(self, rule, members, outcome):
        if outcome is None or outcome['status'] not in {'complete', 'censored'}:
            raise ValueError('Failure or incomplete execution cannot support a drop')
        evidence = rule['evidence']
        if outcome['observation_id'] not in evidence['observation_refs']:
            raise ValueError('Drop must cite the current measured outcome')
        criterion = rule.get('criterion', 'performance')
        quality_enabled = bool(self.policy.get('quality'))
        if criterion == 'quality_failure':
            if not quality_enabled or self.quality(outcome)['status'] != 'failed':
                raise ValueError('Quality exclusion requires complete measured quality below the baseline floors')
            anchor = outcome
            basis = ('quality_failure_scoped_heuristic' if members - {outcome.get('combination_id')}
                     else 'measured_quality_failure')
            for observation in self.state['observations'].values():
                if (observation.get('combination_id') in members
                        and self.quality(observation)['status'] == 'passed'):
                    raise ValueError('Quality exclusion contradicts a quality-passing measured member')
            if PREMISE_ID in evidence['knowledge_refs']:
                closure = self.quality_failure_closure(outcome)
                if closure is None or not closure['applicable']:
                    raise ValueError('User quality premise is unavailable or contradicted by measured evidence')
                if not members <= set(closure['combination_ids']):
                    raise ValueError('Quality exclusion exceeds the user-premise closure; nondefault alternatives are unordered')
                basis = 'user_assumed_quality_monotonicity'
        else:
            control_id = self.state['pending']['design']['control_observation_id']
            if control_id not in evidence['observation_refs']:
                raise ValueError('Drop must cite the matched control as well as treatment')
            control = self.performance_control(rule, outcome)
            if outcome['status'] == 'censored':
                if (self.policy['objective'] != 'wall_time_s' or self.policy['direction'] != 'minimize'
                        or outcome['censoring']['lower_bound_s'] <= control['objective_value']):
                    raise ValueError('Censored exclusion requires a strictly worse comparable runtime lower bound')
                winner, anchor = control, outcome
                basis = 'measured_scoped_heuristic'
            elif self.strictly_better(control['objective_value'], outcome['objective_value']):
                winner, anchor = control, outcome
                basis = 'measured_scoped_heuristic'
            elif (quality_enabled and self.strictly_better(outcome['objective_value'], control['objective_value'])
                    and self.quality(outcome)['status'] == 'passed'):
                winner, anchor = outcome, control
                basis = 'feasible_winner_scoped_heuristic'
            else:
                raise ValueError('A competitive performance outcome cannot justify dropping its set')
            if quality_enabled and self.quality(winner)['status'] != 'passed':
                raise ValueError('Performance exclusion requires a quality-passing measured winner')
            if winner.get('combination_id') in members:
                raise ValueError('Performance exclusion cannot delete its winning witness')
            if quality_enabled:
                for observation in self.state['observations'].values():
                    if (observation.get('combination_id') in members
                            and self.quality(observation)['status'] == 'passed'
                            and not self.strictly_better(winner['objective_value'], observation['objective_value'])):
                        raise ValueError('Performance exclusion contradicts a competitive quality-passing measured member')
        if members - {anchor.get('combination_id')}:
            if not rule['untested_conditions']:
                raise ValueError('Extrapolation must disclose untested conditions')
            if quality_enabled and not evidence['fact_refs']:
                raise ValueError('Scoped extrapolation must cite workload context facts')
        return basis

    def refs(self, evidence):
        for ref in evidence['fact_refs']:
            if ref not in self.evidence['facts'] or self.evidence['facts'][ref]['value'] is None:
                raise ValueError(f'Unknown or unavailable fact: {ref}')
        if not set(evidence['knowledge_refs']) <= self.evidence['knowledge'].keys():
            unknown = sorted(set(evidence['knowledge_refs']) - self.evidence['knowledge'].keys())
            misplaced = [r for r in unknown if r in self.evidence['facts']]
            raise ValueError(f'Unknown knowledge references: {unknown}. '
                             f'These are workload/runtime facts and belong in fact_refs instead: {misplaced}. '
                             f'Valid knowledge IDs: {sorted(self.evidence["knowledge"])}')
        if not set(evidence['observation_refs']) <= self.state['observations'].keys():
            raise ValueError('Unknown observation reference')

    def members(self, predicate, *, allow_already_deleted=False):
        for key, values in predicate.items():
            known = {json.dumps(c['mechanisms'][key]) for c in self.catalog.values()}
            if not {json.dumps(v) for v in values} <= known:
                raise ValueError(f'Unknown predicate values for {key}')
        result = {k for k in self.remaining() if matches(self.catalog[k]['mechanisms'], predicate)}
        if not result:
            if allow_already_deleted and any(matches(c['mechanisms'], predicate) for c in self.catalog.values()):
                return set()
            raise ValueError(f'Predicate {json.dumps(predicate)} matches no remaining combinations. '
                             'Remove this empty rule; do not invent compatibility exceptions. '
                             'Preserve independently valid decisions for existing selected members.')
        return result

    def rules(self, rules, scope, *, initial=False, outcome=None, reviewed_ids=(), ignore_already_deleted=False):
        if not initial and outcome is not None:
            scope = self.assessment_scope(scope, outcome)
        seen, result, predicates = {}, [], set()
        for rule in rules:
            criterion = rule.get('criterion', 'performance')
            if criterion not in {'performance', 'quality_failure'}:
                raise ValueError('Unknown screening criterion')
            if criterion == 'quality_failure' and (initial or not self.policy.get('quality')):
                raise ValueError('Quality exclusion is unavailable without measured quality evidence')
            self.refs(rule['evidence'])
            if initial and set(rule['predicate']) & {'enable_prefix_caching', 'chunked_prefill'} and 'cacheblend_mode' not in rule['predicate']:
                raise ValueError('Initial cache-switch rules must explicitly scope cacheblend_mode to preserve alternative execution branches')
            members = self.members(rule['predicate'], allow_already_deleted=ignore_already_deleted and rule['action'] == 'drop')
            if not members:
                continue  # Valid already-deleted scope: audited raw response, no new decision.
            signature = digest(rule['predicate'])
            if signature in predicates:
                raise ValueError('Duplicate rule predicate')
            predicates.add(signature)
            if not members <= scope or any(k in seen and seen[k] != rule['action'] for k in members):
                raise ValueError('Rules must stay within scope; overlapping actions must agree')
            seen.update({k: rule['action'] for k in members})
            if rule['action'] in {'drop', 'deprioritize'} and criterion == 'performance':
                restricted = [k for k in set(rule['predicate']) - PROTECTED
                              if {json.dumps(c['mechanisms'][k]) for c in self.catalog.values()}
                              - {json.dumps(v) for v in rule['predicate'][k]}]
                if not restricted and not self.policy.get('contextual_precision_screening'):
                    raise ValueError('Precision-only performance pruning is prohibited; wildcard mechanism guards do not count')
            if rule['action'] == 'drop':
                evidence = rule['evidence']
                if not evidence['knowledge_refs']:
                    raise ValueError('Drop requires mechanism knowledge')
                if initial:
                    if not evidence['fact_refs']:
                        raise ValueError('Initial drop requires actual workload features')
                else:
                    basis = self.validate_measured_drop(rule, members, outcome)
            result.append(rule | {'combination_ids': sorted(members),
                                  'basis': 'knowledge_prior' if initial else
                                  basis if rule['action'] == 'drop' else 'measured_scoped_heuristic'})
        dropping = {k for r in result if r['action'] == 'drop' for k in r['combination_ids']}
        if not initial and outcome is not None and self.policy.get('quality'):
            control = self.state['observations'][self.state['pending']['design']['control_observation_id']]
            if (self.quality(outcome)['status'] == 'passed' and outcome['status'] == 'complete'
                    and self.strictly_better(outcome['objective_value'], control['objective_value'])
                    and control.get('combination_id') in self.remaining()
                    and control['combination_id'] not in seen
                    and control['combination_id'] not in reviewed_ids):
                raise ValueError('A feasible winner requires an explicit decision for the losing control '
                                 f'combination {control["combination_id"]}: {control["configuration"]}. '
                                 'Assess that other-branch member with drop, retain or deprioritize and a reason; '
                                 'do not restrict the review to selected_ids. Deletion is not mandatory.')
        if self.remaining() <= dropping:
            raise ValueError('Cannot prune the entire candidate space')
        return result

    def call(self, stage, context, validator, stem):
        import jsonschema
        envelope = request(stage, context)
        folder = self.output / 'prompts'
        input_path = folder / f'{stem}.input.json'
        if input_path.exists() and load(input_path) != envelope:
            raise ValueError(f'Resume prompt changed: {stem}')
        write(input_path, envelope)
        last_error = None
        # Independent citation and predicate errors can require separate repairs.
        # Keep a finite budget and preserve every rejected response for audit.
        for attempt in range(4):
            path = folder / f'{stem}.{attempt}.response.json'
            if path.exists():
                raw = load(path)
            else:
                user = deepcopy(envelope['user'])
                if attempt:
                    user['validation_feedback'] = {
                        'error': last_error, 'rejected_response': previous,
                        'instruction': (
                            'Return the complete corrected response. For predicate scope or overlap errors, '
                            'preserve supported decision actions and narrow the exclusion to its applicable '
                            'branch, leaving explicitly retained alternative paths outside it. Do not weaken '
                            'drop to deprioritize solely to resolve a scope conflict. Initial broad retain '
                            'statements belong in summary because uncovered combinations are retained by '
                            'default. For evidence errors reconsider the affected reasoning; never merely '
                            'backfill citations. For comparison/configuration errors, abandon the false '
                            'hypothesis and use the supplied exact configurations and computed comparisons '
                            'to choose a genuinely different valid pair, or stop/defer. Never repeat the '
                            'rejected pair with the same invented differences.'
                        ),
                    }
                    if stage.startswith('performance_simple_'):
                        user['validation_feedback']['instruction'] = (
                            'Return the complete corrected response using only output_schema fields. '
                            'remaining_catalog is already mechanically legal; do NOT propose exclusions '
                            'for illegal combinations or re-check legality using the baseline backend. '
                            'Remove predicates with no members. Keep independently supported performance '
                            'exclusions; missing benefit is not evidence of harm. For assessment predicates, '
                            'only the current remaining_catalog is updated. Assessment questions must concern the space after its exclusions. For selection use a runnable ID and '
                            'feasible control; do not add changed_parameters or varying_dimensions.'
                        )
                    elif stage == 'performance_interaction_initial':
                        user['validation_feedback']['instruction'] = (
                            'Return the complete corrected response and fix the reported error. '
                            'For EACH mechanism, scopes must cover EVERY remaining legal combination. '
                            'A missing background must receive an explicit scope, using unresolved '
                            'when evidence is insufficient. Do not put uncovered scopes only in summary. '
                            'For example, chunked_prefill scopes restricted to prefix=true or '
                            'CacheBlend=on OMIT prefix=false AND CacheBlend=off, for BOTH legal '
                            'chunked_prefill values. Add that conditional unresolved scope or use '
                            'broader predicates. Preserve required interaction-specific splits. '
                            'rules contains proposed exclusions only; empty rules is valid. '
                            'Do not invent citations, measurements, categorical values, or confidence.'
                        )
                        compact = context.get('remaining_catalog',{})
                        columns = compact.get('columns',[])
                        user['validation_feedback']['allowed_predicate_values'] = {
                            name:list({json.dumps(row[columns.index(name)]):row[columns.index(name)]
                                       for row in compact.get('rows',[])}.values())
                            for name in MECHANISMS if name in columns}
                    elif stage == 'performance_interaction_decision':
                        user['validation_feedback']['instruction'] = (
                            'This is NEXT-EXPERIMENT SELECTION, not an exclusion-rule repair. '
                            'Discard a rejected measured/deleted representative; never repeat it. '
                            'Choose a NEW combination_id from context.experiment_options only, '
                            'and use its actual mechanism values in predicate. Historical decisions '
                            'are evidence, NOT runnable choices or response templates. Return only '
                            'output_schema fields; omit program-derived fields from old decisions. '
                            'If no useful unattempted representative exists, action=stop with decision=null. '
                            'Do not invent measurements or compatibility exceptions.'
                            ' For varying-dimension errors, keep a valid representative and fix the '
                            'scope explanation: omitted predicate dimensions allow ANY legal value. '
                            'Explain every dimension listed in the error exactly once in '
                            'varying_dimensions, with shared_effect and reversal_condition. '
                            'A fixed background in prose does not restrict predicate membership. '
                            'Only narrow predicate when that restriction is causally necessary.'
                        )
                        user['validation_feedback']['eligible_representatives'] = [
                            {'combination_id':row['combination_id'],'mechanisms':row['mechanisms']}
                            for row in context.get('experiment_options',[])]
                    # Give a complete membership diagnosis, rather than making the
                    # model discover multiple impossible predicates one at a time.
                    membership = []
                    for rule in previous.get('exclusions', previous.get('decisions', previous.get('rules', []))):
                        try:
                            members = self.members(rule['predicate'], allow_already_deleted=True)
                            membership.append({'predicate': rule['predicate'], 'remaining_count': len(members)})
                        except (ValueError, KeyError, TypeError) as exc:
                            membership.append({'predicate': rule.get('predicate'), 'error': str(exc),
                                               'instruction': 'Omit this impossible rule entirely. It has no legal members.'})
                    user['validation_feedback']['predicate_membership'] = membership
                    # Supply exact available IDs/values when a citation typo cannot
                    # be resolved from a large context. These are suggestions only;
                    # the model must reconsider whether they support its reasoning.
                    from difflib import get_close_matches
                    citation_hints = []
                    def find_invalid_refs(value):
                        if isinstance(value, dict):
                            for field, bucket in (('fact_refs','facts'),('knowledge_refs','knowledge')):
                                available = context.get('evidence',{}).get(bucket,{})
                                for ref in value.get(field,[]) if isinstance(value.get(field,[]),list) else []:
                                    if isinstance(ref,str) and ref not in available:
                                        hints = {k:available[k] for k in get_close_matches(ref,available,n=3,cutoff=.6)}
                                        item = {'invalid_reference':ref,'available_similar_references':hints}
                                        if item not in citation_hints:
                                            citation_hints.append(item)
                            for child in value.values():find_invalid_refs(child)
                        elif isinstance(value,list):
                            for child in value:find_invalid_refs(child)
                    find_invalid_refs(previous)
                    if citation_hints:
                        user['validation_feedback']['citation_hints'] = citation_hints
                    user = {'validation_feedback': user.pop('validation_feedback'), **user}
                write(folder / f'{stem}.{attempt}.request.json', {'system': envelope['system'], 'user': user})
                raw = self.llm.complete_json(system=envelope['system'], user=json.dumps(user, ensure_ascii=False))
                write(path, raw)
            try:
                jsonschema.validate(raw, SCHEMAS[stage])
                validator(raw)
            except (jsonschema.ValidationError, ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)[:2500]
                previous = raw
                write(folder / f'{stem}.{attempt}.validation.json', {'valid': False, 'error': last_error})
                continue
            write(folder / f'{stem}.{attempt}.validation.json', {'valid': True})
            write(folder / f'{stem}.accepted.json', raw)
            return raw
        raise ValueError(f'{stage}: rejected after bounded schema repair: {last_error}')

    def apply(self, rules):
        for rule in rules:
            for identifier in rule['combination_ids']:
                self.state['states'][identifier] = rule['action']

    def validate_select(self, raw):
        self.refs(raw['evidence'])
        if raw['action'] == 'stop':
            if raw['predicate'] is not None or not raw['stop_reason'] or raw['comparison'] is not None:
                raise ValueError('stop requires null predicate/comparison and a reason')
        else:
            if raw['predicate'] is None or raw['stop_reason'] is not None:
                raise ValueError('investigate requires predicate and null stop_reason')
            selected = self.members(raw['predicate'])
            comparison = raw['comparison']
            if comparison is None:
                raise ValueError('investigate requires a concrete comparison')
            anchor = self.catalog.get(comparison['combination_id'])
            if anchor is not None and comparison['combination_id'] not in selected:
                conflicts = {k: {'predicate_allows': values, 'actual_treatment': anchor['mechanisms'][k]}
                             for k, values in raw['predicate'].items()
                             if anchor['mechanisms'][k] not in values}
                raise ValueError(f'Comparison anchor is outside the treatment set. Conflicts: {conflicts}. '
                                 f'Actual treatment mechanisms: {anchor["mechanisms"]}. '
                                 'The predicate describes TREATMENTS, not the control. Revise the predicate '
                                 'to contain the intended treatment, or choose an in-scope anchor; '
                                 'already dropped members cannot be selected.')
            row, control = self.validate_comparison(comparison, selected)
            existing = [o for o in self.state['observations'].values()
                        if o['configuration_hash'] == row['configuration_hash']]
            if existing:
                complete = [o for o in existing if o['status'] == 'complete']
                if not complete:
                    raise ValueError('Comparison anchor was attempted without a complete reusable result')
                self.require_new_comparison(row, control, selected)
            elif row['blockers'] or self.state['attempts'] >= self.policy['max_runs']:
                raise ValueError('Comparison anchor must be runnable with available budget')

    def assessed_comparisons(self):
        """Coverage is directional: reversing treatment/control is a new question."""
        covered = {}
        for item in self.state['history']:
            if item.get('stage') != 'experiment':
                continue
            treatment = self.state['observations'][item['observation_id']]
            control = self.state['observations'][item['design']['control_observation_id']]
            key = (treatment['configuration_hash'], control['configuration_hash'])
            covered.setdefault(key, set()).update(item.get('assessed_combination_ids', item['selected_ids']))
        return [{'treatment_configuration_hash': treatment,
                 'control_configuration_hash': control,
                 'covered_combination_ids': sorted(members)}
                for (treatment, control), members in sorted(covered.items())]

    @staticmethod
    def configuration_difference(treatment, control):
        return {k: {'treatment': treatment.get(k), 'control': control.get(k)}
                for k in sorted(treatment.keys() | control.keys())
                if k not in treatment or k not in control or digest(treatment[k]) != digest(control[k])}

    def single_mechanism_comparisons(self):
        comparisons = []
        for control_id, control in self.state['observations'].items():
            if control['status'] != 'complete':
                continue
            for identifier in sorted(self.remaining()):
                row = self.catalog[identifier]
                difference = self.configuration_difference(row['configuration'], control['configuration'])
                if len(set(difference) & set(MECHANISMS)) != 1 or row['blockers']:
                    continue
                existing = [o for o in self.state['observations'].values()
                            if o['configuration_hash'] == row['configuration_hash']]
                if existing:
                    if not any(o['status'] == 'complete' for o in existing):
                        continue
                    try:
                        self.require_new_comparison(row, control, {identifier})
                    except ValueError:
                        continue
                elif self.state['attempts'] >= self.policy['max_runs']:
                    continue
                comparisons.append({'combination_id': identifier, 'control_observation_id': control_id,
                                    'changed_parameters': list(difference), 'differences': difference})
        return comparisons

    def require_new_comparison(self, row, control, selected):
        for comparison in self.assessed_comparisons():
            if (comparison['treatment_configuration_hash'] == row['configuration_hash']
                    and comparison['control_configuration_hash'] == control['configuration_hash']
                    and set(selected) <= set(comparison['covered_combination_ids'])):
                raise ValueError('Already assessed this treatment/control pair over the selected scope; '
                                 'choose a new comparison, expand to unassessed members, or stop/defer')

    def validate_comparison(self, raw, selected):
        identifier = raw['combination_id']
        if identifier not in selected:
            raise ValueError(f'Comparison {identifier} must belong to selected_ids')
        control = self.state['observations'].get(raw['control_observation_id'])
        if not control or control['status'] != 'complete':
            raise ValueError('Control must be a successful supplied observation')
        row = self.catalog[identifier]
        treatment_config, control_config = row['configuration'], control['configuration']
        actual = self.configuration_difference(treatment_config, control_config)
        differences = list(actual)
        if not differences:
            raise ValueError(f'Treatment {identifier} and control {raw["control_observation_id"]} '
                             f'are the SAME configuration: {control_config}. There is no intervention. '
                             'Choose a different combination using single_mechanism_comparisons or stop/defer.')
        if sorted(raw['changed_parameters']) != differences:
            raise ValueError(f'changed_parameters must match the actual configuration differences: {actual}')
        return row, control

    def validate_design(self, raw, selected):
        self.refs(raw['evidence'])
        if raw['action'] == 'defer':
            if any(raw[k] is not None for k in ('combination_id', 'control_observation_id', 'reuse_observation_id')):
                raise ValueError('defer requires null IDs')
            if raw['changed_parameters']:
                raise ValueError('defer requires empty changed_parameters')
            return
        identifier = raw['combination_id']
        if identifier not in selected:
            raise ValueError(f'Experiment {identifier} must belong to selected_ids: {sorted(selected)}. '
                             'Do not substitute a configuration from another set. If no useful '
                             'valid experiment exists here, return defer with all IDs null; '
                             'the next selection round can choose another set.')
        control = self.state['observations'].get(raw['control_observation_id'])
        if not control or control['status'] != 'complete':
            raise ValueError('Control must be a successful supplied observation')
        row = self.catalog[identifier]
        self.validate_comparison(raw, selected)
        existing = [o for o in self.state['observations'].values()
                    if o['configuration_hash'] == row['configuration_hash']]
        if raw['action'] == 'reuse':
            reused = self.state['observations'].get(raw['reuse_observation_id'])
            if reused not in existing or reused['status'] != 'complete':
                raise ValueError('Reuse requires an exact complete configuration match')
            if reused['observation_id'] == control['observation_id']:
                raise ValueError('Treatment and control must be different observations')
            self.require_new_comparison(row, control, selected)
        elif (raw['reuse_observation_id'] is not None or row['blockers'] or existing
              or self.state['attempts'] >= self.policy['max_runs']):
            raise ValueError('Run requires unattempted runnable configuration and available budget')
        if control['configuration_hash'] == row['configuration_hash']:
            raise ValueError('Treatment and control configurations must differ')

    def execute(self, design):
        if design['action'] == 'reuse':
            return self.state['observations'][design['reuse_observation_id']]
        if self.executor is None:
            return None
        row = self.catalog[design['combination_id']]
        number = self.state['attempts'] + 1
        identifier = f'probe_{number:03d}'
        observation = {
            'observation_id': identifier, 'kind': 'measured', 'status': 'started',
            'combination_id': row['combination_id'], 'configuration': row['configuration'],
            'configuration_hash': row['configuration_hash'], 'objective_value': None,
            'result': None,
        }
        # Persist before launching: an interrupted attempt can never be silently rerun.
        self.state['attempts'] = number
        self.state['observations'][identifier] = observation
        self.state['pending']['observation_id'] = identifier
        self.save()
        execution = self.executor.execute(row['configuration'],
                                          active_parameters=list(row['configuration']), trial_id=identifier)
        observation['execution'] = json.loads(json.dumps(asdict(execution)))
        if execution.status == 'censored' and execution.censoring:
            observation.update(status='censored', censoring=dict(execution.censoring))
            self.save()
            return observation
        result = dict(execution.result or {})
        value = metric(result, self.policy) if execution.status == 'complete' else None
        if value is not None and (self.policy.get('quality') or {}).get('hotpotqa_dataset'):
            result = enrich_hotpotqa(result, self.policy['quality'])
        reported = canonical(result)
        mismatches = [key for key, expected in row['configuration'].items()
                      if key in reported and reported[key] != expected]
        if mismatches:
            value = None
            observation['configuration_mismatches'] = mismatches
        observation.update(status='complete' if value is not None else 'failed', result=result,
                           objective_value=value)
        observation['quality'] = self.quality(observation)
        self.save()
        return observation

    def export(self, reason):
        self.state['status'] = 'complete'
        self.state['stop_reason'] = reason
        self.save()
        result = {
            'schema_version': VERSION, 'status': 'complete', 'stop_reason': reason,
            'baseline_config_digest': (digest(self.input_identity['baseline_config'])
                if 'baseline_config' in self.input_identity else None),
            'additional_run_attempts': self.state['attempts'],
            'retained_combinations': [self.catalog[k] | {'priority': self.state['states'][k]}
                                      for k in sorted(self.remaining())],
            'dropped_combination_ids': sorted(set(self.catalog) - self.remaining()),
            'history': self.state['history'], 'observations': self.state['observations'],
            'quality_policy': self.policy.get('quality'),
            'quality_assessments': {k: self.quality(v) for k, v in self.state['observations'].items()},
            'scope': 'Current workload and frozen reference, plus explicitly recorded mechanical dependencies; heuristic exclusions are reopenable.',
            'handoff': 'Consume retained full combinations, never reconstruct a Cartesian product from independent retained values.',
        }
        write(self.output / 'screening_result.json', result)
        return result

    def run(self):
        with (self.output / '.workflow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = load(self.output / 'state.json')
            if self.state['identity'] != self.identity:
                raise ValueError('Concurrent workflow has different inputs')
            return self._run()

    def _run(self):
        if self.state['status'] == 'complete':
            return load(self.output / 'screening_result.json') if (self.output / 'screening_result.json').exists() else self.export(self.state['stop_reason'])
        if self.state['status'] == 'initial':
            raw = self.call('initial', self.context(),
                            lambda r: self.rules(r['rules'], self.remaining(), initial=True), 'initial')
            rules = self.rules(raw['rules'], self.remaining(), initial=True)
            self.apply(rules)
            self.state['history'].append({'stage': 'initial', 'rules': rules, 'summary': raw['summary']})
            self.state['status'] = 'iterating'
            self.save()
        while self.state['round'] < self.policy['max_rounds']:
            index = self.state['round']
            stem = f'round_{index:03d}'
            if 'pending' not in self.state:
                if self.state['attempts'] >= self.policy['max_runs']:
                    return self.export('additional_run_budget_exhausted')
                selection = self.call('select', self.context(), self.validate_select, stem + '_select')
                if selection['action'] == 'stop':
                    return self.export(selection['stop_reason'])
                selected = sorted(self.members(selection['predicate']))
                design = self.call('design', self.context(selection=selection, selected_ids=selected,
                                                          experiment_catalog=[self.catalog[k] for k in selected]),
                                   lambda r: self.validate_design(r, set(selected)), stem + '_design')
                if design['action'] == 'defer':
                    self.state['history'].append({'stage': 'defer', 'selection': selection,
                                                  'selected_ids': selected, 'design': design})
                    self.state['round'] += 1
                    self.save()
                    continue
                self.state['pending'] = {'selection': selection, 'selected_ids': selected, 'design': design}
                self.save()
            pending = self.state['pending']
            if 'observation_id' in pending:
                outcome = self.state['observations'][pending['observation_id']]
                if outcome['status'] == 'started':
                    outcome.update(status='interrupted', reason='Interrupted attempt retained as unknown; never automatically retried')
                    self.save()
            else:
                outcome = self.execute(pending['design'])
                if outcome is None:
                    write(self.output / 'pending_experiment.json', {
                        'design': pending['design'], 'combination': self.catalog[pending['design']['combination_id']],
                        'note': 'Planning only; resume with --execute to run this exact experiment',
                    })
                    return {'status': 'awaiting_execution', 'additional_run_attempts': self.state['attempts']}
            outcome_summary = self.context()['observations'][outcome['observation_id']]
            raw = self.call('assess', self.context(**pending, outcome=outcome_summary,
                            assessment_scope_ids=sorted(self.assessment_scope(pending['selected_ids'], outcome))),
                            lambda r: self.rules(r['decisions'], set(pending['selected_ids']), outcome=outcome),
                            stem + '_assess')
            decisions = self.rules(raw['decisions'], set(pending['selected_ids']), outcome=outcome)
            self.apply(decisions)
            self.state['history'].append({
                'stage': 'experiment', **pending, 'observation_id': outcome['observation_id'],
                'decisions': decisions, 'interpretation': raw['interpretation'],
                'coverage': raw['coverage'], 'limitations': raw['limitations'],
                'assessed_combination_ids': sorted(set(pending['selected_ids']) |
                                                  {k for r in decisions for k in r['combination_ids']}),
            })
            del self.state['pending']
            self.state['round'] += 1
            self.save()
        return self.export('screening_round_budget_exhausted')


def main(argv=None):
    from agents.tuning.dqbo.src.services.engine.space import FixedSearchSpace
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('domain-spec', 'baseline-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('baseline', 'static-features', 'knowledge'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--dependencies', type=Path, help='Explicit existing mechanical background bindings, JSON array')
    parser.add_argument('--trial-argv', type=Path, help='Original workload runner argv as a JSON array')
    parser.add_argument('--objective', default='wall_time_s')
    parser.add_argument('--direction', choices=('minimize', 'maximize'), default='minimize')
    parser.add_argument('--max-runs', type=int, default=3)
    parser.add_argument('--deletion-confidence-threshold', type=float, default=.9,
                        help='Interaction workflow: minimum group deletion confidence in [0,1] (default 0.9)')
    parser.add_argument('--workflow', choices=('simple', 'interaction-first', 'cluster-first', 'decision-first', 'legacy'), default='simple',
                        help='Simple iterative screening (default), or previous interaction/cluster/decision/legacy workflows')
    parser.add_argument('--downstream-optimization', type=Path,
                        help='JSON describing later optimization parameters/constraints beyond the seven mechanisms')
    parser.add_argument('--max-rounds', type=int, default=6)
    parser.add_argument('--assessment-batch-size', type=int, default=64,
                        help='Simple workflow: maximum surviving combinations per assessment batch')
    parser.add_argument('--assessment-workers', type=int, default=4,
                        help='Simple workflow: maximum concurrent assessment calls')
    parser.add_argument('--trial-timeout', type=float, default=900)
    parser.add_argument('--slowdown-cutoff', type=float, default=3.0,
                        help='Stop when observed query-time lower bound exceeds this multiple of baseline wall time; 0 disables')
    parser.add_argument('--workload-timeout', type=float,
                        help='Explicit query-execution time cutoff in seconds, overrides slowdown-cutoff')
    quality_options = parser.add_mutually_exclusive_group()
    quality_options.add_argument('--hotpotqa-dataset', type=Path,
                                 help='Score raw HotpotQA answers for both EM and F1 quality floors')
    quality_options.add_argument('--quality-metric', action='append', default=[], metavar='NAME=RESULT.PATH',
                                 help='Higher-is-better normalized quality metric; repeat for conjunctive gates')
    parser.add_argument('--quality-retention', type=float, default=.9)
    parser.add_argument('--quality-monotonic-defaults', type=Path,
                        help='Five quality defaults: legacy monotonic premise; context only for decision-first exact-tuple quality scope')
    parser.add_argument('--execute', action='store_true', help='Execute budgeted probes using the existing BO trial executor')
    parser.add_argument('--enumerate-only', action='store_true', help='No LLM calls or workload executions')
    args = parser.parse_args(argv)
    if args.assessment_batch_size < 1 or args.assessment_workers < 1:
        parser.error('assessment batch size and workers must be positive integers')
    if not math.isfinite(args.quality_retention) or not 0 < args.quality_retention <= 1:
        parser.error('--quality-retention must be in (0, 1]')
    if quality_field(args.objective):
        parser.error('This screening stage accepts performance objectives only')
    if not args.enumerate_only and any(getattr(args, key) is None for key in ('baseline', 'static_features', 'knowledge')):
        parser.error('--baseline, --static-features and --knowledge are required for LLM screening')
    if args.max_runs < 0 or args.max_rounds < 0 or not math.isfinite(args.trial_timeout) or args.trial_timeout <= 0:
        parser.error('Budgets must be nonnegative; timeout must be positive and finite')
    if not math.isfinite(args.deletion_confidence_threshold) or not 0 <= args.deletion_confidence_threshold <= 1:
        parser.error('--deletion-confidence-threshold must be finite and between 0 and 1')
    if not math.isfinite(args.slowdown_cutoff) or args.slowdown_cutoff < 0:
        parser.error('--slowdown-cutoff must be finite and nonnegative')
    if args.workload_timeout is not None and (not math.isfinite(args.workload_timeout) or args.workload_timeout <= 0):
        parser.error('--workload-timeout must be positive and finite')
    space = FixedSearchSpace.from_json(args.domain_spec)
    reference = reference_config(load(args.baseline_config))
    dependencies = load(args.dependencies) if args.dependencies else []
    enumeration = enumerate_combinations(space, reference, dependencies)
    if args.enumerate_only:
        if args.baseline:
            enumeration = apply_measured_runtime_constraints(enumeration, load(args.baseline))
        destination = args.output / 'legal_combinations.json'
        if destination.exists() and load(destination) != enumeration:
            raise ValueError('Refusing to overwrite a different enumeration')
        write(destination, enumeration)
        print(json.dumps({'legal': len(enumeration['combinations']), 'excluded': len(enumeration['excluded'])}))
        return
    baseline, static, knowledge = load(args.baseline), load(args.static_features), load(args.knowledge)
    baseline_values = canonical(baseline)
    for key in reference:
        if key in baseline_values and baseline_values[key] != reference[key]:
            raise ValueError(f'Baseline result/config mismatch: {key}')
    completion = {k: baseline[k] for k in ('successful_query_count', 'request_success_total_delta')}
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0 for v in completion.values()):
        raise ValueError('Baseline completion counters must be positive')
    policy = {'max_runs': args.max_runs, 'max_rounds': args.max_rounds,
              'objective': args.objective, 'direction': args.direction,
              'completion': completion, 'screening_scope': 'performance_only',
              'protected_dimensions': sorted(PROTECTED), 'trial_timeout': args.trial_timeout}
    if args.workflow == 'interaction-first':
        policy['deletion_confidence_threshold'] = args.deletion_confidence_threshold
    if args.workflow == 'simple':
        policy.update(assessment_batch_size=args.assessment_batch_size, assessment_workers=args.assessment_workers)
    if args.downstream_optimization:
        downstream = load(args.downstream_optimization)
        if not isinstance(downstream, dict) or not isinstance(downstream.get('parameters'), dict):
            parser.error('--downstream-optimization requires an object with a parameters mapping')
        policy['downstream_optimization'] = downstream
    if args.hotpotqa_dataset:
        policy['quality'] = hotpotqa_config(args.hotpotqa_dataset)
    elif args.quality_metric:
        metrics = {}
        for item in args.quality_metric:
            name, separator, path = item.partition('=')
            if not separator or not name or not path or name in metrics:
                parser.error('--quality-metric requires unique NAME=RESULT.PATH entries')
            metrics[name] = path
        policy['quality'] = {'metrics': metrics}
    if policy.get('quality'):
        policy['quality']['fraction'] = args.quality_retention
    if args.quality_monotonic_defaults:
        if not policy.get('quality'):
            parser.error('--quality-monotonic-defaults requires a quality policy')
        policy['quality']['monotonic_defaults'] = load(args.quality_monotonic_defaults)
    cutoff = None
    if args.objective == 'wall_time_s' and args.direction == 'minimize':
        cutoff = args.workload_timeout or (args.slowdown_cutoff * baseline['wall_time_s'] if args.slowdown_cutoff else None)
    elif args.workload_timeout is not None:
        parser.error('Query runtime censoring requires minimizing wall_time_s')
    policy['query_execution_cutoff_s'] = cutoff
    trial_argv = load(args.trial_argv) if args.trial_argv else []
    if not isinstance(trial_argv, list) or any(not isinstance(x, str) for x in trial_argv):
        raise ValueError('trial-argv must be a JSON array of strings')
    executor = None
    if args.execute:
        if not args.trial_argv:
            parser.error('--execute requires --trial-argv from the original workload')
        from agents.tuning.dqbo.src.services.engine.executor import BOTrialExecutor
        from .combination_timeout import QueryTimeCutoff
        monitor = QueryTimeCutoff(
            args.output / 'query_progress', cutoff,
            expected_sql_count=completion['successful_query_count'],
            expected_request_count=completion['request_success_total_delta'],
        ) if cutoff else None
        executor = BOTrialExecutor(output_dir=args.output, baseline_config=args.baseline_config,
                                   trial_argv=trial_argv, timeout_seconds=args.trial_timeout,
                                   stop_condition=monitor,
                                   trial_environment=monitor.environment if monitor else None)
    workflow_type = Workflow
    if args.workflow == 'simple':
        from .simple_screen import SimpleWorkflow
        workflow_type = SimpleWorkflow
    if args.workflow == 'decision-first':
        from .decision_screen import DecisionWorkflow
        workflow_type = DecisionWorkflow
    if args.workflow == 'cluster-first':
        from .cluster_screen import ClusterWorkflow
        workflow_type = ClusterWorkflow
    if args.workflow == 'interaction-first':
        from .interaction_screen import InteractionWorkflow
        workflow_type = InteractionWorkflow
    workflow = workflow_type(enumeration=enumeration, reference=reference,
                        evidence=make_evidence(baseline, static, knowledge), baseline=baseline,
                        policy=policy, output=args.output, llm=client(), executor=executor,
                        input_identity={'domain_spec': load(args.domain_spec), 'dependencies': dependencies,
                                        'baseline_config': load(args.baseline_config), 'trial_argv': trial_argv,
                                        'prompts': {s: request(s, {}) for s in SCHEMAS}})
    result = workflow.run()
    print(json.dumps({'status': result['status'], 'additional_run_attempts': result['additional_run_attempts']}))


if __name__ == '__main__':
    main()
