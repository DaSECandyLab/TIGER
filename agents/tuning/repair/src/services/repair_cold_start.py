"""Two measured cold-start slots with bounded, evidence-driven recommendation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from agents.tuning.experience_summarizer.src.services.local_memory import search_observations, linked_experiences
from agents.tuning.candidate_optimizer.src.services.agent_static_knowledge import retrieve_static_parameter_knowledge
from agents.tuning.candidate_optimizer.src.services.direct_inference import default_candidate_structured_inference_client
from agents.tuning.dqbo.src.services.engine.history import write_json_atomic
from agents.tuning.candidate_optimizer.src.services.prompt_format import render_tagged_sections
from .repair_bo import _complete_configuration_schema, _complete_range_contract, _compact_trial_context

POLICY = {
    "enabled": True, "version": "repair-cold-start.v4", "trial_slots": [1, 2],
    "observation_top_k": 3, "max_recommendation_passes": 2,
    "memory_source": "local_bo_memory_active_observations",
    "best_category": "high_performance_pattern",
    "quality_categories": ["negative_triplet", "quality_triplet_preference"],
    "failure_policy": "same_slot_native_sobol_with_recorded_reason",
}


def enabled(context):
    return (context.get('repair_bo', {}).get('enabled') is True
            and context.get('repair_cold_start', {}).get('enabled') is True)


def recommend(*, context, history, space, validator, slot, client=None,
              search=search_observations, expand=linked_experiences,
              static=retrieve_static_parameter_knowledge):
    """No workload execution here. Invalid/unavailable proposals consume no extra slot."""
    policy = context['repair_cold_start']
    if not enabled(context) or slot not in policy['trial_slots']:
        raise ValueError('outside RepairBO cold-start slots')
    directory = Path(context['output_dir']) / 'repair_cold_start' / f'slot_{slot:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(json.dumps({
        'implementation_version': POLICY['version'],
        'history': {k: history.get(k) for k in ('baseline_observation', 'trials', 'quality_constraint')},
        'policy': policy, 'workload': context.get('static_workload'),
        'range': _complete_range_contract(space, context.get('repair_bo', {}).get('scope', {}).get('runtime_constraints', [])),
    }, sort_keys=True).encode()).hexdigest()
    saved = directory / 'receipt.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous.get('input_hash') == input_hash:
            return previous
    receipt = {'schema': POLICY['version'], 'slot': slot, 'attempts': [],
               'input_hash': input_hash, 'recommended_candidate': None, 'status': 'fallback_to_native_sobol'}
    memory_path = Path(context['output_dir']) / 'repair_cold_start' / 'memory_snapshot.json'
    try:
        # Freeze retrieval across the two slots and process restarts.
        if memory_path.exists():
            memory = json.loads(memory_path.read_text())
        else:
            workload = context.get('static_workload', {})
            kwargs = {'root': policy.get('memory_root')}
            best = search(workload, categories=[policy['best_category']], top_k=3, **kwargs)
            quality = search(workload, categories=policy['quality_categories'], top_k=100000, **kwargs)
            memory = {'best_observations': best, 'quality_observations': quality,
                      'workload': workload, 'top_k': 3,
                      'policy': 'historical_conditions_only_not_current_measurements_or_global_bans'}
            write_json_atomic(memory_path, memory)
        receipt['memory_snapshot_path'] = str(memory_path)
        receipt['observation_ids'] = [r['observation']['memory_id'] for r in memory['best_observations']]
        if not memory['best_observations']:
            receipt['reason'] = 'no_similar_best_observation'
            return receipt
        baseline = history.get('baseline_observation', {}).get('candidate', {})
        excluded = [
            {'trial_id': r.get('trial_id', 'baseline'), 'status': r.get('status'),
             'configuration': space.bo_to_canonical(r['candidate'])}
            for r in [history.get('baseline_observation', {}), *history.get('trials', [])]
            if isinstance(r.get('candidate'), dict)
        ]
        names = {p.name for p in space.parameters}
        schema = {'type': 'object', 'additionalProperties': False, 'properties': {
            'evidence_sufficient': {'type': 'boolean'},
            'evidence_gaps': {'type': 'array', 'items': {'type': 'string'}},
            'requested_parameters': {'type': 'array', 'items': {'enum': sorted(names)}},
            'cited_observation_ids': {'type': 'array', 'items': {'enum': receipt['observation_ids']}},
            'configuration': {'anyOf': [_complete_configuration_schema(space)['properties']['target_values'], {'type': 'null'}]},
            'rationale': {'type': 'string'},
        }}
        schema['required'] = list(schema['properties'])
        system = (
            '你是 RepairBO 冷启动配置推荐员。依据相似 workload 的高性能 observation、当前 workload 特征、'
            '质量门槛和完整冻结参数域，返回完整配置，不输出候选编号或参数差异。'
            '冻结的是搜索域，不是 baseline 的取值。输入配置仅为已测参考，所有非 fixed 参数都可在域内调整。'
            '例如 TP×DP=2 不固定 TP 或 DP；如各自域允许，则(1,2)和(2,1)均满足该约束。'
            '禁止把 baseline 拓扑误认为唯一允许拓扑，也不要重新推荐 baseline。'
            '每个实测槽最多两次推荐：第一遍证据不足时 evidence_sufficient=false，configuration=null，'
            '列出具体证据缺口和相关参数；程序检索静态旋钮知识及所选 observation 关联 experience 后进行第二遍。'
            'requested_parameters 只列解决缺口必需的旋钮，通常1至4项，不要机械列出全部参数。'
            '证据足够时直接推荐，不必强行追加检索。第二遍仍不足时保持 false，不捏造证据。'
            '历史成功不保证当前质量或性能；比较 workload、环境、质量合同与适用配置条件。'
            '历史经验是推荐参考，structured_claim.conditions 描述历史证据的适用范围，不是当前搜索空间的硬约束。'
            '结合当前workload的输入输出长度、算子及查询混合、并发和请求规模等实际可用特征，'
            '以及当前硬件和质量目标，选择、组合或调整历史配置；允许推荐不满足部分历史conditions的新配置。'
            '不必照搬历史配置，也不必为了匹配历史条件而放弃有依据的迁移；不要仅因偏离历史条件就判定证据不足。'
            '准确阅读历史条件，尤其量化方案、模型和KV精度、并行拓扑及调度范围，不得把历史取值说成其他取值。'
            'rationale简要区分历史观测与本次推断：引用了什么经验、哪些当前workload特征支持采用或调整它、'
            '主要保留或改变了什么；若偏离经验条件，说明迁移理由和待验证之处。'
            '参数也可依据当前workload、运行约束、静态知识或参考配置选择，不得将这些推断虚构为历史实测结论。'
            '通过数及质量余量只属于原始支持配置；不得写成推荐配置已被验证，也不得表述为未来稳定通过的保证。'
            '完整配置须满足全部范围、固定值、条件依赖及机械约束；禁止与已执行完整配置重复。'
            '第二实测槽应参考第一槽实际结果并推荐不同配置。所有建议需实测，不得把预测写成实测。'
            '禁止重复推荐区列出本次已测配置，包括 baseline、成功、质量失败和运行失败的点。'
            '即使历史 observation 推荐该点、该点质量达标或性能最优，也不能再次推荐。'
            '必须至少改变一个当前域内可调且实际生效的参数，最终可执行配置与每个禁用点都不同；'
            '只改字段顺序、别名、派生字段或未激活参数不算新配置。'
            '收到重复冲突后必须针对该完整配置修改；找不到有依据的新配置时返回证据不足，不要重复原方案。'
            '只返回 schema JSON；理由简短，证据与建议分开。'
        )
        base_sections = [
            ('任务目标', {'stage': 'repair_cold_start', 'slot': slot, 'maximum_passes': 2}),
            ('静态workload特征', context.get('static_workload', {})),
            ('运行环境', context.get('runtime_constraint_scope', {})),
            ('质量硬门槛', history.get('quality_constraint', {})),
            ('冻结Range', _complete_range_contract(space, context.get('repair_bo', {}).get('scope', {}).get('runtime_constraints', []))),
            ('输入配置', {'role': 'measured_baseline_reference_not_frozen_values', 'configuration': baseline}),
            ('可调参数与固定参数', {
                'mutable': [p.name for p in space.parameters if p.decision != 'fixed'],
                'fixed': {p.name: p.fixed_value for p in space.parameters if p.decision == 'fixed'},
            }),
            ('历史observation', memory['best_observations']),
            ('禁止重复推荐', {'scope': 'current_run_including_baseline', 'configurations': excluded,
                         'requirement': 'new_executable_configuration_not_equal_to_any_listed_configuration'}),
            ('当前运行实测', [_compact_trial_context(r) for r in history.get('trials', [])]),
        ]
        client = client if client is not None else default_candidate_structured_inference_client()
        extra = []
        for attempt in range(1, 3):
            payload = None
            request = {'system': system, 'user': render_tagged_sections(base_sections + extra, compact=True),
                       'response_schema': schema}
            write_json_atomic(directory / f'request_{attempt}.json', request)
            try:
                response = client.complete_structured(**request)
                write_json_atomic(directory / f'response_{attempt}.json', response)
                payload = response.get('structured_output', response)
                if not isinstance(payload, dict) or set(payload) != set(schema['required']):
                    raise ValueError('output fields do not match schema')
                if type(payload['evidence_sufficient']) is not bool or not isinstance(payload['rationale'], str):
                    raise ValueError('invalid evidence_sufficient or rationale')
                for key in ('evidence_gaps', 'requested_parameters', 'cited_observation_ids'):
                    if not isinstance(payload[key], list) or any(not isinstance(x, str) for x in payload[key]):
                        raise ValueError(f'invalid {key}')
                if not set(payload['requested_parameters']) <= names:
                    raise ValueError('requested_parameters contains unknown parameter')
                if not set(payload['cited_observation_ids']) <= set(receipt['observation_ids']):
                    raise ValueError('unknown observation citation')
                if not payload['evidence_sufficient']:
                    if payload['configuration'] is not None or not payload['evidence_gaps']:
                        raise ValueError('insufficient evidence requires null configuration and explicit gaps')
                    receipt['attempts'].append({'pass': attempt, 'status': 'insufficient_evidence', 'output': payload})
                    if attempt == 1:
                        chosen = payload['cited_observation_ids'] or receipt['observation_ids']
                        observations = [r['observation'] for r in memory['best_observations'] if r['observation']['memory_id'] in chosen]
                        evidence = expand(observations, root=policy.get('memory_root'))
                        knowledge = static(payload['requested_parameters'], task='repair_cold_start_evidence_expansion')
                        extra = [('上游Agent结构化输出', payload), ('历史experience memory', evidence), ('静态知识', knowledge)]
                        write_json_atomic(directory / 'expanded_evidence.json', {'experiences': evidence, 'static': knowledge})
                    continue
                if not payload['cited_observation_ids']:
                    raise ValueError('recommendation must cite a retrieved best observation')
                raw = payload['configuration']
                if not isinstance(raw, dict) or set(raw) != names:
                    raise ValueError('configuration must contain every canonical Range parameter exactly once')
                canonical = space.normalize(raw)
                errors = space.validate(raw)
                if canonical != raw:
                    errors.append('fixed or inactive parameter values changed by normalization: ' + json.dumps(
                        {k: {'requested': raw[k], 'required': canonical[k]} for k in raw if raw[k] != canonical[k]}, ensure_ascii=False))
                candidate = dict(baseline)
                for p in space.parameters:
                    candidate.pop(p.name, None); candidate.pop(p.bo_name, None)
                candidate.update(space.canonical_to_bo(canonical))
                checked = validator(candidate)
                if checked.get('valid') is not True:
                    errors.extend(checked.get('validation_errors') or ['mechanical check failed'])
                actual = checked.get('candidate', {})
                if space.bo_to_canonical(actual) != canonical:
                    errors.append('mechanical validation changed requested configuration')
                conflicts = [r for r in excluded if r['configuration'] == space.bo_to_canonical(actual)]
                if conflicts:
                    errors.append('complete configuration duplicates an already measured configuration: ' + json.dumps(conflicts, ensure_ascii=False))
                if errors:
                    raise ValueError('; '.join(map(str, errors)))
                receipt['attempts'].append({'pass': attempt, 'status': 'valid', 'output': payload, 'mechanical_check': checked})
                receipt.update(status='recommended', recommended_candidate=dict(actual))
                return receipt
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                receipt['attempts'].append({'pass': attempt, 'status': 'invalid', 'error': error, 'output': payload})
                extra.append(('纠错信息', {'error': error, 'previous_output': payload}))
        receipt['reason'] = 'no_valid_evidence_supported_candidate_after_two_passes'
        return receipt
    except Exception as exc:
        receipt['reason'] = f'{type(exc).__name__}: {exc}'
        return receipt
    finally:
        write_json_atomic(directory / 'receipt.json', receipt)


def load_historical_memory(context, search=search_observations):
    """Freeze quality observations independently of LLM initialization."""
    path = Path(context['output_dir']) / 'historical_memory' / 'memory_snapshot.json'
    if not path.exists():
        rows = search(context.get('static_workload', {}),
                      categories=POLICY['quality_categories'], top_k=100000)
        write_json_atomic(path, {'observations': rows,
                                'workload': context.get('static_workload', {})})
    return {'observations': json.loads(path.read_text())['observations'],
            'source': str(path),
            'policy': 'historical_conditions_only; do_not_treat_source_trial_ids_as_current_run_ids; no_global_triplet_ban'}


def repair_memory(output_dir):
    """External observations remain distinct from current-run measured cases."""
    path = Path(output_dir) / 'repair_cold_start' / 'memory_snapshot.json'
    if not path.exists():
        return {}
    memory = json.loads(path.read_text())
    return {'observations': memory['quality_observations'], 'source': str(path),
            'policy': 'historical_conditions_only; do_not_treat_source_trial_ids_as_current_run_ids; no_global_triplet_ban'}
