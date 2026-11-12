"""Bounded pre-execution review; assessments never become measured BO rows."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from agents.tuning.candidate_optimizer.src.services.direct_inference import StructuredInferenceClient
from agents.tuning.candidate_optimizer.src.services.prompt_format import render_tagged_sections


def memory_failure_comparisons(proposal: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose measured failure conditions without turning them into a ban."""
    configuration = proposal.get("configuration", {})
    records = evidence.get("records", {})
    fields = ("tp", "tensor_parallel_size", "data_parallel_size", "enforce_eager",
              "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
              "max_model_len", "quantization", "dtype", "kv_cache_dtype")
    comparisons = []
    for failure in evidence.get("runtime_failures", []):
        reason = str(failure.get("failure_reason", ""))
        lowered = reason.lower()
        if "sampler" in lowered and ("out of memory" in lowered or "oom" in lowered):
            stage = "sampler_warmup_temporary_memory"
        elif "kv" in lowered and any(x in lowered for x in ("memory", "budget", "capacity")):
            stage = "kv_cache_capacity"
        else:
            continue
        trial_id = str(failure.get("trial_id", ""))
        previous = records.get(trial_id, {}).get("configuration", {})
        comparisons.append({
            "trial_id": trial_id, "failure_stage": stage, "failure_reason": reason,
            "conditions": {field: {"failed": previous.get(field), "proposed": configuration.get(field)}
                           for field in fields if field in previous or field in configuration},
            "is_failure_prediction": False,
        })
    return comparisons


class RepairProposalReviewer:
    def __init__(self, client: StructuredInferenceClient | None) -> None:
        self.client = client

    def evaluate(self, *, proposal: Mapping[str, Any], evidence: Mapping[str, Any],
                 performance: Mapping[str, Any], scope: Mapping[str, Any],
                 transitions: list[dict[str, Any]], route: str | None = None) -> dict[str, Any]:
        feedback_properties = {
            "category": {"type": "string", "enum": ["quality", "performance", "runtime", "scope"]},
            "evidence_type": {"type": "string", "enum": ["measured", "predicted", "unknown"]},
            "parameters": {"type": "array", "items": {"type": "string"}},
            "cited_trial_ids": {"type": "array", "items": {"type": "string"}},
            "cited_memory_ids": {"type": "array", "items": {"type": "string"}},
            "finding": {"type": "string"},
            "suggestion": {"type": "string"},
        }
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": ["accept", "revise"]},
                **{name: {"type": "string", "enum": ["low", "medium", "high", "unknown"]}
                   for name in ("quality_risk", "performance_risk", "runtime_risk")},
                "feedback": {"type": "array", "maxItems": 4, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": feedback_properties, "required": list(feedback_properties),
                }},
                "cited_trial_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
        schema["required"] = list(schema["properties"])
        system = (
            "你是 RepairBO 执行前评审员，不是实验执行器。判断提案是否值得完整实测，"
            "目标是质量达标且改进可行 Pareto 前沿。只能使用给定证据。"
            "性能均值、不确定性、EHVI 和运行概率都是预测，不是实测，更不是可行性证明；"
            "没有质量代理，不能伪造质量数值。不能把未知或低置信度预测作为淘汰理由。"
            "仅当实测历史支持具体问题且可提出范围内的修订方向时返回 revise；否则 accept 并注明不确定性。"
            "相同三组合在不同配置下结果可以不同，不得因单次质量失败或 OOM 禁用整个组合。"
            "先核对 parameter_changes 中每个实际改动，不能把历史修复的改动说成当前提案的改动；"
            "历史代价是整个配置变更的观测关联，不能归因于某个单独参数。"
            "检查父点的性能优势能否承受历史修复代价，不能只看修复后是否更快于父点。"
            "质量门槛见 quality_gate，可行前沿见 performance，必须据此评估目标。"
            "历史 Trial 是否达标以 quality_feasible 为准；false 表示未通过所有质量门槛，"
            "null 表示未知。不得把部分指标通过、运行完成或性能好误称为质量达标。"
            "只有 repair_warmup_guard.applied 为真才表示显存检查实际调低了预算；"
            "触发阈值不是参数下界，不得把 0.94 等触发条件解读为 gpu_memory_utilization 最小值。"
            "显存失败条件对照不是失败预测：sampler预热OOM也可能发生在中等batch，"
            "不能因bs低于保护阈值、采用GPTQ/AWQ或guard未触发而断言安全。"
            "对照DP/TP、enforce_eager、batch、利用率和三参数；不同执行模式的成功不能直接证明当前配置安全。"
            "sampler临时显存不足与KV容量不足必须区分：降低利用率可给前者留余量，却减少KV预算，"
            "不得把降利用率当成KV不足的修复或保证0.92一定成功。"
            "仅在当前scope允许时建议改利用率/并发/执行模式；否则指出范围限制及可选三组合，不能越权改参数。"
            "必须按实际配置评估。反馈简短具体，最多四项。"
            "feedback 每项是结构化对象：category 为 quality/performance/runtime/scope；"
            "evidence_type 为 measured/predicted/unknown；parameters 列出关联参数；"
            "cited_trial_ids 列出该项引用的实测 Trial，并纳入顶层引用列表；"
            "cited_memory_ids 列出该项引用的 historical_quality_memory observation memory_id，无引用时用空数组。"
            "外部 observation 的 source Trial 不是本次 Trial，禁止混用编号；保留相似度、环境、质量合同、适用区域及反例，"
            "负面或优选三参数历史经验只辅助判断，不是全局禁用规则或当前成功保证。"
            "finding 用一句话描述具体判断及适用条件，suggestion 用一句话描述范围内的修订方向，"
            "没有修订建议时用空字符串。避免重复抄写完整配置和历史。"
            "measured 表示该项依据历史实测，不表示当前提案已被实测或因果已获证明；"
            "revise 至少需要一项 measured 反馈，包含实测引用和非空的具体修订建议。"
            "这是 proposal_review 阶段；只返回符合 schema 的 JSON。"
        )
        if route is not None:
            system = (
                "你是RepairBO执行前审计员。本次只执行" + route + "审计。仅使用给定证据。"
                "输出既定JSON；verdict=accept表示未发现足够拒绝依据，不保证质量或运行成功；"
                "verdict=revise表示拒绝当前完整配置并请求repair二次修改。"
                "feedback必须结构化，引用所给Trial或memory，建议必须在scope内；"
                "无域内修订方案时明确说明范围限制。不得编造实测或因果关系。"
            )
            if route == "runtime":
                system += (
                    "只判断运行失败，不以质量或性能差为由拒绝。仅本次真实runtime_failures可支持拒绝。"
                    "逐条比较运行失败条件表：DP/TP、模型/权重/KV表示、后端、eager、并发、token预算、"
                    "模型长度、显存利用率和workload。相同三组合或一次OOM不意味着整个组合不可运行。"
                    "失败迁移必须有明确机制和具体条件；必须讨论差异为何保留或解除失败机制，并检查成功反例。"
                    "missing_conditions非空、关键条件未知、机制不明或存在无法解释的成功反例时accept并标记unknown。"
                    "不得自行扩大阈值或把缺失值视作相同。sampler临时OOM和KV容量不足分开分析。"
                    "每条拒绝feedback类别只能是runtime，evidence_type=measured，parameters必须列举运行失败条件表中的全部字段，缺失字段时不能拒绝，"
                    "finding说明适用条件、差异及反例，suggestion给出域内修订方向。"
                    "performance_risk和quality_risk填unknown。"
                )
            else:
                system += (
                    "只审计性能潜力。综合判断尚未执行的repair完整候选是否可能成为或扩展Pareto前沿点，"
                    "不得把审计缩减为父节点的某一个优势方向。candidate_pareto_potential是程序使用三目标"
                    "预测均值及逐轴两倍标准差乐观界与当前实测前沿计算的辅助说明，不是拒绝结论或概率保证。"
                    "必须联合审计完整的QPS、平均延迟和P95三个性能方向及其权衡。"
                    "结合workload、current_pareto_front中的当前可行前沿实测性能、父点实测性能、"
                    "尚未实测修复完整配置的当前BO预测均值/方差/标准差判断。三个坐标均越大越好，延迟坐标使用负值。"
                    "比较的是修复后对前沿的潜力，不要求比父点更快，也不要求同时胜过每个前沿点。"
                    "只预测修复配置；父点和前沿仅使用实测值，不需要也不得推算它们的预测值或预测误差。"
                    "只有有证据说明考虑不确定性后仍缺乏前沿增益机会，才revise。"
                    "不能仅凭预测均值暂时落后、单指标轻微变差或低EHVI拒绝；标准差不是已校准保证。"
                    "缺少修复预测或前沿时accept并标记unknown；不要请求父点或前沿的预测，也不使用历史代理预测。"
                    "每条拒绝feedback类别只能是performance，可以用predicted证据，但须引用提供的父点/前沿Trial，"
                    "finding说明三个指标、候选性来源、标准差和前沿的比较以及为何无潜力，suggestion给出域内修改方向。"
                    "runtime_risk和quality_risk填unknown。"
                )
        prompt = render_tagged_sections([
            ("审计路径", route), ("待评审提案", proposal), ("修复范围", scope),
            ("实测证据", evidence), ("性能预测非实测", performance),
            ("历史配对修复效果", transitions),
            ("显存失败条件对照", memory_failure_comparisons(proposal, evidence)),
        ], compact=True)
        error = "review_client_unavailable"
        attempts: list[dict[str, Any]] = []
        for attempt in range(2 if self.client is not None else 0):
            response = None
            payload = None
            try:
                response = self.client.complete_structured(system=system, user=prompt if not attempts else prompt + render_tagged_sections([("评审格式校验失败，请修正后重新返回完整JSON", attempts[-1])], compact=True), response_schema=schema)
                payload = response.get("structured_output", response.get("output", response.get("result", response)))
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if not isinstance(payload, dict) or set(payload) != set(schema["required"]):
                    raise ValueError("review fields do not match schema")
                for name in ("verdict", "quality_risk", "performance_risk", "runtime_risk"):
                    if payload[name] not in schema["properties"][name]["enum"]:
                        raise ValueError(f"invalid {name}")
                feedback = payload["feedback"]
                if not isinstance(feedback, list) or len(feedback) > 4:
                    raise ValueError("invalid review feedback: expected <=4 structured items")
                known = set(evidence.get("records", {})) | {
                    str(t[key]) for t in transitions for key in ("parent_trial_id", "trial_id")
                }
                citations = payload["cited_trial_ids"]
                if not isinstance(citations, list) or any(not isinstance(x, str) or x not in known for x in citations):
                    raise ValueError("review cites unknown trials")
                for index, item in enumerate(feedback):
                    if not isinstance(item, dict) or set(item) != set(feedback_properties):
                        raise ValueError(f"invalid review feedback[{index}]: fields do not match schema")
                    for name in ("category", "evidence_type"):
                        if item[name] not in feedback_properties[name]["enum"]:
                            raise ValueError(f"invalid review feedback[{index}].{name}")
                    for name in ("finding", "suggestion"):
                        if not isinstance(item[name], str):
                            raise ValueError(f"invalid review feedback[{index}].{name}: expected string")
                    if not item["finding"].strip():
                        raise ValueError(f"invalid review feedback[{index}]: empty finding")
                    parameters = item["parameters"]
                    if not isinstance(parameters, list) or any(not isinstance(x, str) for x in parameters):
                        raise ValueError(f"invalid review feedback[{index}].parameters")
                    configuration = proposal.get("configuration", {})
                    if any(x not in configuration for x in parameters):
                        raise ValueError(f"invalid review feedback[{index}]: unknown parameter")
                    refs = item["cited_trial_ids"]
                    if not isinstance(refs, list) or any(not isinstance(x, str) or x not in known or x not in citations for x in refs):
                        raise ValueError(f"invalid review feedback[{index}]: unknown trials or missing top-level citation")
                    known_memories = {r['observation']['memory_id'] for r in evidence.get('historical_quality_memory', {}).get('observations', [])}
                    memory_refs = item['cited_memory_ids']
                    if not isinstance(memory_refs, list) or any(not isinstance(x, str) or x not in known_memories for x in memory_refs):
                        raise ValueError(f"invalid review feedback[{index}]: unknown memory citation")
                    if item["evidence_type"] == "measured" and not (refs or memory_refs):
                        raise ValueError(f"invalid review feedback[{index}]: measured finding requires citations")
                confidence = payload["confidence"]
                if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("invalid review confidence")
                if payload["verdict"] == "revise" and not any(
                    (x["evidence_type"] == "measured" or (route == "performance" and x["evidence_type"] == "predicted")) and x["suggestion"].strip() and (x["cited_trial_ids"] or x['cited_memory_ids'])
                    for x in feedback
                ):
                    raise ValueError("revision requires actionable feedback and measured citations")
                return {**payload, "status": "reviewed", "usage": response.get("usage"), "format_retry_attempts": attempts}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:400]
                attempts.append({"attempt": attempt + 1, "error": error,
                                 "invalid_response": str(payload if payload is not None else response)[:2400]})
        # A transport/schema failure is not evidence that a legal point is bad.
        return {
            "status": "unavailable", "error": error, "verdict": None,
            "format_retry_attempts": attempts,
            "quality_risk": "unknown", "performance_risk": "unknown", "runtime_risk": "unknown",
            "feedback": [],
            "cited_trial_ids": [], "confidence": 0.0,
        }


# Explicit execution conditions, never a learned global blacklist.
RUNTIME_CONDITIONS = (
    'tp', 'data_parallel_size', 'quantization', 'dtype', 'kv_cache_dtype',
    'model_impl', 'data_parallel_backend', 'enforce_eager', 'max_num_seqs',
    'max_num_batched_tokens', 'max_model_len', 'gpu_memory_utilization',
)


def runtime_condition_evidence(proposal, evidence):
    configuration = proposal.get('configuration', {})
    result = []
    for failure in evidence.get('runtime_failures', []):
        trial_id = failure['trial_id']
        previous = evidence.get('records', {}).get(trial_id, {}).get('configuration', {})
        conditions = {key: {'failed': previous.get(key), 'proposed': configuration.get(key)}
                      for key in RUNTIME_CONDITIONS}
        result.append({
            'trial_id': trial_id, 'failure_reason': failure.get('failure_reason'),
            'conditions': conditions,
            'missing_conditions': [k for k in RUNTIME_CONDITIONS
                                   if k not in previous or k not in configuration],
            'different_conditions': [k for k in RUNTIME_CONDITIONS
                                     if k in previous and k in configuration and previous[k] != configuration[k]],
        })
    return result


def _has_prediction(value):
    if not isinstance(value, Mapping):
        return False
    predicted = value.get('predicted_objectives')
    uncertainty = value.get('objective_uncertainty')
    if not isinstance(predicted, (list, tuple)) or not predicted:
        return False
    if not isinstance(uncertainty, (list, tuple)) or len(uncertainty) != len(predicted):
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        and math.isfinite(item) and (key != 'objective_uncertainty' or item >= 0)
        for key, values in (
            ('predicted_objectives', predicted),
            ('objective_uncertainty', uncertainty),
        )
        for item in values
    )


PERFORMANCE_OBJECTIVES = (
    {
        'index': 0,
        'coordinate': 'qps',
        'metric': 'query_throughput_qps',
        'direction': 'maximize',
    },
    {
        'index': 1,
        'coordinate': 'negative_mean_latency',
        'metric': 'mean_query_execution_latency_ms',
        'direction': 'minimize',
    },
    {
        'index': 2,
        'coordinate': 'negative_p95_latency',
        'metric': 'p95_query_execution_latency_ms',
        'direction': 'minimize',
    },
)


def _objective_vector_from_measurement(measured: Mapping[str, Any]) -> list[float] | None:
    try:
        vector = [
            float(measured['query_throughput_qps']),
            -float(measured['mean_query_execution_latency_ms']),
            -float(measured['p95_query_execution_latency_ms']),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def candidate_pareto_potential(
    evidence: Mapping[str, Any], performance: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe joint three-objective Pareto potential without deciding the audit."""
    front_rows = evidence.get('current_pareto_front')
    front_rows = front_rows if isinstance(front_rows, list) else []
    front: list[tuple[str, list[float]]] = []
    for row in front_rows:
        if not isinstance(row, Mapping):
            continue
        vector = row.get('objective_vector')
        if not (
            isinstance(vector, (list, tuple)) and len(vector) == 3
            and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(item) for item in vector)
        ):
            vector = _objective_vector_from_measurement(row.get('measured', {}))
        if vector is not None:
            front.append((str(row.get('trial_id', '')), [float(item) for item in vector]))

    if not _has_prediction(performance) or len(performance['predicted_objectives']) != 3 or not front:
        return {
            'status': 'unavailable_missing_prediction_or_front',
            'basis': 'joint_three_objective_candidate_prediction_vs_measured_feasible_pareto',
            'mechanical_hint_only': True,
            'audit_objectives': [dict(item) for item in PERFORMANCE_OBJECTIVES],
        }

    predicted_mean = [float(item) for item in performance['predicted_objectives']]
    uncertainty = [float(item) for item in performance['objective_uncertainty']]
    optimistic = [mean + 1.0 * std for mean, std in zip(predicted_mean, uncertainty)]

    def dominators(candidate: list[float]) -> list[str]:
        return [
            trial_id for trial_id, point in front
            if all(front_value >= candidate_value
                   for front_value, candidate_value in zip(point, candidate))
            and any(front_value > candidate_value
                    for front_value, candidate_value in zip(point, candidate))
        ]

    def duplicates(candidate: list[float]) -> list[str]:
        return [
            trial_id for trial_id, point in front
            if all(math.isclose(front_value, candidate_value, rel_tol=1e-12, abs_tol=1e-12)
                   for front_value, candidate_value in zip(point, candidate))
        ]

    mean_dominated_by = dominators(predicted_mean)
    optimistic_dominated_by = dominators(optimistic)
    mean_duplicates = duplicates(predicted_mean)
    optimistic_duplicates = duplicates(optimistic)
    mean_candidate = not mean_dominated_by and not mean_duplicates
    optimistic_candidate = not optimistic_dominated_by and not optimistic_duplicates
    if mean_candidate:
        status = 'predicted_mean_is_pareto_candidate'
    elif optimistic_candidate:
        status = 'pareto_candidate_within_one_sigma_optimistic_bound'
    else:
        status = 'dominated_or_duplicate_even_at_one_sigma_optimistic_bound'
    return {
        'status': status,
        'basis': 'joint_three_objective_candidate_prediction_vs_measured_feasible_pareto',
        'mechanical_hint_only': True,
        'uncertainty_is_not_a_calibrated_guarantee': True,
        'optimistic_standard_deviation_multiplier': 1.0,
        'audit_objectives': [dict(item) for item in PERFORMANCE_OBJECTIVES],
        'predicted_mean_objective_vector': predicted_mean,
        'one_sigma_optimistic_objective_vector': optimistic,
        'predicted_mean_dominated_by_trial_ids': mean_dominated_by,
        'one_sigma_optimistic_dominated_by_trial_ids': optimistic_dominated_by,
        'predicted_mean_duplicate_trial_ids': mean_duplicates,
        'one_sigma_optimistic_duplicate_trial_ids': optimistic_duplicates,
        'predicted_mean_is_pareto_candidate': mean_candidate,
        'one_sigma_optimistic_is_pareto_candidate': optimistic_candidate,
    }


def performance_audit_inputs(
    evidence: Mapping[str, Any], performance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    potential = candidate_pareto_potential(evidence, performance)
    annotated_evidence = {**dict(evidence), 'candidate_pareto_potential': potential}
    annotated_performance = {
        **dict(performance),
        'audit_objectives': [dict(item) for item in PERFORMANCE_OBJECTIVES],
        'candidate_pareto_potential': potential,
    }
    return annotated_evidence, annotated_performance, potential


class DualPathRepairReviewer(RepairProposalReviewer):
    """Independent runtime and performance audits with conservative rejection gates."""

    def evaluate(self, *, proposal, evidence, performance, scope, transitions):
        conditions = runtime_condition_evidence(proposal, evidence)
        route_evidence = {**evidence, 'runtime_condition_comparisons': conditions}
        performance_evidence, performance_prediction, pareto_potential = performance_audit_inputs(
            route_evidence, performance,
        )
        results = {}
        for route in ('runtime', 'performance'):
            assessment = super().evaluate(
                proposal=proposal,
                evidence=(performance_evidence if route == 'performance' else route_evidence),
                performance=(performance_prediction if route == 'performance' else performance),
                scope=scope, transitions=[], route=route,
            )
            if route == 'performance':
                assessment = {**assessment, 'candidate_pareto_potential': pareto_potential}
            if assessment['verdict'] == 'revise':
                actionable = [f for f in assessment['feedback']
                              if f['category'] == route and f['suggestion'].strip()]
                supported = False
                if route == 'runtime':
                    for item in actionable:
                        for comparison in conditions:
                            # Reject only when the reviewer explicitly covers every known
                            # condition, including differences. Missing conditions abstain.
                            if (item['evidence_type'] == 'measured'
                                and comparison['trial_id'] in item['cited_trial_ids']
                                and comparison['failure_reason']
                                and not comparison['missing_conditions']
                                and set(RUNTIME_CONDITIONS).issubset(item['parameters'])):
                                supported = True
                else:
                    supported = bool(actionable and _has_prediction(performance_prediction)
                                     and performance_evidence.get('current_pareto_front'))
                if not supported:
                    assessment = {**assessment, 'verdict': 'accept',
                                  'rejection_withheld': 'insufficient_route_specific_evidence',
                                  route + '_risk': 'unknown'}
            results[route] = assessment
        rejected = [key for key, value in results.items() if value['verdict'] == 'revise']
        unavailable = [key for key, value in results.items() if value['status'] == 'unavailable']
        return {
            'schema': 'repair-dual-audit.v1', 'audits': results,
            'status': 'reviewed' if rejected or not unavailable else 'unavailable',
            'verdict': 'revise' if rejected else ('accept' if not unavailable else None),
            'rejected_routes': rejected, 'unavailable_routes': unavailable,
            'quality_risk': 'unknown',
            'runtime_risk': results['runtime']['runtime_risk'],
            'performance_risk': results['performance']['performance_risk'],
            'feedback': [{**f, 'audit_route': key} for key in rejected for f in results[key]['feedback']],
            'cited_trial_ids': sorted({ref for value in results.values() for ref in value['cited_trial_ids']}),
            'confidence': min(value['confidence'] for value in results.values()),
            'format_retry_attempts': [dict(a, audit_route=key) for key, value in results.items()
                                      for a in value.get('format_retry_attempts', [])],
            'candidate_pareto_potential': pareto_potential,
        }
