"""Repair patches with complete-configuration validation and concrete feedback."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from agents.tuning.candidate_optimizer.src.services.prompt_format import render_tagged_sections
from .repair_stages import repair_stage


def propose_changes(*, client: Any, strategy: str, parent: Mapping[str, Any],
                    scope: Any, space: Any, evidence: Mapping[str, Any],
                    cases: Mapping[str, Any], examples: list[dict[str, Any]],
                    validate: Callable[[Mapping[str, Any]], tuple[Any, list[str]]]) -> dict[str, Any]:
    properties = {}
    for field in scope.fields:
        parameter = space.by_name[field]
        if getattr(parameter, "decision", None) == "fixed":
            continue
        if parameter.values is not None:
            properties[field] = {"enum": list(parameter.values)}
        else:
            properties[field] = {"type": "integer" if parameter.kind == "integer" else "number",
                                 **{key: parameter.domain[value] for key, value in
                                    (("minimum", "lower"), ("maximum", "upper")) if value in parameter.domain}}
    schema = {"type": "object", "additionalProperties": False, "required": ["changes"],
              "properties": {"changes": {"type": "object", "properties": properties,
                                         "minProperties": 1, "additionalProperties": False}}}
    system = (
        "你是RepairBO修复提案员。输出JSON {\"changes\":{参数名:修改后的值}}，只列真正修改的字段。"
        "不输出候选编号，不必完整列出所有可修参数。未列出的活跃参数保持锚点值。"
        "父开关改变后失活的条件参数由程序归一化为执行器关闭值或Range参考值，无需列入changes，"
        "这不扩大允许修改范围；仍活跃的冻结参数不能改变。"
        "程序会合并成完整配置并校验范围、机械约束、冻结字段和重复配置。"
        "发生冲突时按具体校验反馈修改，不要原样重复失败输出。"
        "仅依据提供的实测证据，质量硬门槛全部通过并改善可行Pareto前沿才算成功。"
        "失败经验有条件，不得因单次OOM禁用整个参数组合；预测不是实测。"
        "historical_quality_memory 是外部负面/优选三参数 observation；比较其 workload、环境、质量合同、适用区域及反例，"
        "不要把外部 Trial 编号当成本次实测，也不要把条件经验变成全局禁用或成功保证。"
        "外部经验已按applicability_ranking机械排序，优先参考适用区域匹配度高的经验；该排序不是质量成功概率。"
        "推荐时优先参考historical_quality_memory中的历史成功经验：先检查quality_triplet_preference的triplet、"
        "applicability_ranking、支持配置的实测性能及反例，再结合本次实测结果决定方案，不要先从末尾示例取前三项。"
        "在满足当前阶段规则的前提下，应优先将与当前workload、环境和锚点配置高度相似、且具有良好实测性能的历史成功组合纳入推荐。"
        "若某历史组合适用性领先、已有质量成功与高性能证据，且无当前条件下的具体反证，应优先给它一个候选位置，"
        "不要仅因本次尚未测过该组合，或泛泛认为量化有风险，就将其排除。"
        "优先参考不等于无条件照搬：存在具体质量反例、环境不兼容、当前阶段禁止或完整配置重复时，才根据这些证据调整取舍。"
        "比较经验及示例时，应结合配置区域相似度、workload特征、质量余量与反例，以及支持配置的实测吞吐、平均延迟和P95延迟。"
        "历史性能属于其完整配置和测量条件，不是当前锚点替换三参数后的性能保证；缺少可比数据时不要编造性能结论。"
        "R1须复用已有质量成功的可修字段取值；成功证据可以来自本次实测，"
        "也可以来自提供的外部历史成功经验。外部经验仅满足R1复用资格，"
        "仍须机械校验和双路审计，不保证当前配置质量或性能成功。"
        "R2须与本次已执行R1的可修字段取值不同。"
        "R3的performance_recovery保持表示参数不变且最多改两个允许字段。"
        "示例只是参考，不要求在示例中选；域、范围、分支规则与校验反馈是约束。"
        "合法修改示例的排列顺序不代表推荐优先级、相似度或性能排名，不得因示例靠前就优先选择。"
        "应查看全部示例及其关联实测证据，根据适用性、相似度和性能权衡选择方案；示例只有参数值时，到实测证据中查找对应组合的支持配置。"
        "prior_repair_reflections 是此前 episode 从实测结果提炼的经验，可包含成功方向、失败反例和适用边界；仅用于当前阶段合法候选的比较与排序。"
        "不能当作硬约束、当前配置的质量或性能保证，并须核对适用条件与反例。"
    )
    prompt = render_tagged_sections([
        ("当前修复阶段", repair_stage(strategy)), ("当前策略", strategy), ("完整锚点配置", parent["candidate"]),
        ("可修改范围与机械约束", scope.contract()), ("实测证据", evidence),
        ("质量硬门槛", cases.get("quality_gate", {})),
        ("参数简介", cases.get("parameter_guide", {})),
        ("系统预热显存检查例外", cases.get("repair_warmup_guard", {})),
        ("prior_repair_reflections", cases.get("prior_repair_reflections", [])),
        ("上下文", {k: cases.get(k) for k in ("workload_features", "paired_repair_effects",
                   "proposal_feedback", "skipped_a_review", "repair_target", "auditor_critique")}),
        ("合法修改示例非候选编号", examples),
    ], compact=True)
    attempts = []
    for index in range(2 if client is not None else 0):
        payload = None
        try:
            response = client.complete_structured(system=system, user=prompt, response_schema=schema)
            payload = response.get("structured_output", response)
            if not isinstance(payload, Mapping) or set(payload) != {"changes"}:
                selected, errors = None, ['顶层必须仅包含changes对象；不接受候选编号或完整目标配置。']
            elif not isinstance(payload["changes"], Mapping) or not payload["changes"]:
                selected, errors = None, ['changes必须是非空参数改动对象。']
            else:
                selected, errors = validate(payload["changes"])
            attempts.append({"attempt": index + 1, "response": payload, "valid": not errors,
                             "validation_errors": errors, "transport_metadata": {"usage": response.get("usage")}})
            if selected is not None and not errors:
                return {"candidate": selected, "agent_output": payload,
                        "selection_mode": "llm_validated_parameter_changes", "llm_attempts": attempts}
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"[:600]]
            attempts.append({"attempt": index + 1, "response": payload, "valid": False, "validation_errors": errors})
        prompt += render_tagged_sections([("完整配置校验反馈", {
            "rejected_changes": payload, "conflicts": errors,
            "instruction": "针对这些具体冲突返回修改后的changes；完整配置与A重复时必须改变至少一个可修字段，而非仅省略字段。",
        })], compact=True)
    return {"candidate": None, "agent_output": None, "selection_mode": "llm_invalid_after_retry", "llm_attempts": attempts}


def propose_three_changes(*, score, **kwargs):
    """Request one structured batch, validate every patch, then score together."""
    original_client = kwargs.pop("client")
    original_validate = kwargs.pop("validate")
    batch = []
    audits = []

    class BatchClient:
        def complete_structured(self, *, system, user, response_schema):
            patch_schema = response_schema["properties"]["changes"]
            schema = {"type": "object", "additionalProperties": False,
                      "required": ["proposals"], "properties": {"proposals": {
                          "type": "array", "minItems": 3, "maxItems": 3,
                          "items": patch_schema}}}
            system = system.replace('输出JSON {"changes":{参数名:修改后的值}}',
                                    '输出JSON {"proposals":[{参数名:修改后的值},{参数名:修改后的值},{参数名:修改后的值}]}')
            system += "本次必须一次提出三个不同的完整配置修改方案，三个方案将由当前BO统一预测性能并择一。每项是changes字段原本对应的参数修改对象；不得重复。"
            system += "不要机械照抄示例列表的前三项；应比较全部可用历史经验和示例，优先考虑适用性强且有实测性能支持的方案，再选择三个不同方案供BO比较。"
            response = original_client.complete_structured(system=system, user=user, response_schema=schema)
            payload = response.get("structured_output", response)
            return {**response, "structured_output": {"changes": {"__batch__": payload}}}

    def validate(changes):
        batch.clear()
        payload = changes.get("__batch__", {})
        patches = payload.get("proposals") if isinstance(payload, Mapping) else None
        if not isinstance(patches, list) or len(patches) != 3:
            return None, ["必须返回proposals数组，恰好包含三个不同修改方案。"]
        errors = []
        round_audits = []
        for i, patch in enumerate(patches):
            if not isinstance(patch, Mapping) or not patch:
                item, issues = None, ["必须是非空修改对象"]
            else:
                item, issues = original_validate(patch)
            issues = list(issues)
            if item is None and not issues:
                issues.append("机械校验没有返回可执行配置")
            if item is not None and not issues and any(item["candidate"] == c["candidate"] for c in batch):
                issues.append("与本批另一方案重复")
            round_audits.append({"proposal_index": i, "changes": patch, "passed": not issues, "errors": issues})
            errors.extend(f"方案{i+1}: {e}" for e in issues)
            if item is not None and not issues:
                batch.append({**item, "proposal_index": i})
        audits.append(round_audits)
        # A partially valid batch is sufficient. Only an entirely invalid batch
        # is sent back with all mechanical findings for the second recommendation.
        return (batch[0], []) if batch else (None, errors)

    choice = propose_changes(client=BatchClient() if original_client else None,
                             validate=validate, **kwargs)
    choice["mechanical_audit_batches"] = audits
    if not choice.get("candidate"):
        return {**choice, "proposal_batch": batch, "selection_mode": "incomplete_three_proposal_batch"}
    try:
        ranked = score(batch)
        import math
        if len(ranked) != len(batch) or not ranked or any(not isinstance(c.get("ehvi"), (int, float))
                                   or not math.isfinite(c["ehvi"])
                                   or c.get("predicted_objectives") is None for c in ranked):
            raise ValueError("Finite current BO EHVI predictions are required for every passing candidate")
        best = max(range(len(ranked)), key=lambda i: ranked[i]["ehvi"])
    except Exception as exc:
        return {**choice, "candidate": None, "selection_mode": "batch_prediction_unavailable",
                "prediction_error": str(exc), "proposal_batch": batch}
    return {**choice, "candidate": ranked[best],
            "selection_mode": "three_llm_proposals_bo_ehvi",
            "proposal_batch": ranked, "selected_batch_index": best}
