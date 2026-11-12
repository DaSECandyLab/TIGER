"""Canonical tagged prompt layout for every Candidate BO LLM stage."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


PROMPT_FORMAT_SCHEMA = "sembench.bo.tagged-prompt.v1"

_FIELD_SECTION = {
    "instruction": "任务目标",
    "task": "任务目标",
    "objective_role": "任务目标",
    "optimization_direction": "任务目标",
    "decision_stage": "任务目标",
    "attempt": "任务目标",
    "current_round": "任务目标",
    "optimization_objectives": "任务目标",
    "quality_constraint": "注意事项",
    "candidate_preflight_contract": "注意事项",
    "historical_reflection_policy": "注意事项",
    "correction_instruction": "纠错信息",
    "claim_revision_contract": "纠错信息",
    "reflect_check_review": "纠错信息",
    "reflect_check_reviews": "纠错信息",
    "previous_rejected_output": "纠错信息",
    "previous_validation_error": "纠错信息",
    "previous_validation_errors": "纠错信息",
    "review_reasons": "纠错信息",
    "mechanical_validation_errors": "纠错信息",
    "guidance_preservation_errors": "纠错信息",
    "multiple_changes_require_minimality_review": "纠错信息",
    "rejected_execution_feedback": "纠错信息",
    "rejected_repeated_range": "纠错信息",
    "repetition_audit": "纠错信息",
    "semantic_change_contract": "纠错信息",
    "static_workload": "静态workload特征",
    "static_workload_features": "静态workload特征",
    "dynamic_workload": "动态workload特征",
    "dynamic_workload_features": "动态workload特征",
    "latest_dynamic_workload_features": "动态workload特征",
    "recent_three_trial_trend": "动态workload特征",
    "previous_trial_dynamic_anchor": "动态workload特征",
    "performance_observation_dynamic_signatures": "动态workload特征",
    "pareto_anchor": "当前帕累托前沿点",
    "pareto_structural_stability_guardrails": "当前帕累托前沿点",
    "current_pareto_front": "当前帕累托前沿点",
    "prior_structured_reflections": "历史结构化reflect",
    "historical_executed_trial_reflections": "历史结构化reflect",
    "temp_observations": "历史结构化reflect",
    "frozen_range": "冻结Range",
    "frozen_range_domain": "冻结Range",
    "frozen_range_action_contract": "冻结Range",
    "required_candidate_parameters": "冻结Range",
    "required_structural_guardrail_parameters": "冻结Range",
    "current_candidate": "输入配置",
    "baseline_observation": "当前执行Trial",
    "current_trial": "当前执行Trial",
    "current_result_analysis": "当前执行Trial",
    "draft_cause_analysis": "上游Agent结构化输出",
    "mechanical_evidence": "机械事实证据",
    "static_knob_semantics": "静态参数语义",
    "directional_outputs": "三方向Agent结构化输出",
    "merged_output": "汇总Agent结构化输出",
    "prior_assessment": "上游Agent结构化输出",
    "assessment": "上游Agent结构化输出",
    "metric_guidance_strategy": "上游Agent结构化输出",
    "diagnosed_bottleneck_mechanisms": "瓶颈诊断",
    "selected_advisory_observation_type": "瓶颈诊断",
    "selected_dynamic_observation_type": "瓶颈诊断",
    "dynamic_mechanism_evidence_catalog": "瓶颈诊断",
    "retrieved_dynamic_observations": "动态workload observation",
    "raw_capacity_observations": "历史失败observation",
    "raw_quality_failure_observations": "历史失败observation",
    "current_run_capacity_failure_trajectory": "历史失败observation",
    "current_run_quality_failure_trajectory": "历史失败observation",
    "failure_evidence": "历史失败observation",
    "quality_evidence": "历史失败observation",
    "candidate_evidence_gate": "历史失败observation",
    "cleaned_capacity_failure_experience": "历史失败observation",
    "cleaned_quality_failure_experience": "历史失败observation",
    "observations": "历史observation",
    "random_raw_observations": "历史observation",
    "retrieved_source_catalog": "历史observation",
    "linked_experience_evidence": "历史experience memory",
    "static_parameter_knowledge_rag": "静态知识",
    "environment": "运行环境",
    "environment_and_workload": "运行环境与workload上下文",
    "bo_progress": "BO运行进度",
    "guidance_progress_audit": "BO运行进度",
    "search_scope_audit": "BO运行进度",
    "round_state": "BO运行进度",
    "routing_state_after_classification": "BO运行进度",
    "available_metric_agents": "BO运行进度",
    "classification_contract": "注意事项",
    "workload_evidence_contract": "注意事项",
    "termination_constraints": "注意事项",
    "upstream_llm_guidance_preference": "上游LLM guidance范围",
    "protected_by_upstream_quality_repair": "上游质量保护",
    "repair_scope": "Repair范围",
    "repair_strategy": "Repair任务",
    "current_parent_trial": "待修复Trial",
    "quality_success_cases": "质量成功案例",
    "quality_failure_cases": "质量失败案例",
    "eligible_candidates": "Repair候选",
    "auditor_critique": "Repair审计反馈",
    "repair_decisions": "Repair决策",
    "measured_repair_results": "Repair实测结果",
}

_SECTION_ORDER = (
    "任务目标",
    "注意事项",
    "Repair任务",
    "Repair范围",
    "运行环境",
    "静态workload特征",
    "动态workload特征",
    "当前帕累托前沿点",
    "输入配置",
    "当前执行Trial",
    "待修复Trial",
    "质量成功案例",
    "质量失败案例",
    "Repair候选",
    "Repair决策",
    "Repair实测结果",
    "Repair审计反馈",
    "三方向Agent结构化输出",
    "汇总Agent结构化输出",
    "上游Agent结构化输出",
    "瓶颈诊断",
    "历史结构化reflect",
    "历史observation",
    "动态workload observation",
    "历史失败observation",
    "历史experience memory",
    "静态知识",
    "机械事实证据",
    "静态参数语义",
    "上游LLM guidance范围",
    "上游质量保护",
    "冻结Range",
    "BO运行进度",
    "运行环境与workload上下文",
    "纠错信息",
    "其它上下文",
    "输出结构",
)

_TAGGED_SECTION = re.compile(
    r"<(?P<tag>[^<>/]+)>\n(?P<body>.*?)\n</(?P=tag)>", re.DOTALL
)
_NAMED_JSON = re.compile(
    r"(?P<prefix>.*?)(?P<label>[A-Z][A-Z0-9_]+)="
    r"(?P<payload>\{.*\}|\[.*\])\s*$",
    re.DOTALL,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _is_empty(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value):
        return True
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, (list, tuple)):
        return not value
    return False


def render_tagged_sections(
    sections: Sequence[tuple[str, Any]],
    *,
    compact: bool = False,
) -> str:
    """Render ordered XML-like sections with JSON bodies."""

    rendered: list[str] = []
    for tag, value in sections:
        if _is_empty(value):
            continue
        body = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if compact else _json(value)
        )
        rendered.append(f"<{tag}>\n{body}\n</{tag}>")
    return "\n\n".join(rendered)


def render_system_prompt(
    *,
    role: str,
    responsibilities: str,
    attention: Sequence[str] = (),
) -> str:
    """Render the common role/attention system-message contract."""

    common_attention = [
        "Only use evidence explicitly supplied in the user prompt.",
        "Treat the frozen Range and mechanical facts as authoritative.",
        "Return only a JSON object matching the API-provided response schema.",
        "Do not add Markdown, prose outside the JSON object, or undeclared fields.",
    ]
    return render_tagged_sections(
        (
            (
                "角色与职责",
                {
                    "role": role,
                    "responsibilities": responsibilities,
                },
            ),
            ("注意事项", {"rules": [*common_attention, *map(str, attention)]}),
        )
    )


def _split_environment_and_workload(
    value: Mapping[str, Any], grouped: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    remaining = dict(value)
    for key, section in (
        ("environment", "运行环境"),
        ("static_workload", "静态workload特征"),
        ("latest_dynamic_workload_features", "动态workload特征"),
    ):
        if key in remaining:
            item = remaining.pop(key)
            if not _is_empty(item):
                grouped.setdefault(section, {})[key] = item
    for key in (
        "current_run_capacity_failures",
        "current_run_quality_failures",
    ):
        if key in remaining:
            item = remaining.pop(key)
            if not _is_empty(item):
                grouped.setdefault("历史失败observation", {})[key] = item
    return remaining


def render_user_prompt(
    payload: Mapping[str, Any],
    *,
    response_schema: Mapping[str, Any],
    task_name: str | None = None,
    attention: Sequence[str] = (),
    section_order: Sequence[str] | None = None,
) -> str:
    """Group a payload using the calling Agent's own semantic section layout."""

    grouped: dict[str, dict[str, Any]] = {}
    if task_name:
        grouped.setdefault("任务目标", {})["stage"] = str(task_name)
    if attention:
        grouped.setdefault("注意事项", {})["rules"] = list(map(str, attention))
    for raw_key, value in payload.items():
        key = str(raw_key)
        if _is_empty(value):
            continue
        if key == "environment_and_workload" and isinstance(value, Mapping):
            remaining = _split_environment_and_workload(value, grouped)
            if remaining:
                grouped.setdefault("运行环境与workload上下文", {})[key] = remaining
            continue
        section = _FIELD_SECTION.get(key, "其它上下文")
        grouped.setdefault(section, {})[key] = value
    grouped["输出结构"] = {
        "prompt_schema": PROMPT_FORMAT_SCHEMA,
        "response_schema": dict(response_schema),
        "output_contract": (
            "Return exactly one schema-valid JSON object; do not reproduce prompt tags."
        ),
    }
    requested_order = [
        str(section)
        for section in (section_order or _SECTION_ORDER)
        if str(section) != "输出结构"
    ]
    # Agent-specific layouts own their primary order. Any genuinely supplied
    # extension field remains visible instead of being silently dropped.
    trailing = [
        section
        for section in _SECTION_ORDER
        if section not in requested_order
        and section != "输出结构"
        and grouped.get(section)
    ]
    final_order = [*requested_order, *trailing, "输出结构"]
    return render_tagged_sections(
        tuple((section, grouped.get(section)) for section in final_order)
    )


def parse_named_json_prompt(raw: str) -> tuple[dict[str, Any], str | None]:
    """Recover legacy ``PROSE LABEL={...}`` prompts during migration."""

    match = _NAMED_JSON.fullmatch(str(raw).strip())
    if match is None:
        return {"instruction": str(raw)}, None
    try:
        value = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return {"instruction": str(raw)}, None
    payload = dict(value) if isinstance(value, Mapping) else {"input": value}
    prefix = match.group("prefix").strip()
    if prefix:
        payload = {"instruction": prefix, **payload}
    return payload, match.group("label")


def ensure_tagged_user_prompt(
    raw: str | Mapping[str, Any],
    *,
    response_schema: Mapping[str, Any],
    task_name: str | None = None,
    section_order: Sequence[str] | None = None,
) -> str:
    """Tag a prompt payload, preserving already-tagged prompts unchanged."""

    if isinstance(raw, Mapping):
        return render_user_prompt(
            raw,
            response_schema=response_schema,
            task_name=task_name,
            section_order=section_order,
        )
    if str(raw).lstrip().startswith("<任务目标>"):
        return str(raw)
    payload, parsed_task = parse_named_json_prompt(str(raw))
    return render_user_prompt(
        payload,
        response_schema=response_schema,
        task_name=task_name or parsed_task,
        section_order=section_order,
    )


def append_tagged_section(prompt: str, tag: str, value: Any) -> str:
    """Append one retry/correction section without flattening the base prompt."""

    suffix = render_tagged_sections(((tag, value),))
    return f"{prompt}\n\n{suffix}" if suffix else prompt


def parse_tagged_user_prompt(prompt: str) -> dict[str, Any]:
    """Test/audit helper that reconstructs the original structured input fields."""

    result: dict[str, Any] = {}
    for match in _TAGGED_SECTION.finditer(str(prompt)):
        if match.group("tag") == "输出结构":
            continue
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            result.update(payload)
    return result


__all__ = [
    "PROMPT_FORMAT_SCHEMA",
    "append_tagged_section",
    "ensure_tagged_user_prompt",
    "parse_named_json_prompt",
    "parse_tagged_user_prompt",
    "render_system_prompt",
    "render_tagged_sections",
    "render_user_prompt",
]
