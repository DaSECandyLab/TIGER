"""Short, measured RepairBO lessons for subsequent repair proposals."""
from __future__ import annotations

from typing import Any, Mapping

from agents.tuning.dqbo.src.services.engine.history import is_timeout_record


REFLECT_SYSTEM = """一、任务
维护可供后续 Repair 使用的经验。核心是：在什么情况下，采用什么修复方向，结果如何，是否值得尝试或需要权衡。
将本轮证据融入已有认识，不为每次 Repair 单独写实验报告。

二、证据
executed_results 为本轮实测，prior_evidence 为历史实测，prior_lessons 为已有经验。
每次改动相对自己的 anchor 比较；changed_fields 是实际改动，change_kind 明确单改或联合改参。
precision_before/precision_after 给出改动前后的量化、dtype、KV 精度；不要把 before 的取值误当成修复后的取值。
performance_direction 是程序按命名指标比较得到的变化方向，幅度仍根据原始指标判断。
quality_feasible 表示质量是否达标；命名 performance_metrics 用于判断相对父点的吞吐和延迟变化，R3 可与修复锚点比较。
repair_success 用于判断修复成功；feasible_frontier_improved 为程序按质量可行且实际 HVI 为正计算的前沿改善标记。
质量不可行时不能声称刷新可行 Pareto 前沿，即使原始 actual_hv_gain 数值为正。
刷新 Pareto 前沿与相对父点有性能损失可以同时成立：前沿来自质量可行点，父点可能质量不达标。
不同父点不是单变量对照，联合改参不单独归因。不把未执行提案当结果，不推断未测瓶颈。

三、经验表述
用一两句自然语言写清“适用情形—修复方向—结果与权衡”，不要求固定句式。
保留有助于执行的参数名或关键类别取值（如精度格式、开关状态），但不要复述完整配置、逐次案例、查询编号或质量分数。
经验必须保留必要背景：从 anchor_candidate 与 changed_fields 中识别修复前提、实际改动和保持不变的执行条件，
选择与该修复方向最相关的一两项作为适用范围，例如有证据支持的并发/批处理背景、分块预填充是否开启、同步或异步调度、缓存模式或并行执行方式。
并发只是候选背景之一，不要求固定罗列；配置只给上限时只能描述配置约束，不能声称实际处于高并发。
这些背景表示证据在哪些条件下成立，不表示已经证明它们是收益原因或必要条件；没有对照不能宣称无关、独立或可跨条件泛化。
背景用定性表述，不列并发、batch、token 阈值、显存比例或并行度的具体数值；精度格式名称中的数字可以保留。
没有明确比较基准，不把配置数值自行标成高并发、低预算、显存紧张，也不把配置上限当作实际负载。
关闭功能下的从属参数不作为有效背景。仅有配置不能推断瓶颈；不编造请求长度、查询语义或运行负载。
R3 应保留进入该阶段的精度/质量状态及相关调度功能，参数变更用提高、降低、开启、关闭等方向表达。
合并或更新经验时，把背景绑定到相应修复分支；保留旧分支仍成立的前提，不能用最新一次的背景覆盖所有历史证据。
性能采用定性描述，例如几乎无性能损失、有一定损失、明显改善；不写百分比、数值范围或绝对性能值。
根据实测说明质量是否恢复，以及该方向是否值得优先尝试、需要权衡或尚不足以推荐，不强行给正面建议。
例如，恢复质量且几乎无性能损失时可作为优先候选；刷新可行 Pareto 前沿但相对父点有性能损失时，应同时保留收益与代价。
这些只是表述思路，不能套用到不符合证据的结果。多个性能目标方向不一致时保留关键差异。
只概括证据支持的适用条件，不把单次结果写成普遍保证。

四、按修复方向创建或更新经验
一条经验只描述一种修复方向，先匹配方向，再比较适用条件和结果；不能仅因同属精度修复或都能恢复质量就合并。
方向由实际修改的参数集合及其转换方向共同决定：单改与联合改参是不同方向，联合修改的参数集合不同也是不同方向；
同一精度参数的不同源/目标格式转换也是不同方向。数值型参数按提高/降低等操作方向区分，不为不同数值步长创建新方向。
保持不变的参数和运行负载属于适用背景，不作为新方向的依据。

create：本轮有实测支持的修复方向尚无对应经验时，必须创建独立经验，不能向其他方向的旧经验追加分支。
例如单改 KV 精度、联合改量化与 KV 精度、联合改量化/dtype/KV 精度、单改量化分别建条；
这些只是划分方式，不预设它们有效或保证本轮一定存在这些操作。
同一调用内同方向的多条证据合成一条，不同方向分别创建，不因同属 R1_R2 阶段而合并。

update：存在同方向经验，而本轮直接证据需要补充/修正背景条件、放宽或收窄适用范围、修正收益或代价时，更新原 ID。
同类操作在先前未覆盖的必要背景下得到支持，即使收益方向未改变，也应更新适用条件；无需强行证明推荐优先级发生变化。
放宽范围只限新增证据支持的背景，不宣称对任意并发、预算或执行模式普遍有效，也不声称背景变化造成收益变化。
更新时重新组织成简洁完整的经验，保留仍成立的旧条件和权衡，不按时间追加“新增”“另一条方向”等案例段落。
若历史旧条目混合多个方向，不能据此继续合并；对当前方向创建独立经验，保留旧条目供后续清理，不凭本轮证据臆改其他方向。

covered：只有修复方向、必要背景以及结果对建议的影响都已覆盖时才不变；重复成功、无关配置差异或换措辞不值得更新。
没有写出背景限制不代表证据支持任意背景，应结合 prior_evidence 核对；缺少必要背景时可基于证据更新补齐。
insufficient：实际证据不足以形成或修正结论时说明缺口，不编造方向收益或必要条件。
不同父点不是单变量对照，不能用“其余条件相同”掩盖未比较的配置差异；应只列明确共同的背景。
写“在该配置下质量不达标”，不写未经验证的“因某参数导致质量不达标”。half 与 bfloat16 表述为切换，不称为精度升级。

五、逐项决策及输出
先为每个本轮 trial 写一条 assessments，简短说明处理决定（不是案例正文）：
trial_id；decision（change/covered/insufficient）；
prior_lesson_id（covered 时必须是已覆盖该结果的经验 ID，其余可为 null）；reason（指出新认识或已覆盖的具体旧结论）。
必须处理所有本轮结果，不能忽略质量已达标但性能有代价的分支，也不能忽略 R3 的独立修复方向。
按 reflection_stage 组织经验：R1 与 R2 结果同属 R1_R2，不能写入 R3 经验；R3 只引用 R3 结果。
同一阶段且同一修复方向的结果才可合写一条，不同方向必须分条。同一次调用的 change 结果须出现在本阶段 lessons 的 evidence_trial_ids 中。
covered 不是“存在相关经验”，而是旧文已经包含本次结果对建议的影响；需要修改旧判断就选 change。
lessons 的条数不超过本轮实测结果数，按独立修复方向分别输出变更，每条包含 action（create/update）、target_lesson_id（create 为 null；update 为已有 ID）、
stage（R1_R2/R3，不跨阶段更新）、lesson（完整经验文本）、evidence_trial_ids（至少一条本轮实际证据，可附目标经验已有证据）。
返回 {"assessments": [...], "lessons": [...]}；无新增认识时 lessons 为空，但仍说明各结果已被哪条经验覆盖或为何证据不足。
update 后仅保留新版，历史证据引用由程序合并。"""


def _lesson_id(episode: Mapping[str, Any], index: int, item: Mapping[str, Any]) -> str:
    return str(item.get("lesson_id") or f"{episode.get('episode_id')}:lesson:{index}")


def compact_reflections(history: dict[str, Any]) -> None:
    """Keep only the latest text per lesson ID, including legacy episode lessons."""
    seen: set[str] = set()
    for episode in reversed(history.get("repair_episodes", [])):
        reflection = episode.get("reflection", {})
        if not isinstance(reflection, dict) or reflection.get("status") != "complete":
            continue
        kept = []
        for index, item in reversed(list(enumerate(reflection.get("lessons", [])))):
            if not isinstance(item, Mapping):
                continue
            lesson_id = _lesson_id(episode, index, item)
            if lesson_id not in seen:
                kept.append({**item, "lesson_id": lesson_id})
                seen.add(lesson_id)
        reflection["lessons"] = list(reversed(kept))


def prior_reflections(history: Mapping[str, Any], stage: str | None = None, *, limit: int | None = 8) -> list[dict[str, Any]]:
    wanted = "R3" if stage == "R3" else "R1_R2" if stage else None
    lessons: dict[str, dict[str, Any]] = {}
    for episode in history.get("repair_episodes", []):
        reflection = episode.get("reflection", {}) if isinstance(episode, Mapping) else {}
        if not isinstance(reflection, Mapping) or reflection.get("status") != "complete":
            continue
        for index, item in enumerate(reflection.get("lessons", [])):
            if isinstance(item, Mapping):
                lesson_id = _lesson_id(episode, index, item)
                lessons.pop(lesson_id, None)
                lessons[lesson_id] = {**item, "lesson_id": lesson_id,
                                      "source_episode_id": episode.get("episode_id")}
    selected = [item for item in lessons.values() if wanted is None or item.get("stage") == wanted]
    return selected if limit is None else selected[-limit:]


def reflection_inference_client(client: Any) -> Any:
    """Use reasoning for reflection without mutating the shared proposal client."""
    from dataclasses import replace
    from agents.tuning.candidate_optimizer.src.services.direct_inference import OpenAICompatibleStructuredClient

    if isinstance(client, OpenAICompatibleStructuredClient):
        return replace(client, thinking_mode="enabled")
    return client


def reflect_episode(client: Any, episode: Mapping[str, Any], history: Mapping[str, Any]) -> dict[str, Any]:
    measured = [item for round_record in episode.get("rounds", [])
                for item in round_record.get("results", [])
                if isinstance(item, Mapping) and not is_timeout_record(item)]
    if not measured:
        return {"status": "complete", "lessons": [], "reason": "no_executed_repair_trial"}
    if client is None:
        return {"status": "unavailable", "lessons": [], "reason": "structured_client_unavailable"}
    trials = {str(row.get("trial_id")): row for row in history.get("trials", []) if isinstance(row, Mapping)}
    parent = trials.get(str(episode.get("parent_trial_id")), {})
    def fact_for(item: Mapping[str, Any], parent_id: Any) -> dict[str, Any]:
        anchor = trials.get(str(item.get("repair_anchor_trial_id") or parent_id), {})
        before, after = anchor.get("candidate", {}), item.get("candidate", {})
        changes = {name: {"before": before.get(name), "after": after.get(name)}
                   for name in after if before.get(name) != after.get(name)}
        precision_names = ("quantization", "dtype", "kv_cache_dtype")
        directions = {}
        for name, higher_is_better in (("query_throughput_qps", True),
                                       ("mean_query_execution_latency_ms", False),
                                       ("p95_query_execution_latency_ms", False)):
            old_value = (anchor.get("metrics") or {}).get(name)
            new_value = (item.get("metrics") or {}).get(name)
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                directions[name] = ("unchanged" if old_value == new_value else
                                    "improved" if (new_value > old_value) == higher_is_better else "regressed")
        return {"trial_id": str(item.get("trial_id")), "stage": item.get("repair_stage"),
                "reflection_stage": "R3" if item.get("repair_stage") == "R3" else "R1_R2",
                "change_kind": "single" if len(changes) == 1 else "joint" if changes else "unchanged",
                "precision_before": {key: before.get(key) for key in precision_names},
                "precision_after": {key: after.get(key) for key in precision_names},
                "performance_direction": directions,
                "status": item.get("status"), "changed_fields": changes,
                "anchor_trial_id": anchor.get("trial_id"), "anchor_candidate": before,
                "anchor_performance_metrics": anchor.get("metrics"),
                "anchor_quality_feasible": anchor.get("quality_feasible"),
                "performance_metrics": item.get("metrics"),
                "quality_feasible": item.get("quality_feasible"),
                "actual_hv_gain": item.get("repair_actual_hv_gain"),
                "feasible_frontier_improved": (item.get("quality_feasible") is True
                    and isinstance(item.get("repair_actual_hv_gain"), (int, float))
                    and item["repair_actual_hv_gain"] > 0),
                "repair_success": item.get("repair_success")}

    facts = [fact_for(item, episode.get("parent_trial_id")) for item in measured]
    evidence_ids = {item["trial_id"] for item in facts}
    prior = prior_reflections(history, limit=None)
    prior_ids = {trial_id for item in prior for trial_id in item.get("evidence_trial_ids", [])}
    prior_evidence = {}
    for previous in history.get("repair_episodes", []):
        for round_record in previous.get("rounds", []):
            for item in round_record.get("results", []):
                if (isinstance(item, Mapping) and not is_timeout_record(item)
                        and str(item.get("trial_id")) in prior_ids):
                    prior_evidence[str(item["trial_id"])] = fact_for(item, previous.get("parent_trial_id"))
    prior_by_id = {item["lesson_id"]: item for item in prior}
    payload = {"parent": {"trial_id": parent.get("trial_id"), "candidate": parent.get("candidate"),
                           "performance_metrics": parent.get("metrics"),
                           "quality_feasible": parent.get("quality_feasible")},
               "executed_results": facts,
               "unexecuted_proposals_are_not_outcomes": True,
               "prior_lessons": prior, "prior_evidence": list(prior_evidence.values())}
    schema = {"type": "object", "additionalProperties": False, "required": ["assessments", "lessons"],
              "properties": {"lessons": {"type": "array", "maxItems": len(evidence_ids),
                  "items": {"type": "object", "additionalProperties": False,
                      "required": ["action", "target_lesson_id", "stage", "lesson", "evidence_trial_ids"],
                      "properties": {"action": {"enum": ["create", "update"]},
                                     "target_lesson_id": {"type": ["string", "null"]},
                                     "stage": {"enum": ["R1_R2", "R3"]},
                                     "lesson": {"type": "string"},
                                     "evidence_trial_ids": {"type": "array", "items": {"type": "string"}}}}}}}
    schema["properties"]["assessments"] = {
        "type": "array", "minItems": len(evidence_ids), "maxItems": len(evidence_ids),
        "items": {"type": "object", "additionalProperties": False,
            "required": ["trial_id", "decision", "prior_lesson_id", "reason"],
            "properties": {"trial_id": {"enum": sorted(evidence_ids)},
                "decision": {"enum": ["change", "covered", "insufficient"]},
                "prior_lesson_id": {"type": ["string", "null"]},
                "reason": {"type": "string", "minLength": 1}}}}
    try:
        import json
        response = reflection_inference_client(client).complete_structured(system=REFLECT_SYSTEM, user=json.dumps(payload, ensure_ascii=False),
                                              response_schema=schema)
        output = response.get("structured_output", response)
        assessments = output.get("assessments")
        proposals = output.get("lessons")
        if not isinstance(assessments, list) or not isinstance(proposals, list):
            raise ValueError("assessments and lessons must be lists")
        assessed = set()
        changed_trials: set[str] = set()
        for assessment in assessments:
            if not isinstance(assessment, Mapping):
                raise ValueError("invalid assessment")
            trial_id = assessment.get("trial_id")
            if not isinstance(trial_id, str) or trial_id not in evidence_ids or trial_id in assessed:
                raise ValueError("assessment must identify each executed trial exactly once")
            assessed.add(trial_id)
            if not isinstance(assessment.get("reason"), str) or not assessment["reason"].strip():
                raise ValueError("assessment requires a reason")
            decision = assessment.get("decision")
            if decision == "change":
                changed_trials.add(trial_id)
            elif decision in ("covered", "insufficient"):
                if decision == "covered":
                    covered = prior_by_id.get(str(assessment.get("prior_lesson_id")))
                    fact = next(f for f in facts if f["trial_id"] == trial_id)
                    stage = "R3" if fact["stage"] == "R3" else "R1_R2"
                    if covered is None or covered.get("stage") != stage:
                        raise ValueError("covered assessment must reference an existing same-stage lesson")
            else:
                raise ValueError("invalid assessment decision")
        if assessed != evidence_ids:
            raise ValueError("assessment omitted an executed result")
        for trial_id in changed_trials:
            stage = next(f["reflection_stage"] for f in facts if f["trial_id"] == trial_id)
            if not any(isinstance(proposal, Mapping) and proposal.get("stage") == stage
                       and isinstance(proposal.get("evidence_trial_ids"), list)
                       and trial_id in proposal["evidence_trial_ids"] for proposal in proposals):
                raise ValueError("change assessment has no same-stage evidence proposal")
        accepted = []
        updated = set()
        for index, item in enumerate(output.get("lessons", [])):
            if not isinstance(item, Mapping):
                continue
            sentence = item.get("lesson")
            stage = item.get("stage")
            ids = item.get("evidence_trial_ids")
            if (stage not in ("R1_R2", "R3") or not isinstance(sentence, str) or not sentence.strip()
                    or not isinstance(ids, list) or not ids
                    or any(not isinstance(value, str) for value in ids)
                    or not evidence_ids.intersection(ids)
                    or not changed_trials.intersection(ids)):
                continue
            # Known stages must agree with the claimed supporting evidence.
            if any(fact["trial_id"] in ids and fact["stage"] in ("R1", "R2", "R3")
                   and ("R3" if fact["stage"] == "R3" else "R1_R2") != stage for fact in facts):
                continue
            target = item.get("target_lesson_id")
            if item.get("action") == "update":
                if not isinstance(target, str) or target not in prior_by_id or target in updated:
                    continue
                old = prior_by_id[target]
                if (old.get("stage") != stage
                        or not set(ids).issubset(evidence_ids | set(old.get("evidence_trial_ids", [])))):
                    continue
                # Repeated evidence with unchanged text does not warrant a new version.
                if sentence.strip() == str(old.get("lesson", "")).strip():
                    continue
                lesson_id = target
                ids = list(dict.fromkeys([*old.get("evidence_trial_ids", []), *ids]))
            elif item.get("action") == "create" and target is None and set(ids).issubset(evidence_ids):
                lesson_id = f"{episode.get('episode_id')}:lesson:{index}"
            else:
                continue
            if any(other.get("stage") == stage and other.get("lesson", "").strip() == sentence.strip()
                   for other in [*[old for old in prior
                                  if item.get("action") != "update" or old["lesson_id"] != target],
                                 *accepted]):
                continue
            if item.get("action") == "update":
                updated.add(target)
            accepted.append({"lesson_id": lesson_id, "stage": stage, "lesson": sentence.strip(),
                             "evidence_trial_ids": ids})
            if len(accepted) == len(evidence_ids):
                break
        return {"status": "complete", "lessons": accepted, "executed_trial_ids": sorted(evidence_ids),
                "assessments": assessments}
    except Exception as exc:
        return {"status": "unavailable", "lessons": [], "reason": f"{type(exc).__name__}: {exc}"[:400]}
