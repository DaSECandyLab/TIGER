"""检索带来源的静态机制知识，并把论文证据与 workload 观测对齐。

本地 JSON 是权威事实库，本地混合 RAG 只负责召回 claim ID。进入 LLM prompt
的内容必须重新解析到权威 claim，并附带来源定位，不能直接信任索引副本。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .paths import DEFAULT_KNOWLEDGE_BASE_PATH


DEFAULT_MECHANISM_EVIDENCE_PATH = (
    DEFAULT_KNOWLEDGE_BASE_PATH.parent / "mechanism_evidence.json"
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


class StaticEvidenceRetriever(Protocol):
    """机制证据层依赖的最小静态 RAG 接口，便于测试和替换实现。"""

    def retrieve(
        self,
        *,
        query: str,
        filters: Mapping[str, Any],
        top_k: int = 8,
    ) -> dict[str, Any]:
        """按查询和 metadata 过滤条件返回带 item_id 的结果。"""


def _object_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    """读取对象数组，并在知识库结构异常时尽早失败。"""

    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"mechanism evidence field {field} must be an object array")
    return [dict(item) for item in value]


def load_mechanism_evidence(path: Path = DEFAULT_MECHANISM_EVIDENCE_PATH) -> dict[str, Any]:
    """加载并校验来源、机制、观测定义和原子 claim 的引用完整性。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("mechanism evidence knowledge base must be a JSON object")
    sources = _object_list(payload.get("sources"), field="sources")
    mechanisms = _object_list(payload.get("mechanisms"), field="mechanisms")
    claims = _object_list(payload.get("claims"), field="claims")

    source_ids = [str(item.get("source_id", "")).strip() for item in sources]
    if not all(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("mechanism evidence source_id values must be unique and nonempty")
    mechanism_ids = [
        str(item.get("mechanism_id", "")).strip() for item in mechanisms
    ]
    group_ids = [str(item.get("group_id", "")).strip() for item in mechanisms]
    if (
        not all(mechanism_ids)
        or len(mechanism_ids) != len(set(mechanism_ids))
        or not all(group_ids)
        or len(group_ids) != len(set(group_ids))
    ):
        raise ValueError("mechanism_id and group_id values must be unique and nonempty")

    observations_by_mechanism: dict[str, set[str]] = {}
    for mechanism in mechanisms:
        observations = _object_list(
            mechanism.get("observations", []),
            field=f"mechanisms.{mechanism['mechanism_id']}.observations",
        )
        observation_ids = [
            str(item.get("observation_id", "")).strip() for item in observations
        ]
        source_paths = [str(item.get("source_path", "")).strip() for item in observations]
        if (
            not all(observation_ids)
            or len(observation_ids) != len(set(observation_ids))
            or not all(source_paths)
        ):
            raise ValueError(
                f"mechanism {mechanism['mechanism_id']} has invalid observations"
            )
        invalid_kinds = [
            str(item.get("observation_kind", ""))
            for item in observations
            if str(item.get("observation_kind", ""))
            not in {"workload_signal", "matched_intervention"}
        ]
        if invalid_kinds:
            raise ValueError(
                f"mechanism {mechanism['mechanism_id']} has invalid observation "
                f"kinds: {invalid_kinds}"
            )
        observations_by_mechanism[str(mechanism["mechanism_id"])] = set(
            observation_ids
        )

    claim_ids: list[str] = []
    known_sources = set(source_ids)
    known_mechanisms = set(mechanism_ids)
    known_groups = set(group_ids)
    for claim in claims:
        claim_id = str(claim.get("claim_id", "")).strip()
        claim_ids.append(claim_id)
        if not claim_id or not str(claim.get("statement", "")).strip():
            raise ValueError("every mechanism claim needs a claim_id and statement")
        if str(claim.get("source_id")) not in known_sources:
            raise ValueError(f"claim {claim_id} references an unknown source")
        if str(claim.get("mechanism_id")) not in known_mechanisms:
            raise ValueError(f"claim {claim_id} references an unknown mechanism")
        raw_claim_groups = claim.get("group_ids")
        if not isinstance(raw_claim_groups, list) or not raw_claim_groups:
            raise ValueError(f"claim {claim_id} requires group_ids")
        if not set(str(item) for item in raw_claim_groups).issubset(known_groups):
            raise ValueError(f"claim {claim_id} references an unknown group")
        if not str(claim.get("evidence_role", "")).strip():
            raise ValueError(f"claim {claim_id} requires evidence_role")
        if not str(claim.get("source_locator", "")).strip():
            raise ValueError(f"claim {claim_id} requires source_locator")
        decision_use = str(claim.get("decision_use", "evidence"))
        if decision_use not in {"evidence", "context_only"}:
            raise ValueError(f"claim {claim_id} has invalid decision_use")
        relations = _object_list(
            claim.get("observation_relations", []),
            field=f"claims.{claim_id}.observation_relations",
        )
        known_observations = observations_by_mechanism[
            str(claim["mechanism_id"])
        ]
        for relation in relations:
            observation_id = str(relation.get("observation_id", ""))
            allowed_uses = relation.get("allowed_uses")
            if observation_id not in known_observations:
                raise ValueError(
                    f"claim {claim_id} relation references unknown observation "
                    f"{observation_id}"
                )
            if (
                not isinstance(allowed_uses, list)
                or not allowed_uses
                or not set(str(item) for item in allowed_uses).issubset(
                    {"support", "counter"}
                )
            ):
                raise ValueError(
                    f"claim {claim_id} relation has invalid allowed_uses"
                )
            state_options = relation.get("state_options")
            if (
                not isinstance(state_options, Mapping)
                or not state_options
                or not set(str(value) for value in state_options.values()).issubset(
                    {"support", "counter", "uncertainty"}
                )
            ):
                raise ValueError(
                    f"claim {claim_id} relation has invalid state_options"
                )
            decision_states = {
                str(value)
                for value in state_options.values()
                if str(value) != "uncertainty"
            }
            if not decision_states.issubset(
                set(str(item) for item in allowed_uses)
            ):
                raise ValueError(
                    f"claim {claim_id} state_options exceed allowed_uses"
                )
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("mechanism evidence claim_id values must be unique")
    return payload


def _value_at_path(root: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """从 workload 对象读取点分路径，并区分缺失字段与显式 null。"""

    current: Any = root
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return False, None
        current = current[component]
    return True, current


def collect_workload_observations(
    mechanism: Mapping[str, Any],
    workload_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """按机制声明提取原始 workload 事实，不施加人工正负阈值。"""

    observations: list[dict[str, Any]] = []
    for raw in mechanism.get("observations", []):
        observation = dict(raw)
        source_path = str(observation["source_path"])
        exists, value = _value_at_path(workload_context, source_path)
        observations.append(
            {
                "observation_id": str(observation["observation_id"]),
                "source_path": f"workload_context.{source_path}",
                "available": bool(exists and value is not None),
                "value": value if exists else None,
                "semantics": str(observation.get("semantics", "")),
                "observation_kind": str(
                    observation.get("observation_kind", "workload_signal")
                ),
            }
        )
    return observations


def build_mechanism_recall_query(
    *,
    mechanism: Mapping[str, Any],
    group: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
) -> str:
    """用机制、参数和当前观测构造面向“适用条件”的语义查询。"""

    # 检索 query 只描述“需要哪类知识”。具体数值完整保留在返回 bundle 中供
    # LLM 对照，避免长字典稀释向量语义。
    compact_observations = {
        str(item["observation_id"]): bool(item["available"])
        for item in observations
    }
    model = scenario.get("model", {})
    hardware = scenario.get("hardware", {})
    deployment = scenario.get("deployment", {})
    objective = scenario.get("objective", {})
    selected_scenario = {
        "scenario_id": scenario.get("scenario_id"),
        "model": {
            key: model.get(key)
            for key in ("name", "task", "architecture")
        }
        if isinstance(model, Mapping)
        else model,
        "hardware": {
            key: hardware.get(key)
            for key in ("gpu_count", "gpu_model", "node_count")
        }
        if isinstance(hardware, Mapping)
        else hardware,
        "deployment": {
            key: deployment.get(key)
            for key in (
                "tensor_parallel_size",
                "data_parallel_size",
                "pipeline_parallel_size",
            )
        }
        if isinstance(deployment, Mapping)
        else deployment,
        "objective": {
            key: objective.get(key)
            for key in ("phase", "optimize_performance_quality_tradeoff")
        }
        if isinstance(objective, Mapping)
        else objective,
    }
    selected_group = {
        key: group.get(key)
        for key in (
            "group_id",
            "title",
            "parameters",
            "optimization_mechanism",
        )
    }
    return (
        "Find primary-paper or official-document claims about this inference "
        "mechanism's target workload, benefit conditions, limitations, "
        "implementation, and experimental scope.\n"
        f"Mechanism: {mechanism['mechanism_id']}\n"
        f"Terms: {json.dumps(mechanism.get('retrieval_terms', []), ensure_ascii=False)}\n"
        f"Group: {json.dumps(selected_group, ensure_ascii=False, default=str)}\n"
        f"Observations: {json.dumps(compact_observations, ensure_ascii=False, default=str)}\n"
        f"Scenario: {json.dumps(selected_scenario, ensure_ascii=False, default=str)}"
    )


def _tokens(value: Any) -> set[str]:
    """为本地兜底检索提取轻量词项；它只参与排序，不产生决策。"""

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(serialized)}


def _local_ranked_claim_ids(
    claims: Sequence[Mapping[str, Any]],
    *,
    query: str,
) -> list[str]:
    """在静态索引不可用时用词项重合度确定稳定的本地候选顺序。"""

    query_tokens = _tokens(query)
    scored: list[tuple[int, str]] = []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        overlap = len(query_tokens.intersection(_tokens(claim)))
        strength_bonus = 2 if str(claim.get("strength", "")).startswith(
            ("direct", "analytical", "official")
        ) else 0
        scored.append((overlap + strength_bonus, claim_id))
    return [claim_id for _, claim_id in sorted(scored, key=lambda x: (-x[0], x[1]))]


def _retrieved_claim_ids(response: Mapping[str, Any]) -> list[str]:
    """从静态 RAG 响应中提取稳定 item ID，并忽略结构异常的结果。"""

    results = response.get("results", [])
    if not isinstance(results, list):
        raise TypeError("static RAG results must be an array")
    claim_ids: list[str] = []
    for raw in results:
        if not isinstance(raw, Mapping):
            continue
        claim_id = str(raw.get("item_id") or "")
        if claim_id:
            claim_ids.append(claim_id)
    return list(dict.fromkeys(claim_ids))


def _select_claim_ids(
    ordered_ids: Sequence[str],
    *,
    claims_by_id: Mapping[str, Mapping[str, Any]],
    required_roles: Sequence[str],
    max_claims: int,
) -> list[str]:
    """先保证证据角色完整，再按语义排序补足剩余上下文预算。"""

    limit = max(max_claims, len(required_roles))
    selected: list[str] = []
    for role in required_roles:
        for claim_id in ordered_ids:
            claim = claims_by_id.get(claim_id)
            if (
                claim is not None
                and str(claim.get("evidence_role")) == str(role)
                and claim_id not in selected
            ):
                selected.append(claim_id)
                break
    for claim_id in ordered_ids:
        if claim_id in claims_by_id and claim_id not in selected:
            selected.append(claim_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def retrieve_group_mechanism_evidence(
    *,
    knowledge: Mapping[str, Any],
    group: Mapping[str, Any],
    scenario: Mapping[str, Any],
    workload_context: Mapping[str, Any],
    retriever: StaticEvidenceRetriever | None = None,
    max_claims: int = 8,
) -> dict[str, Any] | None:
    """为一个参数组召回均衡证据，并返回可直接进入 prompt 的溯源包。"""

    group_id = str(group.get("group_id", ""))
    mechanisms = {
        str(item["group_id"]): dict(item)
        for item in knowledge.get("mechanisms", [])
    }
    mechanism = mechanisms.get(group_id)
    if mechanism is None:
        return None

    observations = collect_workload_observations(mechanism, workload_context)
    query = build_mechanism_recall_query(
        mechanism=mechanism,
        group=group,
        observations=observations,
        scenario=scenario,
    )
    candidate_claims = [
        dict(claim)
        for claim in knowledge.get("claims", [])
        if group_id in [str(item) for item in claim.get("group_ids", [])]
    ]
    claims_by_id = {str(item["claim_id"]): item for item in candidate_claims}
    local_ids = _local_ranked_claim_ids(candidate_claims, query=query)

    retrieval_response: dict[str, Any] | None = None
    retrieval_error: str | None = None
    retrieved_ids: list[str] = []
    if retriever is not None:
        try:
            retrieval_response = retriever.retrieve(
                query=query,
                filters={
                    "knowledge_type": "mechanism_claim",
                    "vllm_version": str(knowledge["vllm_version"]),
                    "knowledge_base_id": str(knowledge["knowledge_base_id"]),
                    "group_ids": group_id,
                },
                top_k=max_claims,
            )
            retrieved_ids = _retrieved_claim_ids(retrieval_response)
        except Exception as exc:  # 索引失败不能让非权威副本污染权威知识库。
            retrieval_error = f"{type(exc).__name__}: {exc}"

    known_retrieved_ids = [item for item in retrieved_ids if item in claims_by_id]
    unknown_retrieved_ids = [item for item in retrieved_ids if item not in claims_by_id]
    ordered_ids = list(dict.fromkeys([*known_retrieved_ids, *local_ids]))
    required_roles = [str(item) for item in mechanism.get("required_evidence_roles", [])]
    selected_ids = _select_claim_ids(
        ordered_ids,
        claims_by_id=claims_by_id,
        required_roles=required_roles,
        max_claims=max_claims,
    )

    sources_by_id = {
        str(item["source_id"]): dict(item) for item in knowledge.get("sources", [])
    }
    selected_claims = []
    for claim_id in selected_ids:
        claim = claims_by_id[claim_id]
        source = sources_by_id[str(claim["source_id"])]
        selected_claims.append(
            {
                **claim,
                "source": {
                    key: source.get(key)
                    for key in (
                        "source_id",
                        "title",
                        "source_kind",
                        "authority",
                        "version_scope",
                        "url",
                        "pdf_url",
                    )
                    if source.get(key) is not None
                },
            }
        )
    covered_roles = sorted(
        {str(item["evidence_role"]) for item in selected_claims}
    )
    return {
        "knowledge_base_id": knowledge["knowledge_base_id"],
        "mechanism_id": mechanism["mechanism_id"],
        "optional_mechanism": bool(mechanism.get("optional_mechanism")),
        "retrieval_mode": (
            str(retrieval_response.get("retrieval_mode", "static_hybrid_rag"))
            if known_retrieved_ids and retrieval_response is not None
            else "local_metadata_and_lexical_fallback"
        ),
        "retrieval_query": query,
        "retrieved_claim_ids": retrieved_ids,
        "unknown_retrieved_claim_ids": unknown_retrieved_ids,
        "retrieval_error": retrieval_error,
        "required_evidence_roles": required_roles,
        "covered_evidence_roles": covered_roles,
        "missing_evidence_roles": [
            role for role in required_roles if role not in covered_roles
        ],
        "workload_observations": observations,
        "claims": selected_claims,
        "usage_instruction": (
            "Treat claims as sourced mechanism evidence, not executable rules. "
            "Compare them with workload_observations, cite claim IDs and exact "
            "workload paths, and do not invent thresholds beyond the sources."
        ),
        "static_rag_trace": retrieval_response,
    }
