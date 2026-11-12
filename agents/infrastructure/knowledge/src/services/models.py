"""参数知识库与筛选结果的数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterRecord:
    """一项版本化参数，以及筛选所需的结构化可调性信息。"""

    name: str
    flags: tuple[str, ...]
    group: str
    description: str
    default: str | None
    choices: tuple[str, ...]
    source_url: str
    source_anchor: str
    namespace: str = "vllm"
    source_kind: str = "official_documentation"
    tunability: str = "unspecified"
    parent_parameters: tuple[str, ...] = ()
    active_when: tuple[str, ...] = ()
    search_domain: tuple[str, ...] = ()
    conflicts_or_overrides: tuple[str, ...] = ()
    validation_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化参数记录，并把不可变元组转换为 JSON 数组。"""

        data = asdict(self)
        data["flags"] = list(self.flags)
        data["choices"] = list(self.choices)
        data["parent_parameters"] = list(self.parent_parameters)
        data["active_when"] = list(self.active_when)
        data["search_domain"] = list(self.search_domain)
        data["conflicts_or_overrides"] = list(self.conflicts_or_overrides)
        data["validation_evidence"] = list(self.validation_evidence)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterRecord":
        """从知识库 JSON 恢复一项参数记录。"""

        return cls(
            name=str(data["name"]),
            flags=tuple(str(item) for item in data.get("flags", [])),
            group=str(data["group"]),
            description=str(data.get("description", "")),
            default=(
                None if data.get("default") is None else str(data["default"])
            ),
            choices=tuple(str(item) for item in data.get("choices", [])),
            source_url=str(data["source_url"]),
            source_anchor=str(data.get("source_anchor", "")),
            namespace=str(data.get("namespace", "vllm")),
            source_kind=str(
                data.get("source_kind", "official_documentation")
            ),
            tunability=str(data.get("tunability", "unspecified")),
            parent_parameters=tuple(
                str(item) for item in data.get("parent_parameters", [])
            ),
            active_when=tuple(
                str(item) for item in data.get("active_when", [])
            ),
            search_domain=tuple(
                str(item) for item in data.get("search_domain", [])
            ),
            conflicts_or_overrides=tuple(
                str(item)
                for item in data.get("conflicts_or_overrides", [])
            ),
            validation_evidence=tuple(
                str(item) for item in data.get("validation_evidence", [])
            ),
        )


@dataclass(frozen=True)
class ParameterGroup:
    """官方文档中的一个 Config 参数组。"""

    name: str
    description: str
    parameters: tuple[ParameterRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化 Config 组及其全部参数。"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": [item.to_dict() for item in self.parameters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterGroup":
        """从知识库 JSON 恢复 Config 组。"""

        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            parameters=tuple(
                ParameterRecord.from_dict(item)
                for item in data.get("parameters", [])
            ),
        )


@dataclass(frozen=True)
class KnowledgeBase:
    """可复现的 vLLM 版本化参数知识库。"""

    schema_version: int
    vllm_version: str
    source_url: str
    retrieved_at: str
    source_sha256: str
    groups: tuple[ParameterGroup, ...]
    knowledge_sha256: str | None = None

    @property
    def identity_sha256(self) -> str:
        """返回合并知识哈希；旧数据缺失时退回官方文档哈希。"""

        return self.knowledge_sha256 or self.source_sha256

    @property
    def parameter_count(self) -> int:
        """统计知识库中合并后的参数总数。"""

        return sum(len(group.parameters) for group in self.groups)

    def to_dict(self) -> dict[str, Any]:
        """序列化知识库及其可追溯元数据。"""

        return {
            "schema_version": self.schema_version,
            "vllm_version": self.vllm_version,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "source_sha256": self.source_sha256,
            "knowledge_sha256": self.identity_sha256,
            "group_count": len(self.groups),
            "parameter_count": self.parameter_count,
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeBase":
        """从持久化 JSON 恢复完整知识库。"""

        return cls(
            schema_version=int(data["schema_version"]),
            vllm_version=str(data["vllm_version"]),
            source_url=str(data["source_url"]),
            retrieved_at=str(data["retrieved_at"]),
            source_sha256=str(data["source_sha256"]),
            groups=tuple(
                ParameterGroup.from_dict(item)
                for item in data.get("groups", [])
            ),
            knowledge_sha256=(
                None
                if data.get("knowledge_sha256") is None
                else str(data["knowledge_sha256"])
            ),
        )


@dataclass(frozen=True)
class ClassificationResult:
    """LLM 对单个参数的结构化性能相关性判定。"""

    name: str
    group: str
    classification: str
    mechanisms: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    interacts_with: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 0.0
    audited: bool = False
    original_classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化单参数高召回分类结果。"""

        data = asdict(self)
        data["mechanisms"] = list(self.mechanisms)
        data["metrics"] = list(self.metrics)
        data["conditions"] = list(self.conditions)
        data["interacts_with"] = list(self.interacts_with)
        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, group: str | None = None
    ) -> "ClassificationResult":
        """从 LLM/结果 JSON 恢复分类，并规范化置信度。"""

        confidence = float(data.get("confidence", 0.0))
        return cls(
            name=str(data["name"]),
            group=str(group if group is not None else data["group"]),
            classification=str(data["classification"]),
            mechanisms=tuple(str(item) for item in data.get("mechanisms", [])),
            metrics=tuple(str(item) for item in data.get("metrics", [])),
            conditions=tuple(str(item) for item in data.get("conditions", [])),
            interacts_with=tuple(
                str(item) for item in data.get("interacts_with", [])
            ),
            reason=str(data.get("reason", "")),
            confidence=max(0.0, min(1.0, confidence)),
            audited=bool(data.get("audited", False)),
            original_classification=(
                None
                if data.get("original_classification") is None
                else str(data["original_classification"])
            ),
        )
