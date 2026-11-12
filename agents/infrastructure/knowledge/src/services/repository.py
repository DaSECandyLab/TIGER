"""静态知识模块 自己拥有的静态知识库与 RAG 服务。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from .docs import load_knowledge_base
from .mechanism_evidence import load_mechanism_evidence
from .paths import (
    DEFAULT_EXTERNAL_SCREENING_KNOWLEDGE_PATH,
    DEFAULT_KNOWLEDGE_BASE_PATH,
    DEFAULT_MECHANISM_EVIDENCE_PATH,
    DEFAULT_STATIC_RAG_INDEX_DIR,
)
from .static_rag import (
    StaticKnowledgeRAG,
)


_STATIC_RAG_CACHE: dict[tuple[str, bool, str], StaticKnowledgeRAG] = {}
_STATIC_RAG_CACHE_LOCK = Lock()
_STATIC_RAG_RETRIEVAL_LOCK = Lock()


def _path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default.resolve()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """加载 静态知识模块 拥有的顶层 JSON 对象。"""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return dict(raw)


def _cached_static_rag(
    index_dir: Path,
    *,
    dense: bool,
    device: str,
) -> StaticKnowledgeRAG:
    """Load one validated immutable index/encoder instance per process."""

    key = (str(index_dir), bool(dense), str(device))
    with _STATIC_RAG_CACHE_LOCK:
        cached = _STATIC_RAG_CACHE.get(key)
        if cached is None:
            cached = StaticKnowledgeRAG.load(
                index_dir,
                device=device,
                dense=dense,
                validate_sources=True,
            )
            _STATIC_RAG_CACHE[key] = cached
        return cached


def knowledge_status(
    knowledge_base_path: str | None = None,
    index_dir: str | None = None,
) -> dict[str, Any]:
    """返回权威 JSON 与可重建索引的只读状态。"""

    # 第 1 步：解析权威知识 JSON 与静态 RAG 索引的实际路径。
    knowledge_path = _path(knowledge_base_path, DEFAULT_KNOWLEDGE_BASE_PATH)
    rag_dir = _path(index_dir, DEFAULT_STATIC_RAG_INDEX_DIR)
    manifest_path = rag_dir / "manifest.json"
    manifest: dict[str, Any] | None = None

    # 第 2 步：只读加载索引 manifest；状态检查绝不修改知识或索引。
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = dict(raw) if isinstance(raw, Mapping) else None

    # 第 3 步：返回权威源摘要和就绪状态，供依赖方做机械观察。
    return {
        "backend": "local_versioned_static_rag",
        "authoritative_source": str(knowledge_path),
        "authoritative_source_exists": knowledge_path.is_file(),
        "authoritative_source_sha256": (
            hashlib.sha256(knowledge_path.read_bytes()).hexdigest()
            if knowledge_path.is_file()
            else None
        ),
        "index_dir": str(rag_dir),
        "index_manifest_exists": manifest_path.is_file(),
        "index_manifest": manifest,
        "ready": knowledge_path.is_file() and manifest_path.is_file(),
    }


def get_knowledge_snapshot(
    knowledge_base_path: str | None = None,
    mechanism_evidence_path: str | None = None,
    external_screening_knowledge_path: str | None = None,
) -> dict[str, Any]:
    """返回 调用方可直接使用的只读权威知识快照。"""

    # 第 1 步：解析并加载与 Legacy loader 相同的版本化知识文件。
    knowledge_path = _path(knowledge_base_path, DEFAULT_KNOWLEDGE_BASE_PATH)
    mechanism_path = _path(
        mechanism_evidence_path, DEFAULT_MECHANISM_EVIDENCE_PATH
    )
    external_path = _path(
        external_screening_knowledge_path,
        DEFAULT_EXTERNAL_SCREENING_KNOWLEDGE_PATH,
    )
    knowledge = load_knowledge_base(knowledge_path)
    mechanism = load_mechanism_evidence(mechanism_path)
    external = _load_json_object(
        external_path,
        label="external screening knowledge",
    )
    if str(external.get("vllm_version")) != str(knowledge.vllm_version):
        raise ValueError(
            "external screening knowledge and parameter knowledge have "
            "different vLLM versions"
        )

    # 第 2 步：返回原 schema，不在 接口层 中筛选、重排或补写业务字段。
    return {
        "schema_version": 1,
        "operation": "knowledge_snapshot",
        "vllm_version": knowledge.vllm_version,
        "knowledge": knowledge.to_dict(),
        "mechanism_evidence": mechanism,
        "external_screening_knowledge": external,
        "sources": {
            "knowledge_base": {
                "path": str(knowledge_path),
                "sha256": hashlib.sha256(knowledge_path.read_bytes()).hexdigest(),
            },
            "mechanism_evidence": {
                "path": str(mechanism_path),
                "sha256": hashlib.sha256(mechanism_path.read_bytes()).hexdigest(),
            },
            "external_screening_knowledge": {
                "path": str(external_path),
                "sha256": hashlib.sha256(external_path.read_bytes()).hexdigest(),
            },
        },
    }


def get_parameter_record(
    name: str,
    knowledge_base_path: str | None = None,
) -> dict[str, Any]:
    """从本地权威知识 JSON 按参数名解析记录。"""

    # 第 1 步：用 静态知识模块 自己的 loader 加载版本化权威知识。
    knowledge_path = _path(knowledge_base_path, DEFAULT_KNOWLEDGE_BASE_PATH)
    knowledge = load_knowledge_base(knowledge_path)

    # 第 2 步：按参数名遍历分组，并携带版本、分组和来源路径返回。
    for group in knowledge.groups:
        for record in group.parameters:
            if record.name == name:
                return {
                    "vllm_version": knowledge.vllm_version,
                    "group": group.name,
                    "parameter": record.to_dict(),
                    "source": str(knowledge_path),
                }
    raise KeyError(f"unknown vLLM parameter: {name}")


def search_static_knowledge(
    query: str,
    *,
    top_k: int = 8,
    filters: Mapping[str, Any] | None = None,
    index_dir: str | None = None,
    retrieval_mode: str = "sparse",
    device: str = "cpu",
) -> dict[str, Any]:
    """调用既有 BM25/混合 RAG，返回可回映射 ID 及源指纹校验内容。"""

    # 第 1 步：验证检索模式与查询，避免把空请求传入旧 RAG。
    if retrieval_mode not in {"sparse", "hybrid"}:
        raise ValueError("retrieval_mode must be 'sparse' or 'hybrid'")
    if not query.strip():
        raise ValueError("query must not be empty")

    # 第 2 步：加载 模块自身的静态 RAG，并校验索引仍匹配权威源摘要。
    rag = _cached_static_rag(
        _path(index_dir, DEFAULT_STATIC_RAG_INDEX_DIR),
        device=device,
        dense=retrieval_mode == "hybrid",
    )

    # 第 3 步：执行与 Legacy 等价的检索逻辑并返回可回映射的证据记录。
    # Three Pareto sub-agents can request parameter knowledge concurrently.
    # The cached Transformer is shared read-only, but serializing its short CPU
    # query encodes avoids thread oversubscription and backend thread-safety
    # surprises without serializing the surrounding LLM work.
    with _STATIC_RAG_RETRIEVAL_LOCK:
        return rag.retrieve(
            query=query,
            filters=dict(filters or {}),
            top_k=max(1, int(top_k)),
        )
