"""Ephemeral hybrid static-parameter RAG for Candidate BO agents."""

from __future__ import annotations

import re
from threading import Lock
from typing import Any, Mapping, Sequence

from agents.infrastructure.knowledge.src.services import search_static_knowledge


_PARAMETER_RAG_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_PARAMETER_RAG_CACHE_LOCK = Lock()


def _retrieve_one_parameter(
    name: str,
    *,
    index_dir: str | None,
) -> dict[str, Any]:
    """Retrieve one exact parameter through hybrid RAG, once per process."""

    cache_key = (str(index_dir or "<default>"), str(name))
    with _PARAMETER_RAG_CACHE_LOCK:
        cached = _PARAMETER_RAG_CACHE.get(cache_key)
        if cached is not None:
            return cached
        result = search_static_knowledge(
            (
                f"vLLM parameter {name} meaning semantics activation interactions "
                "and documented performance tradeoffs"
            ),
            top_k=2,
            filters={
                "knowledge_type": "parameter",
                "parameter_name": str(name),
            },
            index_dir=index_dir,
            retrieval_mode="hybrid",
            device="cpu",
        )
        if result.get("retrieval_mode") != "static_hybrid_rag":
            raise RuntimeError(
                "hybrid static RAG did not return BM25+dense fused retrieval"
            )
        rows = result.get("results")
        if not isinstance(rows, list) or not rows:
            raise LookupError(f"no static RAG document for parameter {name}")
        if any(
            str(dict(row.get("metadata") or {}).get("parameter_name")) != str(name)
            for row in rows
            if isinstance(row, Mapping)
        ):
            raise RuntimeError(
                f"static RAG returned a different parameter for exact request {name}"
            )
        _PARAMETER_RAG_CACHE[cache_key] = result
        return result


def parameter_mentions(
    value: Any,
    allowed_parameters: Sequence[str],
    *,
    include_mapping_keys: bool = True,
) -> list[str]:
    """Find exact frozen-Range parameter mentions without LLM inference."""

    allowed = {str(item) for item in allowed_parameters if str(item)}
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key)
                if include_mapping_keys and name in allowed:
                    found.add(name)
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            return
        for name in allowed:
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                item,
                re.IGNORECASE,
            ):
                found.add(name)

    visit(value)
    return sorted(found)


def retrieve_static_parameter_knowledge(
    parameter_names: Sequence[str],
    *,
    task: str,
    top_k: int = 8,
    index_dir: str | None = None,
) -> dict[str, Any]:
    """Run real BM25+dense RRF only when an agent has parameter needs."""

    selected = sorted({str(item) for item in parameter_names if str(item)})[:20]
    if not selected:
        return {
            "schema": "sembench.bo.agent-static-parameter-rag.v2",
            "status": "not_requested",
            "task": str(task),
            "parameter_names": [],
            "retrieval_mode_required": "static_hybrid_rag",
            "result": None,
            "error": None,
        }
    retrievals: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in selected:
        try:
            retrievals[name] = _retrieve_one_parameter(
                name,
                index_dir=index_dir,
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    status = (
        "retrieved"
        if retrievals and not errors
        else "partially_retrieved"
        if retrievals
        else "unavailable"
    )
    return {
        "schema": "sembench.bo.agent-static-parameter-rag.v2",
        "status": status,
        "task": str(task),
        "parameter_names": selected,
        "retrieval_mode_required": "static_hybrid_rag",
        "retrieval_granularity": "one_exact_hybrid_query_per_parameter",
        "per_parameter_results": retrievals,
        "errors": errors,
        "evidence_policy": (
            "Use retrieved documents for stated parameter semantics only. Do not infer an "
            "undocumented performance direction. Frozen Range, current measurements, exact "
            "runtime logs, and mechanical constraints remain authoritative."
        ),
    }


__all__ = ["parameter_mentions", "retrieve_static_parameter_knowledge"]
