"""为版本固定的参数与来源 claim 构建可重建的本地混合 RAG 索引。"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

# Keep diagnostics on stderr so CLI results on stdout remain machine-readable.
print = functools.partial(print, file=sys.stderr)

from .docs import load_knowledge_base
from .paths import DEFAULT_KNOWLEDGE_BASE_PATH


DEFAULT_MECHANISM_EVIDENCE_PATH = (
    DEFAULT_KNOWLEDGE_BASE_PATH.parent / "mechanism_evidence.json"
)
DEFAULT_STATIC_RAG_INDEX_DIR = DEFAULT_KNOWLEDGE_BASE_PATH.parent / "static_rag"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_DENSE_MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _sha256_path(path: Path) -> str:
    """计算静态源文件内容摘要。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> list[str]:
    """提取兼容参数名、英文术语和中文查询的 BM25 词项。"""

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(value.lower()):
        token = match.group(0)
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


@dataclass(frozen=True)
class StaticRAGDocument:
    """静态知识库中的一个原子、可回映射文档。"""

    document_id: str
    item_id: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为稳定 JSON 对象。"""

        return {
            "document_id": self.document_id,
            "item_id": self.item_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StaticRAGDocument":
        """从索引文档清单恢复记录。"""

        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("static RAG document metadata must be an object")
        return cls(
            document_id=str(raw["document_id"]),
            item_id=str(raw["item_id"]),
            text=str(raw["text"]),
            metadata=dict(metadata),
        )


class DenseEncoder(Protocol):
    """静态 RAG 所需的最小 dense encoder 接口。"""

    model_id: str
    revision: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """编码 passage 文档。"""

    def encode_query(self, query: str) -> np.ndarray:
        """编码单条检索 query。"""


class E5DenseEncoder:
    """直接用 Transformers 加载本地 multilingual-E5，不依赖向量数据库。"""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_DENSE_MODEL,
        revision: str = DEFAULT_DENSE_MODEL_REVISION,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        local_files_only: bool = True,
    ) -> None:
        """延迟加载 tokenizer/model，并固定 revision 和本地缓存保证可复现。"""

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        ).to(device)
        self._model.eval()

    def _encode(self, texts: Sequence[str], *, prefix: str) -> np.ndarray:
        """执行带 attention-mask mean pooling 的归一化编码。"""

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            selected = [f"{prefix}: {text}" for text in texts[start : start + self.batch_size]]
            inputs = self._tokenizer(
                selected,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.size()).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """按 E5 约定使用 passage 前缀编码静态文档。"""

        return self._encode(texts, prefix="passage")

    def encode_query(self, query: str) -> np.ndarray:
        """按 E5 约定使用 query 前缀编码查询。"""

        return self._encode([query], prefix="query")[0]


def build_static_documents(
    *,
    parameter_knowledge_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
    mechanism_evidence_path: Path = DEFAULT_MECHANISM_EVIDENCE_PATH,
) -> list[StaticRAGDocument]:
    """把参数记录和来源 claim 投影为统一、原子、可过滤的静态文档。"""

    parameter_knowledge = load_knowledge_base(parameter_knowledge_path)
    mechanism_knowledge = json.loads(
        mechanism_evidence_path.read_text(encoding="utf-8")
    )
    sources = {
        str(item["source_id"]): dict(item)
        for item in mechanism_knowledge.get("sources", [])
    }
    documents: list[StaticRAGDocument] = []

    for group in parameter_knowledge.groups:
        for record in group.parameters:
            payload = record.to_dict()
            text = json.dumps(
                {
                    "kind": "vLLM parameter documentation",
                    "parameter": payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            documents.append(
                StaticRAGDocument(
                    document_id=f"parameter:{record.name}",
                    item_id=record.name,
                    text=text,
                    metadata={
                        "knowledge_type": "parameter",
                        "vllm_version": parameter_knowledge.vllm_version,
                        "parameter_name": record.name,
                        "config_group": record.group,
                        "source_kind": record.source_kind,
                    },
                )
            )

    version = str(mechanism_knowledge["vllm_version"])
    knowledge_base_id = str(mechanism_knowledge["knowledge_base_id"])
    for raw_claim in mechanism_knowledge.get("claims", []):
        claim = dict(raw_claim)
        claim_id = str(claim["claim_id"])
        source = sources[str(claim["source_id"])]
        text = json.dumps(
            {
                "kind": "source-grounded inference mechanism claim",
                "claim": claim,
                "source": source,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        documents.append(
            StaticRAGDocument(
                document_id=f"mechanism_claim:{claim_id}",
                item_id=claim_id,
                text=text,
                metadata={
                    "knowledge_type": "mechanism_claim",
                    "vllm_version": version,
                    "knowledge_base_id": knowledge_base_id,
                    "claim_id": claim_id,
                    "mechanism_id": str(claim["mechanism_id"]),
                    "group_ids": [str(item) for item in claim["group_ids"]],
                    "evidence_role": str(claim["evidence_role"]),
                    "source_id": str(claim["source_id"]),
                    "source_kind": str(source["source_kind"]),
                },
            )
        )

    ids = [item.document_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("static RAG document IDs must be unique")
    return documents


def build_static_rag_index(
    *,
    output_dir: Path = DEFAULT_STATIC_RAG_INDEX_DIR,
    parameter_knowledge_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
    mechanism_evidence_path: Path = DEFAULT_MECHANISM_EVIDENCE_PATH,
    encoder: DenseEncoder | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """重建文档清单与 dense 向量，并写出可校验的索引 manifest。"""

    selected_encoder = encoder or E5DenseEncoder(device=device)
    documents = build_static_documents(
        parameter_knowledge_path=parameter_knowledge_path,
        mechanism_evidence_path=mechanism_evidence_path,
    )
    embeddings = selected_encoder.encode_documents([item.text for item in documents])
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("dense encoder returned an invalid document matrix")
    source_paths = {
        "parameter_knowledge": parameter_knowledge_path,
        "parameter_extensions": parameter_knowledge_path.with_name(
            "parameter_extensions.json"
        ),
        "mechanism_evidence": mechanism_evidence_path,
    }
    # 索引会随仓库一起移动，因此 manifest 保存相对索引目录的路径；索引身份只
    # 依赖角色和内容摘要，不包含开发机绝对路径。
    source_fingerprints = [
        {
            "role": role,
            "path": os.path.relpath(path.resolve(), output_dir.resolve()),
            "sha256": _sha256_path(path),
        }
        for role, path in source_paths.items()
    ]
    identity_payload = {
        "sources": [
            {"role": item["role"], "sha256": item["sha256"]}
            for item in source_fingerprints
        ],
        "model_id": selected_encoder.model_id,
        "model_revision": selected_encoder.revision,
        "document_ids": [item.document_id for item in documents],
    }
    index_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.json"
    embeddings_path = output_dir / "dense_embeddings.npy"
    manifest_path = output_dir / "manifest.json"
    documents_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [item.to_dict() for item in documents],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(embeddings_path, embeddings.astype(np.float32), allow_pickle=False)
    manifest = {
        "schema_version": 2,
        "index_id": index_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "retrieval": "metadata_filter + BM25 + multilingual-E5 + RRF",
        "rrf_k": 60,
        "document_count": len(documents),
        "embedding_dimension": int(embeddings.shape[1]),
        "dense_model": {
            "model_id": selected_encoder.model_id,
            "revision": selected_encoder.revision,
        },
        "source_fingerprints": source_fingerprints,
        "documents_file": documents_path.name,
        "embeddings_file": embeddings_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """执行严格 metadata 过滤，并支持数组字段的成员匹配。"""

    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(actual, list):
            expected_values = expected if isinstance(expected, list) else [expected]
            if not any(value in actual for value in expected_values):
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _bm25_scores(query: str, documents: Sequence[str]) -> np.ndarray:
    """对过滤后的少量静态文档计算标准 Okapi BM25 分数。"""

    tokenized = [_tokens(text) for text in documents]
    query_terms = _tokens(query)
    if not documents:
        return np.empty((0,), dtype=np.float32)
    lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
    average_length = float(lengths.mean()) if lengths.size else 1.0
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens)
    )
    scores = np.zeros(len(documents), dtype=np.float32)
    k1, b = 1.2, 0.75
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * lengths[index] / max(average_length, 1.0)
            )
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
    return scores


class StaticKnowledgeRAG:
    """加载本地静态索引并执行 metadata + BM25 + dense + RRF 检索。"""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        documents: Sequence[StaticRAGDocument],
        embeddings: np.ndarray,
        encoder: DenseEncoder | None,
    ) -> None:
        """保存只读索引并校验向量与文档一一对应。"""

        self.manifest = dict(manifest)
        self.documents = list(documents)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.encoder = encoder
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(documents):
            raise ValueError("static RAG documents and embeddings are inconsistent")

    @classmethod
    def load(
        cls,
        index_dir: Path = DEFAULT_STATIC_RAG_INDEX_DIR,
        *,
        device: str = "cpu",
        dense: bool = True,
        encoder: DenseEncoder | None = None,
        validate_sources: bool = True,
    ) -> "StaticKnowledgeRAG":
        """加载索引；默认校验静态源文件未在构建后发生变化。"""

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        documents_payload = json.loads(
            (index_dir / str(manifest["documents_file"])).read_text(encoding="utf-8")
        )
        documents = [
            StaticRAGDocument.from_dict(item)
            for item in documents_payload.get("documents", [])
        ]
        embeddings = np.load(
            index_dir / str(manifest["embeddings_file"]),
            allow_pickle=False,
        )
        if validate_sources:
            raw_fingerprints = manifest.get("source_fingerprints", [])
            if not isinstance(raw_fingerprints, list):
                raise TypeError(
                    "static RAG manifest source_fingerprints must be an array"
                )
            stale: list[str] = []
            for raw in raw_fingerprints:
                if not isinstance(raw, Mapping):
                    raise TypeError("static RAG source fingerprint must be an object")
                source_path = Path(str(raw["path"]))
                if not source_path.is_absolute():
                    source_path = index_dir / source_path
                expected = str(raw["sha256"])
                if (
                    not source_path.exists()
                    or _sha256_path(source_path) != expected
                ):
                    stale.append(str(source_path))
            if stale:
                raise ValueError(
                    "static RAG index is stale; rebuild after source changes: "
                    + ", ".join(stale)
                )
        selected_encoder = encoder
        if dense and selected_encoder is None:
            model = manifest["dense_model"]
            selected_encoder = E5DenseEncoder(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                device=device,
            )
        return cls(
            manifest=manifest,
            documents=documents,
            embeddings=embeddings,
            encoder=selected_encoder,
        )

    def retrieve(
        self,
        *,
        query: str,
        filters: Mapping[str, Any],
        top_k: int = 8,
    ) -> dict[str, Any]:
        """融合稀疏/稠密排名并返回完整 trace，不让索引内容成为事实源。"""

        candidate_indices = [
            index
            for index, document in enumerate(self.documents)
            if _metadata_matches(document.metadata, filters)
        ]
        if not candidate_indices:
            return {
                "retrieval_mode": "static_hybrid_rag",
                "query": query,
                "filters": dict(filters),
                "results": [],
            }
        candidate_documents = [self.documents[index] for index in candidate_indices]
        sparse_scores = _bm25_scores(
            query,
            [document.text for document in candidate_documents],
        )
        sparse_order = sorted(
            range(len(candidate_documents)),
            key=lambda index: (-float(sparse_scores[index]), candidate_documents[index].document_id),
        )
        dense_scores: np.ndarray | None = None
        dense_order: list[int] = []
        if self.encoder is not None:
            query_embedding = self.encoder.encode_query(query)
            selected_embeddings = self.embeddings[candidate_indices]
            dense_scores = selected_embeddings @ query_embedding.astype(np.float32)
            dense_order = sorted(
                range(len(candidate_documents)),
                key=lambda index: (-float(dense_scores[index]), candidate_documents[index].document_id),
            )
        sparse_ranks = {index: rank + 1 for rank, index in enumerate(sparse_order)}
        dense_ranks = {index: rank + 1 for rank, index in enumerate(dense_order)}
        rrf_k = int(self.manifest.get("rrf_k", 60))
        fused: list[tuple[float, int]] = []
        for index in range(len(candidate_documents)):
            score = 1.0 / (rrf_k + sparse_ranks[index])
            if index in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[index])
            fused.append((score, index))
        fused.sort(
            key=lambda item: (-item[0], candidate_documents[item[1]].document_id)
        )
        results = []
        for fused_score, index in fused[: max(1, top_k)]:
            document = candidate_documents[index]
            try:
                retrieved_content = json.loads(document.text)
            except json.JSONDecodeError:
                retrieved_content = {"text": document.text}
            results.append(
                {
                    "document_id": document.document_id,
                    "item_id": document.item_id,
                    "metadata": document.metadata,
                    "rrf_score": fused_score,
                    "sparse_rank": sparse_ranks[index],
                    "sparse_score": float(sparse_scores[index]),
                    "dense_rank": dense_ranks.get(index),
                    "dense_score": (
                        None if dense_scores is None else float(dense_scores[index])
                    ),
                    # The index is accepted only after source fingerprints are
                    # validated in ``load``. Returning the exact indexed payload
                    # lets the downstream LLM cleaner compare semantics while the
                    # stable document/item IDs preserve authoritative re-mapping.
                    "retrieved_content": retrieved_content,
                    "retrieved_content_sha256": hashlib.sha256(
                        document.text.encode("utf-8")
                    ).hexdigest(),
                }
            )
        return {
            "retrieval_mode": (
                "static_hybrid_rag" if self.encoder is not None else "static_bm25"
            ),
            "index_id": self.manifest.get("index_id"),
            "query": query,
            "filters": dict(filters),
            "candidate_count": len(candidate_documents),
            "results": results,
        }


def _build_parser() -> argparse.ArgumentParser:
    """构造静态索引构建/查询命令。"""

    parser = argparse.ArgumentParser(description="Build or query the static knowledge RAG index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_STATIC_RAG_INDEX_DIR)
    build.add_argument("--device", default="cpu")
    query = subparsers.add_parser("query")
    query.add_argument("query")
    query.add_argument("--index-dir", type=Path, default=DEFAULT_STATIC_RAG_INDEX_DIR)
    query.add_argument("--device", default="cpu")
    query.add_argument("--knowledge-type")
    query.add_argument("--group-id")
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument("--sparse-only", action="store_true")
    return parser


def main() -> int:
    """执行显式构建或人工查询，静态知识变化时必须重新 build。"""

    args = _build_parser().parse_args()
    if args.command == "build":
        manifest = build_static_rag_index(
            output_dir=args.output_dir,
            device=args.device,
        )
        print(
            f"built static RAG index: documents={manifest['document_count']}, "
            f"index_id={manifest['index_id']}, output={args.output_dir}"
        )
        return 0
    rag = StaticKnowledgeRAG.load(
        args.index_dir,
        device=args.device,
        dense=not args.sparse_only,
    )
    filters = {
        key: value
        for key, value in {
            "knowledge_type": args.knowledge_type,
            "group_ids": args.group_id,
        }.items()
        if value is not None
    }
    print(
        json.dumps(
            rag.retrieve(query=args.query, filters=filters, top_k=args.top_k),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
