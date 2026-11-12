"""静态知识模块 自己的版本固定路径；不依赖 Legacy 包。"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
VERSION_ROOT = (
    REPOSITORY_ROOT / "my_storage" / "static_knowledge_base" / "vllm-0.20.0"
)
DEFAULT_KNOWLEDGE_BASE_PATH = VERSION_ROOT / "serve_parameters.json"
DEFAULT_MECHANISM_EVIDENCE_PATH = VERSION_ROOT / "mechanism_evidence.json"
DEFAULT_EXTERNAL_SCREENING_KNOWLEDGE_PATH = (
    VERSION_ROOT / "external_screening_knowledge.json"
)
DEFAULT_STATIC_RAG_INDEX_DIR = VERSION_ROOT / "static_rag"
