"""从 vLLM 0.20.0 官方 CLI 文档构建参数知识库。"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import KnowledgeBase, ParameterGroup, ParameterRecord

DEFAULT_VLLM_VERSION = "0.20.0"
DEFAULT_SERVE_DOC_URL = "https://docs.vllm.ai/en/v0.20.0/cli/serve/"
DEFAULT_EXTENSION_FILENAME = "parameter_extensions.json"
USER_AGENT = "SCOOT-vLLM-parameter-screening/1.0"

_FLAG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(--[A-Za-z0-9][A-Za-z0-9-]*|-[A-Za-z][A-Za-z0-9-]*)"
)
_DEFAULT_PATTERN = re.compile(r"^Default:\s*(.*)$", re.IGNORECASE)
_CHOICES_PATTERN = re.compile(r"^Possible choices:\s*(.*)$", re.IGNORECASE)
_VERSION_PATTERN = re.compile(r"/en/v([^/]+)/")


def _normalize_text(value: str) -> str:
    """清理 HTML 文本中的实体、空白和锚点符号。"""

    lines = []
    for raw_line in html.unescape(value).splitlines():
        line = " ".join(raw_line.split()).strip()
        if line and line != "¶":
            lines.append(line)
    return "\n".join(lines)


def _canonical_parameter_name(flags: Iterable[str]) -> str:
    """从一组 CLI flag 中选出稳定的蛇形参数名。"""

    flag_list = list(flags)
    long_flags = [flag for flag in flag_list if flag.startswith("--")]
    candidates = long_flags or flag_list
    if not candidates:
        raise ValueError("parameter heading contains no CLI flags")
    positive = [
        flag
        for flag in candidates
        if not flag.startswith("--no-") and not flag.startswith("--disable-")
    ]
    chosen = (positive or candidates)[0]
    return chosen.lstrip("-").replace("-", "_")


def _parse_choices(value: str) -> tuple[str, ...]:
    """解析官方文档渲染的逗号分隔候选值。"""

    # 官方页面会将 choices 渲染为逗号分隔的 code 文本。
    return tuple(
        item.strip().strip("`'\"")
        for item in value.split(",")
        if item.strip()
    )


class _ServeDocumentParser(HTMLParser):
    """按 article 内的 h3(Config)/h4(parameter) 流式抽取文档。"""

    _BLOCK_END_TAGS = frozenset(
        {"p", "dd", "dt", "li", "pre", "ul", "ol", "dl", "div"}
    )

    def __init__(self, source_url: str) -> None:
        """初始化流式解析状态与来源 URL。"""

        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self._in_article = False
        self._article_depth = 0
        self._ignored_depth = 0
        self._capture_heading: str | None = None
        self._heading_buffer: list[str] = []
        self._current_group = "General"
        self._current_group_description: list[str] = []
        self._group_descriptions: OrderedDict[str, str] = OrderedDict()
        self._current_flags: tuple[str, ...] | None = None
        self._current_anchor = ""
        self._current_body: list[str] = []
        self.records: list[ParameterRecord] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        """从 HTML 属性中提取 class 集合。"""

        attr_map = dict(attrs)
        return set((attr_map.get("class") or "").split())

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """处理开始标签并切换组标题、参数标题或忽略区状态。"""

        classes = self._classes(attrs)
        if not self._in_article:
            if tag == "article" and "md-content__inner" in classes:
                self._in_article = True
                self._article_depth = 1
            return

        self._article_depth += 1
        if self._ignored_depth:
            self._ignored_depth += 1
            return
        if "headerlink" in classes or tag in {"script", "style", "svg"}:
            self._ignored_depth = 1
            return

        if tag == "h3":
            self._finalize_parameter()
            self._capture_heading = "group"
            self._heading_buffer = []
            return
        if tag == "h4":
            self._finalize_parameter()
            self._finalize_group_description()
            self._capture_heading = "parameter"
            self._heading_buffer = []
            self._current_anchor = dict(attrs).get("id") or ""

    def handle_endtag(self, tag: str) -> None:
        """处理结束标签，并在边界处固化组或参数记录。"""

        if not self._in_article:
            return
        if self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "h3" and self._capture_heading == "group":
            group = _normalize_text("".join(self._heading_buffer))
            if group:
                self._current_group = group
                self._current_group_description = []
            self._capture_heading = None
            self._heading_buffer = []
        elif tag == "h4" and self._capture_heading == "parameter":
            heading = _normalize_text("".join(self._heading_buffer))
            flags = tuple(dict.fromkeys(_FLAG_PATTERN.findall(heading)))
            self._current_flags = flags or None
            self._current_body = []
            self._capture_heading = None
            self._heading_buffer = []
        elif tag in self._BLOCK_END_TAGS:
            self._append_text("\n")

        self._article_depth -= 1
        if tag == "article" and self._article_depth == 0:
            self._finalize_parameter()
            self._finalize_group_description()
            self._in_article = False

    def handle_data(self, data: str) -> None:
        """把文本送入当前标题或正文缓冲区。"""

        if not self._in_article or self._ignored_depth:
            return
        if self._capture_heading is not None:
            self._heading_buffer.append(data)
            return
        self._append_text(data)

    def close(self) -> None:
        """结束解析并提交尚未闭合的最后一条记录。"""

        super().close()
        self._finalize_parameter()
        self._finalize_group_description()

    def _append_text(self, data: str) -> None:
        """按当前状态追加参数正文或组说明。"""

        if self._current_flags is not None:
            self._current_body.append(data)
        elif self._current_group:
            self._current_group_description.append(data)

    def _finalize_group_description(self) -> None:
        """规范化并保存当前 Config 组说明。"""

        if not self._current_group:
            return
        description = _normalize_text("".join(self._current_group_description))
        if description and self._current_group not in self._group_descriptions:
            self._group_descriptions[self._current_group] = description
        self._current_group_description = []

    def _finalize_parameter(self) -> None:
        """从当前标题与正文构造一条 ParameterRecord。"""

        if self._current_flags is None:
            return
        description = _normalize_text("".join(self._current_body))
        lines = description.splitlines()
        default: str | None = None
        choices: tuple[str, ...] = ()
        description_lines: list[str] = []
        for line in lines:
            default_match = _DEFAULT_PATTERN.match(line)
            if default_match:
                default = default_match.group(1).strip() or None
                continue
            choices_match = _CHOICES_PATTERN.match(line)
            if choices_match:
                choices = _parse_choices(choices_match.group(1))
                continue
            description_lines.append(line)

        source_anchor = self._current_anchor
        record_url = (
            f"{self.source_url.rstrip('/')}/#{source_anchor}"
            if source_anchor
            else self.source_url
        )
        self.records.append(
            ParameterRecord(
                name=_canonical_parameter_name(self._current_flags),
                flags=self._current_flags,
                group=self._current_group,
                description="\n".join(description_lines).strip(),
                default=default,
                choices=choices,
                source_url=record_url,
                source_anchor=source_anchor,
            )
        )
        self._current_flags = None
        self._current_anchor = ""
        self._current_body = []

    @property
    def group_descriptions(self) -> dict[str, str]:
        """返回已解析的 Config 组说明副本。"""

        return dict(self._group_descriptions)


def fetch_document(
    source_url: str = DEFAULT_SERVE_DOC_URL, *, timeout: float = 60.0
) -> bytes:
    """下载版本固定的官方 serve 文档原始字节。"""

    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_serve_document(
    document: bytes | str,
    *,
    source_url: str = DEFAULT_SERVE_DOC_URL,
    retrieved_at: str | None = None,
) -> KnowledgeBase:
    """解析官方 HTML，校验重名并构造可复现知识库。"""

    raw = document.encode("utf-8") if isinstance(document, str) else document
    text = raw.decode("utf-8")
    parser = _ServeDocumentParser(source_url)
    parser.feed(text)
    parser.close()

    if not parser.records:
        raise ValueError(
            "no vLLM parameters found; the official document layout may have changed"
        )

    seen_names: set[str] = set()
    duplicates: set[str] = set()
    for record in parser.records:
        if record.name in seen_names:
            duplicates.add(record.name)
        seen_names.add(record.name)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate canonical parameter names: {duplicate_list}")

    grouped: OrderedDict[str, list[ParameterRecord]] = OrderedDict()
    for record in parser.records:
        grouped.setdefault(record.group, []).append(record)

    version_match = _VERSION_PATTERN.search(source_url)
    version = version_match.group(1) if version_match else DEFAULT_VLLM_VERSION
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    descriptions = parser.group_descriptions
    groups = tuple(
        ParameterGroup(
            name=name,
            description=descriptions.get(name, ""),
            parameters=tuple(parameters),
        )
        for name, parameters in grouped.items()
    )
    return KnowledgeBase(
        schema_version=1,
        vllm_version=version,
        source_url=source_url,
        retrieved_at=timestamp,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        groups=groups,
    )


def build_knowledge_base(
    *,
    source_url: str = DEFAULT_SERVE_DOC_URL,
    document_path: Path | None = None,
    timeout: float = 60.0,
) -> KnowledgeBase:
    """从本地快照或远程文档构建基础知识库。"""

    document = (
        document_path.read_bytes()
        if document_path is not None
        else fetch_document(source_url, timeout=timeout)
    )
    return parse_serve_document(document, source_url=source_url)


def save_knowledge_base(knowledge_base: KnowledgeBase, output_path: Path) -> None:
    """把知识库以稳定、可读的 JSON 格式写入磁盘。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(knowledge_base.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_parameter_extensions(
    knowledge_base: KnowledgeBase,
    extension_path: Path,
) -> KnowledgeBase:
    """合并版本固定的扩展参数与官方参数元数据覆盖。

    官方 serve 文档仍可独立复现；同目录扩展文件补充连接器与编排层参数，
    它们参与同一推理路径，但不是 vLLM 顶层 CLI 参数。
    """

    payload = json.loads(extension_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"knowledge extension must be an object: {extension_path}")
    extension_version = str(payload.get("vllm_version", ""))
    if extension_version != knowledge_base.vllm_version:
        raise ValueError(
            "knowledge extension and base have different vLLM versions: "
            f"{extension_version!r} != {knowledge_base.vllm_version!r}"
        )

    overrides = payload.get("parameter_overrides", [])
    if not isinstance(overrides, list):
        raise TypeError("parameter_extensions.parameter_overrides must be an array")
    override_by_name: dict[str, dict[str, object]] = {}
    for raw_override in overrides:
        if not isinstance(raw_override, dict):
            raise TypeError("each parameter override must be an object")
        name = str(raw_override.get("name", "")).strip()
        if not name or name in override_by_name:
            raise ValueError(f"invalid or duplicate parameter override: {name!r}")
        override_by_name[name] = raw_override

    known_names = {
        record.name
        for group in knowledge_base.groups
        for record in group.parameters
    }
    unknown_overrides = set(override_by_name) - known_names
    if unknown_overrides:
        raise ValueError(
            "parameter overrides refer to unknown base parameters: "
            + ", ".join(sorted(unknown_overrides))
        )

    merged_groups: list[ParameterGroup] = []
    for group in knowledge_base.groups:
        records: list[ParameterRecord] = []
        for record in group.parameters:
            override = override_by_name.get(record.name)
            if override is None:
                records.append(record)
                continue
            changes: dict[str, object] = {}
            for field_name in (
                "namespace",
                "source_kind",
                "tunability",
            ):
                if field_name in override:
                    changes[field_name] = str(override[field_name])
            for field_name in (
                "parent_parameters",
                "active_when",
                "search_domain",
                "conflicts_or_overrides",
                "validation_evidence",
            ):
                if field_name in override:
                    raw_values = override[field_name]
                    if not isinstance(raw_values, list):
                        raise TypeError(
                            f"override {record.name}.{field_name} must be an array"
                        )
                    changes[field_name] = tuple(str(item) for item in raw_values)
            records.append(replace(record, **changes))
        merged_groups.append(replace(group, parameters=tuple(records)))

    raw_groups = payload.get("groups", [])
    if not isinstance(raw_groups, list):
        raise TypeError("parameter_extensions.groups must be an array")
    extension_groups = [ParameterGroup.from_dict(item) for item in raw_groups]
    extension_names = [
        record.name for group in extension_groups for record in group.parameters
    ]
    duplicates = known_names.intersection(extension_names)
    if len(extension_names) != len(set(extension_names)):
        duplicates.update(
            name for name in extension_names if extension_names.count(name) > 1
        )
    if duplicates:
        raise ValueError(
            "duplicate parameter names across base and extensions: "
            + ", ".join(sorted(duplicates))
        )
    merged_groups.extend(extension_groups)
    effective_hash = hashlib.sha256(
        (
            knowledge_base.source_sha256
            + "\n"
            + hashlib.sha256(extension_path.read_bytes()).hexdigest()
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        knowledge_base,
        groups=tuple(merged_groups),
        knowledge_sha256=effective_hash,
    )


def load_knowledge_base(
    path: Path,
    *,
    include_extensions: bool = True,
) -> KnowledgeBase:
    """加载基础知识库，并按需合并同目录扩展文件。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    knowledge_base = KnowledgeBase.from_dict(data)
    if knowledge_base.schema_version != 1:
        raise ValueError(
            f"unsupported knowledge-base schema: {knowledge_base.schema_version}"
        )
    extension_path = path.with_name(DEFAULT_EXTENSION_FILENAME)
    if include_extensions and extension_path.exists():
        knowledge_base = _load_parameter_extensions(
            knowledge_base,
            extension_path,
        )
    elif knowledge_base.knowledge_sha256 is None:
        knowledge_base = replace(
            knowledge_base,
            knowledge_sha256=knowledge_base.source_sha256,
        )
    return knowledge_base


def validate_scenario_parameter_coverage(
    knowledge_base: KnowledgeBase,
    scenario: Mapping[str, Any],
) -> None:
    """校验场景搜索契约已经映射到真实、有限的知识库叶子。

    该检查专门防止“LLM 正确删除复合 JSON 容器，但本地知识库没有录入
    对应叶子”的静默漏参。这里只检查结构覆盖与有限域一致性，不替代后续
    场景筛选和五道门对性能意义、可执行性的判断。
    """

    records = {
        record.name: record
        for group in knowledge_base.groups
        for record in group.parameters
    }
    missing_parents = sorted(
        {
            parent
            for record in records.values()
            for parent in record.parent_parameters
            if parent not in records
        }
    )
    if missing_parents:
        raise ValueError(
            "knowledge-base parameter dependencies reference unknown parents: "
            + ", ".join(missing_parents)
        )

    contracts = scenario.get("search_contracts", [])
    if not isinstance(contracts, list):
        raise TypeError("scenario.search_contracts must be an array")
    seen_primary: set[str] = set()
    for contract in contracts:
        if not isinstance(contract, Mapping):
            raise TypeError("scenario.search_contracts items must be objects")
        contract_id = str(contract.get("id", "")).strip()
        primary = str(contract.get("primary_parameter", "")).strip()
        if not contract_id or not primary:
            raise ValueError(
                "each scenario search contract needs id and primary_parameter"
            )
        if primary in seen_primary:
            raise ValueError(
                f"duplicate scenario search contract primary parameter: {primary}"
            )
        seen_primary.add(primary)
        record = records.get(primary)
        if record is None:
            raise ValueError(
                f"scenario search contract {contract_id} references missing "
                f"knowledge-base leaf {primary}"
            )
        if record.tunability == "generated_container":
            raise ValueError(
                f"scenario search contract {contract_id} uses generated container "
                f"{primary}; declare its bounded executable leaves instead"
            )
        candidate_values = contract.get("candidate_values")
        if not isinstance(candidate_values, list) or len(candidate_values) < 2:
            raise ValueError(
                f"scenario search contract {contract_id} must provide at least "
                "two candidate_values"
            )
        normalized_candidates = {str(value) for value in candidate_values}
        if len(normalized_candidates) < 2:
            raise ValueError(
                f"scenario search contract {contract_id} needs two distinct values"
            )
        if record.search_domain and normalized_candidates != set(
            record.search_domain
        ):
            raise ValueError(
                f"scenario search contract {contract_id} candidate_values differ "
                f"from {primary}.search_domain"
            )
