"""本地执行流程共享的机械审计结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditReport:
    """记录校验结果和可重试错误。"""

    passed: bool
    audit_type: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retryable: bool = False
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
