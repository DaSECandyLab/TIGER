"""Aggregate ROUGE-L gate for native Scoot and LLMConf trials."""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_ROUGE_L = 0.23289767303593403
QUALITY_RATIO = 0.88
QUALITY_MIN = BASELINE_ROUGE_L * QUALITY_RATIO


def aggregate_quality(result: dict) -> float:
    metrics = result.get("sembench_metrics") or {}
    values = [float(row["rougeL"]) for row in metrics.values()
              if isinstance(row, dict) and row.get("rougeL") is not None]
    if len(values) != 12:
        raise ValueError(f"Expected ROUGE-L from 12 SQLs, received {len(values)}")
    return sum(values) / len(values)


def annotate_result(path: str | Path) -> dict:
    path = Path(path)
    result = json.loads(path.read_text())
    quality = aggregate_quality(result)
    gate = {"metric": "mean_rougeL_across_12_sql", "value": quality,
            "baseline": BASELINE_ROUGE_L, "ratio": QUALITY_RATIO,
            "threshold": QUALITY_MIN, "passed": quality >= QUALITY_MIN}
    result["aggregate_quality_gate"] = gate
    result.setdefault("sembench_summary", {})["quality_gate_passed"] = gate["passed"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return result
