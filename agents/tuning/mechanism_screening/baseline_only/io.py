"""Shared repository paths and atomic JSON output for mechanism screening."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)
