"""把 Baseline 锚点和本轮候选应用到 Candidate 自有 Trial namespace。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any


_OFFICIAL_TO_TRIAL = {
    "chunked_prefill": "enable_chunked_prefill",
    "tensor_parallel_size": "tp",
    "pipeline_parallel_size": "pp",
}
_VALUE_ALIASES = {
    "dtype": {"bf16": "bfloat16", "float16": "half", "fp16": "half"},
    "quantization": {None: "none"},
    # 原 BO 的完整组合用字符串 ``none`` 表示关闭 speculative decoding；
    # Candidate trial namespace 则用 None。这里是表示层转换，不改变搜索点。
    "speculative_method": {"none": None},
}


def _normalise_value(name: str, value: Any) -> Any:
    aliases = _VALUE_ALIASES.get(name, {})
    try:
        normalized = aliases.get(value, value)
    except TypeError:
        # JSON arrays/dicts are legitimate vLLM values but cannot be dict keys.
        normalized = value
    if name == "cacheblend_blend_check_layers" and isinstance(normalized, str):
        return [int(item) for item in normalized.split(",") if item.strip()]
    if (
        name == "cacheblend_blend_recompute_ratios"
        and isinstance(normalized, str)
    ):
        return [float(item) for item in normalized.split(",") if item.strip()]
    return normalized


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _render(name: str, value: Any) -> list[str]:
    flag = _flag(name)
    if value is None:
        return []
    if isinstance(value, bool):
        return [flag if value else "--no-" + flag[2:]]
    if isinstance(value, Mapping):
        return [flag, json.dumps(dict(value), sort_keys=True)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [flag, *[str(item) for item in value]]
    return [flag, str(value)]


def _replace_extra_argument(extra: list[str], name: str, rendered: list[str]) -> list[str]:
    positive = _flag(name)
    negative = "--no-" + positive[2:]
    result: list[str] = []
    index = 0
    while index < len(extra):
        if extra[index] not in {positive, negative}:
            result.append(extra[index])
            index += 1
            continue
        index += 1
        while index < len(extra) and not str(extra[index]).startswith("--"):
            index += 1
    result.extend(rendered)
    return result


def apply_parameter_values(namespace: Any, values: Mapping[str, Any]) -> None:
    """优先覆盖 Candidate 自有 runner 字段，其余原样传给 vLLM。"""

    extra = list(getattr(namespace, "extra_vllm_args", []) or [])
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        value = _normalise_value(name, raw_value)
        target = _OFFICIAL_TO_TRIAL.get(name, name)
        if hasattr(namespace, target):
            setattr(namespace, target, value)
        else:
            extra = _replace_extra_argument(extra, name, _render(name, value))
    namespace.extra_vllm_args = extra


def apply_baseline_config_to_namespace(namespace: Any, payload: Mapping[str, Any]) -> None:
    """读取已审计 Baseline；Candidate 随后可在固定域内覆盖其参数。"""

    if payload.get("purpose") != "pre_measurement_baseline_configuration":
        raise ValueError("baseline config has an unsupported purpose")
    merged: dict[str, Any] = {}
    for label in (
        "parameters",
        "runner_parameters",
        "safety_overrides",
        "experiment_overrides",
    ):
        values = payload.get(label, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"baseline config {label} must be an object")
        merged.update(values)
    apply_parameter_values(namespace, merged)
    setattr(
        namespace,
        "_baseline_config_metadata",
        {
            "schema_version": payload.get("schema_version"),
            "purpose": payload.get("purpose"),
            "generator": payload.get("generator"),
        },
    )


def apply_candidate_request(namespace: Any) -> None:
    candidate = getattr(namespace, "_candidate_parameters", None)
    if not isinstance(candidate, Mapping):
        raise TypeError("Candidate trial request lacks candidate parameters")
    active = getattr(namespace, "_candidate_active_parameters", None)
    if active is not None:
        active_names = {str(item) for item in active}
        candidate = {
            name: value for name, value in candidate.items() if name in active_names
        }
    apply_parameter_values(namespace, candidate)
