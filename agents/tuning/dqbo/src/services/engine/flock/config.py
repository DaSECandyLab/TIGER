"""集中处理 Flock 调用 vLLM 时的有效配置解析。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

HALF_ONLY_QUANTIZATION_METHODS = frozenset({"awq", "gptq"})

MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP = {
    "fp8": "fp8_e4m3",
    "nvfp4": "nvfp4",
}

MODEL_DTYPE_ALIASES = {
    "half": "half",
    "float16": "half",
    "bfloat16": "bfloat16",
    "float": "float",
    "float32": "float",
    "auto": "auto",
}

KV_CACHE_DTYPE_ALIASES = {
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "fp8": "fp8_e4m3",
    "float8_per_token_head": "fp8_per_token_head",
}

ACTUAL_MODEL_DTYPES = frozenset({"half", "bfloat16", "float"})
ACTUAL_KV_CACHE_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "fp8_per_token_head",
        "nvfp4",
    }
)


def quantization_requires_half_dtype(value: Any) -> bool:
    return str(value or "none") in HALF_ONLY_QUANTIZATION_METHODS


def canonicalize_model_dtype(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return MODEL_DTYPE_ALIASES.get(normalized, normalized)


def canonicalize_kv_cache_dtype(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return KV_CACHE_DTYPE_ALIASES.get(normalized, normalized)


def is_actual_model_dtype(value: Any) -> bool:
    return canonicalize_model_dtype(value) in ACTUAL_MODEL_DTYPES


def is_actual_kv_cache_dtype(value: Any) -> bool:
    return canonicalize_kv_cache_dtype(value) in ACTUAL_KV_CACHE_DTYPES


def resolve_model_path_for_quantization(
    *,
    model_path: str,
    awq_model_path: str,
    gptq_model_path: str,
    quantization: Any,
) -> str:
    normalized_quantization = str(quantization or "none")
    if normalized_quantization in {"awq", "awq_marlin"}:
        return awq_model_path
    if normalized_quantization in {"gptq", "gptq_marlin"}:
        return gptq_model_path
    return model_path


@lru_cache(maxsize=None)
def load_model_config_json(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def preferred_auto_model_dtype() -> str:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "half"


def resolve_effective_dtype(
    raw_dtype: Any,
    *,
    quantization: Any,
    model_path: str,
    awq_model_path: str,
    gptq_model_path: str,
) -> str | None:
    normalized_dtype = canonicalize_model_dtype(raw_dtype)
    normalized_quantization = str(quantization or "none")
    if normalized_dtype is None:
        return None
    if normalized_dtype == "auto":
        resolved_model_path = resolve_model_path_for_quantization(
            model_path=model_path,
            awq_model_path=awq_model_path,
            gptq_model_path=gptq_model_path,
            quantization=normalized_quantization,
        )
        config = load_model_config_json(resolved_model_path)
        config_dtype = canonicalize_model_dtype(config.get("torch_dtype"))
        if config_dtype in {"half", "bfloat16"}:
            normalized_dtype = config_dtype
        else:
            normalized_dtype = preferred_auto_model_dtype()
    if quantization_requires_half_dtype(normalized_quantization):
        return "half"
    return normalized_dtype


def resolve_quantized_kv_cache_dtype_from_config(
    quantization_config: dict[str, Any] | None,
) -> str | None:
    if not quantization_config:
        return None
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if not quant_method.startswith("modelopt"):
        return None
    inner = quantization_config.get("quantization", quantization_config)
    kv_algo = (
        inner.get("kv_cache_scheme")
        or quantization_config.get("kv_cache_scheme")
        or inner.get("kv_cache_quant_algo")
        or quantization_config.get("kv_cache_quant_algo")
    )
    if isinstance(kv_algo, dict):
        if (
            kv_algo.get("dynamic") is False
            and kv_algo.get("num_bits") == 8
            and kv_algo.get("type") == "float"
        ):
            return "fp8_e4m3"
        if kv_algo.get("num_bits") == 4 and kv_algo.get("type") == "float":
            return "nvfp4"
        return "auto"
    if isinstance(kv_algo, str):
        normalized = kv_algo.strip().lower()
        if normalized in MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP:
            return MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP[normalized]
        return canonicalize_kv_cache_dtype(normalized) or "auto"
    return None


def resolve_effective_kv_cache_dtype(
    raw_kv_cache_dtype: Any,
    *,
    dtype: Any,
    quantization: Any,
    model_path: str,
    awq_model_path: str,
    gptq_model_path: str,
) -> str | None:
    normalized_kv_cache_dtype = canonicalize_kv_cache_dtype(raw_kv_cache_dtype)
    normalized_quantization = str(quantization or "none")
    if normalized_kv_cache_dtype is None:
        return None
    if normalized_kv_cache_dtype != "auto":
        return normalized_kv_cache_dtype

    resolved_model_path = resolve_model_path_for_quantization(
        model_path=model_path,
        awq_model_path=awq_model_path,
        gptq_model_path=gptq_model_path,
        quantization=normalized_quantization,
    )
    config = load_model_config_json(resolved_model_path)
    quantized_kv_cache_dtype = resolve_quantized_kv_cache_dtype_from_config(
        config.get("quantization_config")
    )
    if quantized_kv_cache_dtype not in {None, "auto"}:
        return quantized_kv_cache_dtype

    effective_dtype = resolve_effective_dtype(
        dtype if dtype is not None else "auto",
        quantization=normalized_quantization,
        model_path=model_path,
        awq_model_path=awq_model_path,
        gptq_model_path=gptq_model_path,
    )
    if effective_dtype == "bfloat16":
        return "bfloat16"
    return "float16"
