"""Local workload identity and runtime features; no external memory service."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

DYNAMIC_WORKLOAD_FEATURE_NAMES = (
    "gpu_utilization_mean",
    "gpu_utilization_p90",
    "kv_cache_usage_mean",
    "kv_cache_usage_p90",
    "hbm_usage_mean",
    "hbm_usage_p90",
    "mean_ttft_ms",
    "mean_e2e_latency_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "actual_batch_size_mean",
    "actual_batch_size_p95",
    "actual_global_in_flight_peak",
    "prefix_cache_hit_rate",
    "lmcache_cache_hit_rate",
)


WORKLOAD_PROFILE_SCHEMA = "sembench.bo.workload-profile.v1"


STATIC_WORKLOAD_CONTEXT_SCHEMA = "sembench.bo.static-workload-context.v1"


WORKLOAD_SIGNATURE_SCHEMA = "sembench.bo.workload-signature.v1"


WORKLOAD_ANCHOR_SIMILARITY_METHOD = (
    "deterministic_static_workload_feature_similarity_v1"
)


WORKLOAD_SIMILARITY_FEATURE_NAMES = (
    "terminal_count",
    "request_control_mode",
    "max_in_flight",
    "effective_per_terminal_max_in_flight",
    "request_cap_strategy",
    "arrival_mode",
    "request_rate",
    "concurrent_query_mix",
    "terminal_query_slot_count",
    "terminal_unique_query_count",
    "system_prompt_len_p50",
    "system_prompt_len_p95",
    "system_prompt_len_max",
    "output_len_p50",
    "output_len_p95",
    "output_len_max",
    "input_output_ratio",
    "theoretical_global_in_flight_upper",
    "candidate_rows_p50",
    "candidate_rows_p90",
    "candidate_rows_max",
    "data_prompt_len_p50",
    "data_prompt_len_p95",
    "data_prompt_len_max",
    "prefix_reuse_ratio",
    "shared_prefix_len_p50",
    "shared_prefix_len_p90",
    "shared_prefix_len_max",
    "shared_cacheblend_len_p50",
    "shared_cacheblend_len_p90",
    "shared_cacheblend_len_max",
    "cacheblend_reuse_ratio",
    "generation_tokens_per_request",
    "prompt_tokens_per_request",
    "mean_llm_requests_per_query",
    "p50_llm_requests_per_query",
    "p95_llm_requests_per_query",
    "max_llm_requests_per_query",
    "query_instance_count",
    "operator_mix",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _workload_similarity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _workload_similarity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_workload_similarity_value(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def build_workload_profile(
    baseline_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the full retained workload profile and its stable comparison view."""

    baseline = baseline_result if isinstance(baseline_result, Mapping) else {}
    features = baseline.get("flock_workload_features")
    groups = baseline.get("flock_workload_feature_groups")
    full_features = dict(features) if isinstance(features, Mapping) else {}
    full_groups = dict(groups) if isinstance(groups, Mapping) else {}
    similarity_features = {
        name: _workload_similarity_value(full_features[name])
        for name in WORKLOAD_SIMILARITY_FEATURE_NAMES
        if name in full_features and full_features[name] is not None
    }
    canonical = json.dumps(
        similarity_features,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": WORKLOAD_PROFILE_SCHEMA,
        "available": bool(full_features),
        "source": baseline.get("workload_feature_json_path")
        or baseline.get("trial_result_path"),
        "features": _json_value(full_features),
        "feature_groups": _json_value(full_groups),
        "similarity_feature_policy": "static_workload_semantics_v1",
        "similarity_features": similarity_features,
        "similarity_feature_count": len(similarity_features),
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def build_static_workload_context(
    baseline_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the canonical static-only workload view supplied to BO agents.

    ``build_workload_profile`` deliberately retains the complete feature payload for
    memory/audit consumers.  Agent prompts must not mistake per-Trial runtime
    measurements in that payload for immutable workload facts, so this view exposes
    only the features covered by ``WORKLOAD_SIMILARITY_FEATURE_NAMES``.
    """

    profile = build_workload_profile(baseline_result)
    features = dict(profile.get("similarity_features") or {})
    return {
        "schema": STATIC_WORKLOAD_CONTEXT_SCHEMA,
        "available": bool(features),
        "source": profile.get("source"),
        "feature_policy": "static_workload_semantics_v1",
        "features": features,
        "feature_count": len(features),
        "fingerprint": profile.get("fingerprint"),
    }


def _dynamic_workload_source(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the feature-bearing object without treating objectives as features."""

    direct = value.get("dynamic_workload_features")
    if isinstance(direct, Mapping):
        nested = direct.get("features")
        return nested if isinstance(nested, Mapping) else direct
    flock_features = value.get("flock_workload_features")
    if isinstance(flock_features, Mapping):
        return flock_features
    features = value.get("features")
    if isinstance(features, Mapping):
        return features
    for wrapper_name in ("result", "dynamic_workload", "workload_context"):
        wrapped = value.get(wrapper_name)
        if isinstance(wrapped, Mapping):
            resolved = _dynamic_workload_source(wrapped)
            if resolved:
                return resolved
    result_path = value.get("result_path")
    if isinstance(result_path, str) and result_path.strip():
        path = Path(result_path).expanduser()
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            if isinstance(payload, Mapping):
                return _dynamic_workload_source(payload)
    return {}


def extract_dynamic_workload_features(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Extract the canonical non-objective runtime feature vector for one Trial."""

    source = _dynamic_workload_source(value or {})
    extracted: dict[str, Any] = {}
    for name in DYNAMIC_WORKLOAD_FEATURE_NAMES:
        if name not in source or source[name] is None:
            continue
        feature = source[name]
        if (
            isinstance(feature, (int, float))
            and not isinstance(feature, bool)
            and not math.isfinite(float(feature))
        ):
            continue
        extracted[name] = _json_value(feature)
    return extracted


def normalize_workload_anchor(
    value: Mapping[str, Any] | None,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Normalize every workload-feature retrieval anchor to one stable profile.

    Callers may hold a Trial result, a workload extraction bundle, an already
    materialized workload profile, or an Agent context that wraps one of those
    shapes.  Keeping this normalization here prevents each consumer from
    selecting a different subset of workload features.
    """

    raw = value if isinstance(value, Mapping) else {}
    similarity_features = raw.get("similarity_features")
    if isinstance(similarity_features, Mapping):
        normalized_features = {
            name: _workload_similarity_value(similarity_features[name])
            for name in WORKLOAD_SIMILARITY_FEATURE_NAMES
            if name in similarity_features and similarity_features[name] is not None
        }
        canonical = json.dumps(
            normalized_features,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        full_features = raw.get("features")
        feature_groups = raw.get("feature_groups")
        return {
            "schema": WORKLOAD_PROFILE_SCHEMA,
            "available": bool(normalized_features),
            "source": source or raw.get("source"),
            "features": _json_value(
                dict(full_features) if isinstance(full_features, Mapping) else {}
            ),
            "feature_groups": _json_value(
                dict(feature_groups) if isinstance(feature_groups, Mapping) else {}
            ),
            "similarity_feature_policy": "static_workload_semantics_v1",
            "similarity_features": normalized_features,
            "similarity_feature_count": len(normalized_features),
            "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    flock_features = raw.get("flock_workload_features")
    if isinstance(flock_features, Mapping):
        payload = dict(raw)
        if source:
            payload["workload_feature_json_path"] = source
        return build_workload_profile(payload)

    extracted_features = raw.get("features")
    if isinstance(extracted_features, Mapping):
        return build_workload_profile(
            {
                "flock_workload_features": extracted_features,
                "flock_workload_feature_groups": raw.get("groups")
                or raw.get("feature_groups")
                or {},
                "workload_feature_json_path": source,
            }
        )

    for wrapper_name in ("workload_context", "dynamic_workload", "workload"):
        wrapped = raw.get(wrapper_name)
        if isinstance(wrapped, Mapping):
            profile = normalize_workload_anchor(wrapped, source=source)
            if profile["available"]:
                return profile
    return build_workload_profile(
        {"workload_feature_json_path": source} if source else {}
    )


def _flatten_similarity_features(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for name, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            path = f"{prefix}.{name}" if prefix else str(name)
            flattened.update(_flatten_similarity_features(item, prefix=path))
        return flattened
    return {prefix: value} if prefix else {}


def _scalar_similarity(left: Any, right: Any) -> float:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        left_number = float(left)
        right_number = float(right)
        scale = max(abs(left_number), abs(right_number), 1e-12)
        return max(0.0, 1.0 - abs(left_number - right_number) / scale)
    if isinstance(left, list) and isinstance(right, list):
        left_set = {
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in left
        }
        right_set = {
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in right
        }
        union = left_set | right_set
        return len(left_set & right_set) / len(union) if union else 1.0
    return 1.0 if left == right else 0.0


def similar_workload_anchor_similarity(
    target: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two static workload anchors on a deterministic ``[0, 1]`` scale.

    Numeric features use relative distance, lists use Jaccard similarity, and
    other scalar values use exact equality.  Missing target features reduce the
    score through a coverage factor instead of silently disappearing.  Exact
    fingerprint equality remains separately visible through ``matched`` so an
    exact-only consumer can keep its stronger contract.
    """

    target_profile = normalize_workload_anchor(target)
    candidate_kind: str
    descriptors: dict[str, Mapping[str, Any]] = {}
    if candidate.get("schema") == WORKLOAD_SIGNATURE_SCHEMA:
        candidate_kind = "workload_signature"
        raw_features = candidate.get("representative_features", [])
        if not isinstance(raw_features, list):
            raise TypeError(
                "workload signature representative_features must be an array"
            )
        target_values = dict(target_profile.get("similarity_features") or {})
        for raw_feature in raw_features:
            if not isinstance(raw_feature, Mapping):
                continue
            name = str(raw_feature.get("name") or "")
            descriptor = raw_feature.get("descriptor")
            if name and isinstance(descriptor, Mapping):
                descriptors[name] = descriptor
        candidate_feature_count = len(raw_features)
    else:
        candidate_kind = "workload_profile"
        candidate_profile = normalize_workload_anchor(candidate)
        target_values = _flatten_similarity_features(
            target_profile.get("similarity_features", {})
        )
        candidate_values = _flatten_similarity_features(
            candidate_profile.get("similarity_features", {})
        )
        descriptors = {
            name: {"kind": "constant", "value": value}
            for name, value in candidate_values.items()
        }
        candidate_feature_count = len(candidate_values)

    contributions: dict[str, float] = {}
    for name in sorted(set(target_values) & set(descriptors)):
        descriptor = descriptors[name]
        target_value = target_values[name]
        kind = descriptor.get("kind")
        if kind == "constant" and "value" in descriptor:
            feature_score = _scalar_similarity(target_value, descriptor["value"])
        elif (
            kind == "numeric_range"
            and isinstance(target_value, (int, float))
            and not isinstance(target_value, bool)
        ):
            lower = float(descriptor["min"])
            upper = float(descriptor["max"])
            value = float(target_value)
            if lower <= value <= upper:
                feature_score = 1.0
            else:
                nearest = lower if value < lower else upper
                scale = max(abs(value), abs(nearest), 1e-12)
                feature_score = max(0.0, 1.0 - abs(value - nearest) / scale)
        elif kind == "categorical_set" and isinstance(descriptor.get("values"), list):
            feature_score = max(
                (
                    _scalar_similarity(target_value, item)
                    for item in descriptor["values"]
                ),
                default=0.0,
            )
        else:
            continue
        contributions[name] = round(feature_score, 6)
    exact_descriptor_contract = candidate_kind != "workload_signature" or (
        len(descriptors) == len(target_values)
        and all(
            descriptor.get("kind") == "constant" and "value" in descriptor
            for descriptor in descriptors.values()
        )
    )
    exact_match = (
        bool(target_values)
        and exact_descriptor_contract
        and len(contributions) == len(target_values)
        and candidate_feature_count == len(target_values)
        and all(value == 1.0 for value in contributions.values())
    )
    shared_feature_count = len(contributions)
    target_feature_count = len(target_values)
    mean_feature_similarity = (
        sum(contributions.values()) / shared_feature_count
        if shared_feature_count
        else 0.0
    )
    target_coverage = (
        shared_feature_count / target_feature_count if target_feature_count else 0.0
    )
    score = mean_feature_similarity * target_coverage
    result = {
        "method": WORKLOAD_ANCHOR_SIMILARITY_METHOD,
        "feature_policy": "static_workload_semantics_v1",
        "match_mode": "feature_similarity",
        "matched": exact_match,
        "exact_match": exact_match,
        "candidate_anchor_kind": candidate_kind,
        "score": round(score, 6),
        "mean_feature_similarity": round(mean_feature_similarity, 6),
        "target_feature_coverage": round(target_coverage, 6),
        "shared_feature_count": shared_feature_count,
        "target_feature_count": target_feature_count,
        "shared_features": sorted(contributions),
        "feature_similarities": contributions,
        "target_fingerprint": target_profile.get("fingerprint"),
        "reason": (
            "exact_static_workload_match"
            if exact_match
            else (
                "similar_static_workload_features"
                if score > 0.0
                else "no_similar_static_workload_features"
            )
        ),
    }
    if candidate_kind == "workload_signature":
        result["signature_feature_count"] = candidate_feature_count
    else:
        result["candidate_feature_count"] = candidate_feature_count
        result["candidate_fingerprint"] = candidate_profile.get("fingerprint")
    return result


def build_system_scope(baseline_result: Mapping[str, Any] | None) -> dict[str, str]:
    """Build the stable system/model/hardware scope copied into Observations."""

    baseline = baseline_result if isinstance(baseline_result, Mapping) else {}
    model_path = (
        baseline.get("model")
        or baseline.get("served_model_name")
        or baseline.get("resolved_model_path")
        or "unknown"
    )
    model = Path(str(model_path)).name or str(model_path)
    runtime = baseline.get("runtime_environment")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    inventory = runtime.get("gpu_inventory")
    gpu_names = (
        [
            str(item.get("name") or "unknown")
            for item in inventory
            if isinstance(item, Mapping)
        ]
        if isinstance(inventory, list)
        else []
    )
    if gpu_names:
        counts: dict[str, int] = {}
        for name in gpu_names:
            counts[name] = counts.get(name, 0) + 1
        hardware = ", ".join(
            f"{name} x{count}" for name, count in sorted(counts.items())
        )
    else:
        hardware = str(baseline.get("hardware") or "unknown")
    return {
        "system": str(baseline.get("system") or baseline.get("model_impl") or "vLLM"),
        "model": model,
        "hardware": hardware,
    }


