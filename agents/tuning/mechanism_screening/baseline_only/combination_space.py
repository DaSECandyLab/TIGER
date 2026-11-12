"""Enumerate mechanisms using the existing Range/BO mechanical validator."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json

from agents.tuning.dqbo.src.services.engine.space import FixedSearchSpace
from .combination_prompts import MECHANISMS

# These existing constraint IDs can be evaluated on the seven-dimensional projection.
# All other constraints are still checked against every full experiment configuration.
PROJECTED_CONSTRAINTS = {
    "representation_template", "cacheblend_compatibility", "prefix_hash_parent",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def canonical(values):
    result = dict(values)
    speculative = result.get('speculative_config')
    if isinstance(speculative, dict):
        result.setdefault('speculative_method', speculative.get('method'))
        result.setdefault('num_speculative_tokens', speculative.get('num_speculative_tokens'))
    for old, new in (("enable_chunked_prefill", "chunked_prefill"),
                     ("tp", "tensor_parallel_size"), ("pp", "pipeline_parallel_size")):
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError(f"conflicting aliases: {old}, {new}")
            result[new] = result.pop(old)
    if result.get("dtype") == "float16":
        result["dtype"] = "half"
    if result.get("speculative_method") is None:
        result["speculative_method"] = "none"
    return result


def reference_config(payload):
    if payload.get("purpose") != "pre_measurement_baseline_configuration":
        raise ValueError("Expected the original pre-measurement baseline configuration")
    result = {}
    for key in ("parameters", "runner_parameters", "safety_overrides", "experiment_overrides"):
        result.update(payload.get(key, {}))
    return canonical(result)


def matches(values, predicate):
    return all(values.get(key) in allowed for key, allowed in predicate.items())


def apply_measured_runtime_constraints(enumeration, baseline):
    """Carry the existing BO backend constraint into screening without repairing IDs."""
    from agents.tuning.candidate_optimizer.src.services.workflow import _runtime_constraint_scope
    from agents.tuning.dqbo.src.services.engine.constraints import uncompressed_kv_dtype_compatible

    result = deepcopy(enumeration)
    scope = _runtime_constraint_scope(baseline)
    result['runtime_constraint_scope'] = scope
    kept = []
    for row in result['combinations']:
        config = row['configuration']
        backend = config.get('attention_backend') or scope['resolved_attention_backend']
        if (backend == 'FLASH_ATTN'
                and not uncompressed_kv_dtype_compatible(config.get('dtype'), config.get('kv_cache_dtype'))):
            result['excluded'].append({
                'combination_id': row['combination_id'], 'mechanisms': row['mechanisms'],
                'reasons': ['attention_backend_uncompressed_kv_dtype_alignment: '
                            'FLASH_ATTN requires FP16/FP16 or BF16/BF16 for uncompressed KV'],
                'constraint_source': scope,
            })
        else:
            kept.append(row)
    if not kept:
        raise ValueError('No legal combinations remain under measured runtime constraints')
    result['combinations'] = kept
    return result


def enumerate_combinations(space, reference, dependency_rules=()):
    """Never infer background legality from a performance hypothesis.

    A mechanism-compatible combination that cannot run at the fixed reference stays
    in the catalog with blockers; it is NOT removed from downstream exploration.
    Dependency rules are explicit input configuration, never LLM-generated.
    """
    missing = set(MECHANISMS) - space.by_name.keys()
    if missing:
        raise ValueError(f"Domain spec must explicitly include all seven mechanisms: {sorted(missing)}")
    domains = {}
    for key in MECHANISMS:
        p = space.by_name[key]
        values = list(p.values) if p.tuned and p.values else [p.fixed_value]
        if not values or any(value is None for value in values):
            raise ValueError(f"Mechanism {key} needs an explicit finite domain")
        domains[key] = values
    # Constructor also checks duplicates and whether any dimensions are tunable.
    projected = FixedSearchSpace(
        [space.by_name[k] for k in MECHANISMS],
        [c for c in space.joint_constraints if c.get("id") in PROJECTED_CONSTRAINTS],
        source_path=space.source_path, source_parameter_results=space.source_parameter_results,
    )
    base = deepcopy(reference)
    missing_background = set(space.by_name) - set(base) - set(MECHANISMS)
    if missing_background:
        raise ValueError(f"Baseline must explicitly resolve background values: {sorted(missing_background)}")
    if any(key not in base for key in MECHANISMS):
        raise ValueError("Baseline must explicitly resolve all seven mechanisms")
    for rule in dependency_rules:
        if set(rule) != {"id", "when", "set", "reason"} or not rule["id"] or not rule["reason"]:
            raise ValueError("Dependency rule requires id, when, set, reason")
        if not rule["when"] or not set(rule["when"]) <= set(MECHANISMS):
            raise ValueError("Dependency conditions must use mechanism dimensions")
        if not rule["set"] or not set(rule["set"]) <= (set(space.by_name) - set(MECHANISMS)):
            raise ValueError("Dependencies may only modify declared background domain parameters")
    catalog, excluded = [], []
    for values in itertools.product(*(domains[k] for k in MECHANISMS)):
        mechanism = dict(zip(MECHANISMS, values))
        identifier = "c_" + digest(mechanism)[:16]
        errors = projected.validate(mechanism)
        if projected.normalize(mechanism) != mechanism:
            errors.append("Inactive mechanism values normalize to another combination")
        if errors:
            excluded.append({"combination_id": identifier, "mechanisms": mechanism, "reasons": errors})
            continue
        full = {**base, **mechanism}
        reasons = {}
        for rule in dependency_rules:
            if matches(mechanism, rule["when"]):
                for key, value in rule["set"].items():
                    if key in reasons and full[key] != value:
                        raise ValueError(f"Conflicting mechanical dependencies for {identifier}: {key}")
                    full[key] = value
                    reasons[key] = {"rule_id": rule["id"], "reason": rule["reason"]}
        normalized = space.normalize(full)
        blockers = space.validate(full)
        if any(normalized[k] != mechanism[k] for k in MECHANISMS):
            blockers.append("Full domain normalization changes a selected mechanism")
        for key, value in normalized.items():
            if value != full.get(key):
                if key in MECHANISMS:
                    continue
                reasons[key] = {"rule_id": "domain_activation", "reason": "Existing domain inactive/reference normalization"}
                full[key] = value
        diff = {k: {"before": base.get(k), "after": v,
                    "basis": "mechanism" if k in MECHANISMS else reasons[k]}
                for k, v in full.items() if v != base.get(k)}
        catalog.append({"combination_id": identifier, "mechanisms": mechanism,
                        "configuration": full, "configuration_hash": digest(full),
                        "mechanical_diff": diff, "blockers": blockers})
    if not catalog:
        raise ValueError("Mechanical enumeration produced no legal combinations")
    return {"domains": domains, "combinations": catalog, "excluded": excluded,
            "scope": "Mechanism legality; blockers preserve combinations not runnable at the frozen reference"}
