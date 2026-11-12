"""Materialize fixed workload domains without invoking the former Range agent."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/workload_tuning_ranges.json"
SCHEMA = "sembench.workload-tuning-ranges.v1"
ALIASES = {"movie": "movies", "cnn": "cnn_dailymail"}


def _load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ranges(workload: str, model: str, config=DEFAULT_CONFIG) -> dict[str, Any]:
    payload = _load(config)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported fixed workload range configuration")
    workload = ALIASES.get(workload, workload)
    try:
        return deepcopy(payload["workloads"][workload][model])
    except KeyError:
        raise ValueError(f"No fixed tuning ranges for {workload!r}, {model!r}") from None


def _domain(entry, screening_result=None, baseline_config=None):
    parameters = deepcopy(entry["parameters"])
    constraints = deepcopy(entry["joint_constraints"])
    if screening_result is not None:
        # Preserve complete joint alternatives and their mechanical bindings;
        # flattening them into independent categories would resurrect deletions.
        from agents.tuning.mechanism_screening.handoff import (
            retained_alternatives, combination_constraint,
        )
        alternatives = retained_alternatives(_load(screening_result))
        by_name = {p["name"]: p for p in parameters}
        # Historical domains omit mechanisms frozen outside BO. Carry them as
        # explicit baseline constants so the seven-mechanism joint constraint
        # remains executable without introducing any new searchable values.
        from agents.tuning.mechanism_screening.baseline_only.combination_space import reference_config
        missing_frozen = [name for name in ("chunked_prefill", "cacheblend_mode", "speculative_method")
                          if name not in by_name and not (name == "chunked_prefill" and "enable_chunked_prefill" in by_name)]
        baseline = reference_config(_load(baseline_config)) if missing_frozen and baseline_config else {}
        for name in ("chunked_prefill", "cacheblend_mode", "speculative_method"):
            if name in by_name or (name == "chunked_prefill" and "enable_chunked_prefill" in by_name):
                continue
            if name not in baseline:
                raise ValueError(f"Screened parameter is absent from fixed ranges and explicit baseline: {name}")
            value = baseline[name]
            if any(row["mechanisms"][name] != value for row in alternatives):
                raise ValueError(f"Screened value changes a baseline-frozen mechanism: {name}")
            record = {"name": name, "bo_name": name, "decision": "fixed", "domain": None,
                      "fixed_value": value, "reference_value": value, "activation_conditions": []}
            parameters.append(record)
            by_name[name] = record
        for alternative in alternatives:
            for name, value in {**alternative["mechanisms"], **alternative["required_bindings"]}.items():
                record = by_name.get(name)
                if record is None and name == "chunked_prefill":
                    record = by_name.get("enable_chunked_prefill")
                if record is None:
                    raise ValueError(f"Screened parameter is absent from the fixed ranges: {name}")
                # Validation below checks each retained value against its
                # original domain, including fixed decisions and integer steps.
                candidate = _parameter_domain(record)
                if not candidate.contains(value):
                    raise ValueError(f"Screened value exceeds fixed ranges: {name}={value!r}")
            # Some historical domains use the equivalent vLLM spelling.
            if "chunked_prefill" not in by_name and "enable_chunked_prefill" in by_name:
                value = alternative["mechanisms"].pop("chunked_prefill")
                alternative["mechanisms"]["enable_chunked_prefill"] = value
        constraints = [c for c in constraints if c.get("id") != "screened_mechanism_combinations"]
        constraint = combination_constraint(alternatives)
        if "chunked_prefill" not in by_name and "enable_chunked_prefill" in by_name:
            constraint["parameters"] = ["enable_chunked_prefill" if n == "chunked_prefill" else n
                                        for n in constraint["parameters"]]
        constraints.append(constraint)
    return parameters, constraints


def _parameter_domain(record):
    from .src.services.engine.space import ParameterDomain, FixedSearchSpace
    domain = record.get("domain")
    return ParameterDomain(
        name=record["name"], bo_name=record.get("bo_name", record["name"]),
        decision=record["decision"],
        kind=domain["kind"] if domain else FixedSearchSpace._infer_kind(record.get("fixed_value")),
        domain=domain, fixed_value=record.get("fixed_value"),
        reference_value=record.get("reference_value"),
        activation_conditions=tuple(record.get("activation_conditions", [])),
    )


def materialize_ranges(*, workload: str, model: str, baseline_config: str,
                       output_dir: str | Path, config=DEFAULT_CONFIG,
                       screening_result: str | None = None) -> dict[str, str]:
    """Write the legacy-compatible domain and parameter manifest for DQBO."""
    from .src.services.engine.space import FixedSearchSpace
    from .src.services.engine.history import write_json_atomic

    config = Path(config).expanduser().resolve()
    baseline = Path(baseline_config).expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    output = Path(output_dir).expanduser().resolve()
    entry = load_ranges(workload, model, config)
    parameters, constraints = _domain(entry, screening_result, baseline)
    parameter_path = output / "parameter_results.json"
    domain_path = output / "bo_domain_spec.json"
    manifest = {
        "schema_version": "sembench.fixed-range-parameters.v1",
        "parameters": [{"name": p["name"], "enters_bo": True} for p in parameters],
    }
    domain = {
        "module": "vllm_parameter_domain_design", "screening_source": "fixed_config",
        "source_parameter_results": str(parameter_path), "source_baseline_config": str(baseline),
        "manual_search_ranges_used": True, "screening_candidate_values_used": False,
        "parameters": parameters, "joint_constraints": constraints,
        "fixed_range_source": {
            "config": str(config), "sha256": _hash(config),
            "workload": ALIASES.get(workload, workload), "model": model,
            "provenance": entry["provenance"],
            "screening_result": str(Path(screening_result).resolve()) if screening_result else None,
            "screening_sha256": _hash(screening_result) if screening_result else None,
            "baseline_sha256": _hash(baseline) if screening_result else None,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    # Never overwrite artifacts belonging to an existing, frozen BO context.
    for path, payload in [(parameter_path, manifest), (domain_path, domain)]:
        if path.exists() and _load(path) != payload:
            raise ValueError(f"Fixed range inputs changed; use a new output directory: {path}")
    for path, payload in [(parameter_path, manifest), (domain_path, domain)]:
        if not path.exists():
            write_json_atomic(path, payload)
    FixedSearchSpace.from_json(domain_path)
    return {"parameter_results": str(parameter_path), "domain_spec": str(domain_path)}


def validate_fixed_domain(payload: dict[str, Any]) -> None:
    """Reject edited manifests or screened combinations at the BO boundary."""
    source = payload["fixed_range_source"]
    if _hash(source["config"]) != source["sha256"]:
        raise ValueError("Fixed workload range configuration changed")
    screening = source.get("screening_result")
    if screening and _hash(screening) != source["screening_sha256"]:
        raise ValueError("Mechanism screening result changed")
    if screening and _hash(payload["source_baseline_config"]) != source.get("baseline_sha256"):
        raise ValueError("Baseline configuration changed after screening handoff")
    entry = load_ranges(source["workload"], source["model"], source["config"])
    parameters, constraints = _domain(entry, screening, payload.get("source_baseline_config"))
    if payload["parameters"] != parameters or payload.get("joint_constraints") != constraints:
        raise ValueError("DQBO domain differs from the configured fixed workload ranges")
