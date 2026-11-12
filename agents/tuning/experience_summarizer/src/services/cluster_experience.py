"""Post-run, static-workload-anchored consolidation of native BO observations.

No experiments are executed and trial order is never interpreted as a repair.
All empirical claims are executable predicates audited against the full snapshot.
The optional structured LLM selects hypotheses, then revises their disposition;
it cannot supply counts, remove counterexamples, or change measured outcomes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from agents.tuning.dqbo.src.services.workload_facts import (
    build_system_scope,
    normalize_workload_anchor,
    similar_workload_anchor_similarity,
)
from agents.tuning.dqbo.src.services.engine.history import write_json_atomic

SCHEMA = "sembench.cluster-experience.v1"
ALGORITHM = "complete-link-pattern-counterexample.v1.11"
QUALITY_FIELDS = ("quantization", "dtype", "kv_cache_dtype")
# Explicit allowlist prevents paths, derived model artifacts, metrics and free text
# from silently leaking into the representation. Extend via policy, not heuristics.
CONFIG_FIELDS = (
    "tp", "data_parallel_size", "model_impl", "data_parallel_backend",
    "decode_context_parallel_size", "cp_kv_cache_interleave_size",
    "disable_nccl_for_dp_synchronization", "cpu_offload_gb",
    "kv_offloading_backend", "kv_offloading_size", "cacheblend_mode",
    "cacheblend_blend_check_layers",
    "speculative_method", "num_speculative_tokens", "max_num_seqs",
    "max_num_batched_tokens", "block_size", "long_prefill_token_threshold",
    "max_model_len", "gpu_memory_utilization", "enforce_eager",
    "enable_chunked_prefill", "scheduler_delay_factor", "enable_prefix_caching",
    "prefix_caching_hash_algo", "scheduler_reserve_full_isl", "async_scheduling",
    "disable_cascade_attn", "enable_flashinfer_autotune", "disable_custom_all_reduce",
) + QUALITY_FIELDS
PARTITION_FIELDS = ("tp", "data_parallel_size", "model_impl", "data_parallel_backend")
NUMERIC_CATEGORIES = {"block_size", "decode_context_parallel_size", "cp_kv_cache_interleave_size"}
METRICS = ("query_throughput_qps", "mean_query_execution_latency_ms",
           "p95_query_execution_latency_ms")


@dataclass(frozen=True)
class Policy:
    cluster_diameter: float = 0.38
    neighbor_distance: float = 0.48
    near_front_tolerance: float = 0.15
    negative_min_relative_gap: float = 0.05
    minimum_unique_support: int = 2
    max_pattern_conditions: int = 3
    max_high_performance_patterns: int = 8
    quality_fields: tuple[str, ...] = QUALITY_FIELDS
    preference_ignore_runtime_failures: bool = False

    def __post_init__(self):
        if not 0 < self.cluster_diameter <= self.neighbor_distance <= 1:
            raise ValueError("require 0 < cluster_diameter <= neighbor_distance <= 1")
        if not 0 <= self.near_front_tolerance < 1:
            raise ValueError("invalid near-front tolerance")
        if not 0 < self.negative_min_relative_gap <= 1:
            raise ValueError("negative relative gap must be in (0, 1]")
        if self.minimum_unique_support < 2 or not 1 <= self.max_pattern_conditions <= 3:
            raise ValueError("require >=2 independent configurations and 1..3 conditions")
        if self.max_high_performance_patterns < 1:
            raise ValueError("max patterns must be positive")
        if not self.quality_fields or not set(self.quality_fields) <= set(CONFIG_FIELDS):
            raise ValueError("quality fields must be known configuration parameters")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def summary_client(mode="auto", env_path=None):
    """Read project dotenv configuration without executing shell or exporting secrets."""
    if mode not in ("auto", "required", "off"):
        raise ValueError("invalid LLM mode")
    if mode == "off":
        return None
    import os
    from dotenv import dotenv_values
    from agents.tuning.candidate_optimizer.src.services.direct_inference import default_candidate_structured_inference_client

    root = Path(__file__).resolve().parents[5]
    paths = [root / "SemBench" / ".env", root / ".env"]
    if env_path is not None:
        explicit = Path(env_path).resolve()
        if not explicit.is_file():
            raise ValueError("explicit dotenv file does not exist")
        paths.append(explicit)
    allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL"}
    values = {}
    for path in paths:
        if path.is_file():
            values.update({k: v for k, v in dotenv_values(path).items() if k in allowed and v})
    values.update({k: v for k, v in os.environ.items() if k in allowed and v})
    if mode == "auto" and not values.get("DEEPSEEK_API_KEY"):
        return None
    return default_candidate_structured_inference_client(environ=values)


def _effective(candidate):
    c = {k: candidate[k] for k in CONFIG_FIELDS if k in candidate}
    if c.get("cacheblend_mode") == "off":
        c.pop("cacheblend_blend_check_layers", None)
    if "tp" not in c and "tensor_parallel_size" in candidate:
        c["tp"] = candidate["tensor_parallel_size"]
    if c.get("enable_prefix_caching") is False:
        c.pop("prefix_caching_hash_algo", None)
    if c.get("enable_chunked_prefill") is False:
        c.pop("long_prefill_token_threshold", None)
    if c.get("speculative_method") in (None, "none"):
        c.pop("num_speculative_tokens", None)
    if not c.get("kv_offloading_size"):
        c.pop("kv_offloading_backend", None)
    if c.get("data_parallel_size") == 1:
        c.pop("disable_nccl_for_dp_synchronization", None)
    return c


def extract_evidence(history):
    """Recompute frozen quality gates; missing/invalid measurements are unknown."""
    contract = history.get("quality_constraint") or {}
    thresholds = contract.get("query_metric_thresholds") or {}
    expected = {f"{q}.{m}": limits for q, metrics in thresholds.items()
                for m, limits in metrics.items()}
    if not expected:
        # Native scalar histories store the measured baseline-relative quality
        # in quality_margins, with an explicit frozen tau_q on that same scale.
        metric, threshold = contract.get("metric"), contract.get("tau_q")
        if (contract.get("rule") != "quality / baseline_quality >= 1 - alpha"
                or contract.get("inclusive") is not True
                or not isinstance(metric, str) or not metric
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)):
            raise ValueError("frozen per-query quality thresholds or an explicit scalar quality gate are required")
        expected = {metric: {"min": threshold}}
    rows, ids = [], set()
    for trial in history.get("trials", []):
        tid = trial["trial_id"]
        if tid in ids:
            raise ValueError(f"duplicate trial id: {tid}")
        ids.add(tid)
        candidate = trial["candidate"]
        measurements = {m["metric"]: m.get("value") for m in trial.get("quality_margins", [])}
        margins, missing = [], []
        for metric, limits in sorted(expected.items()):
            value = measurements.get(metric)
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
                missing.append(metric)
                continue
            for op, threshold in limits.items():
                if op not in ("min", "max"):
                    raise ValueError(f"unsupported quality operator {op}")
                raw = value - threshold if op == "min" else threshold - value
                margins.append(dict(metric=metric, value=value, operator=op,
                                    threshold=threshold, raw_margin=raw,
                                    normalized_margin=raw / max(abs(threshold), 1e-12),
                                    passed=raw >= 0))
        complete = trial.get("status") == "complete"
        quality = all(m["passed"] for m in margins) if complete and not missing else None
        metrics = trial.get("metrics") or {}
        performance = [metrics.get(k) for k in METRICS]
        valid_perf = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                         and math.isfinite(v) and v > 0 for v in performance)
        rows.append(dict(trial_id=tid, candidate=candidate, config=_effective(candidate),
                         configuration_id=_hash(_effective(candidate)),
                         status=trial.get("status"), quality_passed=quality,
                         quality_margins=margins, missing_quality_metrics=missing,
                         reported_quality_passed=trial.get("quality_feasible"),
                         performance=performance if complete and valid_perf else None,
                         result_path=trial.get("result_path")))
    return sorted(rows, key=lambda r: r["trial_id"])


def reassess_quality(history, retention_overrides):
    """Change only the summary's quality contract, using measured baseline values."""
    revised = deepcopy(history)
    original = deepcopy(history["quality_constraint"])
    baseline = {m["metric"]: m.get("value") for m in history.get("baseline_observation", {}).get("quality_margins", [])}
    changes = []
    for metric, factor in sorted(retention_overrides.items()):
        if isinstance(factor, bool) or not isinstance(factor, (float, int)) or not 0 < factor <= 1:
            raise ValueError(f"invalid retention factor: {metric}")
        query, name = metric.split(".", 1)
        gate = revised["quality_constraint"]["query_metric_thresholds"].get(query, {}).get(name)
        value = baseline.get(metric)
        if not gate or set(gate) != {"min"}:
            raise ValueError(f"expected an existing minimum quality gate: {metric}")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"measured baseline value missing: {metric}")
        before = gate["min"]
        gate["min"] = value * factor
        changes.append(dict(metric=metric, baseline_value=value, retention_factor=factor,
                            previous_threshold=before, new_threshold=gate["min"]))
    revised["quality_constraint"]["threshold_source"] = "post_run_baseline_relative_reassessment"
    rows = extract_evidence(revised)
    prior_rows = {r["trial_id"]: r for r in extract_evidence(history)}
    changed = [dict(trial_id=r["trial_id"], previous=prior_rows[r["trial_id"]]["quality_passed"],
                    current=r["quality_passed"]) for r in rows
               if r["quality_passed"] != prior_rows[r["trial_id"]]["quality_passed"]]
    return revised, dict(scope="summary_only_original_trial_history_unchanged",
        source_quality_contract=original, retention_overrides=dict(retention_overrides),
        threshold_changes=changes, changed_trial_decisions=changed,
        previous_counts=statistics(list(prior_rows.values())), current_counts=statistics(rows))


def _unique(rows):
    return len({r["configuration_id"] for r in rows})


def _numeric(values):
    return all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in values)


def space(rows, excluded=()):
    fields = {}
    for key in CONFIG_FIELDS:
        if key in excluded or key in PARTITION_FIELDS:
            continue
        values = [r["config"].get(key) for r in rows]
        if len({_hash(v) for v in values}) < 2:
            continue
        fields[key] = ({"kind": "numeric", "min": min(values), "max": max(values)}
                       if _numeric(values) and key not in NUMERIC_CATEGORIES else {"kind": "categorical"})
    return fields


def _partition(row):
    return tuple(row["config"].get(k) for k in PARTITION_FIELDS)


def distance(a, b, fields):
    if _partition(a) != _partition(b):
        return math.inf
    deltas = []
    for key, spec in fields.items():
        av, bv = a["config"].get(key), b["config"].get(key)
        if spec["kind"] == "numeric":
            deltas.append(min(1., abs(av - bv) / (spec["max"] - spec["min"])))
        else:
            deltas.append(float(av != bv))
    return sum(deltas) / len(deltas) if deltas else 0.


def cluster(rows, fields, diameter):
    """Deterministic complete linkage. Each cluster's maximum pair distance is bounded."""
    groups = [[r] for r in rows]
    pair = {(a["trial_id"], b["trial_id"]): distance(a, b, fields)
            for a in rows for b in rows}
    while len(groups) > 1:
        best = None
        for i, a in enumerate(groups):
            for j in range(i + 1, len(groups)):
                d = max(pair[x["trial_id"], y["trial_id"]] for x in a for y in groups[j])
                if d <= diameter and (best is None or (d, i, j) < best):
                    best = d, i, j
        if best is None:
            break
        _, i, j = best
        groups[i] += groups.pop(j)
    return sorted(groups, key=lambda g: min(r["trial_id"] for r in g))


def _dominates(a, b):
    return a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2] and a != b


def front_and_good(rows, tolerance):
    feasible = [r for r in rows if r["quality_passed"] is True and r["performance"]]
    front = [r for r in feasible if not any(_dominates(s["performance"], r["performance"])
                                           for s in feasible)]
    good = [r for r in feasible if any(
        r["performance"][0] >= p["performance"][0] * (1 - tolerance)
        and all(r["performance"][i] <= p["performance"][i] * (1 + tolerance) for i in (1, 2))
        for p in front)]
    return front, good


def statistics(rows):
    passed = [r for r in rows if r["quality_passed"] is True]
    failed = [r for r in rows if r["quality_passed"] is False]
    quality = {}
    for row in rows:
        if row["quality_passed"] is None:
            continue
        for m in row["quality_margins"]:
            key = m["metric"] + ":" + m["operator"]
            bucket = quality.setdefault(key, dict(threshold=m["threshold"], values=[],
                                                  normalized_margins=[], failure_trial_ids=[]))
            bucket["values"].append(m["value"])
            bucket["normalized_margins"].append(m["normalized_margin"])
            if not m["passed"]:
                bucket["failure_trial_ids"].append(row["trial_id"])
    for bucket in quality.values():
        values = bucket.pop("values")
        margins = bucket.pop("normalized_margins")
        bucket.update(min=min(values), median=median(values), max=max(values),
                      min_normalized_margin=min(margins))
    perf = {}
    for i, name in enumerate(METRICS):
        values = [r["performance"][i] for r in passed if r["performance"]]
        if values:
            perf[name] = dict(min=min(values), median=median(values), max=max(values))
    return dict(sample_count=len(rows), unique_configuration_count=_unique(rows),
                passed=len(passed), failed=len(failed),
                unique_passed=_unique(passed), unique_failed=_unique(failed),
                runtime_failed=sum(r["status"] == "runtime_failed" for r in rows),
                unknown_quality=sum(r["quality_passed"] is None for r in rows),
                observed_pass_fraction=len(passed) / (len(passed) + len(failed)) if passed or failed else None,
                quality_metrics=quality, feasible_performance=perf)


def _observed(rows, fields):
    result = {}
    for k in fields:
        values = {json.dumps(r["config"].get(k), sort_keys=True) for r in rows}
        result[k] = [json.loads(v) for v in sorted(values)]
    return result


def _matches(row, conditions):
    for key, condition in conditions.items():
        value = row["config"].get(key)
        if "eq" in condition and value != condition["eq"]:
            return False
        if "range" in condition:
            lo, hi = condition["range"]
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                return False
    return True


def severe_metrics(row, policy):
    """Engineering effect size against the gate, not statistical significance."""
    if row["quality_passed"] is not False:
        return []
    return [m for m in row["quality_margins"]
            if not m["passed"] and -m["normalized_margin"] + 1e-12 >= policy.negative_min_relative_gap]


def _preference_rows(rows, policy):
    return [r for r in rows if not (
        policy.preference_ignore_runtime_failures and r["status"] == "runtime_failed")]


def propose(rows, policy):
    preference_rows = _preference_rows(rows, policy)
    xspace = space(preference_rows, policy.quality_fields)
    xgroups = cluster(preference_rows, xspace, policy.cluster_diameter)
    front, good = front_and_good(rows, policy.near_front_tolerance)
    good_ids = {r["trial_id"] for r in good}
    fullspace = space(rows)
    good_groups = cluster(good, fullspace, policy.cluster_diameter)
    # Also consider shared patterns across good clusters in the same topology;
    # otherwise minor scheduling differences could hide a reusable common mode.
    partition_good = {}
    for row in good:
        partition_good.setdefault(_partition(row), []).append(row)
    pattern_groups = good_groups + list(partition_good.values())
    groups, drafts = [], []
    for n, members in enumerate(xgroups):
        cid = f"region_{n:03d}"
        groups.append(dict(cluster_id=cid, member_trial_ids=[r["trial_id"] for r in members],
                           observed_values=_observed(members, xspace),
                           partition=dict(zip(PARTITION_FIELDS, _partition(members[0])))))
        combos = {}
        for row in members:
            z = {k: row["config"].get(k) for k in policy.quality_fields}
            combos.setdefault(_hash(z), (z, []))[1].append(row)
        for z, samples in combos.values():
            for kind, label in (("quality_triplet_preference", True),):
                support = [r for r in samples if r["quality_passed"] is label]
                if not support:
                    continue
                drafts.append(dict(type=kind, cluster_id=cid, triplet=z,
                                   seed_trial_ids=[r["trial_id"] for r in support],
                                   **({"runtime_failure_policy": "excluded_from_preference_assessment"}
                                      if policy.preference_ignore_runtime_failures else {})))
    # Negative evidence pools the entire triplet, independently of X regions.
    negative_groups = {}
    for row in rows:
        z = {k: row["config"].get(k) for k in policy.quality_fields}
        negative_groups.setdefault(_hash(z), (z, []))[1].append(row)
    triplet_groups = []
    for key, (z, samples) in sorted(negative_groups.items()):
        support = [r for r in samples if severe_metrics(r, policy)]
        triplet_groups.append(dict(triplet=z, member_trial_ids=[r["trial_id"] for r in samples],
                                   severe_trial_ids=[r["trial_id"] for r in support]))
        if _unique(support) >= policy.minimum_unique_support:
            drafts.append(dict(type="negative_triplet", triplet=z,
                               scope="static_workload_triplet",
                               seed_trial_ids=[r["trial_id"] for r in support]))
    # Mine short conjunctions common to >=2 good configurations. Numerics are
    # observed seed ranges (hypotheses), never asserted as Cartesian guarantees.
    patterns = {}
    for members in pattern_groups:
        if _unique(members) < policy.minimum_unique_support:
            continue
        literals = []
        for key, spec in fullspace.items():
            values = [r["config"].get(key) for r in members]
            if spec["kind"] == "numeric":
                # Outcome-independent coarse bins, rather than exact bounds
                # around a handful of winners (which would overfit those points).
                span = spec["max"] - spec["min"]
                lo = max(0, math.floor(4 * (min(values)-spec["min"]) / span))
                hi = min(4, math.ceil(4 * (max(values)-spec["min"]) / span))
                if lo == hi:
                    lo, hi = max(0, lo-1), min(4, hi+1)
                bounds = [round(spec["min"] + lo * span / 4, 12), round(spec["min"] + hi * span / 4, 12)]
                if bounds != [spec["min"], spec["max"]]:
                    literals.append((key, {"range": bounds}))
            elif len({_hash(v) for v in values}) == 1:
                literals.append((key, {"eq": values[0]}))
        base = {k: {"eq": members[0]["config"].get(k)} for k in PARTITION_FIELDS}
        for length in range(1, min(policy.max_pattern_conditions, len(literals)) + 1):
            for combo in itertools.combinations(literals, length):
                conditions = {**base, **dict(combo)}
                matched = [r for r in rows if _matches(r, conditions)]
                support = [r for r in matched if r["trial_id"] in good_ids]
                if _unique(support) < policy.minimum_unique_support:
                    continue
                pool = [r for r in rows if _partition(r) == _partition(members[0])]
                # Unique configuration fractions limit adaptive/repeat count inflation.
                precision = _unique(support) / _unique(matched)
                prior = _unique([r for r in pool if r["trial_id"] in good_ids]) / _unique(pool)
                if precision <= prior:
                    continue
                key = tuple(r["trial_id"] for r in matched)
                score = precision, _unique(support), -length
                old = patterns.get(key)
                if old is None or score > old[0]:
                    patterns[key] = (score, dict(type="high_performance_pattern",
                        conditions=conditions, seed_trial_ids=[r["trial_id"] for r in support]))
    ordered = sorted(patterns.values(), key=lambda p: (p[0], _hash(p[1])), reverse=True)
    drafts += [p[1] for p in ordered[:policy.max_high_performance_patterns]]
    for draft in drafts:
        draft["experience_id"] = "experience_" + _hash(draft)[:20]
    return dict(configuration_space=xspace, full_configuration_space=fullspace,
                distance_normalization="observed_run_min_max; categorical_mismatch; equal_weights",
                clusters=groups, negative_triplet_groups=triplet_groups,
                high_performance_seed_clusters=[[r["trial_id"] for r in g] for g in good_groups],
                feasible_pareto_trial_ids=[r["trial_id"] for r in front],
                near_front_trial_ids=sorted(good_ids), drafts=drafts)


def verify(draft, rows, clustering, policy):
    """Evaluate every matching row, including contradictions outside seed clusters."""
    good = set(clustering["near_front_trial_ids"])
    by_id = {r["trial_id"]: r for r in rows}
    kind = draft["type"]
    if kind == "high_performance_pattern":
        matched = [r for r in rows if _matches(r, draft["conditions"])]
        support = [r for r in matched if r["trial_id"] in good]
        counter = [r for r in matched if r["quality_passed"] is False or
                   (r["quality_passed"] is True and r["performance"] and r["trial_id"] not in good)]
        neighbors = []
    elif kind == "negative_triplet":
        matched = [r for r in rows if all(r["config"].get(k) == v for k, v in draft["triplet"].items())]
        support = [r for r in matched if severe_metrics(r, policy)]
        # Mild failures contradict a blanket claim of severe failure too.
        counter = [r for r in matched if r["quality_passed"] is not None and not severe_metrics(r, policy)]
        neighbors = []
    else:
        group = next(g for g in clustering["clusters"] if g["cluster_id"] == draft["cluster_id"])
        members = [by_id[k] for k in group["member_trial_ids"]]
        same_z = [r for r in _preference_rows(rows, policy)
                  if all(r["config"].get(k) == v for k, v in draft["triplet"].items())]
        # Region membership is distance to ALL prototypes (complete-link bound),
        # not a rectangular min/max envelope and not only the original cluster IDs.
        matched, neighbors = [], []
        for row in same_z:
            distances = [distance(row, m, clustering["configuration_space"]) for m in members]
            if max(distances) <= policy.cluster_diameter:
                matched.append(row)
            elif min(distances) <= policy.neighbor_distance:
                neighbors.append(row)
        positive = kind == "quality_triplet_preference"
        support = [r for r in matched if r["quality_passed"] is positive]
        counter = [r for r in matched if r["quality_passed"] is (not positive)]
    negative = kind == "negative_triplet"
    boundary_counter = [r for r in neighbors if r["quality_passed"] is negative]
    runtime = [r for r in matched if r["status"] == "runtime_failed"]
    unknown = [r for r in matched if r["quality_passed"] is None or
               (kind == "high_performance_pattern" and not r["performance"])]
    strong = _unique(support) >= policy.minimum_unique_support
    status = ("instance_only" if not strong else
              "mixed_evidence" if counter or boundary_counter or unknown else "observed_pattern")
    # Always qualify causality, independent validation and adaptive sampling.
    report = {**draft, "evidence_status": status,
              "support_trial_ids": [r["trial_id"] for r in support],
              "counterexample_trial_ids": [r["trial_id"] for r in counter],
              "boundary_counterexample_trial_ids": [r["trial_id"] for r in boundary_counter],
              "runtime_failure_trial_ids": [r["trial_id"] for r in runtime],
              "unknown_trial_ids": [r["trial_id"] for r in unknown],
              "matched_trial_ids": [r["trial_id"] for r in matched],
              "unique_support_configurations": _unique(support),
              "statistics": statistics(matched), "boundary_statistics": statistics(neighbors),
              "validation": "same_history_consistency_only",
              "causal_repair_verified": False,
              "pass_fraction_is_future_probability": False}
    if negative:
        report["severity_criterion"] = dict(min_relative_gap=policy.negative_min_relative_gap,
            formula="-normalized_margin >= min_relative_gap", statistical_significance=False)
        report["severe_failures"] = [dict(trial_id=r["trial_id"], metrics=severe_metrics(r, policy)) for r in support]
        report["mild_failure_trial_ids"] = [r["trial_id"] for r in counter if r["quality_passed"] is False]
        report["passing_counterexample_trial_ids"] = [r["trial_id"] for r in counter if r["quality_passed"] is True]
        report["other_configuration_observations"] = {
            "severe": _observed(support, clustering["configuration_space"]),
            "non_severe": _observed(counter, clustering["configuration_space"])} if counter else {}
    feasible = [r for r in support if r["quality_passed"] is True and r["performance"]]
    reps, _ = front_and_good(feasible, 0)
    report["representative_configurations"] = [dict(trial_id=r["trial_id"],
        candidate=r["candidate"], performance=r["performance"]) for r in reps]
    if kind == "high_performance_pattern":
        partitions = {_partition(r) for r in matched}
        pool = [r for r in rows if _partition(r) in partitions]
        contrast = [r for r in pool if not _matches(r, draft["conditions"])]
        report["contrast_statistics"] = statistics(contrast)
        report["contrast_trial_ids"] = [r["trial_id"] for r in contrast]
        report["unique_good_fraction"] = _unique(support) / _unique(matched) if matched else None
        report["partition_unique_good_fraction"] = _unique([r for r in pool if r["trial_id"] in good]) / _unique(pool) if pool else None
    if kind == "quality_triplet_preference":
        group = next(g for g in clustering["clusters"] if g["cluster_id"] == draft["cluster_id"])
        members = [by_id[k] for k in group["member_trial_ids"]]
        local_front, local_good = front_and_good(members, policy.near_front_tolerance)
        local_good_ids = {r["trial_id"] for r in local_good}
        report["local_high_performance_support_ids"] = [r["trial_id"] for r in support if r["trial_id"] in local_good_ids]
        report["performance_role"] = ("local_front_or_near_front" if report["local_high_performance_support_ids"]
                                       else "quality_option_only")
        pairs = []
        for source in members:
            if source["quality_passed"] is not False or not feasible:
                continue
            if all(source["config"].get(k) == v for k, v in draft["triplet"].items()):
                continue
            target = min(feasible, key=lambda r: (distance(source, r, clustering["configuration_space"]), r["trial_id"]))
            changes = {k: [source["config"].get(k), target["config"].get(k)]
                       for k in set(source["config"]) | set(target["config"])
                       if k not in policy.quality_fields and source["config"].get(k) != target["config"].get(k)}
            delta = None
            if source["performance"]:
                a, b = source["performance"], target["performance"]
                delta = dict(qps_loss=1-b[0]/a[0], mean_latency_increase=b[1]/a[1]-1,
                             p95_latency_increase=b[2]/a[2]-1)
            pairs.append(dict(source_trial_id=source["trial_id"], target_trial_id=target["trial_id"],
                evidence_grade="matched_other_config" if not changes else "approximate_comparison",
                other_changes=changes, distance=distance(source, target, clustering["configuration_space"]),
                observed_performance_difference=delta, is_executed_repair=False))
        report["candidate_comparisons"] = pairs
        report["independent_comparison_targets"] = len({p["target_trial_id"] for p in pairs})
    return report


def _llm_selection(client, drafts, evidence, stage):
    """LLM chooses/refines disposition; executable hypotheses and statistics stay frozen."""
    ids = [d["experience_id"] for d in drafts]
    schema = {"type": "object", "properties": {"decisions": {"type": "array", "items": {
        "type": "object", "properties": {"experience_id": {"type": "string", "enum": ids},
            "action": {"type": "string", "enum": ["keep", "downgrade", "discard"]},
            "explanation": {"type": "string"}},
        "required": ["experience_id", "action", "explanation"], "additionalProperties": False}}},
        "required": ["decisions"], "additionalProperties": False}
    response = client.complete_structured(
        system="你是BO历史经验总结器。只依据给定实测证据选择有复用价值的规则。"
               "不能把观测频率当未来概率，不能把近似配置对比说成真实修复。"
               "三类主张都是限定条件下的观测规律，不要求因果证明、独立验证或真实修复轨迹。"
               "尤其负面组合：按候选 triplet 中列出的全部参数直接汇总，支持必须达到给定的质量相对缺口判据，轻微失败不是支持。多个不同配置明显失败是有用证据；"
               "不需要存在达标反例，也不需要证明该组合是唯一失败的组合。"
               "其他指标成败不完全一致，不否定已经共同失败的指标；运行失败单列，不否定有效质量评测证据。"
               "不要仅因没有因果证明、没有达标反例或失败只集中在个别指标而discard负面经验。"
               "单点可保留实例，但不能称为多样本规律；边界反例也是适用性风险，不得忽略。"
               "优选候选若标记runtime_failure_policy=excluded_from_preference_assessment，运行失败不属于其评估范围，"
               "不得引用其他候选中的运行失败来降级或撤回该优选经验；质量反例和完成评测但质量未知仍需考虑。"
               "用中文解释保留、降级或撤回的理由。解释只写证据强度和限制，不复述数值、参数条件或trial编号，"
               "这些事实由程序精确展示。不得发明因果机制或适用条件。"
               "反例未解释时降级或撤回，不添加逐点条件排除反例。",
        user=json.dumps(dict(stage=stage, hypotheses=drafts, evidence=evidence), ensure_ascii=False),
        response_schema=schema)
    payload = response.get("structured_output", response)
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or {d.get("experience_id") for d in decisions} != set(ids) or len(decisions) != len(ids):
        raise ValueError("LLM must decide exactly once for every supplied experience")
    if any(d.get("action") not in ("keep", "downgrade", "discard") or not isinstance(d.get("explanation"), str) for d in decisions):
        raise ValueError("invalid LLM disposition")
    return decisions


def recommended_triplets(rows, clustering, experiences, policy):
    """Recommendation value is distinct from confidence in quality reliability.

    Keep measured feasible Pareto combinations visible even if their regional
    generalizations were downgraded. Pooled counts are workload-level context,
    not a replacement for the original conditional experience records.
    """
    front_ids = set(clustering["feasible_pareto_trial_ids"])
    groups = {}
    for experience in experiences:
        if experience["type"] != "quality_triplet_preference":
            continue
        z = experience["triplet"]
        groups.setdefault(_hash(z), (z, []))[1].append(experience)
    recommendations = []
    for z, sources in groups.values():
        matched = [r for r in _preference_rows(rows, policy)
                   if all(r["config"].get(k) == v for k, v in z.items())]
        passed = [r for r in matched if r["quality_passed"] is True]
        failed = [r for r in matched if r["quality_passed"] is False]
        front = [r for r in passed if r["trial_id"] in front_ids]
        if not front and not any(s["disposition"] == "keep" for s in sources):
            continue
        role = "performance_priority" if front else "quality_reference"
        status = "mixed_evidence" if failed else (
            "instance_only" if _unique(passed) < policy.minimum_unique_support else "observed_passes_only")
        representatives, _ = front_and_good(passed, 0)
        recommendations.append(dict(
            triplet=z, recommendation_role=role, quality_confidence=status,
            quality_check_required=True, causal_repair_verified=False,
            pooled_statistics_scope="same_workload_context_only_not_region_generalization",
            statistics=statistics(matched),
            support_trial_ids=[r["trial_id"] for r in passed],
            counterexamples=[dict(trial_id=r["trial_id"], failed_metrics=[m for m in r["quality_margins"] if not m["passed"]]) for r in failed],
            feasible_pareto_trial_ids=[r["trial_id"] for r in front],
            representative_configurations=[dict(trial_id=r["trial_id"], candidate=r["candidate"], performance=r["performance"]) for r in representatives],
            conditional_experience_ids=[s["experience_id"] for s in sources],
            applicable_region_ids=sorted({s["cluster_id"] for s in sources}),
            rationale=("保留实测达标 Pareto 组合为性能优先候选；质量失败反例降低可靠性，不抹去性能推荐价值。"
                       if front else "已有达标实例及保留的区域经验，作为质量选择参考；应用时检查对应区域条件。")))
    return sorted(recommendations, key=lambda r: (r["recommendation_role"] != "performance_priority", _hash(r["triplet"])))


def build_bundle(history, anchor, policy=None, client=None):
    policy = policy or Policy()
    anchor = normalize_workload_anchor(anchor)
    if not anchor.get("available"):
        raise ValueError("static workload feature anchor is missing; refusing unanchored memory")
    rows = extract_evidence(history)
    if not rows:
        raise ValueError("no trial observations")
    clustering = propose(rows, policy)
    reports, llm_receipts, llm_errors = [], [], []
    drafts = clustering["drafts"]
    # Bounded batches keep exact evidence IDs and avoid whole raw-log prompts.
    for start in range(0, len(drafts), 8):
        batch = drafts[start:start+8]
        initial = []
        if client:
            try:
                initial = _llm_selection(client, batch,
                    {"static_workload_features": anchor["similarity_features"],
                     "seed_statistics": [{"experience_id": d["experience_id"], "statistics": statistics([
                        r for r in rows if r["trial_id"] in d["seed_trial_ids"]])} for d in batch]}, "synthesis")
            except Exception as exc:
                llm_errors.append(dict(stage="synthesis", error_type=type(exc).__name__))
        verified = [verify(d, rows, clustering, policy) for d in batch]
        revised = []
        if client:
            try:
                compact = [{k: v for k, v in r.items() if k not in
                           ("representative_configurations", "candidate_comparisons")} for r in verified]
                revised = _llm_selection(client, batch, {"initial_decisions": initial,
                    "static_workload_features": anchor["similarity_features"],
                    "full_history_verification": compact}, "counterexample_revision")
            except Exception as exc:
                llm_errors.append(dict(stage="counterexample_revision", error_type=type(exc).__name__))
        decisions = {d["experience_id"]: d for d in revised}
        for report in verified:
            decision = decisions.get(report["experience_id"])
            report["disposition"] = decision["action"] if decision else "keep"
            # Live model validation exposed conflation of seed counts and full
            # predicate-match counts. Keep free text only in audit receipts;
            # render all user-facing factual prose from verified measurements.
            if decision and decision["action"] == "downgrade" and report["evidence_status"] == "observed_pattern":
                report["evidence_status"] = "tentative_pattern"
            s = report["statistics"]
            report["interpretation"] = (
                f"处理：{report['disposition']}；证据状态：{report['evidence_status']}。"
                f"完整条件匹配样本中，质量通过 {s['passed']}，质量未通过 {s['failed']}，"
                f"运行失败 {s['runtime_failed']}。"
                f"支持来自 {report['unique_support_configurations']} 个不同配置；"
                f"条件内反例 {len(report['counterexample_trial_ids'])}，"
                f"边界反例 {len(report['boundary_counterexample_trial_ids'])}。"
                "仅支持本次 workload 下的观测结论，未验证因果修复效果。"
            )
            report["interpretation_is_model_generated"] = False
            report["model_decision_provided"] = bool(decision)
        reports += verified
        if initial or revised:
            llm_receipts.append(dict(batch_start=start, initial=initial, revised=revised))
    llm_validated = sum(r["model_decision_provided"] for r in reports)
    mode = ("deterministic_rules" if client is None else "deterministic_fallback" if not llm_validated
            else "hybrid_partial_fallback" if llm_errors else "structured_llm")
    bundle = dict(schema=SCHEMA, algorithm=ALGORITHM, policy=asdict(policy),
        workload_anchor=anchor, quality_contract=history["quality_constraint"],
        evidence=rows, clustering=clustering, experiences=reports,
        synthesis=dict(mode=mode,
                       receipts=llm_receipts, errors=llm_errors),
        limitations=["single-workload observational evidence; no causal repair trajectory",
                     "adaptive BO samples are not independent success-rate trials",
                     "observed numeric ranges are not guarantees for their Cartesian product",
                     "same-snapshot counterexample checks are not independent validation"])
    from .observation_selection import select_observations
    bundle["observations"], bundle["observation_generation"] = select_observations(bundle)
    bundle["recommended_triplets"] = final_recommended_triplets(bundle)
    bundle["audit"] = audit_bundle(bundle)
    if not bundle["audit"]["passed"]:
        raise ValueError(f"experience audit failed: {bundle['audit']['errors']}")
    return bundle


def final_recommended_triplets(bundle):
    candidates = recommended_triplets(bundle["evidence"], bundle["clustering"], bundle["experiences"], Policy(**bundle["policy"]))
    ranks = {_hash(r["triplet"]): r["rank"] for r in bundle.get("observations", []) if r["type"] == "quality_triplet_preference"}
    return sorted([r for r in candidates if _hash(r["triplet"]) in ranks], key=lambda r: ranks[_hash(r["triplet"])])


def audit_bundle(bundle):
    errors = []
    policy = Policy(**bundle["policy"])
    rows, clustering = bundle["evidence"], bundle["clustering"]
    reassessment = bundle.get("quality_reassessment")
    if reassessment:
        expected_contract = deepcopy(reassessment["source_quality_contract"])
        expected_contract["threshold_source"] = "post_run_baseline_relative_reassessment"
        for change in reassessment["threshold_changes"]:
            query, metric = change["metric"].split(".", 1)
            gate = expected_contract["query_metric_thresholds"][query][metric]
            if gate["min"] != change["previous_threshold"] or change["new_threshold"] != change["baseline_value"] * change["retention_factor"]:
                errors.append("invalid baseline-relative reassessment calculation")
            gate["min"] = change["new_threshold"]
        if expected_contract != bundle["quality_contract"]:
            errors.append("reassessment changed an unrelated quality gate")
    if set(clustering["configuration_space"]) & set(policy.quality_fields):
        errors.append("quality fields leaked into X clustering")
    if not normalize_workload_anchor(bundle["workload_anchor"]).get("available"):
        errors.append("missing static workload anchor")
    drafts = {d["experience_id"]: d for d in clustering["drafts"]}
    if len(bundle["experiences"]) != len(drafts) or {r["experience_id"] for r in bundle["experiences"]} != set(drafts):
        errors.append("missing or duplicated experiences")
    if len({r["trial_id"] for r in rows}) != len(rows):
        errors.append("duplicate evidence trial IDs")
    if propose(rows, policy) != clustering:
        errors.append("clustering or hypotheses differ from deterministic evidence replay")
    for row in rows:
        if _effective(row["candidate"]) != row["config"] or _hash(row["config"]) != row["configuration_id"]:
            errors.append("configuration evidence was altered")
        if row["status"] != "complete" and row["quality_passed"] is not None:
            errors.append("non-complete trial was assigned a quality outcome")
    if "observations" in bundle or bundle.get("algorithm") == ALGORITHM:
        from .observation_selection import select_observations
        expected_observations, expected_generation = select_observations(bundle)
        if bundle.get("observations") != expected_observations or bundle.get("observation_generation") != expected_generation:
            errors.append("final observation merge/delete differs from evidence replay")
        if bundle.get("recommended_triplets") != final_recommended_triplets(bundle):
            errors.append("final triplet recommendations differ from top4 selection")
    for report in bundle["experiences"]:
        draft = drafts.get(report["experience_id"])
        if draft is None:
            errors.append("unknown experience")
            continue
        expected = verify(draft, rows, clustering, policy)
        for key, value in expected.items():
            if key == "evidence_status" and report.get(key) == "tentative_pattern" and value == "observed_pattern":
                continue
            if report.get(key) != value:
                errors.append(f"{report['experience_id']}: altered {key}")
    return dict(passed=not errors, errors=errors, type="full_snapshot_predicate_replay",
                empirical_trials_rerun=False)


def render_markdown(bundle):
    lines = ["# BO 历史经验总结", "", f"静态 workload 锚点：`{bundle['workload_anchor']['fingerprint']}`",
             f"总结方式：{bundle['synthesis']['mode']}；审计通过：{bundle['audit']['passed']}", "",
             "通过次数是本次自适应搜索的观测频次，不是未来成功概率。候选配置对比不是已执行修复。", ""]
    if bundle["policy"].get("preference_ignore_runtime_failures"):
        lines += ["优选组合仅评估非运行失败样本；运行失败不参与聚类、优选判定或排名，原始记录保留在证据中。", ""]
    reassessment = bundle.get("quality_reassessment")
    if reassessment:
        counts = reassessment["current_counts"]
        lines += ["## 本次总结质量口径", "", "仅对历史实测值重判并重新总结；原始实验记录及旧版经验保留。", "",
                  f"新口径：质量通过 {counts['passed']}，质量未通过 {counts['failed']}，运行失败 {counts['runtime_failed']}。", "",
                  "| 指标 | 实测 baseline | 保留比例 | 旧门槛 | 本次门槛 |",
                  "|---|---:|---:|---:|---:|"]
        for change in reassessment["threshold_changes"]:
            lines.append(f"| {change['metric']} | {change['baseline_value']:.9f} | {change['retention_factor']:.1%} | {change['previous_threshold']:.9f} | {change['new_threshold']:.9f} |")
        lines += ["", "其他门槛保持原值。判定变化：" + ", ".join(
            f"{c['trial_id']}：{c['previous']} → {c['current']}" for c in reassessment["changed_trial_decisions"]), ""]
    fields_label = " / ".join(bundle["policy"]["quality_fields"])
    lines += ["## 优选参数组合总览", "",
              "性能推荐价值与质量可靠性分别记录。以下计数是本 workload 的去重 trial 汇总；适用条件仍以引用的区域经验为准。", "",
              f"| 参数组合（{fields_label}） | 推荐角色 | 质量达标/完成评测 | 达标 Pareto 实例 | 质量证据 |",
              "|---|---|---|---|---|"]
    for r in bundle.get("recommended_triplets", []):
        s = r["statistics"]
        role = "性能优先，需验证质量" if r["recommendation_role"] == "performance_priority" else "质量参考，需匹配区域"
        z = " / ".join(str(r["triplet"].get(k)) for k in bundle["policy"]["quality_fields"])
        lines.append(f"| {z} | {role} | {s['passed']}/{s['passed'] + s['failed']} | {', '.join(r['feasible_pareto_trial_ids']) or '—'} | {r['quality_confidence']} |")
    lines += [""]
    for r in bundle.get("recommended_triplets", []):
        if r["recommendation_role"] != "performance_priority":
            continue
        lines += ["性能优先组合的质量反例：", ""]
        for counter in r["counterexamples"]:
            details = "; ".join(f"{m['metric']}={m['value']:.9f}，门槛 {m['operator']} {m['threshold']:.9f}" for m in counter["failed_metrics"])
            lines += [f"- {counter['trial_id']}：{details}"]
        lines += [""]
    for kind, title in (("high_performance_pattern", "高性能且质量达标模式"),
                        ("quality_triplet_preference", "优选参数组合及候选修复方向"),
                        ("negative_triplet", "负面参数组合")):
        lines += [f"## {title}", ""]
        matching = [r for r in bundle.get("observations", bundle["experiences"]) if r["type"] == kind]
        if not matching:
            lines += ["暂无多样本模式；可查看证据与 Pareto 实例。", ""]
        for r in matching:
            s = r["statistics"]
            lines += [f"### {r['experience_id']}", "",
                f"状态：{r['evidence_status']}；处理：{r['disposition']}", "",
                "条件：`" + json.dumps(r.get("conditions", {"regions": r.get("region_ids", [r.get("cluster_id")]), "triplet": r.get("triplet")}), ensure_ascii=False) + "`", "",
                f"质量通过 {s['passed']} / 失败 {s['failed']} / 运行失败 {s['runtime_failed']}；独立支持配置 {r['unique_support_configurations']}。", "",
                f"支持：{', '.join(r['support_trial_ids']) or '无'}", "",
                f"反例：{', '.join(r['counterexample_trial_ids']) or '无'}；边界反例：{', '.join(r['boundary_counterexample_trial_ids']) or '无'}", "",
                "达标样本性能：`" + json.dumps(s['feasible_performance']) + "`", ""]
            if kind == "negative_triplet":
                lines += [f"显著缺口判据：相对质量门槛至少 {r['severity_criterion']['min_relative_gap']:.0%}（工程判据，非统计检验）；显著失败 {len(r['support_trial_ids'])} 次；轻微失败 {len(r['mild_failure_trial_ids'])} 次。", ""]
                for failure in r["severe_failures"]:
                    details = "; ".join(f"{m['metric']}={m['value']:.6f}，门槛 {m['threshold']:.6f}，相对缺口 {-m['normalized_margin']:.2%}" for m in failure["metrics"])
                    lines += [f"- {failure['trial_id']}：{details}"]
                lines += [""]
            if r.get("interpretation"):
                lines += ["经验结论（程序根据审计结果渲染；模型原始理由见 synthesis.receipts）：" + r["interpretation"], ""]
    return "\n".join(lines)


def consolidate_run(context_path, *, output_dir=None, policy=None, client=None, quality_retention_overrides=None):
    """Idempotent, local publication. No external writes or benchmark execution."""
    context_path = Path(context_path).resolve()
    context = _read(context_path)
    from .local_memory import publish_bundle
    memory_root = context.get("experience_consolidation", {}).get("memory_store_root")
    history_path = Path(context["history_path"])
    history = _read(history_path)
    source_history_sha256 = hashlib.sha256(history_path.read_bytes()).hexdigest()
    result_path = Path(context["result_path"])
    result = _read(result_path)
    if result.get("status") != "completed":
        raise ValueError("consolidation requires a completed run result")
    if any(r.get("status") not in ("complete", "runtime_failed") for r in history.get("trials", [])):
        raise ValueError("history still contains nonterminal trials")
    baseline = _read(context["source_baseline_trial_result"])
    reassessment = None
    if quality_retention_overrides:
        history, reassessment = reassess_quality(history, quality_retention_overrides)
    anchor = normalize_workload_anchor(context.get("static_workload") or baseline)
    effective_policy = policy or Policy()
    key = _hash(dict(history=history, anchor=anchor, baseline=baseline, quality_reassessment=reassessment,
                     runtime_constraint_scope=context.get("runtime_constraint_scope"),
                     workload_contract=context.get("workload_contract"),
                     policy=asdict(effective_policy), algorithm=ALGORITHM,
                     synthesis="llm" if client else "deterministic",
                     model_identity=client.checkpoint_identity() if client and hasattr(client, "checkpoint_identity") else None))
    root = Path(output_dir) if output_dir else context_path.parent / "experience_consolidation"
    target = root / key[:20]
    result_file = target / "experience_memories.json"
    required_files = [result_file, target / "experience_evidence.json", target / "experience_clusters.json",
                      target / "experience_audit.json", target / "summary.md"]
    if all(p.exists() for p in required_files):
        try:
            existing = _read(result_file)
            reusable = existing.get("snapshot_id") == key and audit_bundle(existing)["passed"] and not existing["synthesis"]["errors"]
        except (ValueError, KeyError, TypeError):
            reusable = False
        if reusable:
            write_json_atomic(root / "latest.json", dict(snapshot_id=key, result_path=str(result_file.resolve())))
            return dict(status="completed", result_path=str(result_file.resolve()), idempotent_reuse=True,
                        synthesis_mode=existing["synthesis"]["mode"], llm_errors=[],
                        local_memory=publish_bundle(result_file, memory_root),
                        quality_retention_overrides=quality_retention_overrides or {})
    bundle = build_bundle(history, anchor, effective_policy, client)
    if reassessment:
        bundle["quality_reassessment"] = reassessment
    bundle["source_history_sha256"] = source_history_sha256
    bundle["audit"] = audit_bundle(bundle)
    if not bundle["audit"]["passed"]:
        raise ValueError("quality reassessment audit failed")
    bundle.update(snapshot_id=key, source_run_id=str(context_path.parent.resolve()),
        source_context_path=str(context_path), source_history_path=str(history_path),
        environment={"system_scope": build_system_scope(baseline),
                     "runtime_constraint_scope": context.get("runtime_constraint_scope"),
                     "workload_contract": context.get("workload_contract"),
                     "generation_settings": {k: baseline.get(k) for k in
                         ("generation_config", "structured_outputs_config", "temperature", "top_p", "seed")}},
        source_baseline_trial_result=context["source_baseline_trial_result"])
    target.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target / "experience_evidence.json", dict(schema=SCHEMA, snapshot_id=key,
        workload_anchor=anchor, quality_contract=bundle["quality_contract"],
        quality_reassessment=reassessment, rows=bundle["evidence"]))
    write_json_atomic(target / "experience_clusters.json", bundle["clustering"])
    write_json_atomic(target / "experience_audit.json", bundle["audit"])
    write_json_atomic(result_file, bundle)
    (target / "summary.md").write_text(render_markdown(bundle), encoding="utf-8")
    write_json_atomic(root / "latest.json", dict(snapshot_id=key, result_path=str(result_file.resolve())))
    return dict(status="completed", result_path=str(result_file.resolve()), idempotent_reuse=False,
                synthesis_mode=bundle["synthesis"]["mode"], llm_errors=bundle["synthesis"]["errors"],
                local_memory=publish_bundle(result_file, memory_root),
                quality_retention_overrides=quality_retention_overrides or {})


def consolidate_after_run(context_path, *, client=None, quality_retention_overrides=None):
    """Non-fatal completion hook. Retry from CLI without rerunning BO."""
    context = _read(context_path)
    options = context.get("experience_consolidation", {})
    if options.get("enabled", True) is False:
        return {"status": "disabled"}
    try:
        mode = options.get("llm", "auto")
        if mode == "off":
            client = None
        elif client is None:
            client = summary_client(mode, options.get("env_path"))
        receipt = consolidate_run(context_path, policy=Policy(**options.get("policy", {})), client=client,
                                  quality_retention_overrides=quality_retention_overrides)
    except Exception as exc:
        receipt = dict(status="failed", error_type=type(exc).__name__, error=str(exc), retryable=True)
    write_json_atomic(Path(context_path).parent / "experience_consolidation_status.json", receipt)
    paths = [Path(context["result_path"])]
    if paths[0].exists():
        agentic_path = _read(paths[0]).get("agentic_result_path")
        if agentic_path:
            paths.append(Path(agentic_path))
    for path in paths:
        if path.exists():
            result = _read(path)
            result["experience_consolidation"] = receipt
            write_json_atomic(path, result)
    return receipt


def search_bundles(workload, paths, *, limit=4, threshold=0.85, minimum_shared_features=8):
    """Rank whole run bundles ONLY by canonical static workload similarity."""
    if limit < 1 or not 0 <= threshold <= 1 or minimum_shared_features < 1:
        raise ValueError("invalid retrieval policy")
    anchor = normalize_workload_anchor(workload)
    if not anchor.get("available"):
        raise ValueError("missing static workload query features")
    matches, seen = [], set()
    for path in paths:
        bundle = _read(path)
        if bundle.get("schema") != SCHEMA or not audit_bundle(bundle)["passed"]:
            continue
        key = bundle.get("source_run_id", bundle.get("snapshot_id", str(path)))
        if key in seen:
            continue
        sim = similar_workload_anchor_similarity(anchor, bundle["workload_anchor"])
        shared = len(set(anchor["similarity_features"]) & set(bundle["workload_anchor"]["similarity_features"]))
        if shared >= minimum_shared_features and sim["score"] >= threshold:
            matches.append(dict(result_path=str(path), score=sim["score"], workload_similarity=sim, bundle=bundle))
            seen.add(key)
    return sorted(matches, key=lambda m: (-m["score"], m["result_path"]))[:limit]
