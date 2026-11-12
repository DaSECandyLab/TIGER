"""固定 BO 搜索空间的加载、采样、编码与机械约束处理。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_DECISIONS = frozenset({"tune", "conditional", "composite", "fixed"})
_KINDS = frozenset({"boolean", "integer", "float", "categorical"})
_SIMPLE_CONDITION = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|>|>=|<|<=)\s*(.+?)\s*$"
)
_REPRESENTATION_ARROW = re.compile(
    r"^\s*quantization\s*==\s*(['\"])(?P<quantization>[^'\"]+)\1\s*"
    r"->\s*dtype\s+must\s+be\s*(['\"])(?P<dtype>[^'\"]+)\3\s*$"
)


def _load_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {resolved}")
    return payload


def _json_scalar(raw: str) -> Any:
    normalized = raw.strip()
    aliases = {"true": True, "false": False, "null": None}
    if normalized.lower() in aliases:
        return aliases[normalized.lower()]
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return normalized.strip("'\"")


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return actual == expected
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }[operator]


@dataclass(frozen=True)
class ParameterDomain:
    """Range Agent 输出的一项不可变参数域。"""

    name: str
    bo_name: str
    decision: str
    kind: str
    domain: Mapping[str, Any] | None
    fixed_value: Any
    reference_value: Any
    activation_conditions: tuple[str, ...]
    group_id: str = "ungrouped"

    @property
    def tuned(self) -> bool:
        return self.decision != "fixed"

    @property
    def values(self) -> tuple[Any, ...] | None:
        if self.domain is None or "values" not in self.domain:
            return None
        return tuple(self.domain["values"])

    def inactive_value(self) -> Any:
        if self.reference_value is not None:
            return self.reference_value
        if self.fixed_value is not None:
            return self.fixed_value
        values = self.values
        return values[0] if values else None

    def sample(self, rng: np.random.Generator) -> Any:
        if not self.tuned:
            return self.fixed_value
        if self.domain is None:
            raise ValueError(f"tuned parameter {self.name} has no domain")
        values = self.values
        if values is not None:
            return values[int(rng.integers(0, len(values)))]
        lower = self.domain["lower"]
        upper = self.domain["upper"]
        scale = str(self.domain.get("scale", "linear"))
        if scale == "log":
            draw = math.exp(rng.uniform(math.log(float(lower)), math.log(float(upper))))
        else:
            draw = rng.uniform(float(lower), float(upper))
        if self.kind == "float":
            return float(draw)
        granularity = str(self.domain.get("granularity", "any_integer"))
        if granularity == "powers_of_base":
            base = float(self.domain["base"])
            exponent = int(
                rng.integers(
                    math.ceil(math.log(float(lower), base)),
                    math.floor(math.log(float(upper), base)) + 1,
                )
            )
            return int(round(base**exponent))
        if granularity == "step":
            step = int(self.domain["step"])
            count = (int(upper) - int(lower)) // step
            return int(lower) + step * int(rng.integers(0, count + 1))
        return int(np.clip(round(draw), int(lower), int(upper)))

    def from_unit_interval(self, value: float) -> Any:
        """把 Sobol 的一个 [0,1) 坐标映射到本参数的固定域。"""

        if not self.tuned:
            return self.fixed_value
        if self.domain is None:
            raise ValueError(f"tuned parameter {self.name} has no domain")
        unit = float(np.clip(value, 0.0, np.nextafter(1.0, 0.0)))
        values = self.values
        if values is not None:
            return values[min(int(unit * len(values)), len(values) - 1)]
        lower = float(self.domain["lower"])
        upper = float(self.domain["upper"])
        if str(self.domain.get("scale", "linear")) == "log":
            draw = math.exp(math.log(lower) + unit * (math.log(upper) - math.log(lower)))
        else:
            draw = lower + unit * (upper - lower)
        if self.kind == "float":
            return float(draw)
        granularity = str(self.domain.get("granularity", "any_integer"))
        if granularity == "powers_of_base":
            base = float(self.domain["base"])
            minimum = math.ceil(math.log(lower, base))
            maximum = math.floor(math.log(upper, base))
            exponent = min(
                minimum + int(unit * (maximum - minimum + 1)), maximum
            )
            return int(round(base**exponent))
        if granularity == "step":
            step = int(self.domain["step"])
            count = (int(upper) - int(lower)) // step
            index = min(int(unit * (count + 1)), count)
            return int(lower) + step * index
        return int(np.clip(round(draw), int(lower), int(upper)))

    def contains(self, value: Any) -> bool:
        if not self.tuned:
            return value == self.fixed_value
        if self.domain is None:
            return False
        values = self.values
        if values is not None:
            return value in values
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        lower, upper = float(self.domain["lower"]), float(self.domain["upper"])
        if not lower <= float(value) <= upper:
            return False
        if self.kind == "integer" and type(value) is not int:
            return False
        granularity = self.domain.get("granularity")
        if granularity == "step":
            step = float(self.domain["step"])
            quotient = (float(value) - lower) / step
            return math.isclose(quotient, round(quotient), abs_tol=1e-9)
        if granularity == "powers_of_base":
            base = float(self.domain["base"])
            exponent = math.log(float(value), base)
            return math.isclose(exponent, round(exponent), abs_tol=1e-9)
        return True


class FixedSearchSpace:
    """只消费 Range 产物，不扩大或重写其中任何参数域。"""

    def __init__(
        self,
        parameters: Sequence[ParameterDomain],
        joint_constraints: Sequence[Mapping[str, Any]],
        *,
        source_path: str,
        source_parameter_results: str | None,
    ) -> None:
        self.parameters = tuple(parameters)
        self.joint_constraints = tuple(dict(item) for item in joint_constraints)
        self.source_path = source_path
        self.source_parameter_results = source_parameter_results
        self.by_name = {parameter.name: parameter for parameter in self.parameters}
        self.by_bo_name = {parameter.bo_name: parameter for parameter in self.parameters}
        if len(self.by_name) != len(self.parameters):
            raise ValueError("domain spec contains duplicate parameter names")
        if len(self.by_bo_name) != len(self.parameters):
            raise ValueError("domain spec contains duplicate bo_name values")
        self.tuned_parameters = tuple(p for p in self.parameters if p.tuned)
        if not self.tuned_parameters:
            raise ValueError("domain spec contains no tunable BO dimensions")

    @classmethod
    def from_json(cls, path: str | Path) -> "FixedSearchSpace":
        resolved = Path(path).expanduser().resolve()
        payload = _load_object(resolved)
        if payload.get("module") != "vllm_parameter_domain_design":
            raise ValueError("unsupported Range domain module")
        raw_parameters = payload.get("parameters")
        if not isinstance(raw_parameters, list):
            raise TypeError("domain spec parameters must be an array")
        parameters: list[ParameterDomain] = []
        for raw in raw_parameters:
            if not isinstance(raw, Mapping):
                raise TypeError("every domain parameter must be an object")
            decision = str(raw.get("decision", ""))
            if decision not in _DECISIONS:
                raise ValueError(f"invalid parameter decision: {decision!r}")
            domain = raw.get("domain")
            if decision == "fixed":
                if domain is not None:
                    raise ValueError(f"fixed parameter {raw.get('name')} must not have a domain")
                kind = cls._infer_kind(raw.get("fixed_value"))
            else:
                if not isinstance(domain, Mapping):
                    raise TypeError(f"parameter {raw.get('name')} domain must be an object")
                kind = str(domain.get("kind", ""))
                if kind not in _KINDS:
                    raise ValueError(f"parameter {raw.get('name')} has invalid kind {kind!r}")
                cls._validate_domain(str(raw.get("name")), kind, domain)
            name = str(raw.get("name", "")).strip()
            bo_name = str(raw.get("bo_name", name)).strip()
            if not name or not bo_name:
                raise ValueError("parameter name and bo_name must be non-empty")
            parameters.append(
                ParameterDomain(
                    name=name,
                    bo_name=bo_name,
                    decision=decision,
                    kind=kind,
                    domain=dict(domain) if isinstance(domain, Mapping) else None,
                    fixed_value=raw.get("fixed_value"),
                    reference_value=raw.get("reference_value"),
                    activation_conditions=tuple(
                        str(item) for item in raw.get("activation_conditions", [])
                    ),
                    group_id=str(
                        raw.get("group_id")
                        or raw.get("parameter_group")
                        or raw.get("category")
                        or "ungrouped"
                    ),
                )
            )
        constraints = payload.get("joint_constraints", [])
        if not isinstance(constraints, list) or not all(
            isinstance(item, Mapping) for item in constraints
        ):
            raise TypeError("domain spec joint_constraints must be an object array")
        return cls(
            parameters,
            constraints,
            source_path=str(resolved),
            source_parameter_results=(
                str(payload["source_parameter_results"])
                if payload.get("source_parameter_results")
                else None
            ),
        )

    @staticmethod
    def _infer_kind(value: Any) -> str:
        if type(value) is bool:
            return "boolean"
        if type(value) is int:
            return "integer"
        if type(value) is float:
            return "float"
        return "categorical"

    @staticmethod
    def _validate_domain(name: str, kind: str, domain: Mapping[str, Any]) -> None:
        if "values" in domain:
            values = domain["values"]
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{name} enumerated domain needs at least two values")
            if len({json.dumps(v, sort_keys=True) for v in values}) != len(values):
                raise ValueError(f"{name} domain contains duplicate values")
            return
        if kind not in {"integer", "float"}:
            raise ValueError(f"{name} non-numeric domain requires values")
        lower, upper = domain.get("lower"), domain.get("upper")
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise TypeError(f"{name} numeric bounds cannot be boolean")
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError(f"{name} numeric domain requires lower/upper")
        if float(lower) >= float(upper):
            raise ValueError(f"{name} requires lower < upper")

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for parameter in self.tuned_parameters:
            values = parameter.values
            # Range may deliberately enumerate a sparse integer domain (for
            # example block_size=[16, 32, 64]).  Treat every enumerated domain
            # as a closed discrete choice so the encoder never invents an
            # ordinal distance or requires absent lower/upper bounds.
            if values is not None:
                names.extend(f"{parameter.bo_name}={json.dumps(v, sort_keys=True)}" for v in values)
            else:
                names.append(parameter.bo_name)
            if parameter.decision == "conditional":
                names.append(f"{parameter.bo_name}.__active__")
        return tuple(names)

    def canonical_to_bo(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            parameter.bo_name: candidate[parameter.name]
            for parameter in self.parameters
        }

    def bo_to_canonical(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.bo_name in candidate:
                result[parameter.name] = candidate[parameter.bo_name]
            elif parameter.name in candidate:
                result[parameter.name] = candidate[parameter.name]
            elif parameter.decision == "fixed":
                result[parameter.name] = parameter.fixed_value
            else:
                result[parameter.name] = parameter.inactive_value()
        return self.normalize(result)

    def _condition_active(self, condition: str, candidate: Mapping[str, Any]) -> bool:
        # "A -> B must be C" 描述的是联合有效性，不把子参数本身判为 inactive。
        if "->" in condition or " implies " in condition:
            return True
        match = _SIMPLE_CONDITION.fullmatch(condition)
        if match is None:
            raise ValueError(f"unsupported activation condition: {condition!r}")
        name, operator, raw_expected = match.groups()
        return _compare(candidate.get(name), operator, _json_scalar(raw_expected))

    def representation_dtype_template(self) -> dict[str, str]:
        """Return the concrete quantization→dtype mapping frozen by Range."""

        parameter = self.by_name.get("dtype")
        if parameter is None:
            return {}
        declared: dict[str, str] = {}
        for condition in parameter.activation_conditions:
            match = _REPRESENTATION_ARROW.fullmatch(condition)
            if match is not None:
                dtype = {"float16": "half", "float32": "float"}.get(
                    match.group("dtype"), match.group("dtype")
                )
                declared[match.group("quantization")] = dtype
        return declared

    def active_parameter_names(self, candidate: Mapping[str, Any]) -> set[str]:
        active: set[str] = set()
        for parameter in self.parameters:
            if parameter.decision != "conditional" or all(
                self._condition_active(condition, candidate)
                for condition in parameter.activation_conditions
            ):
                active.add(parameter.name)
        return active

    def inactive_parameter_values(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Return executable inactive values under canonical and BO names.

        Turning a parent off makes these coordinates semantically inactive;
        resetting them is not a new tuning action. The legacy executor clears
        disabled speculation and DP1 synchronization flags, even when Range
        uses an in-domain placeholder to encode those inactive coordinates.
        """
        canonical = self.bo_to_canonical(candidate)
        active = self.active_parameter_names(canonical)
        def executable_value(parameter: ParameterDomain) -> Any:
            if (parameter.name == "num_speculative_tokens"
                    and canonical.get("speculative_method") in {None, "none"}):
                return None
            if (parameter.name == "disable_nccl_for_dp_synchronization"
                    and canonical.get("data_parallel_size") == 1):
                return False
            return parameter.inactive_value()
        return {
            alias: executable_value(parameter)
            for parameter in self.parameters
            if parameter.name not in active
            for alias in {parameter.name, parameter.bo_name}
        }

    def normalize(self, raw_candidate: Mapping[str, Any]) -> dict[str, Any]:
        candidate: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in raw_candidate:
                candidate[parameter.name] = raw_candidate[parameter.name]
            elif parameter.bo_name in raw_candidate:
                candidate[parameter.name] = raw_candidate[parameter.bo_name]
            elif parameter.decision == "fixed":
                candidate[parameter.name] = parameter.fixed_value
            else:
                candidate[parameter.name] = parameter.inactive_value()
        active = self.active_parameter_names(candidate)
        for parameter in self.parameters:
            if parameter.name not in active:
                candidate[parameter.name] = parameter.inactive_value()
        return candidate

    def validate(self, raw_candidate: Mapping[str, Any]) -> list[str]:
        candidate = self.normalize(raw_candidate)
        errors: list[str] = []
        active = self.active_parameter_names(candidate)
        for parameter in self.parameters:
            value = candidate[parameter.name]
            if parameter.name in active and not parameter.contains(value):
                errors.append(f"{parameter.name}={value!r} is outside its fixed domain")
            if parameter.decision == "fixed" and value != parameter.fixed_value:
                errors.append(f"{parameter.name} must equal fixed_value={parameter.fixed_value!r}")
        errors.extend(self._joint_constraint_errors(candidate))
        return errors

    def _joint_constraint_errors(self, candidate: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for constraint in self.joint_constraints:
            identifier = str(constraint.get("id", ""))
            if identifier in {"gpu_topology_budget", "two_gpu_topology"}:
                tp = int(candidate.get("tensor_parallel_size", 1))
                dp = int(candidate.get("data_parallel_size", 1))
                expression = str(constraint.get("expression", ""))
                match = re.search(r"==\s*(\d+)\s*$", expression)
                expected = int(match.group(1)) if match else None
                if expected is None or tp * dp != expected:
                    errors.append(f"{identifier} violated: tp*dp={tp * dp}, expected={expected}")
            elif identifier == "screened_mechanism_combinations":
                alternatives = constraint.get("alternatives", [])
                if not any(
                    all(type(candidate.get(k)) is type(v) and candidate.get(k) == v
                        for k, v in alternative.get("mechanisms", {}).items())
                    and all(type(candidate.get(k)) is type(v) and candidate.get(k) == v
                        for k, v in alternative.get("required_bindings", {}).items())
                    for alternative in alternatives
                ):
                    errors.append("Candidate is outside retained screened mechanism combinations or bindings")
            elif identifier == "dcp_divides_tp":
                tp = int(candidate.get("tensor_parallel_size", 1))
                dcp = int(candidate.get("decode_context_parallel_size", 1))
                if dcp < 1 or tp % dcp != 0:
                    errors.append(f"dcp_divides_tp violated: tp={tp}, dcp={dcp}")
            elif identifier == "async_executor_compatibility":
                if (candidate.get("data_parallel_backend") == "ray" or candidate.get("distributed_executor_backend") == "ray") and bool(
                    candidate.get("async_scheduling", False)
                ):
                    errors.append("async_executor_compatibility violated")
            elif identifier == "representation_template":
                expected_dtype = {
                    "none": {"bfloat16", "half", "float", None},
                    None: {"bfloat16", "half", "float", None},
                    "awq": {"half"},
                    "gptq": {"half"},
                    "awq_marlin": {"half", "bfloat16"},
                    "gptq_marlin": {"half", "bfloat16"},
                }.get(candidate.get("quantization"))
                if expected_dtype is not None and candidate.get("dtype") not in expected_dtype:
                    errors.append("representation_template violated")
            elif identifier == "cacheblend_compatibility":
                if candidate.get("cacheblend_mode") == "on":
                    if constraint.get("require_enforce_eager") and candidate.get("enforce_eager") is not True:
                        errors.append("CacheBlend requires enforce_eager=true in this frozen domain")
                    if candidate.get("chunked_prefill", candidate.get("enable_chunked_prefill")):
                        errors.append("CacheBlend requires chunked prefill off")
                    if candidate.get("enable_prefix_caching") and not constraint.get("allow_prefix_caching", False):
                        errors.append("CacheBlend requires native prefix caching off")
                    expected = "float16" if candidate.get("dtype") in ("half", "float16") else "bfloat16"
                    if candidate.get("kv_cache_dtype") != expected:
                        errors.append("CacheBlend requires uncompressed KV matching model dtype")
                    if int(candidate.get("block_size", 16)) < 16:
                        errors.append("CacheBlend requires block_size >= 16")
                    layers = candidate.get("cacheblend_blend_check_layers", "1")
                    ratios = candidate.get("cacheblend_blend_recompute_ratios", "0.2")
                    def items(value):
                        return list(value) if isinstance(value, (list, tuple)) else str(value).split(",")
                    ratio_values = items(ratios)
                    if not ratio_values or not 0 <= float(ratio_values[0]) <= 1:
                        errors.append("CacheBlend first recompute ratio must be within [0, 1]")
                    elif len(items(layers)) > 1 and float(ratio_values[0]) != 1.0:
                        errors.append("Installed LMCache multiple check layers require full recompute to avoid tensor shape mismatch")
            elif identifier == "full_prefill_token_budget":
                if candidate.get("chunked_prefill", candidate.get("enable_chunked_prefill")) is False:
                    limit = candidate.get("max_model_len") or constraint.get("max_model_len")
                    if limit is not None and candidate.get("max_num_batched_tokens", 0) < limit:
                        errors.append("Full prefill requires max_num_batched_tokens >= max_model_len")
            elif identifier == "prefix_hash_parent":
                # 子参数在 parent 关闭时会被 inactive/reference 归一化，不形成非法点。
                continue
            else:
                errors.append(f"unsupported joint constraint id: {identifier}")
        # Range 的箭头条件有时承载具体表示模板约束。
        quantization = candidate.get("quantization")
        dtype = candidate.get("dtype")
        expected = self.representation_dtype_template().get(str(quantization or "none"))
        arrow_conditions = [
            condition
            for parameter in self.parameters
            for condition in parameter.activation_conditions
            if "->" in condition
        ]
        if arrow_conditions and expected is not None and dtype != expected:
            errors.append(
                f"representation activation constraint violated: {quantization=} requires dtype={expected!r}"
            )
        # vLLM scheduler 的已知运行硬约束。它只裁剪 Range 域的不可执行联合点，
        # 不改变任一旋钮的身份、上下界或候选值；与原 BO 的 hard constraint 一致。
        max_num_seqs = candidate.get("max_num_seqs")
        max_num_batched_tokens = candidate.get("max_num_batched_tokens")
        if (
            isinstance(max_num_seqs, (int, float))
            and not isinstance(max_num_seqs, bool)
            and isinstance(max_num_batched_tokens, (int, float))
            and not isinstance(max_num_batched_tokens, bool)
            and max_num_seqs > max_num_batched_tokens
        ):
            errors.append(
                "scheduler token budget violated: "
                f"max_num_seqs={max_num_seqs} > "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )
        return errors

    def sample(
        self,
        rng: np.random.Generator,
        *,
        count: int = 1,
        exclude: set[str] | None = None,
        max_attempts: int | None = None,
    ) -> list[dict[str, Any]]:
        if count < 1:
            return []
        excluded = set(exclude or ())
        samples: list[dict[str, Any]] = []
        attempts = 0
        attempt_limit = max_attempts or max(1000, count * 200)
        while len(samples) < count and attempts < attempt_limit:
            attempts += 1
            raw = {parameter.name: parameter.sample(rng) for parameter in self.parameters}
            for constraint in self.joint_constraints:
                if constraint.get("id") == "screened_mechanism_combinations":
                    alternatives = constraint["alternatives"]
                    alternative = alternatives[int(rng.integers(0, len(alternatives)))]
                    raw.update(alternative["mechanisms"])
                    raw.update(alternative.get("required_bindings", {}))
            candidate = self.normalize(raw)
            key = self.key(candidate)
            if key in excluded or self.validate(candidate):
                continue
            excluded.add(key)
            samples.append(candidate)
        if len(samples) < count:
            raise RuntimeError(
                f"could only sample {len(samples)}/{count} unique valid candidates "
                f"from the fixed search space after {attempts} attempts"
            )
        return samples

    def from_unit_cube(self, coordinates: Sequence[float]) -> dict[str, Any]:
        """将一行 Sobol 坐标映射成完整候选，不扩大 Range 固定域。"""

        values = list(coordinates)
        if len(values) != len(self.tuned_parameters):
            raise ValueError(
                "unit-cube dimension differs from the logical tuned parameter count: "
                f"{len(values)} != {len(self.tuned_parameters)}"
            )
        iterator = iter(values)
        raw: dict[str, Any] = {}
        for parameter in self.parameters:
            raw[parameter.name] = (
                parameter.from_unit_interval(next(iterator))
                if parameter.tuned
                else parameter.fixed_value
            )
        return self.normalize(raw)

    def encode(self, candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[list[float]] = []
        for raw in candidates:
            candidate = self.normalize(raw)
            active = self.active_parameter_names(candidate)
            row: list[float] = []
            for parameter in self.tuned_parameters:
                value = candidate[parameter.name]
                values = parameter.values
                if values is not None:
                    row.extend(1.0 if value == category else 0.0 for category in values)
                else:
                    if parameter.domain is None:
                        raise ValueError(f"parameter {parameter.name} lacks numeric domain")
                    lower = float(parameter.domain["lower"])
                    upper = float(parameter.domain["upper"])
                    numeric = float(value)
                    if parameter.domain.get("scale") == "log":
                        numeric = math.log(max(numeric, np.finfo(float).tiny))
                        lower, upper = math.log(lower), math.log(upper)
                    row.append((numeric - lower) / (upper - lower))
                if parameter.decision == "conditional":
                    row.append(1.0 if parameter.name in active else 0.0)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def reference_candidate(self) -> dict[str, Any]:
        return self.normalize(
            {
                parameter.name: (
                    parameter.fixed_value
                    if parameter.decision == "fixed"
                    else parameter.reference_value
                )
                for parameter in self.parameters
            }
        )

    def key(self, candidate: Mapping[str, Any]) -> str:
        normalized = self.normalize(candidate)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
