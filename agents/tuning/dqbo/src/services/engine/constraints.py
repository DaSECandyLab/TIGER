"""Candidate Optimizer 的候选已知约束检查与域内修复。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from .flock.config import (
    canonicalize_kv_cache_dtype,
    canonicalize_model_dtype,
    resolve_effective_dtype,
    resolve_effective_kv_cache_dtype,
)
from .space import FixedSearchSpace


COMPRESSED_KV_CACHE_DTYPES = frozenset({"fp8_e4m3", "int8_per_token_head"})
MIN_COMPRESSED_KV_BLOCK_SIZE = 16
HALF_ONLY_QUANTIZATION_METHODS = frozenset({"awq", "gptq"})
KNOWN_CONSTRAINT_IDS = (
    "representation_dtype_template",
    "gpu_topology_budget",
    "dcp_divides_tp",
    "async_executor_compatibility",
    "attention_backend_uncompressed_kv_dtype_alignment",
    "compressed_kv_block_size",
    "scheduler_token_budget",
)


def uncompressed_kv_dtype_compatible(model_dtype: str | None, kv_dtype: str | None) -> bool:
    """Shared predicate; callers must first establish the backend's alignment scope."""
    if model_dtype is None or kv_dtype is None:
        return True
    if kv_dtype == "float16":
        return model_dtype == "half"
    if kv_dtype == "bfloat16":
        return model_dtype == "bfloat16"
    return True


@dataclass(frozen=True)
class ConstraintRepairResult:
    candidate: dict[str, Any]
    repairs: tuple[dict[str, Any], ...]
    effective_values: dict[str, Any]
    validation_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.validation_errors

    @property
    def changed(self) -> bool:
        return bool(self.repairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "changed": self.changed,
            "candidate": dict(self.candidate),
            "repairs": [dict(item) for item in self.repairs],
            "effective_values": dict(self.effective_values),
            "validation_errors": list(self.validation_errors),
            "known_constraint_ids": list(KNOWN_CONSTRAINT_IDS),
        }


class KnownConstraintHandler:
    """在 Range 固定域内规范候选，并在 Trial 前做最后一次机械检查。"""

    def __init__(
        self,
        search_space: FixedSearchSpace,
        *,
        model_path: str | None = None,
        awq_model_path: str | None = None,
        gptq_model_path: str | None = None,
        uncompressed_kv_model_dtype_alignment: bool = False,
        alignment_scope: str | None = None,
    ) -> None:
        self.space = search_space
        self.model_path = model_path
        self.awq_model_path = awq_model_path
        self.gptq_model_path = gptq_model_path
        self.uncompressed_kv_model_dtype_alignment = bool(
            uncompressed_kv_model_dtype_alignment
        )
        self.alignment_scope = alignment_scope

    def _set_within_domain(
        self,
        candidate: dict[str, Any],
        repairs: list[dict[str, Any]],
        *,
        parameter_name: str,
        value: Any,
        constraint_id: str,
        reason: str,
    ) -> bool:
        parameter = self.space.by_name.get(parameter_name)
        if parameter is None or not parameter.contains(value):
            return False
        old_value = candidate.get(parameter_name)
        if old_value == value:
            return True
        candidate[parameter_name] = value
        repairs.append(
            {
                "constraint_id": constraint_id,
                "parameter": parameter_name,
                "from": old_value,
                "to": value,
                "reason": reason,
            }
        )
        return True

    def _repair_representation(
        self, candidate: dict[str, Any], repairs: list[dict[str, Any]]
    ) -> None:
        dtype_parameter = self.space.by_name.get("dtype")
        if dtype_parameter is None or not any(
            "->" in condition for condition in dtype_parameter.activation_conditions
        ):
            return
        quantization = str(candidate.get("quantization") or "none")
        expected = self.space.representation_dtype_template().get(quantization)
        if expected is not None:
            self._set_within_domain(
                candidate,
                repairs,
                parameter_name="dtype",
                value=expected,
                constraint_id="representation_dtype_template",
                reason=f"quantization={quantization!r} requires dtype={expected!r}",
            )

    def _repair_joint_constraints(
        self, candidate: dict[str, Any], repairs: list[dict[str, Any]]
    ) -> None:
        for constraint in self.space.joint_constraints:
            identifier = str(constraint.get("id", ""))
            if (identifier == "cacheblend_compatibility" and constraint.get("require_enforce_eager")
                    and candidate.get("cacheblend_mode") == "on"):
                self._set_within_domain(candidate, repairs, parameter_name="enforce_eager", value=True,
                    constraint_id="cacheblend_compatibility",
                    reason="frozen workload domain requires eager execution when CacheBlend is on")
                # Apply the switch's existing dependencies before validation;
                # otherwise an off->on RepairBO candidate keeps stale off flags.
                targets = {"enable_prefix_caching": False,
                    "kv_cache_dtype": "float16" if candidate.get("dtype") in ("half", "float16") else "bfloat16"}
                for name in ("chunked_prefill", "enable_chunked_prefill"):
                    if name in self.space.by_name:
                        targets[name] = False
                if int(candidate.get("block_size", 16)) < 16:
                    targets["block_size"] = 16
                for gate in self.space.joint_constraints:
                    if gate.get("id") == "full_prefill_token_budget":
                        limit = candidate.get("max_model_len") or gate.get("max_model_len")
                        if limit and candidate.get("max_num_batched_tokens", 0) < limit:
                            targets["max_num_batched_tokens"] = limit
                for name, value in targets.items():
                    if name in self.space.by_name:
                        self._set_within_domain(candidate, repairs, parameter_name=name, value=value,
                            constraint_id="cacheblend_compatibility",
                            reason="apply frozen CacheBlend on-branch execution dependencies")
            if identifier in {"gpu_topology_budget", "two_gpu_topology"}:
                expression = str(constraint.get("expression", ""))
                match = re.search(r"==\s*(\d+)\s*$", expression)
                expected = int(match.group(1)) if match else None
                tp = int(candidate.get("tensor_parallel_size", 1))
                dp = int(candidate.get("data_parallel_size", 1))
                if expected and tp * dp != expected and expected % tp == 0:
                    target_dp = expected // tp
                    self._set_within_domain(
                        candidate,
                        repairs,
                        parameter_name="data_parallel_size",
                        value=target_dp,
                        constraint_id="gpu_topology_budget",
                        reason=f"derive DP so TP*DP equals the GPU budget {expected}",
                    )
            elif identifier == "dcp_divides_tp":
                tp = int(candidate.get("tensor_parallel_size", 1))
                dcp = int(candidate.get("decode_context_parallel_size", 1))
                if dcp < 1 or tp % dcp != 0:
                    self._set_within_domain(
                        candidate,
                        repairs,
                        parameter_name="decode_context_parallel_size",
                        value=1,
                        constraint_id="dcp_divides_tp",
                        reason=f"DCP must be a positive divisor of TP={tp}",
                    )
            elif identifier == "async_executor_compatibility":
                if candidate.get("data_parallel_backend") == "ray" and bool(
                    candidate.get("async_scheduling", False)
                ):
                    self._set_within_domain(
                        candidate,
                        repairs,
                        parameter_name="async_scheduling",
                        value=False,
                        constraint_id="async_executor_compatibility",
                        reason="the legacy Ray executor does not support async scheduling",
                    )

    def _effective_model_dtype(self, candidate: Mapping[str, Any]) -> str | None:
        raw_dtype = candidate.get("dtype")
        quantization = str(candidate.get("quantization") or "none")
        if quantization in HALF_ONLY_QUANTIZATION_METHODS:
            return "half"
        canonical = canonicalize_model_dtype(raw_dtype)
        if canonical != "auto" or not all(
            (self.model_path, self.awq_model_path, self.gptq_model_path)
        ):
            return canonical
        return resolve_effective_dtype(
            raw_dtype,
            quantization=quantization,
            model_path=str(self.model_path),
            awq_model_path=str(self.awq_model_path),
            gptq_model_path=str(self.gptq_model_path),
        )

    def _effective_kv_cache_dtype(
        self, candidate: Mapping[str, Any], effective_model_dtype: str | None
    ) -> str | None:
        raw_value = candidate.get("kv_cache_dtype")
        canonical = canonicalize_kv_cache_dtype(raw_value)
        if canonical != "auto" or not all(
            (self.model_path, self.awq_model_path, self.gptq_model_path)
        ):
            return canonical
        return resolve_effective_kv_cache_dtype(
            raw_value,
            dtype=effective_model_dtype,
            quantization=candidate.get("quantization"),
            model_path=str(self.model_path),
            awq_model_path=str(self.awq_model_path),
            gptq_model_path=str(self.gptq_model_path),
        )

    def _kv_cache_dtype_compatible(
        self,
        effective_model_dtype: str | None,
        effective_kv_cache_dtype: str | None,
    ) -> bool:
        """Check only a measured backend's uncompressed Q/K dtype dependency.

        Explicit model and KV storage dtypes are not universally required to
        match.  In the measured FLASH_ATTN execution path, however, float16 and
        bfloat16 KV tensors become attention keys directly and the kernel
        requires query and key to share a dtype.  Quantized KV formats use a
        separate conversion path and are deliberately outside this rule.
        """

        if not self.uncompressed_kv_model_dtype_alignment:
            return True
        return uncompressed_kv_dtype_compatible(effective_model_dtype, effective_kv_cache_dtype)

    def _repair_dtype_and_cache(
        self, candidate: dict[str, Any], repairs: list[dict[str, Any]]
    ) -> tuple[str | None, str | None]:
        if "dtype" not in self.space.by_name or "kv_cache_dtype" not in self.space.by_name:
            return None, None
        effective_model_dtype = self._effective_model_dtype(candidate)
        effective_kv_cache_dtype = self._effective_kv_cache_dtype(
            candidate, effective_model_dtype
        )
        if not self._kv_cache_dtype_compatible(
            effective_model_dtype, effective_kv_cache_dtype
        ):
            target = {
                "half": "float16",
                "bfloat16": "bfloat16",
            }.get(effective_model_dtype)
            if target is not None:
                self._set_within_domain(
                    candidate,
                    repairs,
                    parameter_name="kv_cache_dtype",
                    value=target,
                    constraint_id=(
                        "attention_backend_uncompressed_kv_dtype_alignment"
                    ),
                    reason=(
                        f"measured attention backend {self.alignment_scope!r} "
                        "requires uncompressed query and key tensors to share "
                        f"a dtype; align KV cache with model dtype "
                        f"{effective_model_dtype!r}"
                    ),
                )
                effective_kv_cache_dtype = self._effective_kv_cache_dtype(
                    candidate, effective_model_dtype
                )

        block_size = candidate.get("block_size")
        if (
            effective_kv_cache_dtype in COMPRESSED_KV_CACHE_DTYPES
            and isinstance(block_size, int)
            and block_size < MIN_COMPRESSED_KV_BLOCK_SIZE
        ):
            self._set_within_domain(
                candidate,
                repairs,
                parameter_name="block_size",
                value=MIN_COMPRESSED_KV_BLOCK_SIZE,
                constraint_id="compressed_kv_block_size",
                reason="compressed KV cache requires block_size >= 16",
            )
        return effective_model_dtype, effective_kv_cache_dtype

    def _repair_scheduler(
        self, candidate: dict[str, Any], repairs: list[dict[str, Any]]
    ) -> None:
        max_num_seqs = candidate.get("max_num_seqs")
        max_num_batched_tokens = candidate.get("max_num_batched_tokens")
        if (
            isinstance(max_num_seqs, int)
            and isinstance(max_num_batched_tokens, int)
            and max_num_seqs > max_num_batched_tokens
        ):
            self._set_within_domain(
                candidate,
                repairs,
                parameter_name="max_num_batched_tokens",
                value=max_num_seqs,
                constraint_id="scheduler_token_budget",
                reason="max_num_batched_tokens must be at least max_num_seqs",
            )

    def repair(self, raw_candidate: Mapping[str, Any]) -> ConstraintRepairResult:
        candidate = self.space.normalize(raw_candidate)
        repairs: list[dict[str, Any]] = []
        self._repair_representation(candidate, repairs)
        self._repair_joint_constraints(candidate, repairs)
        self._repair_scheduler(candidate, repairs)
        candidate = self.space.normalize(candidate)
        effective_model_dtype, effective_kv_cache_dtype = self._repair_dtype_and_cache(
            candidate, repairs
        )
        candidate = self.space.normalize(candidate)
        validation_errors = list(self.space.validate(candidate))
        if not self._kv_cache_dtype_compatible(
            effective_model_dtype, effective_kv_cache_dtype
        ):
            validation_errors.append(
                "measured attention backend requires aligned uncompressed "
                "model/KV cache dtypes, but no Range-valid repair exists"
            )
        return ConstraintRepairResult(
            candidate=candidate,
            repairs=tuple(repairs),
            effective_values={
                "model_dtype": effective_model_dtype,
                "kv_cache_dtype": effective_kv_cache_dtype,
            },
            validation_errors=tuple(validation_errors),
        )

    def sample(
        self,
        rng: np.random.Generator,
        *,
        count: int,
        exclude: set[str] | None = None,
        max_attempts: int | None = None,
    ) -> list[dict[str, Any]]:
        """采样后先修复再去重，使 BO 建模值与实际执行值完全一致。"""

        if count < 1:
            return []
        excluded = set(exclude or ())
        raw_excluded: set[str] = set()
        samples: list[dict[str, Any]] = []
        attempts = 0
        attempt_limit = max_attempts or max(1000, count * 300)
        while len(samples) < count and attempts < attempt_limit:
            attempts += 1
            try:
                raw = self.space.sample(
                    rng, count=1, exclude=raw_excluded, max_attempts=1000
                )[0]
            except RuntimeError:
                break
            raw_excluded.add(self.space.key(raw))
            repaired = self.repair(raw)
            if not repaired.valid:
                continue
            key = self.space.key(repaired.candidate)
            if key in excluded:
                continue
            excluded.add(key)
            samples.append(repaired.candidate)
        if len(samples) < count:
            raise RuntimeError(
                f"could only sample {len(samples)}/{count} unique constraint-safe "
                f"candidates after {attempts} attempts"
            )
        return samples
