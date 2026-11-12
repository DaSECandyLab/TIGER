"""Dependency-aware sampling and positive-definite conditional GP embeddings.

The same activity predicates drive generation and prediction. Coordinates remain
physical HEBO values; historical configurations never become latent genotypes.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

VERSION = "conditional-space.v1"


class ConditionalSpace:
    def __init__(self, space, *, defaults, gpu_count=2, ray_v2=False,
                 align_uncompressed=False, representation_dtypes=None,
                 cacheblend_allow_prefix_caching=False, cacheblend_require_eager=False):
        self.cacheblend_require_eager = cacheblend_require_eager
        self.cacheblend_allow_prefix_caching = cacheblend_allow_prefix_caching
        self.space = space
        self.defaults = dict(defaults)
        self.gpu_count = gpu_count
        self.ray_v2 = ray_v2
        self.align_uncompressed = align_uncompressed
        self.representation_dtypes = representation_dtypes or {}
        self.activity = {
            "data_parallel_backend": ("dp", "mp"),
            "disable_nccl_for_dp_synchronization": ("dp", False),
            "cp_kv_cache_interleave_size": ("dcp", 1),
            "prefix_caching_hash_algo": ("prefix", "sha256"),
        }
        for name in ("lmcache_chunk_size", "cacheblend_blend_min_tokens",
                     "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios",
                     "cacheblend_allow_prefix_caching"):
            if name in space.paras:
                self.activity[name] = ("cacheblend", self.defaults[name])
        if "long_prefill_token_threshold" in space.paras:
            self.activity["long_prefill_token_threshold"] = ("chunked", self.defaults.get("long_prefill_token_threshold", 0))
        self.activity = {k: v for k, v in self.activity.items() if k in space.paras}
        priority = ["tp", "data_parallel_size", "decode_context_parallel_size",
                    "cacheblend_mode", "enable_prefix_caching", "quantization", "dtype", "kv_cache_dtype",
                    "data_parallel_backend", "distributed_executor_backend", "enable_chunked_prefill", "max_num_seqs",
                    "cacheblend_blend_check_layers", "cacheblend_blend_recompute_ratios"]
        self.order = [k for k in priority if k in space.paras]
        self.order += [k for k in space.para_names if k not in self.order]

    @staticmethod
    def cacheblend_active(row):
        return str(row.get("cacheblend_mode", "off")).lower() == "on"

    def active(self, name, row):
        kind = self.activity.get(name, (None, None))[0]
        if kind == "dp":
            return (int(row.get("data_parallel_size", 1)) > 1
                    if "data_parallel_size" in self.space.paras
                    else self.gpu_count // int(row.get("tp", 1)) > 1)
        if kind == "dcp":
            return int(row.get("decode_context_parallel_size", 1)) > 1
        if kind == "cacheblend":
            return self.cacheblend_active(row)
        if kind == "chunked":
            return bool(row.get("enable_chunked_prefill", self.defaults.get("enable_chunked_prefill", True)))
        if kind == "prefix":
            return bool(row.get("enable_prefix_caching", True))
        return True

    def contract(self):
        return {"version": VERSION, "conditional_parameters": list(self.activity),
                "generation_order": self.order, "kernel": "product_matern32_activity_embedding",
                "input_warping": "fixed_domain_scaling_preserves_conditional_geometry",
                "gpu_count": self.gpu_count, "ray_v2": self.ray_v2,
                "align_uncompressed": self.align_uncompressed,
                "cacheblend_require_eager": self.cacheblend_require_eager}

    def _choices(self, name, row, choices):
        if name == "enforce_eager" and self.cacheblend_require_eager and self.cacheblend_active(row):
            return [v for v in choices if bool(v)]
        if self.cacheblend_active(row):
            def length(value):
                return len(value) if isinstance(value, (list, tuple)) else len(str(value).split(","))
            if name == "cacheblend_blend_check_layers" and "cacheblend_blend_recompute_ratios" in self.space.paras:
                ratios = self.space.paras["cacheblend_blend_recompute_ratios"].categories
                full_available = any(float(str(value).split(",")[0]) == 1.0 for value in ratios)
                choices = [value for value in choices if length(value) == 1 or full_available]
            if name == "cacheblend_blend_recompute_ratios":
                layers = row.get("cacheblend_blend_check_layers", self.defaults.get("cacheblend_blend_check_layers", "1"))
                if length(layers) > 1:
                    choices = [value for value in choices if float(str(value).split(",")[0]) == 1.0]
        if name == "enable_chunked_prefill" and self.cacheblend_active(row):
            return [v for v in choices if not v]
        if name == "enable_prefix_caching" and self.cacheblend_active(row) and not self.cacheblend_allow_prefix_caching:
            return [v for v in choices if not v]
        if name == "kv_cache_dtype" and self.cacheblend_active(row):
            required = "float16" if row.get("dtype") in ("half", "float16") else "bfloat16"
            return [v for v in choices if v == required]

        if name == "dtype" and row.get("quantization") in self.representation_dtypes:
            return [v for v in choices if v == self.representation_dtypes[row["quantization"]]]
        if name == "kv_cache_dtype" and self.align_uncompressed:
            wrong = "bfloat16" if row.get("dtype") in ("half", "float16") else "float16"
            return [v for v in choices if v != wrong]
        if name == "async_scheduling" and (row.get("data_parallel_backend") == "ray" or row.get("distributed_executor_backend") == "ray") and not self.ray_v2:
            return [v for v in choices if not v]
        if name == "block_size" and (self.cacheblend_active(row) or row.get("kv_cache_dtype") in {
                "fp8_e4m3", "fp8_e5m2", "int8_per_token_head", "fp8_per_token_head"}):
            return [v for v in choices if int(v) >= 16]
        return choices

    def generate(self, unit, *, fixed=None, bounds=None, proposals=None):
        """Decode independent unit draws in dependency order, without clipping
        invalid children onto boundaries. Mating may retain independent genes;
        dependent genes are drawn afresh under the selected parents.
        """
        unit = np.asarray(unit, dtype=float)
        fixed, bounds = fixed or {}, bounds or {}
        dependent = set(self.activity) | {"dtype", "kv_cache_dtype", "async_scheduling",
                                        "block_size", "max_num_batched_tokens",
                                        "gpu_memory_utilization", "enable_prefix_caching", "cacheblend_mode", "enable_chunked_prefill"}
        if self.cacheblend_require_eager:
            dependent.add("enforce_eager")
        rows = []
        for i, draws in enumerate(unit):
            row = dict(self.defaults)
            for name in self.order:
                p = self.space.paras[name]
                j = self.space.para_names.index(name)
                u = float(np.clip(draws[j], 0, np.nextafter(1., 0.)))
                forced = fixed.get(name)
                has_forced = name in fixed
                if not self.active(name, row):
                    value = self.activity[name][1]
                    if has_forced and forced != value:
                        raise ValueError(f"fixed inactive parameter conflicts: {name}")
                elif p.is_categorical:
                    choices = self._choices(name, row, list(p.categories))
                    if name == "cacheblend_mode":
                        if self.cacheblend_require_eager and "enforce_eager" in fixed and not fixed["enforce_eager"]:
                            choices = [v for v in choices if v != "on"]
                        if fixed.get("enable_chunked_prefill"):
                            choices = [v for v in choices if v != "on"]
                        if fixed.get("enable_prefix_caching") and not self.cacheblend_allow_prefix_caching:
                            choices = [v for v in choices if v != "on"]
                        if fixed.get("kv_cache_dtype") not in (None, "float16", "bfloat16"):
                            choices = [v for v in choices if v != "on"]
                        if any(child in fixed and fixed[child] != inactive
                               for child, (kind, inactive) in self.activity.items() if kind == "cacheblend"):
                            choices = [v for v in choices if v == "on"]
                    if name == "tp" and any(
                        child in fixed and fixed[child] != inactive
                        for child, (kind, inactive) in self.activity.items() if kind == "dp"
                    ):
                        choices = [v for v in choices if self.gpu_count // int(v) > 1]
                    if name in ("data_parallel_backend", "distributed_executor_backend") and fixed.get("async_scheduling") and not self.ray_v2:
                        choices = [v for v in choices if v != "ray"]
                    if name == "dtype" and self.align_uncompressed:
                        kv = fixed.get("kv_cache_dtype")
                        if kv in ("float16", "bfloat16"):
                            choices = [v for v in choices if
                                       (v in ("half", "float16")) == (kv == "float16")]
                    if not choices:
                        raise ValueError(f"empty conditional domain: {name}")
                    value = choices[min(int(u * len(choices)), len(choices)-1)]
                    if has_forced:
                        if forced not in choices:
                            raise ValueError(f"fixed parameter violates dependency: {name}")
                        value = forced
                    elif proposals is not None and name not in dependent:
                        value = proposals.iloc[i][name]
                else:
                    lo, hi = map(float, bounds.get(name, (p.opt_lb, p.opt_ub)))
                    lo, hi = max(lo, float(p.opt_lb)), min(hi, float(p.opt_ub))
                    if name == "enforce_eager" and self.cacheblend_require_eager and self.cacheblend_active(row):
                        lo = max(lo, float(p.transform(np.asarray([True]))[0]))
                    if name == "enable_chunked_prefill" and self.cacheblend_active(row):
                        hi = min(hi, 0.)
                    if name == "enable_prefix_caching" and self.cacheblend_active(row) and not self.cacheblend_allow_prefix_caching:
                        hi = min(hi, 0.)
                    if name == "async_scheduling" and (row.get("data_parallel_backend") == "ray" or row.get("distributed_executor_backend") == "ray") and not self.ray_v2:
                        hi = min(hi, 0.)
                    if name == "max_num_seqs" and "max_num_batched_tokens" in self.space.paras:
                        tokens = self.space.paras["max_num_batched_tokens"]
                        cap = tokens.inverse_transform(np.asarray([
                            bounds.get("max_num_batched_tokens", (tokens.opt_lb, tokens.opt_ub))[1]
                        ]))[0]
                        cap = fixed.get("max_num_batched_tokens", cap)
                        hi = min(hi, float(p.transform(np.asarray([cap]))[0]))
                    if name == "max_num_batched_tokens" and (self.cacheblend_active(row) or not bool(row.get("enable_chunked_prefill", True))):
                        lo = max(lo, float(p.transform(np.asarray([(row.get("max_model_len") or 32768)]))[0]))
                    if name == "max_num_batched_tokens":
                        lo = max(lo, float(p.transform(np.asarray([row["max_num_seqs"]]))[0]))
                    if name == "block_size" and (self.cacheblend_active(row) or row.get("kv_cache_dtype") in {
                            "fp8_e4m3", "fp8_e5m2", "int8_per_token_head", "fp8_per_token_head"}):
                        lo = max(lo, float(p.transform(np.asarray([16]))[0]))
                    if lo > hi:
                        raise ValueError(f"empty conditional domain: {name}")
                    # Existing executor/HEBO contract uses integer percentages.
                    if p.is_discrete_after_transform or name == "gpu_memory_utilization":
                        a, b = math.ceil(lo), math.floor(hi)
                        if a > b:
                            raise ValueError(f"empty discrete domain: {name}")
                        z = min(a + int(u * (b-a+1)), b)
                    else:
                        z = lo + u * (hi-lo)
                    value = p.inverse_transform(np.asarray([z]))[0]
                    sampled_value = value
                    if has_forced:
                        value = forced
                    elif proposals is not None and name not in dependent:
                        value = proposals.iloc[i][name]
                    encoded = float(p.transform(np.asarray([value]))[0])
                    if not has_forced and not lo-1e-6 <= encoded <= hi+1e-6:
                        value = sampled_value
                        encoded = float(p.transform(np.asarray([value]))[0])
                    if not lo-1e-6 <= encoded <= hi+1e-6:
                        raise ValueError(f"value outside conditional bounds: {name}")
                row[name] = value
            rows.append({k: row[k] for k in self.space.para_names})
        return pd.DataFrame(rows, columns=self.space.para_names)

    def embed(self, xc, xe):
        """Categorical active states have unit norm, inactive states are zero.
        Conditional numeric values lie on a semicircle, inactive at its centre.
        One isotropic kernel per parameter preserves constant cross-state distance.
        """
        frame = self.space.inverse_transform(xc, xe)
        arrays, groups, offset = [], [], 0
        for name in self.space.para_names:
            p = self.space.paras[name]
            if p.is_categorical:
                ids = p.transform(frame[name].to_numpy()).astype(int)
                values = np.eye(len(p.categories))[ids]
            else:
                z = p.transform(frame[name].to_numpy()).astype(float)
                width = float(p.opt_ub)-float(p.opt_lb)
                z = (z-float(p.opt_lb))/width if width else np.zeros_like(z)
                values = (np.column_stack([np.cos(np.pi*z), np.sin(np.pi*z)])
                          if name in self.activity else z[:, None])
            if name in self.activity:
                mask = np.asarray([self.active(name, {**self.defaults, **r})
                                   for r in frame.to_dict("records")])
                values = values * mask[:, None]
            arrays.append(values)
            groups.append(list(range(offset, offset+values.shape[1])))
            offset += values.shape[1]
        return np.column_stack(arrays), groups
