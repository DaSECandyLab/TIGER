# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import math
import numpy  as np
import pandas as pd
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import random
from ..design_space.design_space import DesignSpace
from ..models.model_factory import get_model, get_model_class
from ..acquisitions.acq import GeneralAcq
from ..acq_optimizers.evolution_optimizer import EvolutionOpt
from .util import ensure_hard_constr
from torch.quasirandom import SobolEngine

from .abstract_optimizer import AbstractOptimizer

from pymoo.indicators.hv import HV
from pymoo.util.dominator import Dominator

    
class GeneralBO(AbstractOptimizer):
    """
    Bayesian optimisation that supports multi-objective and constrained optimization
    """
    def __init__(self,
            space : DesignSpace,
            num_obj:            int   = 1,
            num_model_constr:   int   = 0,
            num_hard_constr:    int   = 0,
            num_hidden_constr:   int  = 1,
            max_sequence_length: int  = 12,       # max_sequence_length default to 2^12=4096
            rand_sample:        int   = None,
            model_name:         str   = 'multi_task',
            model_config:       dict  = None,
            kappa:              float = 2.,
            c_kappa:            float = 0.,
            use_noise:          bool  = False, 
            evo_pop:            int   = 100,
            evo_iters:          int   = 200,
            ref_point:          np.ndarray = None,
            scramble_seed:      Optional[int] = None,
            model_constraints_affect_candidate_generation: bool = True,
            constrained_ehvi_mc_samples: int = 10,
            quality_probability_affects_acquisition: bool = True,
            candidate_supplement_per_source: int = 0,
            candidate_transform: Optional[
                Callable[[pd.DataFrame], pd.DataFrame]
            ] = None,
            ):
        super().__init__(space)
        self.space        = space
        self.num_obj      = num_obj
        self.num_model_constr   = num_model_constr
        self.num_hard_constr    = num_hard_constr
        self.num_hidden_constr  = num_hidden_constr
        self.max_sequence_length = max_sequence_length
        self.rand_sample  = 1 + self.space.num_paras if rand_sample is None else rand_sample
        self.model_name   = model_name
        self.model_config = model_config if model_config is not None else {}
        self.X            = pd.DataFrame(columns = self.space.para_names)
        self.y            = np.zeros((0, num_obj + num_model_constr))
        self.kappa        = kappa
        self.c_kappa      = c_kappa
        self.use_noise    = use_noise
        self.model        = None
        self.quality_model = None
        # Timeout extrapolations train performance only. They never enter
        # self.y's measured quality constraint or the observed Pareto front.
        self.performance_X = pd.DataFrame(columns=self.space.para_names)
        self.performance_y = np.zeros((0, num_obj))
        self.model_constraints_affect_candidate_generation = (
            model_constraints_affect_candidate_generation
        )
        if constrained_ehvi_mc_samples < 1:
            raise ValueError("constrained_ehvi_mc_samples must be positive")
        self.constrained_ehvi_mc_samples = int(constrained_ehvi_mc_samples)
        self.quality_probability_affects_acquisition = bool(
            quality_probability_affects_acquisition
        )
        self.candidate_supplement_per_source = candidate_supplement_per_source
        self.candidate_transform = candidate_transform
        self._attempted_candidate_keys: set[tuple[Any, ...]] = set()
        self.evo_pop      = evo_pop
        self.evo_iters    = evo_iters
        self.iter         = 0
        self.ref_point    = ref_point
        self.sobol       = SobolEngine(self.space.num_paras, scramble = True, seed = scramble_seed)
        if num_obj + num_model_constr > 1:
            assert get_model_class(model_name).support_multi_output

    def _canonicalize(self, candidates: pd.DataFrame) -> pd.DataFrame:
        result = candidates.copy()
        if self.candidate_transform is not None:
            result = self.candidate_transform(result)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("candidate_transform must return a pandas DataFrame")
        if len(result) != len(candidates):
            raise ValueError("candidate_transform must preserve the number of rows")
        missing = [name for name in self.space.para_names if name not in result.columns]
        if missing:
            raise ValueError(
                "candidate_transform omitted design-space columns: "
                + ", ".join(missing)
            )
        return result.loc[:, self.space.para_names].reset_index(drop=True)

    @staticmethod
    def _key_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if math.isnan(value):
                return ("__nan__",)
            return round(value, 12)
        return value

    def _candidate_key(self, row: pd.Series) -> tuple[Any, ...]:
        return tuple(
            self._key_value(row[name]) for name in self.space.para_names
        )

    def _known_candidate_keys(self) -> set[tuple[Any, ...]]:
        known = set(self._attempted_candidate_keys)
        for _, row in self.X.iterrows():
            known.add(self._candidate_key(row))
        return known

    def _filter_new_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        canonical = self._canonicalize(candidates)
        known = self._known_candidate_keys()
        accepted_indices: list[int] = []
        accepted_keys: set[tuple[Any, ...]] = set()
        for index, row in canonical.iterrows():
            key = self._candidate_key(row)
            if key in known or key in accepted_keys:
                continue
            accepted_indices.append(index)
            accepted_keys.add(key)
        return canonical.iloc[accepted_indices].reset_index(drop=True)

    def register_attempted(self, candidates: pd.DataFrame) -> None:
        """Exclude canonical candidates already attempted, including failures."""

        canonical = self._canonicalize(candidates)
        for _, row in canonical.iterrows():
            self._attempted_candidate_keys.add(self._candidate_key(row))

    def _mark_returned(self, candidates: pd.DataFrame) -> pd.DataFrame:
        canonical = self._canonicalize(candidates)
        self.register_attempted(canonical)
        return canonical

    def _ensure_candidate_count(
            self,
            candidates: pd.DataFrame,
            n_suggestions: int,
            fix_input=None,
            temporary_candidate_constraint=None,
            temporary_numeric_bounds=None,
            ) -> pd.DataFrame:
        result = self._filter_new_candidates(candidates)
        for _attempt in range(32):
            if len(result) >= n_suggestions:
                return result
            needed = n_suggestions - len(result)
            extra = self.quasi_sample(
                max(needed * 8, 32),
                fix_input,
                temporary_candidate_constraint,
                temporary_numeric_bounds,
            )
            result = self._filter_new_candidates(
                pd.concat([result, extra], axis=0, ignore_index=True)
            )
        raise RuntimeError(
            "unable to find enough unseen feasible candidates after constraint "
            f"projection: requested={n_suggestions}, found={len(result)}"
        )

    def quasi_sample(
            self,
            n,
            fix_input = None,
            temporary_candidate_constraint = None,
            temporary_numeric_bounds: Optional[Mapping[str, Sequence[float]]] = None,
            ):
        if temporary_candidate_constraint is not None:
            accepted = []
            remaining = int(n)
            attempts = 0
            while remaining > 0 and attempts < 32:
                batch = self.quasi_sample(
                    max(remaining * 8, 32),
                    fix_input,
                    temporary_candidate_constraint=None,
                    temporary_numeric_bounds=temporary_numeric_bounds,
                )
                feasible = np.asarray(
                    temporary_candidate_constraint(batch), dtype=bool
                ).reshape(-1)
                accepted.append(batch.iloc[np.flatnonzero(feasible)])
                remaining = n - sum(len(item) for item in accepted)
                attempts += 1
            result = pd.concat(accepted, axis=0, ignore_index=True).head(n)
            if len(result) < n:
                raise RuntimeError(
                    "temporary Observation search space produced too few "
                    "initial-design candidates"
                )
            return result
        if hasattr(self.space, "conditional_space"):
            samples = self.space.conditional_space.generate(
                self.sobol.draw(n).numpy(), fixed=fix_input,
                bounds=temporary_numeric_bounds,
            )
            return self._canonicalize(samples)
        samp    = self.sobol.draw(n)
        opt_lb  = self.space.opt_lb.clone()
        opt_ub  = self.space.opt_ub.clone()
        if temporary_numeric_bounds is not None:
            numeric_indices = {
                name: index for index, name in enumerate(self.space.numeric_names)
            }
            for name, bounds in temporary_numeric_bounds.items():
                if name not in numeric_indices:
                    raise ValueError(
                        f"temporary numeric bound references non-numeric parameter: {name}"
                    )
                if len(bounds) != 2:
                    raise ValueError(
                        f"temporary numeric bound must contain [lower, upper]: {name}"
                    )
                lower, upper = float(bounds[0]), float(bounds[1])
                index = numeric_indices[name]
                if lower < float(opt_lb[index]) or upper > float(opt_ub[index]):
                    raise ValueError(
                        f"temporary numeric bound escapes native domain: {name}"
                    )
                if lower > upper:
                    raise ValueError(
                        f"temporary numeric bound is empty: {name}"
                    )
                opt_lb[index] = lower
                opt_ub[index] = upper
        samp    = samp * (opt_ub - opt_lb) + opt_lb
        x       = samp[:, :self.space.num_numeric]
        xe      = samp[:, self.space.num_numeric:]
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        if self.num_hard_constr and self.candidate_transform is None:
            df_samp = ensure_hard_constr(
                df_samp,
                max_sequence_length=2**self.max_sequence_length,
            )
        return self._canonicalize(df_samp)

    def fit_current_models(self):
        """Fit the native BO posterior without proposing or registering a candidate."""
        X, Xe      = self.space.transform(self.X)
        y          = torch.FloatTensor(self.y)
        num_uniqs  = None if Xe.shape[1] == 0 else [len(self.space.paras[name].categories) for name in self.space.enum_names]
        if (
            self.num_model_constr
            and not self.model_constraints_affect_candidate_generation
        ):
            # 性能模型必须与原三目标 BO 完全相同。质量模型独立训练，且
            # 保存/恢复全部 RNG 状态，避免它间接改变原 BO 的候选序列。
            assert self.num_model_constr == 1
            performance_y = y[:, :self.num_obj]
            performance_x, performance_xe = X, Xe
            if len(self.performance_X):
                extra_x, extra_xe = self.space.transform(self.performance_X)
                performance_x = torch.cat([X, extra_x], dim=0)
                performance_xe = torch.cat([Xe, extra_xe], dim=0)
                performance_y = torch.cat([performance_y, torch.FloatTensor(self.performance_y)], dim=0)
            self.model = get_model(self.model_name, X.shape[1], Xe.shape[1], performance_y.shape[1], num_uniqs = num_uniqs, **self.model_config)
            self.model.fit(performance_x, performance_xe, performance_y)
            # Measured constraint columns still select the observed Pareto front;
            # performance-only BO must not construct or query a quality GP.
            self.quality_model = None
            if not self.quality_probability_affects_acquisition:
                return X, Xe, y, num_uniqs
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            try:
                quality_y = y[:, self.num_obj:]
                self.quality_model = get_model(self.model_name, X.shape[1], Xe.shape[1], quality_y.shape[1], num_uniqs = num_uniqs, **self.model_config)
                self.quality_model.fit(X, Xe, quality_y)
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
        else:
            self.model = get_model(self.model_name, X.shape[1], Xe.shape[1], y.shape[1], num_uniqs = num_uniqs, **self.model_config)
            self.model.fit(X, Xe, y)
        return X, Xe, y, num_uniqs

    def suggest(
        self,
        n_suggestions=1,
        fix_input=None,
        rf_with_thres=None,
        temporary_candidate_constraint=None,
        temporary_numeric_bounds=None,
    ):
        self.iter += 1
        if self.num_performance_observations < self.rand_sample:
            sample = self.quasi_sample(
                max(n_suggestions, 1),
                fix_input,
                temporary_candidate_constraint,
                temporary_numeric_bounds,
            )
            sample = self._ensure_candidate_count(
                sample,
                n_suggestions,
                fix_input,
                temporary_candidate_constraint,
                temporary_numeric_bounds,
            )
            return self._mark_returned(sample.head(n_suggestions))
        else:
            X, Xe, y, num_uniqs = self.fit_current_models()
            kappa   = self.kappa
            c_kappa = self.c_kappa
            upsi    = 0.1
            delta   = 0.01
            if kappa is None:
                kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(self.iter) + np.log(3 * np.pi**2 / (3 * delta))))
            if c_kappa is None:
                c_kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(self.iter) + np.log(3 * np.pi**2 / (3 * delta))))
            acq = GeneralAcq(
                  self.model,
                  self.num_obj,
                  (
                      self.num_model_constr
                      if self.model_constraints_affect_candidate_generation
                      else 0
                  ),
                  num_hard_constr = self.num_hard_constr, 
                  num_hidden_constr = self.num_hidden_constr, 
                  max_sequence_length = self.max_sequence_length,
                  rf_with_thres = rf_with_thres,
                  space = self.space,
                  kappa     = kappa, 
                  c_kappa   = c_kappa,
                  use_noise = self.use_noise)
            opt     = EvolutionOpt(
                self.space,
                acq,
                pop=self.evo_pop,
                iters=self.evo_iters,
                temporary_candidate_constraint=temporary_candidate_constraint,
                temporary_numeric_bounds=temporary_numeric_bounds,
                candidate_transform=self.candidate_transform,
            )
            evolution_candidates = opt.optimize(fix_input=fix_input)
            evolution_keys = {self._candidate_key(row) for _, row in evolution_candidates.iterrows()}
            suggest = self._ensure_candidate_count(
                evolution_candidates,
                n_suggestions,
                fix_input,
                temporary_candidate_constraint,
                temporary_numeric_bounds,
            )
            candidate_generation = dict(
                getattr(opt, "last_generation", {}) or {}
            )
            candidate_sources = ["lcb_evolution" if self._candidate_key(row) in evolution_keys
                                 else "native_sobol_fallback" for _, row in suggest.iterrows()]
            if self.candidate_supplement_per_source and self.ref_point is not None:
                from ...candidate_supplements import supplement_candidates
                suggest, candidate_sources, supplement_receipt = supplement_candidates(
                    self, suggest, acq, per_source=self.candidate_supplement_per_source,
                    fixed=fix_input, bounds=temporary_numeric_bounds,
                    candidate_constraint=temporary_candidate_constraint,
                    base_sources=candidate_sources,
                )
                candidate_generation["supplements"] = supplement_receipt
            if self.ref_point is None:
                with torch.no_grad():
                    py, ps2 = self.model.predict(*self.space.transform(suggest))
                    largest_uncert_id = np.argmax(np.log(ps2).sum(axis = 1))
                select_id = np.random.choice(suggest.shape[0], n_suggestions, replace = False).tolist()
                if largest_uncert_id not in select_id:
                    select_id[0] = largest_uncert_id
                return self._mark_returned(suggest.iloc[select_id])
            else:
                assert self.num_obj > 1
                if self.num_model_constr == 0:
                    # 原 BO 分支保持逐行一致；质量关闭时不得改变候选序列。
                    n_mc = 10
                    hv   = HV(ref_point = self.ref_point.reshape(-1))
                    with torch.no_grad():
                        py, ps2 = self.model.predict(*self.space.transform(suggest))
                        y_samp  = self.model.sample_y(*self.space.transform(suggest), n_mc).numpy()
                    y_curr    = self.get_pf(self.y).copy()
                    select_id = []
                    for i in range(n_suggestions):
                        ehvi_lst = []
                        base_hv  = hv.do(y_curr)
                        for j in range(suggest.shape[0]):
                            samp    = y_samp[:, j]
                            hvi_est = 0
                            for k in range(n_mc):
                                y_tmp    = np.vstack([y_curr, samp [[k]]])
                                hvi_est += hv.do(y_tmp) - base_hv
                            hvi_est /= n_mc
                            ehvi_lst.append(hvi_est)
                        best_id = np.argmax(ehvi_lst) if max(ehvi_lst) > 0 else np.random.choice(suggest.shape[0])
                        y_curr  = np.vstack([y_curr, y_samp[:, best_id].min(axis = 0, keepdims = True)])
                        select_id.append(best_id)
                else:
                    # 第 4 个独立 GPy 输出与性能后验做联合 MC。候选生成不把
                    # 质量预测均值作为硬约束；每个性能 HVI 样本仅在配对的质量
                    # 样本可行时计入 cEHVI。
                    assert self.num_model_constr == 1
                    n_mc = self.constrained_ehvi_mc_samples
                    hv   = HV(ref_point = self.ref_point.reshape(-1))
                    transformed = self.space.transform(suggest)
                    with torch.no_grad():
                        py, ps2 = self.model.predict(*transformed)
                        y_samp  = self.model.sample_y(*transformed, n_mc).numpy()
                        quality_samp = None
                        if not self.quality_probability_affects_acquisition:
                            pass
                        elif self.model_constraints_affect_candidate_generation:
                            quality_py = py[:, self.num_obj:]
                            quality_ps2 = ps2[:, self.num_obj:]
                            quality_samp = y_samp[:, :, self.num_obj:]
                        else:
                            assert self.quality_model is not None
                            quality_py, quality_ps2 = self.quality_model.predict(
                                *transformed
                            )
                            quality_samp = self.quality_model.sample_y(
                                *transformed, n_mc
                            ).numpy()
                    feasible_history = self.get_pf(self.y).copy()
                    y_curr = feasible_history[:, :self.num_obj]
                    p_feasible = None
                    if quality_samp is not None:
                        quality_std = quality_ps2.sqrt().clamp(
                            min=torch.finfo(quality_ps2.dtype).eps
                        )
                        p_feasible = (
                            torch.distributions.Normal(0.0, 1.0)
                            .cdf(-quality_py[:, 0] / quality_std[:, 0])
                            .cpu()
                            .numpy()
                        )
                    # P_run 仍由原 hidden-constraint 模型用于 GeneralAcq 的
                    # 候选约束；这里只记录诊断值，不作为最终乘数。
                    p_run = np.ones(suggest.shape[0], dtype=float)
                    if (
                        isinstance(rf_with_thres, tuple)
                        and rf_with_thres[0] is not None
                    ):
                        p_run = np.asarray(
                            rf_with_thres[0].predict(suggest), dtype=float
                        )
                    select_id = []
                    for i in range(n_suggestions):
                        ehvi = []
                        constrained_ehvi = []
                        base_hv = hv.do(y_curr)
                        for j in range(suggest.shape[0]):
                            hvi_est = 0.0
                            constrained_hvi_est = 0.0
                            for k in range(n_mc):
                                sample = y_samp[k, j, :self.num_obj]
                                y_tmp = np.vstack(
                                    [y_curr, sample[None, :]]
                                )
                                hvi = hv.do(y_tmp) - base_hv
                                hvi_est += hvi
                                if quality_samp is not None and quality_samp[k, j, 0] <= 0:
                                    constrained_hvi_est += hvi
                            ehvi.append(hvi_est / n_mc)
                            constrained_ehvi.append(
                                constrained_hvi_est / n_mc
                            )
                        score = np.asarray(
                            constrained_ehvi
                            if self.quality_probability_affects_acquisition
                            else ehvi
                        )
                        best_id = (
                            int(np.argmax(score))
                            if float(np.max(score)) > 0
                            else int(np.random.choice(suggest.shape[0]))
                        )
                        y_curr = np.vstack([
                            y_curr,
                            y_samp[:, best_id, :self.num_obj].min(
                                axis=0, keepdims=True
                            ),
                        ])
                        select_id.append(best_id)
                        self.last_quality_acquisition = {
                            "selected_candidate_source": candidate_sources[best_id],
                            "candidate_count": int(suggest.shape[0]),
                            "monte_carlo_samples": n_mc,
                            "ehvi": float(ehvi[best_id]),
                            "constrained_ehvi": float(
                                constrained_ehvi[best_id]
                            ) if quality_samp is not None else None,
                            "p_run": float(p_run[best_id]),
                            "p_run_affects_final_score": False,
                            "score": float(score[best_id]),
                            "quality_mc_indicator": bool(
                                self.quality_probability_affects_acquisition
                            ),
                            "scalar_ehvi_times_p_feasible": False,
                            "quality_mean_hard_threshold": False,
                            "quality_surrogate_affects_candidate_generation": bool(
                                self.model_constraints_affect_candidate_generation
                            ),
                            "quality_surrogate_affects_final_ranking": bool(
                                self.quality_probability_affects_acquisition
                            ),
                            "quality_feasible_mc_samples": int(
                                (quality_samp[:, best_id, 0] <= 0).sum()
                            ) if quality_samp is not None else None,
                            "quality_constraint_mean": float(
                                quality_py[best_id, 0]
                            ) if quality_samp is not None else None,
                            "quality_constraint_variance": float(
                                quality_ps2[best_id, 0]
                            ) if quality_samp is not None else None,
                            "p_feasible": float(
                                p_feasible[best_id]
                            ) if p_feasible is not None else None,
                            "quality_surrogate_fitted": self.quality_model is not None,
                            "candidate_generation": candidate_generation,
                        }
                        if getattr(self, 'retain_ranked_candidate_pool', False) and n_suggestions == 1:
                            # Optional external screening can try the next EHVI
                            # point without fitting or resampling performance.
                            self.ranked_candidate_pool = [
                                (suggest.iloc[[j]].copy(), {
                                    **self.last_quality_acquisition,
                                    'ehvi': float(ehvi[j]),
                                    'score': float(score[j]),
                                    'p_run': float(p_run[j]),
                                    'selected_candidate_source': candidate_sources[j],
                                })
                                for j in sorted(range(len(score)), key=lambda j: (-score[j], j))
                                if j != best_id
                            ]

            select_id = list(set(select_id))
            if len(select_id) < n_suggestions:
                candidate_id = [i for i in range(suggest.shape[0]) if i not in select_id]
                select_id   += np.random.choice(candidate_id, n_suggestions - len(select_id), replace = False).tolist()
            return self._mark_returned(suggest.iloc[select_id])

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        valid_id = np.isfinite(y).all(axis = 1)
        XX       = self._canonicalize(X.iloc[valid_id])
        yy       = y[valid_id]
        self.X   = pd.concat([self.X, XX], axis = 0, ignore_index = True)
        self.y   = np.vstack([self.y, yy])
        assert self.y.shape[1] == self.num_obj + self.num_model_constr

    def observe_performance(self, X, y):
        """Register explicitly estimated objectives without inventing quality."""
        if self.num_model_constr != 1 or self.model_constraints_affect_candidate_generation or self.quality_probability_affects_acquisition:
            raise ValueError("Estimated feedback requires independent performance-only models")
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X), self.num_obj) or not np.isfinite(y).all():
            raise ValueError("Estimated performance targets must be finite and match the objective count")
        self.performance_X = pd.concat([self.performance_X, self._canonicalize(X)], ignore_index=True)
        self.performance_y = np.vstack([self.performance_y, y])
        self.register_attempted(X)

    @property
    def num_performance_observations(self):
        return len(self.X) + len(self.performance_X)

    def select_best(self, rec : pd.DataFrame) -> pd.DataFrame:
        pass

    def get_pf(self, y : np.ndarray, return_optimal = False) -> pd.DataFrame:
        feasible = (y[:, self.num_obj:] <= 0).all(axis = 1)
        y        = y[feasible].copy()
        y_obj    = y[:, :self.num_obj]
        if feasible.any():
            dom_mat = Dominator().calc_domination_matrix(y_obj, None)
            optimal = (dom_mat >= 0).all(axis = 1)
        else:
            optimal = feasible

        if not return_optimal:
            return y[optimal].copy()
        else:
            return optimal

    @property
    def best_x(self):
        optimal = self.get_pf(self.y, return_optimal = True)
        return self.X[optimal].copy()

    @property
    def best_y(self):
        return self.get_pf(self.y)
