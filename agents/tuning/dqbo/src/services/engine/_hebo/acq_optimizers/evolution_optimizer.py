# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy  as np
import pandas as pd
import torch
from typing import Callable, Optional
from torch.quasirandom import SobolEngine
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.config import Config
Config.show_compile_hint = False

from ..design_space.design_space import DesignSpace
from ..acquisitions.acq import Acquisition


def _transform_candidates(
        candidates: pd.DataFrame,
        candidate_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        space: DesignSpace,
        ) -> pd.DataFrame:
    """Map optimizer genotypes onto the canonical feasible search space."""

    result = candidates.copy()
    if candidate_transform is not None:
        result = candidate_transform(result)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("candidate_transform must return a pandas DataFrame")
    if len(result) != len(candidates):
        raise ValueError("candidate_transform must preserve the number of rows")
    missing = [name for name in space.para_names if name not in result.columns]
    if missing:
        raise ValueError(
            "candidate_transform omitted design-space columns: "
            + ", ".join(missing)
        )
    return result.loc[:, space.para_names].reset_index(drop=True)

def space_to_pymoo_vars(
        space,
        temporary_numeric_bounds=None,
        fix_input=None,
        ):
    temporary_numeric_bounds = temporary_numeric_bounds or {}
    fixed_names = set((fix_input or {}).keys())
    vars = {}
    for i, p_name in enumerate(space.para_names):
        p  = space.paras[p_name]
        lb = space.opt_lb[i].item()
        ub = space.opt_ub[i].item()
        if p.is_numeric:
            # Equality conditions use fix_input so pymoo never receives a
            # degenerate [x, x] mutation interval. Genuine numeric ranges must
            # be part of the evolutionary domain, not sparse post-hoc rejection.
            if p_name in temporary_numeric_bounds and p_name not in fixed_names:
                lb, ub = map(float, temporary_numeric_bounds[p_name])
                if lb > ub:
                    raise ValueError(
                        f"temporary numeric bound is empty: {p_name}"
                    )
            if not p.is_discrete_after_transform:
                vars[p_name] = Real(bounds = (lb, ub))
            else:
                vars[p_name] = Integer(bounds = (lb, ub))
        elif p.is_categorical:
            vars[p_name] = Choice(options = list(range(int(lb), int(ub + 1))))
        else:
            raise ValueError('Un recognoized parameter type {p_name}')
    return vars

def get_init_pop(
        space,
        pop,
        initial_suggest : pd.DataFrame = None,
        sobol_init = False,
        temporary_numeric_bounds=None,
        fix_input=None,
        candidate_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        ) -> Population:
    temporary_numeric_bounds = temporary_numeric_bounds or {}
    fixed_names = set((fix_input or {}).keys())
    if hasattr(space, "conditional_space"):
        eng = SobolEngine(space.num_paras, scramble=True)
        init_pop = space.conditional_space.generate(
            eng.draw(pop).numpy(), fixed=fix_input, bounds=temporary_numeric_bounds,
        )
    elif sobol_init and not temporary_numeric_bounds:
        init_pop = space.sample(pop)
    else:
        eng        = SobolEngine(space.num_paras, scramble = True)
        sobol_samp = eng.draw(pop)
        opt_lb     = space.opt_lb.clone()
        opt_ub     = space.opt_ub.clone()
        numeric_indices = {
            name: index for index, name in enumerate(space.numeric_names)
        }
        for name, bounds in temporary_numeric_bounds.items():
            if name in fixed_names:
                continue
            if name not in numeric_indices:
                raise ValueError(
                    f"temporary numeric bound references non-numeric parameter: {name}"
                )
            lower, upper = map(float, bounds)
            if lower > upper:
                raise ValueError(f"temporary numeric bound is empty: {name}")
            index = numeric_indices[name]
            opt_lb[index] = lower
            opt_ub[index] = upper
        sobol_samp = sobol_samp * (opt_ub - opt_lb) + opt_lb
        x          = sobol_samp[:, :space.num_numeric]
        xe         = sobol_samp[:, space.num_numeric:].round().long()
        for i, n in enumerate(space.numeric_names):
            if space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        init_pop = space.inverse_transform(x, xe)
    if fix_input is not None:
        for k, v in fix_input.items():
            init_pop[k] = v
    if initial_suggest is not None:
        init_pop = pd.concat([initial_suggest, init_pop], axis = 0).head(pop)
    init_pop = _transform_candidates(init_pop, candidate_transform, space)
    x, xe    = space.transform(init_pop)
    init_pop = np.hstack([x.numpy(), xe.numpy().astype(float)])
    pop_lst  = []
    for item in init_pop:
        pop_item = {}
        for i, name in enumerate(space.para_names):
            if name in space.enum_names:
                pop_item[name] = int(item[i])
            else:
                pop_item[name] = item[i]
        pop_lst.append(pop_item)
    pop = Population.new(X = pop_lst)
    return pop


def population_to_frame(space, population, fix_input=None) -> pd.DataFrame:
    """Decode a pymoo population without losing contextual fixed values."""

    rows = [item.X for item in population]
    if not rows:
        return pd.DataFrame(columns=space.para_names)
    encoded = pd.DataFrame(rows)[space.para_names].values.astype(float)
    x = torch.from_numpy(encoded[:, :space.num_numeric])
    xe = torch.from_numpy(encoded[:, space.num_numeric:])
    frame = space.inverse_transform(x, xe)
    if fix_input is not None:
        for key, value in fix_input.items():
            frame[key] = value
    return frame

class BOProblem(Problem):
    def __init__(self,
            acq   : Acquisition,
            space : DesignSpace, 
            fix   : dict = None,
            temporary_candidate_constraint = None,
            temporary_numeric_bounds = None,
            candidate_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            ):
        self.acq   = acq
        self.space = space
        self.fix   = fix # NOTE: use self.fix to enable contextual BO
        self.temporary_candidate_constraint = temporary_candidate_constraint
        self.candidate_transform = candidate_transform
        vars       = space_to_pymoo_vars(
            self.space,
            temporary_numeric_bounds=temporary_numeric_bounds,
            fix_input=fix,
        )
        super().__init__(
            vars=vars,
            n_obj=acq.num_obj,
            n_constr=acq.num_constr + (
                1 if temporary_candidate_constraint is not None else 0
            ),
        )

    def _evaluate(self, para : np.ndarray, out : dict, *args, **kwargs):
        ## TODO: Use the repair operator to handle fix_input
        num_x = para.shape[0]
        x  = []
        xe = []
        for n in self.space.numeric_names:
            func = np.vectorize(lambda x : x[n])
            x.append(func(para))
        for n in self.space.enum_names:
            func = np.vectorize(lambda x : x[n])
            xe.append(func(para))
        x  = torch.FloatTensor(x).t().reshape(num_x, -1)
        xe = torch.LongTensor(xe).t().reshape(num_x, -1)
        if self.fix is not None or self.candidate_transform is not None:
            df_x = self.space.inverse_transform(x, xe)
            if self.fix is not None:
                for k, v in self.fix.items():
                    df_x[k] = v
            df_x = _transform_candidates(
                df_x, self.candidate_transform, self.space
            )
            x, xe = self.space.transform(df_x)
        with torch.no_grad():
            acq_v = self.acq(x, xe)
        out['F'] = acq_v[:, :self.acq.num_obj].numpy()
        constraints = acq_v[:, self.acq.num_obj:].numpy()
        if self.temporary_candidate_constraint is not None:
            df_x = self.space.inverse_transform(x, xe)
            df_x = _transform_candidates(
                df_x, self.candidate_transform, self.space
            )
            feasible = np.asarray(
                self.temporary_candidate_constraint(df_x), dtype=bool
            ).reshape(-1, 1)
            temporary_g = np.where(feasible, -1.0, 1.0)
            constraints = np.hstack([constraints, temporary_g])
        out['G'] = constraints

class ConditionalMating(MixedVariableMating):
    """Keep NSGA-II selection and ordinary mixed variation for independent
    genes, then generate dependent genes under the offspring's chosen parents.
    Canonical physical genotypes are stored before duplicate elimination.
    """
    def __init__(self, space, bounds=None, fixed=None, **kwargs):
        super().__init__(**kwargs)
        self.space, self.bounds, self.fixed = space, bounds, fixed

    def _do(self, problem, pop, n_offsprings, parents=False, random_state=None, **kwargs):
        offspring = super()._do(problem, pop, n_offsprings, parents=parents,
                                random_state=random_state, **kwargs)
        proposal = population_to_frame(self.space, offspring, fix_input=self.fixed)
        rng = random_state if random_state is not None else np.random
        frame = self.space.conditional_space.generate(
            rng.random((len(offspring), self.space.num_paras)),
            fixed=self.fixed, bounds=self.bounds, proposals=proposal,
        )
        xc, xe = self.space.transform(frame)
        encoded = np.hstack([xc.numpy(), xe.numpy()])
        for i, individual in enumerate(offspring):
            individual.X = {name: (int(encoded[i, j]) if name in self.space.enum_names
                                   else float(encoded[i, j]))
                            for j, name in enumerate(self.space.para_names)}
        return offspring


class EvolutionOpt:
    def __init__(self,
            design_space : DesignSpace,
            acq          : Acquisition,
            es           : str = None, 
            **conf):
        self.space      = design_space
        self.es         = es 
        self.acq        = acq
        self.pop        = conf.get('pop', 100)
        self.iter       = conf.get('iters',500)
        self.verbose    = conf.get('verbose', False)
        self.repair     = conf.get('repair', None)
        self.sobol_init = conf.get('sobol_init', True)
        self.temporary_candidate_constraint = conf.get(
            'temporary_candidate_constraint', None
        )
        self.temporary_numeric_bounds = conf.get(
            'temporary_numeric_bounds', None
        )
        self.candidate_transform = conf.get('candidate_transform', None)
        self.last_generation = {}
        assert(self.acq.num_obj > 0)

        if self.es is None:
            self.es = 'nsga2' if self.acq.num_obj > 1 else 'ga'


    def optimize(self, initial_suggest : pd.DataFrame = None, fix_input : dict = None, return_pop = False) -> pd.DataFrame:
        lb        = self.space.opt_lb.numpy()
        ub        = self.space.opt_ub.numpy()
        prob      = BOProblem(
            self.acq,
            self.space,
            fix_input,
            self.temporary_candidate_constraint,
            self.temporary_numeric_bounds,
            self.candidate_transform,
        )
        init_pop  = get_init_pop(
            self.space,
            self.pop,
            initial_suggest,
            self.sobol_init,
            temporary_numeric_bounds=self.temporary_numeric_bounds,
            fix_input=fix_input,
            candidate_transform=self.candidate_transform,
        )
        # Preserve a guided seed pool. If pymoo cannot expose a feasible Pareto
        # result for a valid narrow domain, GeneralBO can acquisition-rank this
        # pool instead of silently returning one arbitrary Sobol point.
        guided_seed_frame = population_to_frame(
            self.space, init_pop, fix_input=fix_input
        )
        guided_seed_frame = _transform_candidates(
            guided_seed_frame, self.candidate_transform, self.space
        )
        mating = (ConditionalMating(self.space, bounds=self.temporary_numeric_bounds,
                                    fixed=fix_input,
                                    eliminate_duplicates=MixedVariableDuplicateElimination())
                  if hasattr(self.space, "conditional_space") else
                  MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()))
        if self.acq.num_obj == 1:
            algo = MixedVariableGA(pop_size = self.pop, repair = self.repair, sampling = init_pop, mating=mating)
        else:
            algo = NSGA2(pop_size = self.pop, 
                    sampling = init_pop, 
                    mating   = mating, 
                    eliminate_duplicates = MixedVariableDuplicateElimination()
                    )
        res = minimize(prob, algo, ('n_gen', self.iter), verbose = self.verbose)
        if res.X is not None and not return_pop:
            x = res.X
            if isinstance(x, dict):
                x = [x]
            if isinstance(x, np.ndarray):
                x = x.tolist()
            opt_x = pd.DataFrame(x)[self.space.para_names].values.astype(float)
        else:
            opt_x = pd.DataFrame([p.X for p in res.pop])[self.space.para_names].values.astype(float)
            if self.acq.num_obj == 1 and not return_pop:
                opt_x = opt_x[[np.random.choice(opt_x.shape[0])]]
        
        self.res  = res
        opt_xcont = torch.from_numpy(opt_x[:, :self.space.num_numeric])
        opt_xenum = torch.from_numpy(opt_x[:, self.space.num_numeric:])
        df_opt    = self.space.inverse_transform(opt_xcont, opt_xenum)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_opt[k] = v
        df_opt = _transform_candidates(
            df_opt, self.candidate_transform, self.space
        ).drop_duplicates(ignore_index=True)
        if self.temporary_candidate_constraint is not None:
            feasible = np.asarray(
                self.temporary_candidate_constraint(df_opt), dtype=bool
            ).reshape(-1)
            df_opt = df_opt.iloc[np.flatnonzero(feasible)].reset_index(drop=True)
        evolution_candidate_count = int(len(df_opt))
        seed_feasible_count = 0
        fallback_used = False
        if (
            df_opt.empty
            and self.temporary_candidate_constraint is not None
            and not guided_seed_frame.empty
        ):
            seed_feasible = np.asarray(
                self.temporary_candidate_constraint(guided_seed_frame),
                dtype=bool,
            ).reshape(-1)
            seed_frame = guided_seed_frame.iloc[
                np.flatnonzero(seed_feasible)
            ].reset_index(drop=True)
            seed_feasible_count = int(len(seed_frame))
            if seed_feasible_count:
                df_opt = seed_frame
                fallback_used = True
        self.last_generation = {
            "conditional_generation": hasattr(self.space, "conditional_space"),
            "evolution_candidate_count": evolution_candidate_count,
            "guided_seed_feasible_count": seed_feasible_count,
            "guided_seed_pool_used": fallback_used,
            "returned_candidate_count": int(len(df_opt)),
            "temporary_numeric_bounds_applied": bool(
                self.temporary_numeric_bounds
            ),
            "fixed_inputs_applied_during_evolution": bool(fix_input),
        }
        return df_opt
