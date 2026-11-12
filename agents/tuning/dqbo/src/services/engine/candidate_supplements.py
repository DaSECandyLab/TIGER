"""Small, source-labelled additions to the native LCB candidate pool."""
import numpy as np
import pandas as pd
import torch


def supplement_candidates(bo, base, acq, *, per_source, fixed=None, bounds=None,
                          candidate_constraint=None, base_sources=None):
    names = ("global_sobol", "posterior_uncertainty")
    sources = list(base_sources) if base_sources is not None else ["lcb_evolution"] * len(base)
    # At least 75% of the final pool remains from the native search.
    budget = min(len(names) * per_source, len(base) // 3)
    receipt = {"base_count": len(base), "supplement_budget": budget,
               "added_by_source": dict.fromkeys(names, 0)}
    if not budget or not hasattr(bo.space, "conditional_space"):
        return base, sources, receipt
    fixed, bounds = fixed or {}, bounds or {}
    seen = {bo._candidate_key(row) for _, row in base.iterrows()}

    def legal(frame):
        frame = bo._filter_new_candidates(frame)
        if frame.empty:
            return frame
        if candidate_constraint is not None:
            frame = frame.loc[np.asarray(candidate_constraint(frame), dtype=bool)].reset_index(drop=True)
        if not frame.empty and acq.num_constr:
            with torch.no_grad():
                values = acq(*bo.space.transform(frame)).numpy()
            frame = frame.loc[np.all(values[:, acq.num_obj:] <= 0, axis=1)].reset_index(drop=True)
        return frame

    count = max(24, per_source * 12)
    global_pool = legal(bo.quasi_sample(count, fixed, candidate_constraint, bounds))
    uncertain = global_pool.copy()
    if len(uncertain):
        with torch.no_grad():
            _, variance = bo.model.predict(*bo.space.transform(uncertain))
        scale = np.maximum(np.std(bo.y[:, :bo.num_obj], axis=0), 1e-8)
        score = np.sum(variance.numpy()[:, :bo.num_obj] / scale[None, :]**2, axis=1)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
        uncertain = uncertain.iloc[order]
    pools = dict(zip(names, (global_pool, uncertain)))
    positions = dict.fromkeys(names, 0)
    added = []
    # Round robin gives each source a chance when the LCB pool is small.
    for _ in range(per_source):
        for name in names:
            pool = pools[name]
            while positions[name] < len(pool) and len(added) < budget:
                row = pool.iloc[positions[name]]
                positions[name] += 1
                key = bo._candidate_key(row)
                if key in seen:
                    continue
                seen.add(key)
                added.append(row.to_dict())
                sources.append(name)
                receipt["added_by_source"][name] += 1
                break
    result = pd.concat([base, pd.DataFrame(added, columns=base.columns)], ignore_index=True)
    receipt["final_count"] = len(result)
    return result, sources, receipt
