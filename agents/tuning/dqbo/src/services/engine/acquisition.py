"""性能与质量后验联合 Monte Carlo constrained EHVI。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import FourSurrogateModels, PerformanceSurrogateModels, RunFeasibilityModel
from .pareto import hypervolume_improvement


@dataclass(frozen=True)
class AcquisitionScores:
    ehvi: np.ndarray
    constrained_ehvi: np.ndarray
    p_feasible: np.ndarray
    quality_feasible_mc_samples: np.ndarray
    p_run: np.ndarray
    score: np.ndarray


def performance_mc_ehvi_scores(
    *, surrogate_models: PerformanceSurrogateModels, run_model: RunFeasibilityModel,
    encoded_candidates: np.ndarray, feasible_pareto: np.ndarray,
    reference_point: np.ndarray, monte_carlo_samples: int, rng: np.random.Generator,
) -> AcquisitionScores:
    """EHVI without fitting or sampling a quality regression for classifier gating."""
    mean, std = surrogate_models.predict(encoded_candidates)
    samples = rng.normal(mean[None, :, :], std[None, :, :],
                         size=(monte_carlo_samples, len(encoded_candidates), 3))
    scores = np.asarray([
        np.mean([hypervolume_improvement(feasible_pareto, sample, reference_point)
                 for sample in samples[:, i]])
        for i in range(len(encoded_candidates))
    ])
    return AcquisitionScores(
        ehvi=scores, constrained_ehvi=np.zeros_like(scores),
        p_feasible=np.full_like(scores, np.nan), quality_feasible_mc_samples=np.zeros(len(scores), dtype=int),
        p_run=run_model.predict_probability(encoded_candidates), score=scores,
    )


def joint_mc_constrained_ehvi_scores(
    *,
    surrogate_models: FourSurrogateModels,
    run_model: RunFeasibilityModel,
    encoded_candidates: np.ndarray,
    feasible_pareto: Sequence[Sequence[float]] | np.ndarray,
    reference_point: Sequence[float] | np.ndarray,
    tau_q: float,
    monte_carlo_samples: int,
    rng: np.random.Generator,
    quality_probability_affects_acquisition: bool = True,
) -> AcquisitionScores:
    """计算性能 EHVI，并可选让质量后验样本参与最终 acquisition。"""

    if monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive")
    encoded = np.asarray(encoded_candidates, dtype=float)
    objective_mean, objective_std, quality_mean, quality_std = (
        surrogate_models.predict(encoded)
    )
    p_feasible = surrogate_models.feasible_probability(encoded, tau_q)
    p_run = run_model.predict_probability(encoded)
    objective_samples = rng.normal(
        loc=objective_mean[None, :, :],
        scale=objective_std[None, :, :],
        size=(monte_carlo_samples, encoded.shape[0], 3),
    )
    quality_samples = rng.normal(
        loc=quality_mean[None, :],
        scale=quality_std[None, :],
        size=(monte_carlo_samples, encoded.shape[0]),
    )
    front = np.asarray(feasible_pareto, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    ehvi = np.zeros(encoded.shape[0], dtype=float)
    constrained_ehvi = np.zeros(encoded.shape[0], dtype=float)
    for candidate_index in range(encoded.shape[0]):
        total = 0.0
        constrained_total = 0.0
        for sample_index in range(monte_carlo_samples):
            improvement = hypervolume_improvement(
                front,
                objective_samples[sample_index, candidate_index],
                reference,
            )
            total += improvement
            if quality_samples[sample_index, candidate_index] >= tau_q:
                constrained_total += improvement
        ehvi[candidate_index] = total / monte_carlo_samples
        constrained_ehvi[candidate_index] = (
            constrained_total / monte_carlo_samples
        )
    return AcquisitionScores(
        ehvi=ehvi,
        constrained_ehvi=constrained_ehvi,
        p_feasible=np.asarray(p_feasible, dtype=float),
        quality_feasible_mc_samples=np.sum(
            quality_samples >= tau_q, axis=0, dtype=int
        ),
        p_run=np.asarray(p_run, dtype=float),
        score=(
            constrained_ehvi
            if quality_probability_affects_acquisition
            else ehvi
        ),
    )
