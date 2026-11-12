"""Candidate Optimizer 的四个独立代理模型和运行隐藏约束模型。"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

SURROGATE_NAMES = (
    "qps",
    "mean_latency",
    "p95_latency",
    "quality",
)


def _gp(random_seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_seed,
    )


class FourSurrogateModels:
    """严格以四个独立 GPR 分别建模三个性能目标和一个质量约束。"""

    def __init__(self, *, random_seed: int = 0) -> None:
        template = _gp(random_seed)
        self.models = {name: clone(template) for name in SURROGATE_NAMES}
        self.fitted = False

    def fit(
        self,
        x: np.ndarray,
        objectives: np.ndarray,
        quality: np.ndarray,
    ) -> "FourSurrogateModels":
        x = np.asarray(x, dtype=float)
        objectives = np.asarray(objectives, dtype=float)
        quality = np.asarray(quality, dtype=float).reshape(-1)
        if x.ndim != 2 or objectives.shape != (x.shape[0], 3):
            raise ValueError("surrogate training needs X=(n,d), objectives=(n,3)")
        if quality.shape != (x.shape[0],):
            raise ValueError("quality training vector must have n rows")
        targets = {
            "qps": objectives[:, 0],
            # Objective storage uses maximization coordinates, while the two
            # latency surrogates model the physical positive latency metrics.
            "mean_latency": -objectives[:, 1],
            "p95_latency": -objectives[:, 2],
            "quality": quality,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for name, target in targets.items():
                self.models[name].fit(x, target)
        self.fitted = True
        return self

    def predict(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("surrogate models have not been fitted")
        x = np.asarray(x, dtype=float)
        qps_mean, qps_std = self.models["qps"].predict(x, return_std=True)
        mean_latency, mean_latency_std = self.models["mean_latency"].predict(
            x, return_std=True
        )
        p95_latency, p95_latency_std = self.models["p95_latency"].predict(
            x, return_std=True
        )
        quality_mean, quality_std = self.models["quality"].predict(x, return_std=True)
        return (
            np.column_stack([qps_mean, -mean_latency, -p95_latency]),
            np.column_stack(
                [
                    np.maximum(qps_std, 1e-12),
                    np.maximum(mean_latency_std, 1e-12),
                    np.maximum(p95_latency_std, 1e-12),
                ]
            ),
            np.asarray(quality_mean, dtype=float),
            np.maximum(np.asarray(quality_std, dtype=float), 1e-12),
        )

    def feasible_probability(self, x: np.ndarray, tau_q: float) -> np.ndarray:
        _, _, quality_mean, quality_std = self.predict(x)
        return norm.cdf((quality_mean - float(tau_q)) / quality_std)


class PerformanceSurrogateModels:
    """Three independent performance-only GPs used inside a repair fibre.

    RepairBO deliberately does not fit or consult a quality surrogate. Quality
    remains a measured hard gate; continuous margins are evidence and audit
    diagnostics only.
    """

    def __init__(self, *, random_seed: int = 0) -> None:
        template = _gp(random_seed)
        self.models = {
            name: clone(template) for name in ("qps", "mean_latency", "p95_latency")
        }
        self.fitted = False

    def fit(
        self, x: np.ndarray, objectives: np.ndarray
    ) -> "PerformanceSurrogateModels":
        x = np.asarray(x, dtype=float)
        objectives = np.asarray(objectives, dtype=float)
        if x.ndim != 2 or objectives.shape != (x.shape[0], 3):
            raise ValueError("performance surrogate needs X=(n,d), objectives=(n,3)")
        targets = {
            "qps": objectives[:, 0],
            "mean_latency": -objectives[:, 1],
            "p95_latency": -objectives[:, 2],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for name, target in targets.items():
                self.models[name].fit(x, target)
        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("performance surrogate models have not been fitted")
        x = np.asarray(x, dtype=float)
        qps_mean, qps_std = self.models["qps"].predict(x, return_std=True)
        mean_latency, mean_latency_std = self.models["mean_latency"].predict(
            x, return_std=True
        )
        p95_latency, p95_latency_std = self.models["p95_latency"].predict(
            x, return_std=True
        )
        return (
            np.column_stack([qps_mean, -mean_latency, -p95_latency]),
            np.column_stack(
                [
                    np.maximum(qps_std, 1e-12),
                    np.maximum(mean_latency_std, 1e-12),
                    np.maximum(p95_latency_std, 1e-12),
                ]
            ),
        )


@dataclass
class RunFeasibilityModel:
    """OOM、启动失败等 hidden constraint 的 P_run 模型。"""

    random_seed: int = 0
    n_estimators: int = 200

    def __post_init__(self) -> None:
        self.classifier: RandomForestClassifier | None = None
        self.constant_probability = 1.0
        self.fitted_rows = 0

    def fit(
        self, x: np.ndarray, ran_to_completion: Sequence[bool]
    ) -> "RunFeasibilityModel":
        x = np.asarray(x, dtype=float)
        labels = np.asarray(ran_to_completion, dtype=int)
        if x.ndim != 2 or x.shape[0] != labels.shape[0]:
            raise ValueError("hidden-constraint X/label lengths differ")
        self.fitted_rows = int(labels.size)
        if labels.size == 0:
            self.classifier = None
            self.constant_probability = 1.0
            return self
        # Beta(1,1) 平滑避免早期把尚未探索的区域概率压成严格 0/1。
        self.constant_probability = float((labels.sum() + 1) / (labels.size + 2))
        if labels.size < 4 or np.unique(labels).size < 2:
            self.classifier = None
            return self
        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=max(1, int(math.sqrt(labels.size) // 2)),
            random_state=self.random_seed,
            class_weight="balanced_subsample",
        )
        classifier.fit(x, labels)
        self.classifier = classifier
        return self

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.classifier is None:
            return np.full(x.shape[0], self.constant_probability, dtype=float)
        probabilities = self.classifier.predict_proba(x)
        classes = list(self.classifier.classes_)
        if 1 not in classes:
            return np.zeros(x.shape[0], dtype=float)
        return np.asarray(probabilities[:, classes.index(1)], dtype=float)
