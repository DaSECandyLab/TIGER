"""三目标最大化问题的 Pareto 与超体积实现。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pareto_mask(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """返回最大化语义下的非支配点 mask；重复点只保留第一项。"""

    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.zeros((0,), dtype=bool)
    if values.ndim != 2:
        raise ValueError("Pareto points must be a two-dimensional array")
    mask = np.ones(values.shape[0], dtype=bool)
    for index, point in enumerate(values):
        if not mask[index]:
            continue
        for other_index, other in enumerate(values):
            if other_index == index:
                continue
            if np.all(other >= point) and (
                np.any(other > point) or other_index < index
            ):
                mask[index] = False
                break
    return mask


def pareto_front(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 3
        return np.empty((0, width), dtype=float)
    return values[pareto_mask(values)]


def _union_area_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """计算从共同左下 reference 出发的二维矩形并集面积。"""

    if points.size == 0:
        return 0.0
    y_edges = sorted(
        {float(reference[0]), *[float(value) for value in points[:, 0]]}
    )
    area = 0.0
    for lower, upper in zip(y_edges, y_edges[1:]):
        if upper <= reference[0]:
            continue
        active = points[points[:, 0] >= upper]
        if active.size == 0:
            continue
        max_z = float(np.max(active[:, 1]))
        if max_z > reference[1]:
            area += (upper - lower) * (max_z - reference[1])
    return float(area)


def hypervolume_3d(
    points: Sequence[Sequence[float]] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> float:
    """精确计算三目标最大化超体积，不依赖现有 BO/HEBO 实现。"""

    values = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if ref.shape != (3,):
        raise ValueError("three-objective hypervolume requires a 3D reference point")
    if values.size == 0:
        return 0.0
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("three-objective hypervolume requires an (n, 3) array")
    eligible = values[np.all(values > ref, axis=1)]
    if eligible.size == 0:
        return 0.0
    eligible = pareto_front(eligible)
    x_edges = sorted({float(ref[0]), *[float(value) for value in eligible[:, 0]]})
    volume = 0.0
    for lower, upper in zip(x_edges, x_edges[1:]):
        if upper <= ref[0]:
            continue
        active = eligible[eligible[:, 0] >= upper]
        if active.size:
            volume += (upper - lower) * _union_area_2d(active[:, 1:], ref[1:])
    return float(volume)


def hypervolume_improvement(
    pareto_points: Sequence[Sequence[float]] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> float:
    front = np.asarray(pareto_points, dtype=float)
    point = np.asarray(candidate, dtype=float).reshape(1, 3)
    base = hypervolume_3d(front, reference)
    combined = point if front.size == 0 else np.vstack([front, point])
    return max(0.0, hypervolume_3d(combined, reference) - base)

