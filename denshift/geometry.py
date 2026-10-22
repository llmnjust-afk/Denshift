"""Density descriptors computed from raw points only.

The whole point of Denshift is that a black-box model hides its weights but
cannot hide its input.  A point cloud exposes a shift axis that images do not:
sampling density.  This module turns a raw set of 3D points into a small,
model-independent descriptor of that axis.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Number of features produced by :func:`density_descriptor`.
N_FEATURES = 6

FEATURE_NAMES = (
    "log_median_knn",
    "log_mean_knn",
    "knn_cv",
    "log_rms_radius",
    "anisotropy",
    "linearity",
)


def _safe_log(value: float, eps: float = 1e-12) -> float:
    return float(np.log(max(value, eps)))


def density_descriptor(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Return a 6-dimensional density descriptor for one point cloud.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)``.  ``N`` must be at least ``k + 1``.
    k:
        Neighbourhood size used for the local density statistics.

    Returns
    -------
    np.ndarray
        Descriptor of shape ``(6,)``.  The first three coordinates measure
        local sampling density, the last three measure global scale and shape.
        Scale and shape are included because a density shift rarely arrives
        alone: sparse clouds also change the apparent anisotropy of a scan.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points of shape (N, 3), got {points.shape}")
    if points.shape[0] < max(k + 1, 4):
        raise ValueError(f"need at least {max(k + 1, 4)} points, got {points.shape[0]}")

    k = int(min(k, points.shape[0] - 1))
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    # Column 0 is the point itself.
    knn = dists[:, 1:]

    mean_knn = float(knn.mean())
    median_knn = float(np.median(knn))
    knn_cv = float(knn.std() / max(mean_knn, 1e-12))

    centred = points - points.mean(axis=0, keepdims=True)
    rms_radius = float(np.sqrt((centred ** 2).sum(axis=1).mean()))

    cov = np.cov(centred, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        anisotropy, linearity = 0.0, 0.0
    else:
        anisotropy = float((eig[0] - eig[2]) / total)
        linearity = float((eig[0] - eig[1]) / total)

    return np.array(
        [
            _safe_log(median_knn),
            _safe_log(mean_knn),
            knn_cv,
            _safe_log(rms_radius),
            anisotropy,
            linearity,
        ],
        dtype=np.float64,
    )


def subsample(points: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return a density-shifted view of the same object.

    Random subsampling preserves the underlying surface while changing the
    sampling rate, which is exactly the nuisance direction Denshift estimates.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    points = np.asarray(points)
    n = points.shape[0]
    keep = max(4, int(round(n * ratio)))
    if keep >= n:
        return points.copy()
    idx = rng.choice(n, size=keep, replace=False)
    return points[np.sort(idx)]


def stratified_subsample(
    points: np.ndarray, ratio: float, bins: int, rng: np.random.Generator
) -> np.ndarray:
    """Subsample uniformly *within* radial shells.

    Plain random subsampling shrinks the outer shell faster than the inner one
    for non-uniform scans.  Stratifying by radius keeps the density change
    closer to a pure resampling of the same surface.
    """
    points = np.asarray(points)
    n = points.shape[0]
    keep = max(bins * 4, int(round(n * ratio)))
    if keep >= n:
        return points.copy()

    centred = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centred, axis=1)
    edges = np.quantile(radius, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1e-9
    shell = np.clip(np.searchsorted(edges, radius, side="right") - 1, 0, bins - 1)

    per_shell = max(1, keep // bins)
    chosen: list[np.ndarray] = []
    for b in range(bins):
        idx = np.flatnonzero(shell == b)
        if idx.size == 0:
            continue
        take = min(per_shell, idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    if not chosen:
        return points.copy()
    idx = np.sort(np.concatenate(chosen))
    return points[idx]
