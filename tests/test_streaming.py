from __future__ import annotations

import numpy as np

from denshift.streaming import RunningStandardizer, StreamingQuantile


def test_standardizer_matches_batch_moments() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(500, 4))
    s = RunningStandardizer(4)
    for row in x:
        s.update(row)
    assert np.allclose(s.mean, x.mean(axis=0), atol=1e-9)
    assert np.allclose(s.std, x.std(axis=0, ddof=1), rtol=1e-6)


def test_standardizer_transform_is_roughly_standard_normal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(-1.0, 4.0, size=(2000, 3))
    s = RunningStandardizer(3)
    for row in x:
        s.update(row)
    z = np.stack([s.transform(row) for row in x])
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05


def test_standardizer_reset() -> None:
    s = RunningStandardizer(2)
    for row in np.ones((10, 2)):
        s.update(row)
    s.reset()
    assert s.count == 0
    assert np.all(s.mean == 0.0)


def test_quantile_tracks_uniform_distribution() -> None:
    q = StreamingQuantile(capacity=4096, seed=0)
    rng = np.random.default_rng(2)
    for v in rng.uniform(0.0, 1.0, size=5000):
        q.update(float(v))
    for p in (0.1, 0.5, 0.9):
        assert abs(float(q.quantiles([p])[0]) - p) < 0.05


def test_quantile_is_monotone() -> None:
    q = StreamingQuantile(capacity=2048, seed=3)
    for v in np.random.default_rng(4).normal(size=3000):
        q.update(float(v))
    grid = q.quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
    assert all(a <= b for a, b in zip(grid, grid[1:]))
