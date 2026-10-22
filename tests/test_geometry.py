from __future__ import annotations

import numpy as np

from denshift.geometry import N_FEATURES, density_descriptor, stratified_subsample, subsample


def make_cloud(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, 3))


def test_descriptor_shape_and_finiteness() -> None:
    d = density_descriptor(make_cloud())
    assert d.shape == (N_FEATURES,)
    assert np.all(np.isfinite(d))


def test_descriptor_is_invariant_to_translation_and_rigid_scale() -> None:
    pts = make_cloud()
    base = density_descriptor(pts)
    moved = density_descriptor(pts * 3.0 + np.array([10.0, -4.0, 2.0]))
    # The density coordinates are logarithmic, so a rigid rescale shifts them by
    # log(3) rather than by a multiplicative factor.
    assert np.isclose(moved[0] - base[0], np.log(3.0), atol=0.05)
    assert np.isclose(moved[1] - base[1], np.log(3.0), atol=0.05)


def test_descriptor_responds_to_density() -> None:
    pts = make_cloud(600)
    dense = density_descriptor(pts)
    sparse = density_descriptor(pts[::2])
    assert sparse[0] > dense[0], "fewer points must register as lower density"


def test_subsample_preserves_bounds() -> None:
    pts = make_cloud(500)
    q = subsample(pts, 0.4, np.random.default_rng(0))
    assert q.shape == (200, 3)


def test_stratified_subsample_is_smaller_and_reproducible() -> None:
    pts = make_cloud(400)
    a = stratified_subsample(pts, 0.5, 4, np.random.default_rng(7))
    b = stratified_subsample(pts, 0.5, 4, np.random.default_rng(7))
    assert a.shape == (200, 3)
    assert np.array_equal(a, b)
