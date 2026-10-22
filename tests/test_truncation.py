from __future__ import annotations

import numpy as np

from denshift.calibrate import DensityBiasModel


def _feed(model: DensityBiasModel, n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_classes = model.n_classes
    truth = rng.normal(size=(model.n_basis, n_classes))
    for _ in range(n):
        d_dense = rng.normal(size=6)
        d_sparse = d_dense * rng.uniform(0.6, 0.9, size=6) - 0.2
        model.observe_pair(
            d_dense, d_sparse, model.phi(d_dense) @ truth, model.phi(d_sparse) @ truth
        )


def test_truncation_limits_the_rank_of_the_coefficient() -> None:
    full = DensityBiasModel(n_classes=4, warmup=0, ridge=1e-6, use_buckets=False)
    _feed(full)
    assert np.linalg.matrix_rank(full.coefficients()) > 2

    cut = DensityBiasModel(
        n_classes=4, warmup=0, ridge=1e-6, use_buckets=False, max_directions=2
    )
    _feed(cut)
    assert np.linalg.matrix_rank(cut.coefficients()) <= 2 + 1e-9


def test_truncation_preserves_the_leading_direction() -> None:
    cut = DensityBiasModel(
        n_classes=4, warmup=0, ridge=1e-6, use_buckets=False, max_directions=2
    )
    _feed(cut)

    values, vectors = np.linalg.eigh(cut.A)
    top = vectors[:, np.argsort(values)[::-1][:2]]
    coef = cut.coefficients()
    residual = coef - top @ (top.T @ coef)
    assert np.linalg.norm(residual) < 1e-8 * max(np.linalg.norm(coef), 1.0)


def test_truncation_is_a_no_op_when_k_covers_the_basis() -> None:
    full = DensityBiasModel(n_classes=3, warmup=0, ridge=1e-6, use_buckets=False)
    _feed(full, seed=1)
    wide = DensityBiasModel(
        n_classes=3,
        warmup=0,
        ridge=1e-6,
        use_buckets=False,
        max_directions=full.n_basis,
    )
    _feed(wide, seed=1)
    assert np.allclose(full.coefficients(), wide.coefficients(), atol=1e-6)
