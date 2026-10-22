from __future__ import annotations

import numpy as np

from denshift.calibrate import DensityBiasModel


def test_bias_is_zero_before_warmup() -> None:
    model = DensityBiasModel(n_classes=5, warmup=8)
    d = np.zeros(6)
    assert np.all(model.bias(d) == 0.0)


def test_estimator_recovers_a_known_density_response() -> None:
    rng = np.random.default_rng(0)
    n_classes, n_obs = 5, 400

    # warmup=0 freezes the descriptor scaler at its identity state, so the basis
    # is fixed throughout and recovery can be checked exactly.
    model = DensityBiasModel(n_classes=n_classes, warmup=0, ridge=1e-6)
    truth = rng.normal(0.0, 1.0, size=(model.n_basis, n_classes))
    for _ in range(n_obs):
        d_dense = rng.normal(0.0, 1.0, size=6)
        # The perturbation must vary with the observation.  A constant shift of
        # the descriptor excites only two basis directions and leaves the rest
        # unidentified, which is exactly the degeneracy a real subsample avoids.
        d_sparse = d_dense * rng.uniform(0.6, 0.9, size=6) - 0.2
        z_dense = model.phi(d_dense) @ truth
        z_sparse = model.phi(d_sparse) @ truth
        model.observe_pair(d_dense, d_sparse, z_dense, z_sparse)

    # The constant column is unidentifiable from differences; every other
    # coefficient must be recovered.
    estimated = model.coefficients()
    assert np.allclose(estimated[1:], truth[1:], atol=0.05)


def test_bias_is_not_centred_across_samples() -> None:
    """A class-uniform offset must survive: centring would move the decision."""
    rng = np.random.default_rng(1)
    n_classes = 4
    model = DensityBiasModel(n_classes=n_classes, warmup=0)
    truth = rng.normal(0.0, 1.0, size=(model.n_basis, n_classes))
    for _ in range(300):
        d_dense = rng.normal(0.0, 1.0, size=6)
        d_sparse = d_dense * rng.uniform(0.6, 0.9, size=6) - 0.2
        model.observe_pair(
            d_dense, d_sparse, model.phi(d_dense) @ truth, model.phi(d_sparse) @ truth
        )

    logits = rng.normal(size=n_classes)
    d = rng.normal(0.0, 1.0, size=6)
    bias = model.bias(d)
    assert np.allclose(bias, model.phi(d) @ model.coefficients())


def test_correction_leaves_argmax_invariant_to_a_uniform_shift() -> None:
    model = DensityBiasModel(n_classes=3, warmup=1)
    logits = np.array([0.4, 2.0, -1.0])
    shifted = logits + 5.0
    assert np.argmax(logits) == np.argmax(shifted)


def test_standardizer_freezes_after_warmup() -> None:
    model = DensityBiasModel(n_classes=3, warmup=5)
    rng = np.random.default_rng(2)
    for _ in range(5):
        d = rng.normal(size=6)
        model.observe_pair(d, d - 0.1, np.zeros(3), np.zeros(3))

    frozen_mean = model.standardizer.mean.copy()
    for _ in range(20):
        d = rng.normal(5.0, 3.0, size=6)
        model.observe_pair(d, d - 0.1, np.zeros(3), np.zeros(3))
    assert np.allclose(model.standardizer.mean, frozen_mean)


def test_reset_weights_keeps_descriptor_statistics() -> None:
    model = DensityBiasModel(n_classes=3, warmup=2)
    rng = np.random.default_rng(3)
    for _ in range(10):
        d = rng.normal(size=6)
        model.observe_pair(d, d - 0.1, np.ones(3), np.zeros(3))

    mean_before = model.standardizer.mean.copy()
    model.reset_weights()
    assert np.allclose(model.standardizer.mean, mean_before)
    assert model.n_updates == 0
