"""Density-Stratified Calibration (DSC).

The estimator fits a *closed-form* nuisance regression in logit space.  It never
touches the model's weights and never backpropagates through the model.

Model of the black-box output
-----------------------------
Write the logits of a frozen model on a cloud ``x`` as

    z(x) = s(x) + B phi(d(x))

where ``s`` is the semantic score we want, ``d`` is a density descriptor that can
be computed from the input alone, and ``phi`` is a small basis expansion.  ``B``
is unknown and is what a model trained on one density regime has silently
memorised.

Free supervision
----------------
Take two views of the *same* cloud at different sampling rates.  The semantic
content is identical, so any change in logits is pure density nuisance:

    z(x_dense) - z(x_sparse) = B (phi(d_dense) - phi(d_sparse))

That is a linear regression with no labels, no source data, and no gradients
through the model.  It is solved by ridge regression on accumulated sufficient
statistics, which makes it streaming and O(1) in memory.

Inference corrects a single view with the learned bias:

    z_hat(x) = z(x) - alpha * B phi(d(x))

``alpha`` is an optional shrinkage factor in ``(0, 1]``.  The basis contains a
constant column whose coefficient the paired design cannot identify, but the
resulting ambiguity adds the same offset to every class and therefore leaves
``argmax`` unchanged.  Centring the correction across samples would instead
subtract a class-dependent vector and measurably hurt; ``phi_mean`` is kept for
diagnostics only.
"""

from __future__ import annotations

import numpy as np

from .streaming import RunningStandardizer, StreamingQuantile

# Index of the descriptor coordinate used for quantile stratification.
DENSITY_COORD = 0


class DensityBiasModel:
    """Streaming closed-form estimator of a density-induced logit bias."""

    def __init__(
        self,
        n_classes: int,
        n_features: int = 6,
        n_buckets: int = 8,
        ridge: float = 0.01,
        forgetting: float = 1.0,
        alpha: float = 1.0,
        warmup: int = 32,
        max_bias_norm: float = 0.0,
        feature_subset: list[int] | None = None,
        use_buckets: bool = False,
    ) -> None:
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.n_buckets = int(n_buckets) if use_buckets else 1
        self.ridge = float(ridge)
        self.max_bias_norm = float(max_bias_norm)
        self.forgetting = float(forgetting)
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.use_buckets = bool(use_buckets)
        self.feature_subset = (
            list(range(n_features)) if feature_subset is None else list(feature_subset)
        )

        n_lin = len(self.feature_subset)
        # [1, z, z^2, bucket one-hot]: a second-order expansion of the density
        # response, piecewise-constant in the density quantile.
        self.n_basis = 1 + 2 * n_lin + (self.n_buckets if self.n_buckets > 1 else 0)
        self.standardizer = RunningStandardizer(self.n_features)
        self.quantile = StreamingQuantile()
        self.A = np.zeros((self.n_basis, self.n_basis), dtype=np.float64)
        self.C = np.zeros((self.n_basis, self.n_classes), dtype=np.float64)
        self.phi_mean = np.zeros(self.n_basis, dtype=np.float64)
        self.phi_mean_count = 0
        self.n_updates = 0
        self._B: np.ndarray | None = None

    # ------------------------------------------------------------------ basis
    def phi(self, descriptor: np.ndarray) -> np.ndarray:
        """Expand a descriptor into the calibration basis."""
        d = np.asarray(descriptor, dtype=np.float64).reshape(self.n_features)
        z = self.standardizer.transform(d)
        sel = z[self.feature_subset]
        parts = [np.ones(1, dtype=np.float64), sel, sel ** 2]
        if self.n_buckets > 1:
            onehot = np.zeros(self.n_buckets, dtype=np.float64)
            onehot[self.quantile.bucket(d[DENSITY_COORD], self.n_buckets)] = 1.0
            parts.append(onehot)
        return np.concatenate(parts)

    # --------------------------------------------------------------- learning
    def observe_pair(
        self,
        d_dense: np.ndarray,
        d_sparse: np.ndarray,
        z_dense: np.ndarray,
        z_sparse: np.ndarray,
        learn: bool = True,
    ) -> None:
        """Accumulate one paired-view observation.

        ``learn=False`` still updates the running statistics, which is what the
        warm-up phase needs: the basis must be well defined before the ridge
        system is worth solving.
        """
        d_dense = np.asarray(d_dense, dtype=np.float64)
        d_sparse = np.asarray(d_sparse, dtype=np.float64)
        # The descriptor scaler is calibrated during warm-up and then frozen.
        # If it keeps moving, the same cloud maps to a different basis vector at
        # different times and the accumulated normal equations no longer
        # describe one regression problem.
        if self.n_updates < self.warmup:
            self.standardizer.update(d_dense)
            self.standardizer.update(d_sparse)
            self.quantile.update(float(d_dense[DENSITY_COORD]))
            self.quantile.update(float(d_sparse[DENSITY_COORD]))

        if not learn or self.n_updates < 0:
            return

        phi_d = self.phi(d_dense)
        phi_s = self.phi(d_sparse)
        # The reference distribution is the *test* distribution, not the pair
        # average: the sparse view is systematically sparser, and using it would
        # bias the correction toward "everything is too sparse".
        self._update_phi_mean(phi_d)

        dphi = phi_d - phi_s
        dz = np.asarray(z_dense, dtype=np.float64).reshape(self.n_classes) - np.asarray(
            z_sparse, dtype=np.float64
        ).reshape(self.n_classes)

        if self.forgetting < 1.0:
            self.A *= self.forgetting
            self.C *= self.forgetting
        self.A += np.outer(dphi, dphi)
        self.C += np.outer(dphi, dz)
        self.n_updates += 1
        self._B = None

    def _update_phi_mean(self, phi: np.ndarray) -> None:
        self.phi_mean_count += 1
        lr = 1.0 / self.phi_mean_count
        self.phi_mean += lr * (phi - self.phi_mean)

    # -------------------------------------------------------------- inference
    @property
    def fitted(self) -> bool:
        return self.n_updates >= self.warmup

    def coefficients(self) -> np.ndarray:
        if self._B is None:
            # Scale the ridge to the accumulated evidence: a fixed absolute
            # penalty is meaningless once the sufficient statistics grow.
            scale = float(np.trace(self.A)) / max(self.n_basis, 1)
            reg = self.ridge * max(scale, 1e-8) * np.eye(self.n_basis)
            self._B = np.linalg.solve(self.A + reg, self.C)
        return self._B

    def predict_delta(self, d_a: np.ndarray, d_b: np.ndarray) -> np.ndarray:
        """Predicted logit difference between two density views."""
        dphi = self.phi(d_a) - self.phi(d_b)
        return dphi @ self.coefficients()

    def z_score(self, d_a: np.ndarray, d_b: np.ndarray, dz: np.ndarray) -> float:
        """Normalised residual: 1.0 means the pair carries no usable signal."""
        dz = np.asarray(dz, dtype=np.float64).reshape(self.n_classes)
        predicted = self.predict_delta(d_a, d_b)
        num = float(np.linalg.norm(dz - predicted))
        den = float(np.linalg.norm(dz)) + 1e-8
        return num / den

    def bias(self, descriptor: np.ndarray) -> np.ndarray:
        """Estimated density-induced logit offset for one point cloud.

        Deliberately *not* mean-centred.  The basis contains a constant column
        whose coefficient the paired design cannot identify, so the correction
        is known only up to a shift that is identical for every class -- and
        such a shift leaves ``argmax`` unchanged.  Centring across samples would
        instead subtract a *class-dependent* vector, which does move the
        decision and measurably hurts.
        """
        if not self.fitted:
            return np.zeros(self.n_classes, dtype=np.float64)
        raw = self.phi(descriptor) @ self.coefficients()
        norm = float(np.linalg.norm(raw))
        if self.max_bias_norm > 0 and norm > self.max_bias_norm:
            raw = raw * (self.max_bias_norm / norm)
        return raw

    def correct(self, logits: np.ndarray, descriptor: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).reshape(self.n_classes)
        if not self.fitted or self.alpha == 0.0:
            return logits.copy()
        return logits - self.alpha * self.bias(descriptor)

    # ----------------------------------------------------------------- state
    def reset_weights(self) -> None:
        """Forget the learned bias but keep the density statistics."""
        self.A[:] = 0.0
        self.C[:] = 0.0
        self.phi_mean[:] = 0.0
        self.phi_mean_count = 0
        self.n_updates = 0
        self._B = None

    def reset(self) -> None:
        self.reset_weights()
        self.standardizer.reset()
        self.quantile.reset()

    def state(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "fitted": bool(self.fitted),
            "bias_norm": float(np.linalg.norm(self.coefficients())) if self.fitted else 0.0,
            "forgetting": self.forgetting,
            "n_buckets": self.n_buckets,
        }
