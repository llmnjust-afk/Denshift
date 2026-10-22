"""Adapter API and the black-box baselines Denshift is compared against.

Every adapter sees exactly what a deployed client sees: the raw point cloud and
the model's logits.  No adapter may call ``backward`` on the model, read its
parameters, or touch source data.  ``blackbox.py`` enforces that contract.

The baselines form a deliberate ladder so the ablation in the paper is honest:

    SourceOnly      no adaptation at all
    EntropyBias     black-box analogue of TENT (output-space entropy descent)
    GlobalBias      learned bias without any density conditioning
    DSCNoBucket     density features, but a single global regressor
    DSCNoDrift      full basis, no forgetting and no change point
    DSC             full method
"""

from __future__ import annotations

import numpy as np

from .calibrate import DensityBiasModel
from .changepoint import ShockDetector
from .geometry import density_descriptor, stratified_subsample


class Adapter:
    """Base class.  ``adapt`` must return logits of shape ``(n_classes,)``."""

    name = "adapter"

    def reset(self) -> None:
        """Called at the start of every stream."""

    def adapt(self, points: np.ndarray, model) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


class SourceOnly(Adapter):
    name = "source_only"

    def adapt(self, points: np.ndarray, model) -> np.ndarray:
        return np.asarray(model(points), dtype=np.float64)


class EntropyBias(Adapter):
    """Black-box analogue of TENT.

    TENT updates BatchNorm statistics by descending output entropy.  A black-box
    client cannot touch BatchNorm, so the closest legal move is to descend the
    same objective over an additive output bias.  It is a strong baseline on
    appearance shifts and, as the paper argues, a weak one on density shifts.
    """

    name = "entropy_bias"

    def __init__(
        self, steps: int = 10, lr: float = 0.5, weight_decay: float = 0.01
    ) -> None:
        self.steps = int(steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def adapt(self, points: np.ndarray, model) -> np.ndarray:
        z = np.asarray(model(points), dtype=np.float64)
        bias = np.zeros_like(z)
        for _ in range(self.steps):
            p = self._softmax(z + bias)
            entropy = float(-(p * np.log(p + 1e-12)).sum())
            grad = p * (np.log(p + 1e-12) + entropy)
            bias -= self.lr * (grad + self.weight_decay * bias)
        return z + bias


class _DensityAdapter(Adapter):
    """Shared plumbing for every density-aware variant."""

    name = "density"

    def __init__(
        self,
        n_classes: int,
        *,
        n_buckets: int = 8,
        feature_subset: list[int] | None = None,
        forgetting: float = 0.995,
        alpha: float = 0.5,
        ridge: float = 0.05,
        warmup: int = 32,
        second_view_ratio: float = 0.5,
        update_every: int = 1,
        max_directions: int | None = None,
        use_changepoint: bool = True,
        changepoint_threshold: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.n_classes = int(n_classes)
        self.model = DensityBiasModel(
            n_classes=n_classes,
            n_buckets=n_buckets,
            feature_subset=feature_subset,
            forgetting=forgetting,
            alpha=alpha,
            ridge=ridge,
            warmup=warmup,
            use_buckets=n_buckets > 1,
            max_directions=max_directions,
        )
        self.detector = ShockDetector(ratio=changepoint_threshold)
        self.use_changepoint = bool(use_changepoint)
        self.second_view_ratio = float(second_view_ratio)
        self.update_every = max(1, int(update_every))
        self._rng = np.random.default_rng(seed)
        self._tick = 0
        self.n_forward_passes = 0
        self.n_resets = 0
        self._last_descriptor: np.ndarray | None = None
        self._last_entropy = 0.0

    def reset(self) -> None:
        self.model.reset()
        self.detector.reset()
        self._tick = 0
        self.n_forward_passes = 0
        self.n_resets = 0
        self._last_descriptor = None
        self._last_entropy = 0.0

    # ------------------------------------------------------------------
    def _paired_update(self, points: np.ndarray, d_dense, z_dense, model) -> None:
        if self._tick % self.update_every != 0:
            return
        view = stratified_subsample(
            points, self.second_view_ratio, bins=4, rng=self._rng
        )
        if view.shape[0] < 8:
            return
        try:
            d_sparse = density_descriptor(view)
        except ValueError:
            return
        z_sparse = np.asarray(model(view), dtype=np.float64)
        self.n_forward_passes += 1

        # Score the pair against the *current* estimate before folding it in;
        # otherwise the model would always find its own residual small.
        if self.use_changepoint and self.model.fitted:
            residual = self.model.z_score(d_dense, d_sparse, z_dense - z_sparse)
            if self.detector.update(residual):
                self.model.reset_weights()
                self.n_resets += 1

        self.model.observe_pair(d_dense, d_sparse, z_dense, z_sparse)

    def adapt(self, points: np.ndarray, model) -> np.ndarray:
        self._tick += 1
        d = density_descriptor(points)
        z = np.asarray(model(points), dtype=np.float64)
        self.n_forward_passes += 1

        self._paired_update(points, d, z, model)

        self._last_descriptor = d
        p = np.exp(z - z.max())
        p /= p.sum()
        self._last_entropy = float(-(p * np.log(p + 1e-12)).sum())
        return self.model.correct(z, d)

    def diagnostics(self) -> dict:
        out = {
            "forward_passes": self.n_forward_passes,
            "resets": self.n_resets,
            "entropy": self._last_entropy,
            "changepoint_score": self.detector.last_score,
        }
        out.update(self.model.state())
        if self._last_descriptor is not None:
            out["density"] = float(self._last_descriptor[0])
        return out


class GlobalBias(_DensityAdapter):
    """Ablation: no density conditioning, only a scalar scale term."""

    name = "global_bias"

    def __init__(self, n_classes: int, **kw) -> None:
        kw.setdefault("n_buckets", 1)
        kw.setdefault("feature_subset", [3])
        super().__init__(n_classes, **kw)


class DSCNoBucket(_DensityAdapter):
    """Ablation: density features, single global regressor."""

    name = "dsc_no_bucket"

    def __init__(self, n_classes: int, **kw) -> None:
        kw.setdefault("n_buckets", 1)
        super().__init__(n_classes, **kw)


class DSCNoDrift(_DensityAdapter):
    """Ablation: full basis, no forgetting and no change point."""

    name = "dsc_no_drift"

    def __init__(self, n_classes: int, **kw) -> None:
        kw.setdefault("forgetting", 1.0)
        kw.setdefault("use_changepoint", False)
        super().__init__(n_classes, **kw)


class DSC(_DensityAdapter):
    """Denshift: density-stratified, closed-form, black-box adaptation."""

    name = "dsc"


class DSCTruncated(_DensityAdapter):
    """Denshift restricted to the two directions the pairs excite most.

    The correction is identified only on the subspace the dense/sparse
    intervention actually explores; projecting the coefficient onto that
    subspace stops it from over-correcting elsewhere.  See
    ``scripts/diagnose_identifiability.py``.
    """

    name = "dsc_top2"

    def __init__(self, n_classes: int, **kw) -> None:
        kw.setdefault("max_directions", 2)
        super().__init__(n_classes, **kw)


ADAPTERS = {
    "source_only": SourceOnly,
    "entropy_bias": EntropyBias,
    "global_bias": GlobalBias,
    "dsc_no_bucket": DSCNoBucket,
    "dsc_no_drift": DSCNoDrift,
    "dsc": DSC,
    "dsc_top2": DSCTruncated,
}


def build_adapter(name: str, n_classes: int, **kwargs) -> Adapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; available: {sorted(ADAPTERS)}")
    cls = ADAPTERS[name]
    if cls is SourceOnly or cls is EntropyBias:
        return cls(**kwargs)
    return cls(n_classes, **kwargs)
