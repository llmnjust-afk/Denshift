"""Change detection, done on the right signal.

The obvious detector watches the descriptor stream.  That is wrong here: in a
real stream the class mix changes constantly, and a class-imbalanced window looks
exactly like a regime change.  Detecting on descriptors therefore fires on
stationary data, which is worse than not detecting at all -- every reset throws
away the estimate Denshift just learned.

The signal that actually indicates a regime change is the *residual* of the
nuisance regression.  Class effects are shared by the dense and the sparse view
and cancel in their logit difference, so a residual spike cannot be explained by
a different class mix.  It means the learned relation between density and logits
no longer holds, which is the definition of a new regime.
"""

from __future__ import annotations

import numpy as np


class ShockDetector:
    """Two-timescale EMA ratio test on a non-negative scalar residual."""

    def __init__(
        self,
        fast: float = 0.2,
        slow: float = 0.02,
        ratio: float = 2.0,
        min_abs: float = 0.25,
        cooldown: int = 24,
    ) -> None:
        self.fast = float(fast)
        self.slow = float(slow)
        self.ratio = float(ratio)
        self.min_abs = float(min_abs)
        self.cooldown = int(cooldown)
        self._fast: float | None = None
        self._slow: float | None = None
        self._n = 0
        self._since = 0
        self.last_score = 0.0

    def update(self, value: float) -> bool:
        value = float(max(value, 0.0))
        self._n += 1
        self._since += 1
        if self._slow is None or self._fast is None:
            self._fast = value
            self._slow = value
            return False

        fast_prev: float = self._fast
        slow_prev: float = self._slow
        self._fast = (1.0 - self.fast) * fast_prev + self.fast * value
        self._slow = (1.0 - self.slow) * slow_prev + self.slow * value
        denom = max(float(self._slow), 1e-8)
        self.last_score = float(self._fast / denom)

        ready = self._n >= 16 and self._slow > self.min_abs
        if ready and self.last_score > self.ratio and self._since > self.cooldown:
            self._since = 0
            # Collapse the fast average onto the new level so the detector does
            # not fire repeatedly on the same step change.
            self._fast = value
            return True
        return False

    def reset(self) -> None:
        self._fast = None
        self._slow = None
        self._n = 0
        self._since = 0
        self.last_score = 0.0


class DescriptorShiftDetector:
    """Chi-square two-window test on raw descriptors.

    Kept for the ablation that asks whether watching the descriptor stream is
    enough.  On class-imbalanced streams it is not, which is the point.
    """

    def __init__(
        self,
        dim: int = 6,
        window: int = 64,
        threshold: float = 28.0,
        cooldown: int = 32,
        eps: float = 1e-6,
    ) -> None:
        self.dim = int(dim)
        self.window = int(window)
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.eps = float(eps)
        self._buf = np.zeros((2 * self.window, self.dim), dtype=np.float64)
        self._pos = 0
        self._n = 0
        self._since = 0
        self.last_score = 0.0

    def update(self, descriptor: np.ndarray) -> bool:
        d = np.asarray(descriptor, dtype=np.float64).reshape(self.dim)
        self._buf[self._pos] = d
        self._pos = (self._pos + 1) % (2 * self.window)
        self._n = min(self._n + 1, 2 * self.window)
        self._since += 1
        if self._n < 2 * self.window:
            return False

        ordered = np.roll(self._buf, -self._pos, axis=0)
        ref, cur = ordered[: self.window], ordered[self.window :]
        diff = cur.mean(axis=0) - ref.mean(axis=0)
        pooled_var = 0.5 * (cur.var(axis=0, ddof=1) + ref.var(axis=0, ddof=1))
        pooled_var = np.clip(pooled_var, self.eps, None)
        self.last_score = float(0.5 * self.window * np.sum(diff ** 2 / pooled_var))

        if self.last_score > self.threshold and self._since > self.cooldown:
            self._since = 0
            return True
        return False

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0
        self._n = 0
        self._since = 0
        self.last_score = 0.0
