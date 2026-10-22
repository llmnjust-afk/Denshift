"""Streaming statistics with bounded memory.

Denshift must run online, so every statistic it keeps is O(1) in the number of
test samples.  Two primitives are enough: a running standardiser and a
reservoir-sampled quantile estimator.
"""

from __future__ import annotations

import numpy as np


class RunningStandardizer:
    """Welford accumulation of per-feature mean and variance."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.dim = int(dim)
        self.eps = float(eps)
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        var = self.m2 / (self.count - 1)
        return np.sqrt(np.clip(var, self.eps, None))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        return (x - self.mean) / self.std

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)


class StreamingQuantile:
    """Reservoir-sampled empirical quantiles, O(capacity) memory.

    Exact streaming quantiles (P-squared) are tempting, but the bucket
    boundaries matter for a *stratified* estimator and are easier to keep
    monotone with a small reservoir.  ``capacity=2048`` costs 16 KiB per
    feature and is stable enough for a test-time stream.
    """

    def __init__(self, capacity: int = 2048, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._buf = np.empty(self.capacity, dtype=np.float64)
        self._n = 0
        self._seen = 0

    def update(self, value: float) -> None:
        value = float(value)
        self._seen += 1
        if self._n < self.capacity:
            self._buf[self._n] = value
            self._n += 1
            return
        # Algorithm R: replace with probability capacity / seen.
        j = int(self._rng.integers(0, self._seen))
        if j < self.capacity:
            self._buf[j] = value

    def quantiles(self, q: np.ndarray | list[float]) -> np.ndarray:
        if self._n == 0:
            return np.zeros(len(np.atleast_1d(q)), dtype=np.float64)
        return np.quantile(self._buf[: self._n], np.asarray(q, dtype=np.float64))

    def bucket(self, value: float, n_buckets: int) -> int:
        """Return the quantile bucket index of ``value`` in ``[0, n_buckets)``."""
        if n_buckets <= 1 or self._n == 0:
            return 0
        edges = self.quantiles(np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
        return int(np.searchsorted(edges, float(value), side="right"))

    @property
    def ready(self) -> bool:
        return self._n >= min(32, self.capacity)

    def reset(self) -> None:
        self._n = 0
        self._seen = 0
