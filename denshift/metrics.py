"""Metrics used by the paper: accuracy, corruption error, resilience, ECE."""

from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    pred = np.argmax(np.asarray(logits), axis=-1)
    return float((pred == np.asarray(labels)).mean())


def per_class_accuracy(
    logits: np.ndarray, labels: np.ndarray, n_classes: int
) -> np.ndarray:
    pred = np.argmax(np.asarray(logits), axis=-1)
    labels = np.asarray(labels)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            out[c] = float((pred[mask] == c).mean())
    return out


def corruption_error(error: float, reference_error: float, eps: float = 1e-8) -> float:
    """CE as defined by ModelNet-C / Robo3D: error relative to a reference model."""
    return float(error / max(reference_error, eps))


def mean_corruption_error(
    acc_by_corruption: dict[str, float],
    reference_acc_by_corruption: dict[str, float],
) -> float:
    """mCE over every corruption present in both dictionaries.

    ``clean`` is excluded, matching the benchmark convention.
    """
    keys = [
        k
        for k in acc_by_corruption
        if k != "clean" and k in reference_acc_by_corruption
    ]
    if not keys:
        raise ValueError("no shared corruption keys")
    ces = [
        corruption_error(
            1.0 - acc_by_corruption[k], 1.0 - reference_acc_by_corruption[k]
        )
        for k in keys
    ]
    return float(np.mean(ces))


def resilience_rate(acc_by_corruption: dict[str, float]) -> float:
    clean = acc_by_corruption.get("clean")
    if clean is None or clean <= 0:
        raise ValueError("acc_by_corruption must contain a positive 'clean' entry")
    keys = [k for k in acc_by_corruption if k != "clean"]
    if not keys:
        raise ValueError("no corruption entries")
    return float(np.mean([acc_by_corruption[k] / clean for k in keys]))


def expected_calibration_error(
    logits: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == np.asarray(labels)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += gap * mask.mean()
    return float(ece)
