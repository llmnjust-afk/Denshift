"""A wrapper that makes the black-box contract executable.

The paper claims a weaker assumption than prior 3D TTA work: no gradients
through the model, no source data, no weight access.  Claims like that are easy
to state and easy to violate by accident, so the wrapper enforces them:

* parameters are frozen and ``requires_grad`` is asserted False,
* every call runs under ``torch.no_grad()``,
* only logits cross the boundary -- never features, never state.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class ContractViolation(RuntimeError):
    """Raised when an adapter tries to reach inside the model."""


class BlackBoxModel:
    """Callable ``(N, 3) points -> (C,) logits`` with enforcement."""

    def __init__(
        self,
        module=None,
        *,
        preprocess: Callable[[np.ndarray], object] | None = None,
        postprocess: Callable[[object], np.ndarray] | None = None,
        device: str = "cpu",
        n_classes: int | None = None,
    ) -> None:
        self.module = module
        self.device = device
        self.n_classes = n_classes
        self._preprocess = preprocess
        self._postprocess = postprocess
        self.n_calls = 0
        self._torch = None

        if module is not None:
            import torch  # local import keeps the numpy core dependency-free

            self._torch = torch
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
            self.verify_contract()

    # ------------------------------------------------------------- contract
    def verify_contract(self) -> None:
        if self._torch is None:
            return
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                raise ContractViolation(f"parameter {name} still requires grad")
        if self.module.training:
            raise ContractViolation("module is in training mode")

    # ----------------------------------------------------------------- call
    def __call__(self, points: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        if self._torch is None:
            raise RuntimeError("BlackBoxModel has no torch module; pass a callable model")

        x = self._preprocess(points) if self._preprocess else points
        if isinstance(x, np.ndarray):
            x = self._torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with self._torch.no_grad():
            out = self.module(x)

        if self._postprocess is not None:
            out = self._postprocess(out)
        if self._torch.is_tensor(out):
            logits = out.detach().float().cpu().numpy()
        else:
            logits = np.asarray(out, dtype=np.float32)
        logits = np.asarray(logits).reshape(-1)
        if self.n_classes is not None and logits.shape[0] != self.n_classes:
            raise ContractViolation(
                f"expected {self.n_classes} logits, got {logits.shape[0]}"
            )
        return logits.astype(np.float64)


class CallableBox:
    """Wrap any plain ``points -> logits`` function as a black box (for tests)."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], n_classes: int) -> None:
        self.fn = fn
        self.n_classes = int(n_classes)
        self.n_calls = 0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        logits = np.asarray(self.fn(points), dtype=np.float64).reshape(-1)
        if logits.shape[0] != self.n_classes:
            raise ContractViolation(
                f"expected {self.n_classes} logits, got {logits.shape[0]}"
            )
        return logits
