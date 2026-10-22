from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from denshift.blackbox import BlackBoxModel, ContractViolation
from denshift.backends.dgcnn import DGCNN, get_graph_feature, knn, to_black_box


def test_knn_returns_k_neighbours_per_point() -> None:
    x = torch.randn(2, 3, 64)
    idx = knn(x, k=8)
    assert idx.shape == (2, 64, 8)
    assert int(idx.max()) < 64


def test_graph_feature_shape() -> None:
    x = torch.randn(2, 3, 32)
    feat = get_graph_feature(x, k=6)
    assert feat.shape == (2, 6, 32, 6)


def test_forward_shape_and_finiteness() -> None:
    model = DGCNN(n_classes=7).eval()
    x = torch.randn(1, 3, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 7)
    assert torch.isfinite(logits).all()


def test_wrapper_returns_one_logit_vector() -> None:
    model = DGCNN(n_classes=5).eval()
    box = to_black_box(model, n_classes=5)
    logits = box(np.random.default_rng(0).normal(size=(96, 3)))
    assert logits.shape == (5,)
    assert np.isfinite(logits).all()


def test_wrapper_freezes_a_live_model() -> None:
    model = DGCNN(n_classes=5)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    box = BlackBoxModel(module=model, n_classes=5)
    assert box.module.training is False
    assert all(not p.requires_grad for p in box.module.parameters())


def test_contract_is_rechecked_on_demand() -> None:
    box = BlackBoxModel(module=DGCNN(n_classes=5), n_classes=5)
    next(box.module.parameters()).requires_grad_(True)
    with pytest.raises(ContractViolation):
        box.verify_contract()

    box = BlackBoxModel(module=DGCNN(n_classes=5), n_classes=5)
    box.module.train()
    with pytest.raises(ContractViolation):
        box.verify_contract()


def test_wrapper_does_not_build_a_graph_on_the_model() -> None:
    model = DGCNN(n_classes=5).eval()
    box = to_black_box(model, n_classes=5)
    box(np.random.default_rng(1).normal(size=(64, 3)))
    assert all(p.grad is None for p in model.parameters())
