from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from denshift.data.modelnet_c import Sample, as_arrays, class_names, discover


def _write(path: Path, n: int = 32, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.random.default_rng(seed).normal(size=(n, 3))
    if path.suffix == ".npy":
        np.save(path, pts)
    else:
        np.savetxt(path, pts)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    _write(tmp_path / "clean" / "airplane" / "a1.npy", seed=1)
    _write(tmp_path / "clean" / "chair" / "c1.npy", seed=2)
    _write(tmp_path / "add_global" / "1" / "airplane" / "a1.npy", seed=3)
    _write(tmp_path / "add_global" / "1" / "chair" / "c1.txt", seed=4)
    _write(tmp_path / "add_global" / "2" / "chair" / "c2.npy", seed=5)
    _write(tmp_path / "jitter" / "1" / "airplane" / "a2.npy", seed=6)
    return tmp_path


def test_discover_reads_labels_from_class_directories(tree: Path) -> None:
    samples = discover(tree, "add_global", 1)
    assert {s.label for s in samples} == {"airplane", "chair"}
    assert len(samples) == 2


def test_discover_filters_by_corruption(tree: Path) -> None:
    assert len(discover(tree, "jitter")) == 1
    assert len(discover(tree, "add_global")) == 3


def test_discover_filters_by_severity(tree: Path) -> None:
    assert len(discover(tree, "add_global", severity=2)) == 1


def test_discover_without_filters_returns_everything(tree: Path) -> None:
    assert len(discover(tree)) == 6


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")


def test_empty_selection_raises(tree: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tree, "does_not_exist")


def test_load_supports_npy_and_txt(tree: Path) -> None:
    samples = {s.label: s for s in discover(tree, "add_global", 1)}
    assert samples["airplane"].load().shape == (32, 3)
    assert samples["chair"].load().shape == (32, 3)


def test_normalize_puts_points_in_the_unit_ball(tree: Path) -> None:
    sample = discover(tree, "clean", None)[0]
    pts = sample.load(normalize=True)
    assert np.linalg.norm(pts, axis=1).max() <= 1.0 + 1e-9


def test_as_arrays_uses_a_shared_class_order(tree: Path) -> None:
    samples = discover(tree, "clean")
    classes = class_names(samples)
    clouds, labels = as_arrays(samples, classes)
    assert len(clouds) == len(labels) == 2
    assert set(labels.tolist()) == {0, 1}
    assert classes == sorted(classes)
