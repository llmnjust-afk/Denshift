"""Loader for ModelNet40-C / ModelNet-C point clouds.

The release ships one directory per corruption and severity, with a
class-named sub-directory below it.  Layouts differ slightly between the
original ModelNet-C release and the merged PointCloud-C bundle, so files are
discovered by walking the tree and reading the label off the nearest
class-named ancestor rather than by assuming a fixed depth.

    data/ModelNet40-C/
        clean/airplane/airplane_0001.npy
        add_global/1/airplane/airplane_0001.npy
        jitter/3/...

``discover`` also accepts ``add_global_1`` style flat directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POINT_SUFFIXES = (".npy", ".txt")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str

    def load(self, normalize: bool = False) -> np.ndarray:
        points = np.load(self.path) if self.path.suffix == ".npy" else np.loadtxt(self.path)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"{self.path}: expected (N, >=3), got {points.shape}")
        points = points[:, :3]
        if normalize:
            points = points - points.mean(axis=0, keepdims=True)
            radius = np.linalg.norm(points, axis=1).max()
            points = points / max(float(radius), 1e-8)
        return points


def _looks_like_split(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"train", "test", "clean"}
        or lowered.isdigit()
        or lowered.startswith("severity")
        or lowered in {"0", "1", "2", "3", "4", "5"}
    )


def discover(
    root: str | Path,
    corruption: str | None = None,
    severity: int | None = None,
) -> list[Sample]:
    """List ``(path, label)`` pairs under ``root``.

    ``corruption`` is matched against any path component (so both
    ``add_global/1/...`` and ``add_global_1/...`` work); ``severity`` is matched
    against numeric components when given.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    files: list[Path] = []
    for suffix in POINT_SUFFIXES:
        files.extend(root.rglob(f"*{suffix}"))
    files = sorted(set(files))

    out: list[Sample] = []
    for path in files:
        rel = path.relative_to(root).parts[:-1]
        if corruption is not None and not any(corruption in part for part in rel):
            continue
        if severity is not None and not any(
            part.isdigit() and int(part) == int(severity) for part in rel
        ):
            continue

        label = None
        for part in reversed(rel):
            if not _looks_like_split(part):
                label = part
                break
        if label is None:
            raise ValueError(f"cannot infer a class label from {path}")
        out.append(Sample(path, label))

    if not out:
        raise FileNotFoundError(
            f"no point clouds found under {root} for corruption={corruption!r}, "
            f"severity={severity!r}"
        )
    return out


def class_names(samples: list[Sample]) -> list[str]:
    return sorted({s.label for s in samples})


def as_arrays(
    samples: list[Sample],
    classes: list[str],
    normalize: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Materialise a split into ``(list of point arrays, integer labels)``."""
    index = {name: i for i, name in enumerate(classes)}
    clouds = [s.load(normalize=normalize) for s in samples]
    labels = np.array([index[s.label] for s in samples], dtype=np.int64)
    return clouds, labels
