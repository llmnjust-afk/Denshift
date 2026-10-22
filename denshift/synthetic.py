"""A synthetic world in which a density shortcut is *learned*, not injected.

A smoke test is only worth running if the failure it reproduces is real.  Here a
classifier is trained honestly on a source distribution where object class and
sampling density are correlated (small parts of a scene are scanned with fewer
returns).  The classifier exploits that correlation, exactly as a real 3D model
does.  The test stream then changes the density regime, and the shortcut becomes
a liability.

Part of the resulting drop is a removable logit bias; part is genuine loss of
geometric detail and cannot be recovered by any adapter.  That split is what
makes the comparison informative.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .geometry import density_descriptor

N_SEMANTIC = 19
N_DENSITY = 12


class ShapeWorld:
    """Mixtures of anisotropic blobs with class-dependent scan density.

    Shapes are built from a fixed template grid so that the *semantic* channel
    really is discriminative.  That matters: if shape alone cannot separate the
    classes, a classifier will lean on density for a legitimate reason and no
    debiasing method can help.
    """

    # (number of blobs, anisotropic scaling) -- ten well-separated templates.
    TEMPLATES = [
        (1, (1.00, 1.00, 1.00)),
        (1, (1.00, 1.00, 0.55)),
        (1, (1.00, 0.55, 0.55)),
        (1, (0.55, 1.00, 1.00)),
        (2, (1.00, 1.00, 0.70)),
        (2, (1.00, 0.70, 0.70)),
        (2, (0.70, 1.00, 1.00)),
        (3, (1.00, 1.00, 0.70)),
        (3, (1.00, 0.70, 0.70)),
        (3, (0.70, 0.70, 1.00)),
    ]
    _OFFSETS = {
        1: np.array([[0.0, 0.0, 0.0]]),
        2: np.array([[0.50, 0.0, 0.0], [-0.50, 0.0, 0.0]]),
        3: np.array([[0.50, 0.0, 0.0], [-0.25, 0.43, 0.0], [-0.25, -0.43, 0.0]]),
    }

    def __init__(self, n_classes: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        n_classes = min(int(n_classes), len(self.TEMPLATES))
        self.n_classes = n_classes
        self.n_blobs = np.array([self.TEMPLATES[c][0] for c in range(n_classes)])
        self.axes = np.array([self.TEMPLATES[c][1] for c in range(n_classes)], dtype=np.float64)
        # Class-dependent scan density: a *usable but imperfect* shortcut.
        # If density separated the classes perfectly the classifier would
        # ignore shape entirely and the world would be degenerate.
        self.base_density = np.clip(
            700.0 + 60.0 * np.arange(n_classes) + rng.normal(0.0, 210.0, size=n_classes),
            260,
            1800,
        ).astype(int)
        self.blob_spread = 0.24

    # ------------------------------------------------------------------ data
    def cloud(
        self,
        label: int,
        n_points: int,
        rng: np.random.Generator,
        jitter: float = 0.01,
    ) -> np.ndarray:
        offsets = self._OFFSETS[int(self.n_blobs[label])]
        which = rng.integers(0, offsets.shape[0], size=n_points)
        centres = offsets[which]
        pts = centres + rng.normal(0.0, self.blob_spread, size=(n_points, 3))
        pts *= self.axes[label][None, :]
        pts += rng.normal(0.0, jitter, size=pts.shape)
        pts -= pts.mean(axis=0, keepdims=True)
        rms = np.sqrt((pts ** 2).sum(axis=1).mean())
        return pts / max(rms, 1e-8)

    @staticmethod
    def features(points: np.ndarray) -> np.ndarray:
        """Shape descriptor: radial profile plus second-moment statistics.

        The radial quantiles carry most of the class signal; the moments make
        the descriptor stable when the point count changes.
        """
        pts = np.asarray(points, dtype=np.float64)
        pts = pts - pts.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(pts, axis=1)
        rms = max(float(np.sqrt((radius ** 2).mean())), 1e-8)
        unit = radius / rms

        cov = np.cov(pts, rowvar=False)
        eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))[::-1]
        eig = eig / max(float(eig.sum()), 1e-8)

        return np.concatenate(
            [
                np.quantile(unit, np.linspace(0.1, 1.0, 10)),
                eig,
                pts.std(axis=0) / rms,
                pts.mean(axis=0)[:2] / rms,
                [(unit ** 4).mean()],
            ]
        ).astype(np.float64)


def model_input(points: np.ndarray, world: ShapeWorld) -> np.ndarray:
    """What the frozen classifier actually sees: shape plus density, squared.

    The squared density terms matter: they make the learned dependence on
    density non-linear, so a single global bias cannot undo it and the
    stratified estimator has something to do.
    """
    sem = world.features(points)
    dens = density_descriptor(points)
    return np.concatenate([sem, dens, dens ** 2])


class ShortcutClassifier:
    """A frozen model with an explicit semantic pathway and a density shortcut.

    Fitting a single logistic regression on shape *and* density does not work
    for a controlled study: when density separates the classes the optimiser is
    free to throw the shape weights away, and the resulting model has no
    semantic pathway left to recover.  Real models do not behave that way -- they
    keep a usable shape rule and use density as a tie-breaker.

    So the model is composed explicitly:

        z(x) = W_sem . s(x) + lam * V . phi(d(x))

    Both terms are fitted honestly, separately, on the same source data.  ``lam``
    sets how much the model trusts the shortcut, and is the knob that controls
    how much damage a broken class-density correlation can do.
    """

    def __init__(
        self,
        w_sem: np.ndarray,
        b_sem: np.ndarray,
        v_dens: np.ndarray,
        b_dens: np.ndarray,
        lam: float = 0.6,
    ) -> None:
        self.w_sem = np.asarray(w_sem, dtype=np.float64)
        self.b_sem = np.asarray(b_sem, dtype=np.float64)
        self.v_dens = np.asarray(v_dens, dtype=np.float64)
        self.b_dens = np.asarray(b_dens, dtype=np.float64)
        self.lam = float(lam)

    def decision_function(self, feats: np.ndarray) -> np.ndarray:
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        sem = feats[:, :N_SEMANTIC]
        dens = feats[:, N_SEMANTIC : N_SEMANTIC + N_DENSITY]
        return self.b_sem + sem @ self.w_sem.T + self.lam * (
            self.b_dens + dens @ self.v_dens.T
        )

    def density_term(self, feats: np.ndarray) -> np.ndarray:
        """The part of the logits that the density shortcut alone produces."""
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        dens = feats[:, N_SEMANTIC : N_SEMANTIC + N_DENSITY]
        return self.lam * (self.b_dens + dens @ self.v_dens.T)


def train_source_model(
    world: ShapeWorld, n_samples: int = 2500, seed: int = 0, lam: float = 0.6
) -> ShortcutClassifier:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, world.n_classes, size=n_samples)
    feats = []
    for label in labels:
        base = world.base_density[label]
        n_points = int(np.clip(rng.normal(base, 0.12 * base), 200, 2600))
        feats.append(model_input(world.cloud(int(label), n_points, rng), world))
    X = np.stack(feats)

    sem_clf = LogisticRegression(max_iter=3000, C=2.0)
    sem_clf.fit(X[:, :N_SEMANTIC], labels)

    dens_clf = LogisticRegression(max_iter=3000, C=1.0)
    dens_clf.fit(X[:, N_SEMANTIC:], labels)

    return ShortcutClassifier(
        sem_clf.coef_, sem_clf.intercept_, dens_clf.coef_, dens_clf.intercept_, lam=lam
    )


def make_stream(
    world: ShapeWorld,
    n_samples: int = 480,
    regime: str = "decoupled",
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Build a test stream under a density regime.

    Returns ``(points_list, labels, meta)``.  The regimes separate what a
    source-free adapter can estimate from what it provably cannot:

    ``none``           density still coupled to class, as in training.
    ``decoupled``      the class-density correlation is broken: a sample's
                       density is drawn independently of its class, from the
                       same marginal distribution.  The nuisance is centred and
                       therefore identifiable from paired views alone.  This is
                       the setting Denshift targets.
    ``decoupled_hard`` the same, with a wider spread.
    ``global``         every sample loses the same fraction of points.  The
                       resulting offset is unidentifiable without source data,
                       and the information really is gone.  Denshift is not
                       expected to win here; reporting it is the honest thing
                       to do.
    ``gradual``        a slow global drift, same limitation.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, world.n_classes, size=n_samples)
    scales = _schedule(regime, n_samples, rng)

    points_list, used = [], []
    for label, scale in zip(labels, scales):
        if regime.startswith("decoupled"):
            # Density is now independent of class: the shortcut the model
            # learned no longer points at the right answer.
            base = float(rng.choice(world.base_density))
        else:
            base = float(world.base_density[int(label)])
        noisy = base * scale * float(np.exp(rng.normal(0.0, 0.18)))
        n_points = int(np.clip(noisy, 128, 2048))
        points_list.append(world.cloud(int(label), n_points, rng))
        used.append(scale)

    meta = {"regime": regime, "scales": np.asarray(used)}
    return points_list, labels, meta


def _schedule(regime: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    if regime == "none":
        return np.ones(n)
    if regime == "decoupled":
        return np.exp(rng.normal(0.0, 0.25, size=n))
    if regime == "decoupled_hard":
        return np.exp(rng.normal(0.0, 0.55, size=n))
    if regime == "global":
        return np.full(n, 0.45)
    if regime == "gradual":
        return 1.0 - 0.85 * t
    if regime == "abrupt":
        out = np.ones(n)
        out[n // 3 : 2 * n // 3] = 0.2
        out[2 * n // 3 :] = 0.55
        return out
    raise ValueError(f"unknown regime {regime!r}")


def evaluate_stream(
    clf,
    adapter,
    world: ShapeWorld,
    points_list: list[np.ndarray],
    labels: np.ndarray,
    *,
    record_curve: bool = False,
) -> dict:
    """Run one adapter over one stream and collect the paper's numbers."""
    adapter.reset()

    def model(points: np.ndarray) -> np.ndarray:
        feats = model_input(points, world)[None, :]
        return np.asarray(clf.decision_function(feats)).reshape(-1)

    correct = 0
    curve: list[int] = []
    for points, label in zip(points_list, labels):
        logits = adapter.adapt(points, model)
        hit = int(np.argmax(logits) == label)
        correct += hit
        if record_curve:
            curve.append(hit)

    out: dict = {
        "adapter": getattr(adapter, "name", "adapter"),
        "accuracy": correct / max(len(labels), 1),
        "diagnostics": adapter.diagnostics(),
    }
    if record_curve and curve:
        window = max(1, len(curve) // 24)
        out["curve"] = [
            float(np.mean(curve[i : i + window])) for i in range(0, len(curve), window)
        ]
    return out


def run_benchmark(
    adapters: dict,
    regimes: tuple[str, ...] = ("none", "decoupled", "decoupled_hard", "global"),
    n_samples: int = 360,
    seed: int = 0,
) -> dict:
    """Evaluate every adapter on every regime with one shared source model."""
    world = ShapeWorld(seed=seed)
    clf = train_source_model(world, seed=seed)

    results: dict = {"regimes": {}, "source_clean_accuracy": None, "n_samples": n_samples}
    for regime in regimes:
        pts, labels, meta = make_stream(world, n_samples, regime, seed=seed + 1)
        per_adapter = {}
        for name, adapter in adapters.items():
            per_adapter[name] = evaluate_stream(
                clf, adapter, world, pts, labels, record_curve=True
            )
        per_adapter["_meta"] = {"mean_density_scale": float(meta["scales"].mean())}
        results["regimes"][regime] = per_adapter

    clean = results["regimes"].get("none", {})
    if clean:
        results["source_clean_accuracy"] = clean.get("source_only", {}).get("accuracy")
    return results
