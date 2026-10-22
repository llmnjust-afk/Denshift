"""Reproduce the identifiability numbers quoted in the README.

    python scripts/diagnose_identifiability.py --n-samples 200 --seed 1

The estimator is fitted on paired dense/sparse views and then compared with the
analytically known density term of the synthetic model.  Three questions are
answered:

1. How much of descriptor space does the intervention actually excite?  (the
   singular spectrum of the design matrix)
2. How well does the fit describe the pairs?  (residual, in-sample)
3. How wrong is the coefficient away from that subspace?  (absolute error,
   per-class scale, and end-to-end accuracy)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from denshift.calibrate import DensityBiasModel
from denshift.geometry import density_descriptor, stratified_subsample
from denshift.synthetic import ShapeWorld, make_stream, model_input, train_source_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--regime", default="decoupled")
    p.add_argument("--warmup", type=int, default=32)
    p.add_argument("--ridge", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    world = ShapeWorld(seed=0)
    clf = train_source_model(world, seed=0, lam=1.0)
    clouds, labels, _ = make_stream(world, args.n_samples, args.regime, seed=args.seed)

    feats = np.stack([model_input(p, world) for p in clouds])
    logits = clf.decision_function(feats)
    truth = clf.density_term(feats)

    model = DensityBiasModel(n_classes=10, warmup=args.warmup, ridge=args.ridge)
    rng = np.random.default_rng(0)

    pairs = []
    for cloud, desc, z in zip(clouds, (density_descriptor(p) for p in clouds), logits):
        view = stratified_subsample(cloud, 0.5, 4, rng)
        d_view = density_descriptor(view)
        z_view = clf.decision_function(model_input(view, world)[None, :]).reshape(-1)
        model.observe_pair(desc, d_view, z, z_view)
        pairs.append((desc, d_view, z, z_view))

    # Rebuild the design matrix with the frozen basis so the spectrum is exact.
    dphi = np.stack([model.phi(a) - model.phi(b) for a, b, _, _ in pairs])
    dz = np.stack([z - zv for _, _, z, zv in pairs])
    coef = model.coefficients()

    svals = np.linalg.svd(dphi, compute_uv=False)
    tol = svals[0] * max(dphi.shape) * np.finfo(float).eps
    rank = int((svals > tol).sum())
    cond = float(svals[0] / max(svals[rank - 1], 1e-12))

    residual = float(np.linalg.norm(dphi @ coef - dz) / max(np.linalg.norm(dz), 1e-12))

    phi_all = np.stack([model.phi(d) for d in (p[0] for p in pairs)])
    estimate = phi_all @ coef
    abs_error = float(
        np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-12)
    )

    scales = []
    for c in range(truth.shape[1]):
        denom = np.linalg.norm(truth[:, c])
        if denom > 1e-9:
            scales.append(np.linalg.norm(estimate[:, c]) / denom)

    cosines = [
        float(np.dot(e, t) / max(np.linalg.norm(e) * np.linalg.norm(t), 1e-12))
        for e, t in zip(estimate, truth)
    ]

    print(f"regime={args.regime}  samples={args.n_samples}  seed={args.seed}")
    print(
        f"design matrix: rank {rank} of {dphi.shape[1]} basis directions, "
        f"cond {cond:.1e} on the identified subspace"
    )
    print(f"pair residual ||A B - dz|| / ||dz||      : {residual:.4f}")
    print(f"absolute error ||Phi B - f|| / ||f||    : {abs_error:.4f}")
    print(f"per-sample cosine, mean / min           : {np.mean(cosines):.3f} / {np.min(cosines):.3f}")
    print(f"per-class scale, mean / min / max       : {np.mean(scales):.2f} / {np.min(scales):.2f} / {np.max(scales):.2f}")

    print("\ntruncation curve (correction restricted to the top-k directions of A):")
    _, _, vt = np.linalg.svd(dphi, full_matrices=False)
    print(f"{'k':>3}  {'accuracy':>9}")
    for k in range(1, min(len(svals), 10) + 1):
        projector = vt[:k].T @ vt[:k]
        corrected = logits - phi_all @ (projector @ coef)
        acc = float(np.mean(np.argmax(corrected, axis=1) == labels))
        print(f"{k:>3}  {acc:>9.3f}")

    baseline = float(np.mean(np.argmax(logits, axis=1) == labels))
    print(f"\nsource model on this stream: {baseline:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
