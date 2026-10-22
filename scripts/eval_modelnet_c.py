"""Evaluate a frozen point-cloud classifier on ModelNet-C with a density adapter.

    python scripts/eval_modelnet_c.py \
        --root data/ModelNet40-C \
        --checkpoint weights/dgcnn_modelnet40.t7 \
        --corruptions add_global jitter scale \
        --severity 1 2 3 4 5 \
        --adapter dsc

Reports accuracy per (corruption, severity) and mCE against the clean split.
DGCNN on CPU is slow -- budget roughly a second per cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from denshift.adapter import ADAPTERS, build_adapter
from denshift.data.modelnet_c import Sample, as_arrays, class_names, discover
from denshift.metrics import accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="ModelNet-C root directory")
    p.add_argument("--checkpoint", default=None, help="DGCNN checkpoint; omit for a random model")
    p.add_argument("--adapter", default="source_only", choices=sorted(ADAPTERS))
    p.add_argument("--corruptions", nargs="+", default=["clean"])
    p.add_argument("--severity", nargs="*", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=200, help="clouds per split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--normalize", action="store_true", help="centre and scale to unit sphere")
    p.add_argument("--json", default=None)
    return p.parse_args()


def take_balanced(samples: list[Sample], n: int, rng: np.random.Generator) -> list[Sample]:
    """Draw up to ``n`` samples with roughly equal class counts."""
    by_class: dict[str, list[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)

    per_class = max(1, n // max(len(by_class), 1))
    picked: list[Sample] = []
    for label in sorted(by_class):
        pool = by_class[label]
        order = rng.permutation(len(pool))[:per_class]
        picked.extend(pool[i] for i in order)
    rng.shuffle(picked)
    return picked[:n]


def run_split(
    box,
    samples: list[Sample],
    classes: list[str],
    adapter_name: str,
    normalize: bool,
    seed: int,
) -> dict:
    clouds, labels = as_arrays(samples, classes, normalize=normalize)
    adapter = build_adapter(adapter_name, len(classes))
    adapter.reset()

    correct = 0
    entropies = []
    for cloud, label in zip(clouds, labels):
        logits = adapter.adapt(cloud, box)
        pred = int(np.argmax(logits))
        correct += int(pred == int(label))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        entropies.append(float(-(p * np.log(p + 1e-12)).sum()))

    diag = adapter.diagnostics()
    return {
        "accuracy": correct / max(len(clouds), 1),
        "n": len(clouds),
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "forward_passes": diag.get("forward_passes"),
    }


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    import torch

    from denshift.backends.dgcnn import DGCNN, load_checkpoint, to_black_box

    clean = discover(args.root, "clean")
    classes = class_names(clean)
    print(f"{len(classes)} classes, {len(clean)} clean clouds under {args.root}")

    model = DGCNN(n_classes=len(classes))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        print(f"loaded {args.checkpoint}")
    else:
        print("WARNING: no checkpoint given; logits are random")

    box = to_black_box(model, device=args.device, n_classes=len(classes))

    started = time.time()
    clean_subset = take_balanced(clean, args.n_samples, rng)
    base = run_split(box, clean_subset, classes, "source_only", args.normalize, args.seed)
    clean_error = max(1.0 - base["accuracy"], 1e-6)
    print(f"clean accuracy (source-only, mCE reference): {base['accuracy']:.4f}")

    results: dict = {"clean": base, "corruptions": {}, "adapter": args.adapter}
    severities = args.severity if args.severity else [None]
    errors = []

    for corruption in args.corruptions:
        if corruption == "clean":
            continue
        for severity in severities:
            samples = discover(args.root, corruption, severity)
            subset = take_balanced(samples, args.n_samples, rng)
            res = run_split(
                box, subset, classes, args.adapter, args.normalize, args.seed
            )
            key = f"{corruption}/{severity}" if severity is not None else corruption
            error = 1.0 - res["accuracy"]
            errors.append(error)
            results["corruptions"][key] = {**res, "error": error, "ce": error / clean_error}
            print(f"{key:<24} acc={res['accuracy']:.4f}  err={error:.4f}  ce={error / clean_error:.3f}")

    if errors:
        mean_error = float(np.mean(errors))
        results["mean_corruption_error"] = mean_error
        results["mCE"] = mean_error / clean_error
        print(f"\nmCE = {results['mCE']:.3f}  (mean error {mean_error:.4f} / clean error {clean_error:.4f})")

    print(f"elapsed {time.time() - started:.0f}s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"raw results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
