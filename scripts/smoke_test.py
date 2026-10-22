"""Run the controlled density-shift benchmark and print the result table.

    python scripts/smoke_test.py --n-samples 150 --seed 0

The harness is a unit test for the estimator, not a leaderboard: it lets the
density/class coupling be varied independently while everything else is held
fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from denshift.adapter import ADAPTERS, build_adapter
from denshift.synthetic import run_benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--adapters",
        nargs="+",
        default=["source_only", "entropy_bias", "global_bias", "dsc"],
        choices=sorted(ADAPTERS),
    )
    p.add_argument("--json", type=str, default=None, help="write raw results here")
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["none", "decoupled", "decoupled_hard", "global"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapters = {name: build_adapter(name, 10) for name in args.adapters}

    started = time.time()
    results = run_benchmark(
        adapters,
        regimes=tuple(args.regimes),
        n_samples=args.n_samples,
        seed=args.seed,
    )
    elapsed = time.time() - started

    regimes = list(results["regimes"])
    names = [n for n in results["regimes"][regimes[0]] if not n.startswith("_")]

    width = max(len(n) for n in names) + 2
    header = "adapter".ljust(width) + "".join(r.rjust(16) for r in regimes)
    print(header)
    print("-" * len(header))
    for name in names:
        row = name.ljust(width)
        for regime in regimes:
            acc = results["regimes"][regime][name]["accuracy"]
            row += f"{acc:>16.3f}"
        print(row)
    print(f"\n{args.n_samples} samples, seed {args.seed}, {elapsed:.0f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"raw results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
