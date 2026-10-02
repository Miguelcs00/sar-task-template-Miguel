#!/usr/bin/env python3
"""Paired comparison of two `swarm benchmark --summary-json-out` files.

    python TASK/tools/compare_runs.py baseline.json candidate.json [--group type4_village]

Only seeds present in BOTH runs are compared, so the comparison is always paired.
Reports, overall and per map group:
  * mean score before / after and the paired mean difference
  * 95% paired bootstrap CI of that difference
  * success flips (lost -> won, won -> lost) with an exact McNemar p-value
  * the seeds whose score changed, so they can be traced individually
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np


def load(path: str) -> dict[tuple[str, int], dict]:
    data = json.load(open(path))
    out = {}
    for group, rows in data["group_results"].items():
        for r in rows:
            out[(group, int(r["seed"]))] = r
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs b (lost->won) and c (won->lost)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(d: np.ndarray, iters: int = 20000, seed: int = 0) -> tuple[float, float]:
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(iters, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(name: str, keys: list, a: dict, b: dict) -> None:
    sa = np.array([a[k]["score"] for k in keys], float)
    sb = np.array([b[k]["score"] for k in keys], float)
    d = sb - sa
    lo, hi = bootstrap_ci(d)
    won = sum(1 for k in keys if not a[k]["success"] and b[k]["success"])
    lost = sum(1 for k in keys if a[k]["success"] and not b[k]["success"])
    changed = int(np.sum(np.abs(d) > 1e-9))
    print(f"\n== {name}  (n={len(keys)} paired seeds)")
    print(f"   mean before {sa.mean():.4f} | after {sb.mean():.4f} | diff {d.mean():+.4f}"
          f"  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"   success {int(sum(a[k]['success'] for k in keys))} -> "
          f"{int(sum(b[k]['success'] for k in keys))} | flips +{won} / -{lost}"
          f" | McNemar exact p={mcnemar_exact(won, lost):.3f}")
    print(f"   seeds with any score change: {changed}/{len(keys)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--group", default=None, help="restrict to one group, e.g. type4_village")
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.candidate)
    keys = sorted(set(a) & set(b))
    if args.group:
        keys = [k for k in keys if k[0] == args.group]
    if not keys:
        raise SystemExit("No common seeds between the two files.")

    report("ALL", keys, a, b)
    for g in sorted({k[0] for k in keys}):
        report(g, [k for k in keys if k[0] == g], a, b)

    print("\n== seeds whose score changed (group, seed, before -> after)")
    for k in keys:
        if abs(b[k]["score"] - a[k]["score"]) > 1e-9:
            print(f"   {k[0]:<16} {k[1]}  {a[k]['score']:.4f} -> {b[k]['score']:.4f}")


if __name__ == "__main__":
    main()
