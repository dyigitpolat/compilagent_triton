"""Compare two model families' results on the same workload + trial budget.

Usage:
    env/bin/python scripts/experiments/plot_models_compare.py \\
        --left  runs/study/20260426T122511/results.jsonl --left-label  "Mistral large" \\
        --right runs/study/opus-12trials/results.jsonl   --right-label "Claude Opus 4.7" \\
        --trials 12

Plots side-by-side bars (one bar per seed) per workload, with the validated
recheck status overlaid as a marker. Saves `models_compare.png` next to the
left jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path, harness: str | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if harness:
        rows = [r for r in rows if r.get("harness") == harness]
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--harness", default="pydantic_ai")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    left = [r for r in load(args.left, args.harness) if r.get("max_candidates") == args.trials]
    right = [r for r in load(args.right, args.harness) if r.get("max_candidates") == args.trials]

    workloads = sorted({r["workload_id"] for r in (left + right)})
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.6 * len(workloads), 4.4),
                             dpi=150, sharey=False)
    if len(workloads) == 1:
        axes = [axes]
    for ax, w in zip(axes, workloads):
        l_rows = sorted([r for r in left if r["workload_id"] == w], key=lambda r: r["seed"])
        r_rows = sorted([r for r in right if r["workload_id"] == w], key=lambda r: r["seed"])
        seeds = sorted({r["seed"] for r in (l_rows + r_rows)})
        x = list(range(len(seeds)))
        l_speed = [_lookup(l_rows, s) for s in seeds]
        r_speed = [_lookup(r_rows, s) for s in seeds]
        l_rec = [_recheck(l_rows, s) for s in seeds]
        r_rec = [_recheck(r_rows, s) for s in seeds]
        width = 0.36
        ax.bar(
            [xi - width / 2 for xi in x], [s if s is not None else 0 for s in l_speed],
            width=width, color="#0ea5e9", label=args.left_label,
            edgecolor="#0b69a1", linewidth=0.8,
        )
        ax.bar(
            [xi + width / 2 for xi in x], [s if s is not None else 0 for s in r_speed],
            width=width, color="#f97316", label=args.right_label,
            edgecolor="#9a3412", linewidth=0.8,
        )
        # Overlay markers for recheck status; X for FAIL, o for OK,
        # nothing for missing/n/a.
        for xi, s, rec in zip(x, l_speed, l_rec):
            _mark(ax, xi - width / 2, s, rec)
        for xi, s, rec in zip(x, r_speed, r_rec):
            _mark(ax, xi + width / 2, s, rec)
        # 1.0× reference line.
        ax.axhline(1.0, color="#94a3b8", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylabel("speedup vs. baseline")
        ax.set_title(f"{w} · {args.trials} trials", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(
        f"{args.left_label} vs. {args.right_label} — pydantic_ai @ {args.trials} trials\n"
        "● recheck OK, ✗ recheck FAIL, blank = n/a (no candidate beat baseline)",
        fontsize=10,
    )
    fig.tight_layout()
    out = args.out or args.left.parent / f"models_compare_t{args.trials}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


def _lookup(rows: list[dict], seed: int) -> float | None:
    for r in rows:
        if r["seed"] == seed:
            return r.get("best_speedup")
    return None


def _recheck(rows: list[dict], seed: int) -> bool | None:
    for r in rows:
        if r["seed"] == seed:
            return r.get("correctness_recheck_ok")
    return None


def _mark(ax, x, y, rec):
    if y is None or y <= 0:
        return
    if rec is True:
        ax.plot(x, y + 0.02, marker="o", color="#16a34a", markersize=6, linestyle="None")
    elif rec is False:
        ax.plot(x, y + 0.02, marker="x", color="#dc2626", markersize=8, mew=2, linestyle="None")


if __name__ == "__main__":
    raise SystemExit(main())
