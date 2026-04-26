"""Plot the effectiveness study results.

Usage:
    env/bin/python scripts/experiments/plot_study.py runs/study/<timestamp>/results.jsonl

Generates four matplotlib PNGs alongside the input file:

  speedup_vs_trials.png    — best_speedup vs max_candidates,
                             one line per (harness, workload), with
                             mean ± stddev across the seeds.
  correctness_rate.png     — fraction of cells where the independent
                             correctness recheck passed AND the speedup
                             was > 1.0, vs max_candidates.
  successful_per_trial.png — successful_count / max_candidates ratio
                             vs max_candidates (how efficiently the
                             agent's budget converts to validated wins).
  elapsed_vs_trials.png    — wall-clock time per run vs max_candidates.

Cells with `error` set are dropped from the plotted aggregates but
counted toward the dropped-cells annotation in the title.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (harness, workload) -> color/marker/linestyle. Keep the palette small so it
# still reads well grayscale-printed.
SERIES_STYLE = {
    ("pydantic_ai", "vit_block"):  ("#0ea5e9", "o", "-"),
    ("pydantic_ai", "vector_add"): ("#22c55e", "s", "-"),
    ("claude_agent_sdk", "vit_block"):  ("#f97316", "^", "--"),
    ("claude_agent_sdk", "vector_add"): ("#a855f7", "D", "--"),
}


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def aggregate(
    rows: list[dict], *, value_fn, drop_errors: bool = True,
) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Group rows into {(harness, workload): {trials: [values across seeds]}}."""

    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if drop_errors and r.get("error"):
            continue
        key = (r["harness"], r["workload_id"])
        v = value_fn(r)
        if v is None:
            continue
        groups[key][int(r["max_candidates"])].append(float(v))
    return groups


def plot_lines(
    groups: dict[tuple[str, str], dict[int, list[float]]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    yref: float | None = None,
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    if yref is not None:
        ax.axhline(yref, color="#94a3b8", linewidth=0.8, linestyle=":", zorder=1)
        ax.text(
            0.99, yref, f" {yref:g} ",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#475569",
        )
    for series, by_t in sorted(groups.items()):
        color, marker, ls = SERIES_STYLE.get(series, ("#475569", "x", ":"))
        xs = sorted(by_t.keys())
        means = [statistics.mean(by_t[t]) for t in xs]
        stds  = [statistics.pstdev(by_t[t]) if len(by_t[t]) > 1 else 0.0 for t in xs]
        label = f"{series[0]} · {series[1]}"
        ax.errorbar(
            xs, means, yerr=stds,
            color=color, marker=marker, linestyle=ls,
            linewidth=1.6, markersize=6, capsize=3,
            label=label,
        )
    ax.set_xlabel("max trials (successful candidates target)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="best", frameon=False)
    if note:
        ax.text(
            0.01, -0.18, note, transform=ax.transAxes,
            fontsize=7, color="#64748b", ha="left", va="top",
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def correctness_rate(rows: list[dict]) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Fraction of seeds where (recheck_ok AND speedup > 1.0) per cell."""

    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            groups[(r["harness"], r["workload_id"])][int(r["max_candidates"])].append(0.0)
            continue
        sp = r.get("best_speedup")
        ok = r.get("correctness_recheck_ok")
        # Treat "no candidate produced a usable plan" as 0; "recheck failed" as 0;
        # otherwise as 1 iff sp > 1.0.
        win = 1.0 if (isinstance(sp, (int, float)) and sp > 1.0 and ok is True) else 0.0
        groups[(r["harness"], r["workload_id"])][int(r["max_candidates"])].append(win)
    return groups


def successful_ratio(rows: list[dict]) -> dict[tuple[str, str], dict[int, list[float]]]:
    """successful_count / max_candidates."""

    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        m = r.get("max_candidates") or 1
        s = r.get("successful_count") or 0
        groups[(r["harness"], r["workload_id"])][int(m)].append(s / max(1, int(m)))
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.results_jsonl)
    if not rows:
        print(f"No rows in {args.results_jsonl}", flush=True)
        return 1

    n_total = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    note = f"n={n_total} runs ({n_err} errored, dropped from speedup curves)"
    out_dir = args.results_jsonl.parent

    plot_lines(
        aggregate(rows, value_fn=lambda r: r.get("best_speedup")),
        title="Best speedup vs trial budget (mean ± stddev across seeds)",
        ylabel="speedup vs. baseline",
        out_path=out_dir / "speedup_vs_trials.png",
        yref=1.0,
        note=note,
    )

    plot_lines(
        correctness_rate(rows),
        title="Validated win rate (recheck_ok ∧ speedup>1.0) vs trial budget",
        ylabel="win fraction across seeds",
        out_path=out_dir / "correctness_rate.png",
        yref=None,
        note=note,
    )

    plot_lines(
        successful_ratio(rows),
        title="Budget efficiency: successful_count / max_candidates",
        ylabel="ratio",
        out_path=out_dir / "successful_per_trial.png",
        yref=None,
        note=note,
    )

    plot_lines(
        aggregate(rows, value_fn=lambda r: (r.get("elapsed_ms") or 0.0) / 1000.0),
        title="Wall-clock time per run vs trial budget",
        ylabel="seconds",
        out_path=out_dir / "elapsed_vs_trials.png",
        yref=None,
        note=note,
    )

    print(f"Wrote 4 PNGs to {out_dir}/")
    for png in (
        "speedup_vs_trials.png", "correctness_rate.png",
        "successful_per_trial.png", "elapsed_vs_trials.png",
    ):
        print(f"  {out_dir / png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
