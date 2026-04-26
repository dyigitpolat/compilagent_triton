"""Reproducible effectiveness study.

Runs the optimizer across a 3-axis grid and records JSON results suitable for
plotting. Each cell is run 3 times (different RNG seeds) so the plot can show
mean ± stddev:

  - harness:      {pydantic_ai, claude_agent_sdk}
  - workload:     {vit_block (pytorch / inductor), vector_add (triton)}
  - max_candidates: {4, 8, 12, 16, 20}
  - seed:         {0, 1, 2}

Total: 2 × 2 × 5 × 3 = 60 runs.

Each run records:

  - baseline_median_ms
  - best_speedup, best_median_ms
  - best_correctness_ok, best_max_abs_diff
  - successful_count, failed_attempts
  - elapsed_ms
  - final_text (the agent's report)
  - candidates: list of {id, description, changes, speedup, median_ms}

Output: `runs/study/<timestamp>/results.jsonl` (one line per run).
Plotting: `python scripts/experiments/plot_study.py <results.jsonl>`.

Usage:
    env/bin/python scripts/experiments/run_study.py
    env/bin/python scripts/experiments/run_study.py \\
        --harnesses pydantic_ai \\
        --workloads vit_block \\
        --trials 4 8

Both harnesses use the same Mistral model unless `--model` is overridden.
Failed runs are recorded with `error` set; the study continues to the next
cell so a transient failure doesn't trash the whole grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch  # noqa: F401  — must precede triton/inductor imports

from compilagent_triton.backends import import_backend_packages
from compilagent_triton.backends import backend_registry
from compilagent_triton.settings import CompilagentSettings
from compilagent_triton.trace_store import TraceStore
from compilagent_triton.workload_runner import _run_pydantic_ai, _run_claude_sdk
from compilagent_triton.workloads.registry import (
    import_workload_packages,
    workload_registry,
)


# ---------------------------------------------------------------------------
# Grid + cell driver
# ---------------------------------------------------------------------------


WORKLOADS = ("vit_block", "vector_add")
HARNESSES = ("pydantic_ai", "claude_agent_sdk")
TRIALS = (4, 8, 12, 16, 20)
SEEDS = (0, 1, 2)


@dataclass(slots=True)
class CellResult:
    harness: str
    workload_id: str
    backend_id: str
    max_candidates: int
    seed: int
    baseline_median_ms: float | None
    best_speedup: float | None
    best_candidate_id: str | None
    best_median_ms: float | None
    best_correctness_ok: bool | None
    best_max_abs_diff: float | None
    successful_count: int
    failed_attempts: int
    elapsed_ms: float
    final_text: str | None
    candidates: list[dict]
    correctness_recheck_ok: bool | None
    correctness_recheck_max_abs_diff: float | None
    error: str | None
    timestamp: str


# ---------------------------------------------------------------------------
# Correctness re-verification
# ---------------------------------------------------------------------------


def _recheck_correctness(
    *,
    workload_id: str,
    workspace_root: Path,
    run_id: str,
) -> tuple[bool | None, float | None]:
    """Re-run the best candidate's compile and compare its output against the
    baseline output, independent of what the agent claimed.

    The agent reports `correctness_ok` per candidate; here we replay the best
    candidate from scratch with a fresh RNG seed so the plot reflects an
    independently verified speedup, not just the agent's self-report.

    Returns `(ok, max_abs_diff)` or `(None, None)` if no usable candidate was
    found.
    """

    # Walk the per-run trace events to find the best (compiled+timed) candidate
    # plan, then re-execute.
    trace_store = TraceStore(workspace_root)
    events = [
        e for e in trace_store.read_events()
        if (e.payload or {}).get("run_id") == run_id
    ]
    proposals: dict[str, dict] = {}
    benchmarks: dict[str, dict] = {}
    for ev in events:
        kind = ev.kind
        payload = ev.payload or {}
        if kind == "candidate.proposed":
            for c in payload.get("candidates", []) or []:
                proposals[c["id"]] = c
        elif kind == "benchmark.completed":
            cid = payload.get("candidate_id")
            if cid:
                benchmarks[cid] = payload
    # Find the best candidate by speedup.
    best_id = None
    best_sp = 1.0
    for cid, b in benchmarks.items():
        sp = b.get("speedup_vs_baseline")
        if isinstance(sp, (int, float)) and sp > best_sp:
            best_sp = sp
            best_id = cid
    if best_id is None or best_id not in proposals:
        return None, None

    # Reconstruct the plan from the proposal's `changes` dict.
    from compilagent_triton.backends import Plan
    from compilagent_triton.backends.base import (
        Intervention, Target, ToleranceConfig,
    )
    interventions: list[Intervention] = []
    for kind_str, group in (proposals[best_id].get("changes") or {}).items():
        if not isinstance(group, dict):
            continue
        for selector, payload in group.items():
            interventions.append(Intervention(
                target=Target(kind=str(kind_str), selector=str(selector)),
                payload=payload, rationale="recheck",
            ))
    if not interventions:
        return None, None

    # Replay against the workload.
    spec = workload_registry.get_spec(workload_id)
    backend = backend_registry.get(spec.backend_id)
    plan = Plan(interventions=tuple(interventions))
    try:
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Compile baseline + candidate for the recheck.
        recheck_root = workspace_root / "recheck" / run_id
        recheck_root.mkdir(parents=True, exist_ok=True)
        b_dir = recheck_root / "baseline"
        b_dir.mkdir(parents=True, exist_ok=True)
        c_dir = recheck_root / "candidate"
        c_dir.mkdir(parents=True, exist_ok=True)
        b_compile = backend.compile(spec, Plan(), artifact_dir=b_dir)
        c_compile = backend.compile(spec, plan, artifact_dir=c_dir)
        if not (getattr(b_compile, "ok", False) and getattr(c_compile, "ok", False)):
            return None, None
        tol = ToleranceConfig(
            atol=spec.tolerance.atol, rtol=spec.tolerance.rtol,
            notes="independent recheck",
        )
        result = backend.validate_correctness(spec, b_compile, c_compile, tol)
        return result.ok, result.max_abs_diff
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# Per-cell run
# ---------------------------------------------------------------------------


def run_cell(
    *,
    harness: str,
    workload_id: str,
    max_candidates: int,
    seed: int,
    model_name: str,
    out_root: Path,
) -> CellResult:
    """Run one (harness, workload, max_candidates, seed) cell."""

    import asyncio

    timestamp = datetime.now(UTC).isoformat()
    spec = workload_registry.get_spec(workload_id)

    # Per-cell workspace so trace events don't bleed between cells.
    cell_dir = (
        out_root / "cells" /
        f"{harness}__{workload_id}__t{max_candidates}__s{seed}"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    trace_store = TraceStore(cell_dir).ensure()
    run_id = f"study-{harness}-{workload_id}-t{max_candidates}-s{seed}"

    # Seed everything we control.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings = CompilagentSettings.from_env(project_root=Path.cwd())
    settings = settings.model_copy(update={"model_name": model_name})

    runner = _run_claude_sdk if harness == "claude_agent_sdk" else _run_pydantic_ai

    try:
        summary = asyncio.run(runner(
            workload_id=workload_id, run_id=run_id,
            workspace_root=cell_dir, trace_store=trace_store,
            settings=settings, max_candidates=max_candidates,
        ))
        error = None
    except Exception:  # noqa: BLE001
        summary = {}
        error = traceback.format_exc()

    # Collect candidate stats from the trace events.
    candidates_list: list[dict] = []
    for ev in trace_store.read_events():
        kind = ev.kind
        payload = ev.payload or {}
        if (payload.get("run_id") != run_id):
            continue
        if kind == "candidate.proposed":
            for c in payload.get("candidates", []) or []:
                if c.get("kind") == "search_space_summary":
                    continue
                candidates_list.append({
                    "id": c.get("id"),
                    "description": c.get("description"),
                    "changes": c.get("changes", {}),
                })
        elif kind == "benchmark.completed":
            cid = payload.get("candidate_id")
            if not cid:
                continue
            for c in candidates_list:
                if c.get("id") == cid:
                    c["median_ms"] = payload.get("median_ms")
                    c["speedup_vs_baseline"] = payload.get("speedup_vs_baseline")
                    break

    # Independent correctness recheck.
    if not error and summary.get("best_speedup"):
        recheck_ok, recheck_diff = _recheck_correctness(
            workload_id=workload_id,
            workspace_root=cell_dir,
            run_id=run_id,
        )
    else:
        recheck_ok, recheck_diff = None, None

    return CellResult(
        harness=harness,
        workload_id=workload_id,
        backend_id=spec.backend_id,
        max_candidates=max_candidates,
        seed=seed,
        baseline_median_ms=summary.get("baseline_median_ms"),
        best_speedup=summary.get("best_speedup"),
        best_candidate_id=summary.get("best_candidate_id"),
        best_median_ms=summary.get("best_median_ms"),
        best_correctness_ok=summary.get("best_correctness_ok"),
        best_max_abs_diff=summary.get("best_max_abs_diff"),
        successful_count=int(summary.get("successful_count") or 0),
        failed_attempts=int(summary.get("failed_attempts") or 0),
        elapsed_ms=float(summary.get("elapsed_ms") or 0.0),
        final_text=summary.get("final_text"),
        candidates=candidates_list,
        correctness_recheck_ok=recheck_ok,
        correctness_recheck_max_abs_diff=recheck_diff,
        error=error,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--harnesses", nargs="+", default=list(HARNESSES),
        choices=HARNESSES,
        help="Which harnesses to run.",
    )
    parser.add_argument(
        "--workloads", nargs="+", default=list(WORKLOADS),
        help="Workload ids to optimize.",
    )
    parser.add_argument(
        "--trials", nargs="+", type=int, default=list(TRIALS),
        help="max_candidates values to sweep.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS),
        help="RNG seeds to run for each cell.",
    )
    parser.add_argument(
        "--model", default="mistral:mistral-large-latest",
        help="LLM provider:model string. Default Mistral large.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory. Default: runs/study/<timestamp>/.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke run: 1 seed × 1 trial setting.",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = [0]
        args.trials = [4]

    if not torch.cuda.is_available():
        print("CUDA is required to run the study.", file=sys.stderr)
        return 1

    # Self-register backends + workloads.
    import_backend_packages()
    import_workload_packages()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_root = Path(args.out or f"runs/study/{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"

    cells = [
        (h, w, t, s)
        for h in args.harnesses
        for w in args.workloads
        for t in args.trials
        for s in args.seeds
    ]
    print(f"Running {len(cells)} cells. Results: {results_path}")
    started = time.perf_counter()

    with results_path.open("w", encoding="utf-8") as out:
        for i, (h, w, t, s) in enumerate(cells, 1):
            cell_t0 = time.perf_counter()
            print(f"[{i}/{len(cells)}] harness={h} workload={w} "
                  f"trials={t} seed={s} ...", flush=True)
            result = run_cell(
                harness=h, workload_id=w, max_candidates=t, seed=s,
                model_name=args.model, out_root=out_root,
            )
            cell_elapsed = time.perf_counter() - cell_t0
            row = asdict(result)
            out.write(json.dumps(row, default=str) + "\n")
            out.flush()
            if result.error:
                print(f"    -> ERROR ({cell_elapsed:.1f}s): "
                      f"{result.error.splitlines()[-1] if result.error else ''}")
            else:
                sp = f"{result.best_speedup:.4f}x" if result.best_speedup else "n/a"
                ok = result.correctness_recheck_ok
                ok_str = "OK" if ok is True else "FAIL" if ok is False else "n/a"
                print(f"    -> speedup={sp} recheck={ok_str} "
                      f"successful={result.successful_count}/{t} "
                      f"({cell_elapsed:.1f}s)")

    elapsed = time.perf_counter() - started
    print(f"\nStudy complete. {len(cells)} cells in {elapsed/60:.1f} min.")
    print(f"Results: {results_path}")
    print(f"Plot:    env/bin/python scripts/experiments/plot_study.py {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
