"""Phase-9 deterministic verification.

Drives the full backend-agnostic optimization loop on `vit_block` *without*
the LLM in the loop, using the derived search-space directly. For each
high-signal lever returned by `Backend.derive_search_space(workload, analysis)`,
we synthesize a small set of `Intervention`s, compile + time, and report the
best speedup.

The script proves the agent-driven path **could** win on this workload — the
LLM run uses the same primitives. Run with::

    env/bin/python scripts/phase9_victory.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import tempfile
from typing import Any

import torch  # noqa: F401 — must be imported before triton/inductor

# Self-register backends + workloads
import compilagent_triton.backends.triton  # noqa: F401
import compilagent_triton.backends.torch_inductor  # noqa: F401
from compilagent_triton.backends import Plan, backend_registry
from compilagent_triton.backends.base import Intervention, Target, ToleranceConfig
from compilagent_triton.workloads.registry import (
    import_workload_packages,
    workload_registry,
)


def _candidate_payloads(lever: Any) -> list[Any]:
    """Pick a small set of payload values to try for a given lever."""

    rng = lever.range
    kind = type(rng).__name__
    if kind == "BooleanFlag":
        return [True, False]
    if kind == "IntRange":
        return list(rng.candidates[:6])
    if kind == "FloatRange":
        return list(rng.candidates[:4])
    if kind == "EnumChoice":
        return list(rng.candidates[:6])
    if kind == "StructuredJsonRange":
        # Structured payloads are agent-territory; deterministically skip.
        return []
    return []


def _is_safe_knob(lever: Any) -> bool:
    """Filter out levers that don't make sense to flip blindly.

    The deterministic loop avoids levers where flipping them would require the
    agent to also supply an additional setup (e.g. fx_node rewrites need a
    callable; `lowering` swaps need a replacement function). Knobs are safe.
    """

    return lever.target_kind in {"knob"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 deterministic verification")
    parser.add_argument("--workload", default="vit_block")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()

    import_workload_packages()
    spec = workload_registry.get_spec(args.workload)
    backend = backend_registry.get(spec.backend_id)

    print(f"workload      : {spec.id} (kind={spec.kind.value}, backend={spec.backend_id})")
    print(f"device        : {backend.device_capability().name} ({backend.device_capability().arch})")

    workspace = Path(tempfile.mkdtemp(prefix="phase9-")) / ".compilagent-triton"
    workspace.mkdir(parents=True)

    # ---- baseline ----
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    b_dir = workspace / "baseline"
    b_dir.mkdir(parents=True)
    b_compile = backend.compile(spec, Plan(), artifact_dir=b_dir)
    b_time = backend.time_workload(
        spec, Plan(),
        warmup=args.warmup, repetitions=args.repetitions,
        max_seconds=spec.budget.max_seconds,
    )
    print()
    print(f"BASELINE      : ok={b_compile.ok} median={b_time.median_ms:.4f}ms "
          f"p20={b_time.p20_ms:.4f} p80={b_time.p80_ms:.4f}")

    # ---- derive ----
    artifacts = []
    for attr in ("output_code_path", "schedule_log_path", "fx_graph_path"):
        p = getattr(b_compile, attr, None)
        if p:
            artifacts.append(p)
    artifacts.extend(getattr(b_compile, "artifacts", ()) or ())
    analysis = backend.analyze(spec, baseline_artifacts=artifacts)
    search_space = backend.derive_search_space(spec, analysis)
    knob_levers = [lev for lev in search_space.levers if _is_safe_knob(lev)]
    print()
    print(f"derived levers: {len(search_space.levers)} total, "
          f"{len(knob_levers)} safe-knob")

    # ---- evaluate top knobs ----
    candidates: list[dict[str, Any]] = []
    examined = 0
    for lever in knob_levers:
        if examined >= args.max_candidates:
            break
        payloads = _candidate_payloads(lever)
        # Skip the payload that equals the default — that's the baseline.
        payloads = [p for p in payloads if p != lever.default]
        if not payloads:
            continue
        # Try the first non-default value.
        payload = payloads[0]
        examined += 1
        cid = f"cand-{examined:02d}-{lever.id.replace(':', '_')}"
        plan = Plan(interventions=(Intervention(
            target=Target(kind=lever.target_kind, selector=lever.target_selector),
            payload=payload,
            rationale=lever.evidence.signal,
        ),))
        # Memory hygiene
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        cdir = workspace / "candidates" / cid
        cdir.mkdir(parents=True)
        try:
            t0 = time.perf_counter()
            compile_res = backend.compile(spec, plan, artifact_dir=cdir)
            timing = backend.time_workload(
                spec, plan,
                warmup=args.warmup, repetitions=args.repetitions,
                max_seconds=spec.budget.max_seconds,
            ) if compile_res.ok else None
            elapsed = (time.perf_counter() - t0) * 1000.0
            speedup = (
                b_time.median_ms / timing.median_ms
                if timing and timing.median_ms and b_time.median_ms else None
            )
            row = {
                "candidate_id": cid,
                "lever": lever.id,
                "target": f"{lever.target_kind}({lever.target_selector})",
                "payload": payload,
                "compile_ok": getattr(compile_res, "ok", False),
                "median_ms": timing.median_ms if timing else None,
                "speedup": speedup,
                "elapsed_total_ms": elapsed,
            }
            candidates.append(row)
            sp_str = f"{speedup:.4f}x" if speedup else "n/a"
            med = f"{timing.median_ms:.4f}" if timing and timing.median_ms else "n/a"
            print(f"  [{examined:02d}] {lever.id:<58s} = {payload!s:<6s} "
                  f"-> median={med:<10s} speedup={sp_str}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{examined:02d}] {lever.id:<58s} = {payload!s:<6s} -> ERROR {exc!r}")
            candidates.append({
                "candidate_id": cid, "lever": lever.id, "payload": payload,
                "error": repr(exc),
            })

    # ---- summary ----
    measured = [c for c in candidates if isinstance(c.get("speedup"), (int, float))]
    measured.sort(key=lambda r: r["speedup"], reverse=True)
    print()
    print("=== TOP CANDIDATES ===")
    for row in measured[:5]:
        sp_str = f"{row['speedup']:.4f}x"
        print(f"  {sp_str:>10s}  {row['lever']:<58s}  payload={row['payload']!r}")
    print()
    if measured and measured[0]["speedup"] >= 1.05:
        print(f"PHASE 9 VICTORY: best speedup = {measured[0]['speedup']:.4f}x "
              f"on {measured[0]['lever']!r} (target was >= 1.05x)")
        return 0
    elif measured and measured[0]["speedup"] >= 1.01:
        print(f"PHASE 9 PASS:    best speedup = {measured[0]['speedup']:.4f}x "
              f"(>= 1.01x but below 1.05 stretch goal); winner: {measured[0]['lever']!r}")
        return 0
    else:
        print("No candidate beat baseline ≥ 1.01x.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
