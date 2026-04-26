from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schemas import BenchmarkResult


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    warmup: int = 3
    repetitions: int = 10
    max_seconds: int = 120
    noise_threshold_pct: float = 2.0


def run_callable_benchmark(
    *,
    kernel_id: str,
    fn: Callable[[], Any],
    correctness_check: Callable[[], bool] | None = None,
    candidate_id: str | None = None,
    budget: BenchmarkBudget | None = None,
) -> BenchmarkResult:
    budget = budget or BenchmarkBudget()
    try:
        correctness = "skipped"
        if correctness_check is not None:
            correctness = "passed" if correctness_check() else "failed"
        if correctness == "failed":
            return BenchmarkResult(
                kernel_id=kernel_id,
                candidate_id=candidate_id,
                correctness=correctness,
                compile_ok=True,
                diagnostics="Correctness check failed; timing skipped.",
            )
        for _ in range(budget.warmup):
            fn()
        timings: list[float] = []
        start_budget = time.perf_counter()
        for _ in range(budget.repetitions):
            if time.perf_counter() - start_budget > budget.max_seconds:
                break
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000)
        median_ms = statistics.median(timings) if timings else None
        return BenchmarkResult(
            kernel_id=kernel_id,
            candidate_id=candidate_id,
            correctness=correctness,
            compile_ok=True,
            timings_ms=timings,
            median_ms=median_ms,
        )
    except Exception as exc:  # pragma: no cover - diagnostics path
        return BenchmarkResult(
            kernel_id=kernel_id,
            candidate_id=candidate_id,
            correctness="failed",
            compile_ok=False,
            diagnostics=f"{exc.__class__.__name__}: {exc}",
        )


def compare_to_baseline(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    noise_threshold_pct: float = 2.0,
) -> BenchmarkResult:
    if baseline.median_ms is None or candidate.median_ms is None or candidate.median_ms <= 0:
        return candidate.model_copy(update={"diagnostics": "Missing median timing for comparison."})
    speedup = baseline.median_ms / candidate.median_ms
    delta_pct = (speedup - 1.0) * 100
    diagnostics = candidate.diagnostics
    if abs(delta_pct) < noise_threshold_pct:
        diagnostics = (
            f"Observed delta {delta_pct:.2f}% is within noise threshold "
            f"{noise_threshold_pct:.2f}%."
        )
    return candidate.model_copy(
        update={
            "speedup_vs_baseline": speedup,
            "diagnostics": diagnostics,
        }
    )
