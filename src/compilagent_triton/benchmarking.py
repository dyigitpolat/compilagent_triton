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


def time_with_cuda_events(fn: Callable[[], Any]) -> float:
    """Time `fn()` end-to-end using CUDA events; returns ms.

    Falls back to `time.perf_counter()` when CUDA is unavailable. Synchronizes
    around the call to defeat asynchronous launch noise.
    """

    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end))
    except Exception:
        pass
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def device_peak_bandwidth_gbps() -> float | None:
    """Estimate the active CUDA device's HBM peak bandwidth in GB/s.

    Used as the denominator for the roofline model. Returns None on CPU.
    """

    try:
        import torch  # type: ignore[import-not-found]
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        # memory_clock_rate is in kHz, memory_bus_width in bits
        mc_khz = float(getattr(props, "memory_clock_rate", 0) or 0)
        bus_bits = float(getattr(props, "memory_bus_width", 0) or 0)
        if mc_khz <= 0 or bus_bits <= 0:
            return None
        # GDDR / HBM rates are double-data-rate; bandwidth = 2 * clock * bus / 8
        # Output GB/s
        return 2.0 * mc_khz * 1e3 * bus_bits / 8.0 / 1e9
    except Exception:
        return None


def compute_profile_metrics(
    *,
    median_ms: float | None,
    bytes_moved: int | None,
    flops: int | None = None,
) -> dict[str, Any]:
    """Build a profile_metrics dict from a measured median + workload bytes."""

    metrics: dict[str, Any] = {}
    if median_ms is None or median_ms <= 0:
        return metrics
    seconds = median_ms / 1000.0
    if bytes_moved is not None and bytes_moved > 0:
        achieved_gbps = bytes_moved / 1e9 / seconds
        metrics["achieved_bandwidth_gbps"] = achieved_gbps
        peak = device_peak_bandwidth_gbps()
        if peak and peak > 0:
            metrics["roofline_bandwidth_ratio"] = achieved_gbps / peak
            metrics["device_peak_bandwidth_gbps"] = peak
    if flops is not None and flops > 0:
        metrics["achieved_tflops"] = flops / 1e12 / seconds
    return metrics


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
            timings.append(time_with_cuda_events(fn))
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
