from __future__ import annotations

from compilagent_triton.benchmarking import (
    BenchmarkBudget,
    compare_to_baseline,
    run_callable_benchmark,
)
from compilagent_triton.schemas import BenchmarkResult


def test_callable_benchmark_records_timings() -> None:
    result = run_callable_benchmark(
        kernel_id="k",
        fn=lambda: sum(range(10)),
        correctness_check=lambda: True,
        budget=BenchmarkBudget(warmup=1, repetitions=2, max_seconds=10),
    )

    assert result.correctness == "passed"
    assert result.compile_ok
    assert len(result.timings_ms) == 2
    assert result.median_ms is not None


def test_compare_to_baseline_sets_speedup() -> None:
    baseline = BenchmarkResult(
        kernel_id="k",
        correctness="passed",
        compile_ok=True,
        timings_ms=[10.0],
        median_ms=10.0,
    )
    candidate = BenchmarkResult(
        kernel_id="k",
        candidate_id="c",
        correctness="passed",
        compile_ok=True,
        timings_ms=[5.0],
        median_ms=5.0,
    )

    compared = compare_to_baseline(baseline, candidate)

    assert compared.speedup_vs_baseline == 2.0
