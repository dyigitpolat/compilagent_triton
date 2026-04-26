from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
import triton.language as tl

import triton

from ...trace_store import TraceStore


@triton.jit
def reduction_sum_kernel(x_ptr, partials_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    reduced = tl.sum(vals, axis=0)
    tl.store(partials_ptr + pid, reduced)


def _compilagent_compile_reduction(meta: dict) -> object:
    """One-shot launch returning the Triton handle for the agent compile path."""

    import torch

    n = int(meta.get("n_elements", 1024 * 1024))
    block_size = int(meta.get("BLOCK_SIZE", 1024))
    num_warps = int(meta.get("num_warps", 4))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for reduction compile.")
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    grid_size = triton.cdiv(n, block_size)
    partials = torch.empty(grid_size, device="cuda", dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    handle = reduction_sum_kernel[grid](
        x, partials, n, BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return handle


reduction_sum_kernel.compilagent_compile = _compilagent_compile_reduction


@dataclass(frozen=True, slots=True)
class ReductionResult:
    candidate_id: str
    n_elements: int
    block_size: int
    num_warps: int
    correctness: bool
    median_ms: float
    p20_ms: float | None
    p80_ms: float | None
    bandwidth_gbps: float | None
    speedup_vs_baseline: float | None = None


def run_reduction_sweep(
    *,
    n_elements: int = 8_388_608,
    block_sizes: tuple[int, ...] = (256, 512, 1024, 2048),
    num_warps_values: tuple[int, ...] = (4, 8),
    repetitions: int = 50,
    warmup: int = 10,
    device: str = "cuda",
) -> list[ReductionResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for reduction sweep.")
    x = torch.randn(n_elements, device=device)
    reference = x.sum()
    results: list[ReductionResult] = []
    for block_size in block_sizes:
        grid_size = triton.cdiv(n_elements, block_size)
        partials = torch.empty((grid_size,), device=device, dtype=x.dtype)
        for num_warps in num_warps_values:

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            def run_once(
                block_size: int = block_size,
                num_warps: int = num_warps,
                partials: torch.Tensor = partials,
            ) -> torch.Tensor:
                reduction_sum_kernel[grid](
                    x,
                    partials,
                    n_elements,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
                torch.cuda.synchronize()
                return partials.sum()

            observed = run_once()
            correctness = bool(torch.allclose(observed, reference, rtol=1e-3, atol=1e-3))
            if not correctness:
                results.append(
                    ReductionResult(
                        candidate_id=_candidate_id(n_elements, block_size, num_warps),
                        n_elements=n_elements,
                        block_size=block_size,
                        num_warps=num_warps,
                        correctness=False,
                        median_ms=float("inf"),
                        p20_ms=None,
                        p80_ms=None,
                        bandwidth_gbps=None,
                    )
                )
                continue
            for _ in range(warmup):
                run_once()
            timings = []
            for _ in range(repetitions):
                start = time.perf_counter()
                run_once()
                timings.append((time.perf_counter() - start) * 1000)
            median = statistics.median(timings)
            results.append(
                ReductionResult(
                    candidate_id=_candidate_id(n_elements, block_size, num_warps),
                    n_elements=n_elements,
                    block_size=block_size,
                    num_warps=num_warps,
                    correctness=True,
                    median_ms=median,
                    p20_ms=_percentile(timings, 0.20),
                    p80_ms=_percentile(timings, 0.80),
                    bandwidth_gbps=_bandwidth_gbps(n_elements, median),
                )
            )
    baseline = next(
        (
            result
            for result in results
            if result.block_size == 1024 and result.num_warps == 4 and result.correctness
        ),
        None,
    )
    if baseline is None:
        baseline = next((result for result in results if result.correctness), None)
    if baseline is None or baseline.median_ms <= 0:
        return results
    return [
        ReductionResult(
            **{
                **asdict(result),
                "speedup_vs_baseline": baseline.median_ms / result.median_ms
                if result.correctness and result.median_ms > 0
                else None,
            }
        )
        for result in results
    ]


def render_reduction_report(results: list[ReductionResult]) -> str:
    best = min((result for result in results if result.correctness), key=lambda item: item.median_ms)
    lines = [
        "# Reduction Sweep",
        "",
        f"- generated: {datetime.now(UTC).isoformat()}",
        f"- best candidate: `{best.candidate_id}`",
        f"- best median: {best.median_ms:.6f} ms",
        f"- speedup vs baseline: {_fmt(best.speedup_vs_baseline, digits=4)}x",
        "",
        "| candidate | block | warps | correct | p20 ms | median ms | p80 ms | GB/s | speedup |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: item.median_ms):
        lines.append(
            f"| `{result.candidate_id}` | {result.block_size} | {result.num_warps} | "
            f"{result.correctness} | {_fmt(result.p20_ms)} | {result.median_ms:.6f} | "
            f"{_fmt(result.p80_ms)} | {_fmt(result.bandwidth_gbps, digits=1)} | "
            f"{_fmt(result.speedup_vs_baseline, digits=4)}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Triton reduction benchmark sweeps.")
    parser.add_argument("--n-elements", type=int, default=8_388_608)
    parser.add_argument("--block-sizes", default="256,512,1024,2048")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--workspace", type=Path, default=Path(".compilagent-triton"))
    args = parser.parse_args()
    run_id = f"reduction-{uuid4().hex[:8]}"
    results = run_reduction_sweep(
        n_elements=args.n_elements,
        block_sizes=_parse_ints(args.block_sizes),
        num_warps_values=_parse_ints(args.num_warps),
        repetitions=args.repetitions,
        warmup=args.warmup,
    )
    args.workspace.mkdir(parents=True, exist_ok=True)
    report_path = args.workspace / f"{run_id}.md"
    json_path = args.workspace / f"{run_id}.json"
    report_path.write_text(render_reduction_report(results), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    TraceStore(args.workspace).ensure().emit(
        "benchmark.completed",
        payload={
            "run_id": run_id,
            "family": "reduction",
            "results": [asdict(result) for result in results],
        },
        artifact_paths=[str(report_path), str(json_path)],
    )


def _candidate_id(n_elements: int, block_size: int, num_warps: int) -> str:
    return f"red-n{n_elements}-b{block_size}-w{num_warps}"


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _bandwidth_gbps(n_elements: int, median_ms: float) -> float | None:
    if median_ms <= 0:
        return None
    bytes_read = n_elements * 4
    return bytes_read / (median_ms / 1000) / 1e9


def _fmt(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    main()


# --- workload registration ---------------------------------------------------

from ...core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from ..registry import register_workload


_REDUCTION_SPEC = WorkloadSpec(
    id="reduction_sum",
    title="Reduction Sum",
    description="Block-level sum reduction Triton kernel.",
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.reduction_sum:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_elements": 8_388_608}),
    tolerance=ToleranceConfig(atol=1e-4, rtol=1e-4),
    budget=BenchmarkBudget(warmup=10, repetitions=50, max_seconds=120),
    metadata={"kernel_symbol": "reduction_sum_kernel", "source_path": __file__},
)


@register_workload(_REDUCTION_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch

    n = int(spec.shape_policy.extra.get("n_elements", 1024 * 1024))
    block_size = int(spec.metadata.get("block_size", 1024))
    num_warps = int(spec.metadata.get("num_warps", 4))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise reduction_sum.")
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    grid_size = triton.cdiv(n, block_size)
    partials = torch.empty(grid_size, device="cuda", dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def forward():
        reduction_sum_kernel[grid](
            x, partials, n, BLOCK_SIZE=block_size, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return partials

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x,),
        metadata={"output_buffer": partials, "n_elements": n},
    )
