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
def copy_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    vals = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)


def _compilagent_compile_copy(meta: dict) -> object:
    """One-shot launch returning the Triton handle for the agent compile path."""

    import torch

    n = int(meta.get("n_elements", 1024 * 1024))
    block_size = int(meta.get("BLOCK_SIZE", 1024))
    num_warps = int(meta.get("num_warps", 4))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for copy compile.")
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    handle = copy_kernel[grid](x, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    return handle


copy_kernel.compilagent_compile = _compilagent_compile_copy


@dataclass(frozen=True, slots=True)
class CopyResult:
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


def run_copy_sweep(
    *,
    n_elements: int = 8_388_608,
    block_sizes: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
    num_warps_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    repetitions: int = 100,
    warmup: int = 20,
    device: str = "cuda",
) -> list[CopyResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for vector-copy sweep.")
    x = torch.randn(n_elements, device=device)
    out = torch.empty_like(x)
    results: list[CopyResult] = []
    for block_size in block_sizes:
        for num_warps in num_warps_values:

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            def run_once(block_size: int = block_size, num_warps: int = num_warps) -> None:
                copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
                torch.cuda.synchronize()

            run_once()
            correctness = bool(torch.allclose(out, x))
            if not correctness:
                results.append(
                    CopyResult(
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
                CopyResult(
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
        CopyResult(
            **{
                **asdict(result),
                "speedup_vs_baseline": baseline.median_ms / result.median_ms
                if result.correctness and result.median_ms > 0
                else None,
            }
        )
        for result in results
    ]


def render_copy_report(results: list[CopyResult]) -> str:
    best = min((result for result in results if result.correctness), key=lambda item: item.median_ms)
    lines = [
        "# Vector Copy Sweep",
        "",
        f"- generated: {datetime.now(UTC).isoformat()}",
        f"- best candidate: `{best.candidate_id}`",
        f"- best median: {best.median_ms:.6f} ms",
        f"- speedup vs baseline: {best.speedup_vs_baseline:.4f}x"
        if best.speedup_vs_baseline is not None
        else "- speedup vs baseline: n/a",
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
    parser = argparse.ArgumentParser(description="Run vector-copy Triton benchmark sweeps.")
    parser.add_argument("--n-elements", type=int, default=8_388_608)
    parser.add_argument("--block-sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--num-warps", default="1,2,4,8,16")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path(".compilagent-triton/reports"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"vector-copy-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    trace_store = TraceStore(args.output_dir.parent).ensure()
    start_payload = {
        "run_id": run_id,
        "family": "vector_copy",
        "n_elements": args.n_elements,
        "block_sizes": args.block_sizes,
        "num_warps": args.num_warps,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
    }
    trace_store.emit("run.started", payload=start_payload)
    trace_store.emit("benchmark.started", payload=start_payload)
    started = time.perf_counter()
    try:
        results = run_copy_sweep(
            n_elements=args.n_elements,
            block_sizes=_parse_int_tuple(args.block_sizes),
            num_warps_values=_parse_int_tuple(args.num_warps),
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
    except Exception as exc:
        trace_store.emit(
            "run.failed",
            payload={**start_payload, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    stem = f"{run_id}_n{args.n_elements}"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    md_path.write_text(render_copy_report(results), encoding="utf-8")
    best = min((result for result in results if result.correctness), key=lambda item: item.median_ms)
    trace_store.emit(
        "benchmark.completed",
        payload={
            "run_id": run_id,
            "family": "vector_copy",
            "best": asdict(best),
            "candidate_count": len(results),
            "results": [asdict(result) for result in results],
            "elapsed_ms": elapsed_ms,
        },
        artifact_paths=[str(json_path), str(md_path)],
    )
    trace_store.emit(
        "comparison.created",
        payload={
            "run_id": run_id,
            "family": "vector_copy",
            "candidate_id": best.candidate_id,
            "speedup_vs_baseline": best.speedup_vs_baseline,
            "delta_percent": (best.speedup_vs_baseline - 1) * 100
            if best.speedup_vs_baseline is not None
            else None,
            "conclusion": _comparison_conclusion(best.speedup_vs_baseline),
        },
    )
    trace_store.emit("artifact.created", payload={"path": str(json_path)}, artifact_paths=[str(json_path)])
    trace_store.emit("artifact.created", payload={"path": str(md_path)}, artifact_paths=[str(md_path)])
    trace_store.emit(
        "loop.summary",
        payload={
            "run_id": run_id,
            "family": "vector_copy",
            "best_candidate_id": best.candidate_id,
            "best_median_ms": best.median_ms,
            "speedup_vs_baseline": best.speedup_vs_baseline,
            "elapsed_ms": elapsed_ms,
        },
        artifact_paths=[str(json_path), str(md_path)],
    )
    trace_store.emit("run.completed", payload={**start_payload, "elapsed_ms": elapsed_ms})
    print(render_copy_report(results))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _candidate_id(n_elements: int, block_size: int, num_warps: int) -> str:
    return f"vector-copy-n{n_elements}-b{block_size}-w{num_warps}"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * fraction), 0), len(ordered) - 1)
    return ordered[index]


def _bandwidth_gbps(n_elements: int, median_ms: float) -> float | None:
    if median_ms <= 0:
        return None
    bytes_moved = n_elements * 2 * 4
    return bytes_moved / (median_ms / 1000) / 1_000_000_000


def _fmt(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _comparison_conclusion(speedup: float | None) -> str:
    if speedup is None:
        return "no baseline"
    if speedup > 1.02:
        return "candidate improved baseline"
    if speedup < 0.98:
        return "candidate regressed baseline"
    return "within noise band"


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


_VECTOR_COPY_SPEC = WorkloadSpec(
    id="vector_copy",
    title="Vector Copy",
    description="Contiguous vector copy Triton kernel.",
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    entrypoint="compilagent_triton.workloads.triton_kernels.vector_copy:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_elements": 8_388_608}),
    tolerance=ToleranceConfig(atol=0.0, rtol=0.0, notes="exact: copy is bit-identical"),
    budget=BenchmarkBudget(warmup=20, repetitions=100, max_seconds=120),
    metadata={"kernel_symbol": "copy_kernel", "source_path": __file__},
)


@register_workload(_VECTOR_COPY_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch

    n = int(spec.shape_policy.extra.get("n_elements", 1024 * 1024))
    block_size = int(spec.metadata.get("block_size", 1024))
    num_warps = int(spec.metadata.get("num_warps", 4))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vector_copy.")
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def forward():
        copy_kernel[grid](x, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec, forward=forward, example_inputs=(x,),
        metadata={"output_buffer": out, "n_elements": n},
    )
