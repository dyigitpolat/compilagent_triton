from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .trace_store import TraceStore

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - GPU-only dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:
    @triton.jit
    def vector_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        LOAD_CACHE_MODIFIER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        if LOAD_CACHE_MODIFIER == "":
            x_vals = tl.load(x_ptr + offsets, mask=mask)
            y_vals = tl.load(y_ptr + offsets, mask=mask)
        else:
            x_vals = tl.load(x_ptr + offsets, mask=mask, cache_modifier=LOAD_CACHE_MODIFIER)
            y_vals = tl.load(y_ptr + offsets, mask=mask, cache_modifier=LOAD_CACHE_MODIFIER)
        out_vals = x_vals + y_vals
        tl.store(out_ptr + offsets, out_vals, mask=mask)

    def _compilagent_compile_vector_add(meta: dict) -> object:
        """Run a single vector_add launch and return the Triton kernel handle.

        Used by `OptimizerRuntime.compile_baseline` / `run_candidate` so the
        agent gets a real `.asm` dict (TTIR / TTGIR / LLIR / PTX) without
        running a full sweep.
        """

        import torch

        n = int(meta.get("n_elements", 1024 * 1024))
        block_size = int(meta.get("BLOCK_SIZE", 1024))
        num_warps = int(meta.get("num_warps", 4))
        cache_mod = str(meta.get("LOAD_CACHE_MODIFIER", ""))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for vector_add compile.")
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        handle = vector_add_kernel[grid](
            x, y, out, n,
            BLOCK_SIZE=block_size,
            LOAD_CACHE_MODIFIER=cache_mod,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return handle

    vector_add_kernel.compilagent_compile = _compilagent_compile_vector_add


@dataclass(frozen=True, slots=True)
class VectorAddCandidate:
    n_elements: int
    block_size: int
    num_warps: int
    load_cache_modifier: str = ""

    @property
    def id(self) -> str:
        cache = self.load_cache_modifier.replace(".", "") or "default"
        return f"vector-add-n{self.n_elements}-b{self.block_size}-w{self.num_warps}-load{cache}"


@dataclass(frozen=True, slots=True)
class VectorAddResult:
    candidate_id: str
    n_elements: int
    block_size: int
    num_warps: int
    load_cache_modifier: str
    correctness: bool
    timings_ms: list[float]
    median_ms: float
    p20_ms: float | None = None
    p80_ms: float | None = None
    bandwidth_gbps: float | None = None
    speedup_vs_baseline: float | None = None


def run_vector_add_sweep(
    *,
    n_elements: int = 8_388_608,
    block_sizes: tuple[int, ...] = (128, 256, 512, 1024, 2048),
    num_warps_values: tuple[int, ...] = (4, 8),
    load_cache_modifiers: tuple[str, ...] = ("",),
    repetitions: int = 20,
    warmup: int = 5,
    device: str = "cuda",
) -> list[VectorAddResult]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for vector-add sweep.")

    x = torch.randn(n_elements, device=device)
    y = torch.randn(n_elements, device=device)
    out = torch.empty_like(x)
    expected = x + y
    results: list[VectorAddResult] = []

    for block_size in block_sizes:
        for num_warps in num_warps_values:
            for load_cache_modifier in load_cache_modifiers:
                candidate = VectorAddCandidate(
                    n_elements=n_elements,
                    block_size=block_size,
                    num_warps=num_warps,
                    load_cache_modifier=load_cache_modifier,
                )

                def grid(meta):
                    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                def run_once(
                    block_size: int = block_size,
                    num_warps: int = num_warps,
                    load_cache_modifier: str = load_cache_modifier,
                ) -> None:
                    vector_add_kernel[grid](
                        x,
                        y,
                        out,
                        n_elements,
                        BLOCK_SIZE=block_size,
                        LOAD_CACHE_MODIFIER=load_cache_modifier,
                        num_warps=num_warps,
                    )
                    torch.cuda.synchronize()

                run_once()
                correctness = bool(torch.allclose(out, expected))
                if not correctness:
                    results.append(
                        VectorAddResult(
                            candidate_id=candidate.id,
                            n_elements=n_elements,
                            block_size=block_size,
                            num_warps=num_warps,
                            load_cache_modifier=load_cache_modifier,
                            correctness=False,
                            timings_ms=[],
                            median_ms=float("inf"),
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
                results.append(
                    VectorAddResult(
                        candidate_id=candidate.id,
                        n_elements=n_elements,
                        block_size=block_size,
                        num_warps=num_warps,
                        load_cache_modifier=load_cache_modifier,
                        correctness=True,
                        timings_ms=timings,
                        median_ms=statistics.median(timings),
                        p20_ms=_percentile(timings, 0.20),
                        p80_ms=_percentile(timings, 0.80),
                        bandwidth_gbps=_bandwidth_gbps(n_elements, statistics.median(timings)),
                    )
                )

    baseline = next(
        (
            result
            for result in results
            if result.block_size == 1024
            and result.num_warps == 4
            and result.load_cache_modifier == ""
            and result.correctness
        ),
        None,
    )
    if baseline is None:
        baseline = next((result for result in results if result.correctness), None)
    if baseline is None or baseline.median_ms <= 0:
        return results
    return [
        VectorAddResult(
            **{
                **asdict(result),
                "speedup_vs_baseline": baseline.median_ms / result.median_ms
                if result.correctness and result.median_ms > 0
                else None,
            }
        )
        for result in results
    ]


def render_vector_add_report(results: list[VectorAddResult]) -> str:
    if not results:
        return "No vector-add benchmark results were recorded.\n"
    best = min((result for result in results if result.correctness), key=lambda item: item.median_ms)
    lines = [
        "# Vector Add Sweep",
        "",
        f"- generated: {datetime.now(UTC).isoformat()}",
        f"- best candidate: `{best.candidate_id}`",
        f"- best median: {best.median_ms:.6f} ms",
        f"- speedup vs baseline: {best.speedup_vs_baseline:.4f}x"
        if best.speedup_vs_baseline is not None
        else "- speedup vs baseline: n/a",
        "",
        "| candidate | block | warps | load cache | correct | p20 ms | median ms | p80 ms | GB/s | speedup |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: item.median_ms):
        speedup = (
            f"{result.speedup_vs_baseline:.4f}x"
            if result.speedup_vs_baseline is not None
            else "n/a"
        )
        p20 = f"{result.p20_ms:.6f}" if result.p20_ms is not None else "n/a"
        p80 = f"{result.p80_ms:.6f}" if result.p80_ms is not None else "n/a"
        bandwidth = f"{result.bandwidth_gbps:.1f}" if result.bandwidth_gbps is not None else "n/a"
        lines.append(
            f"| `{result.candidate_id}` | {result.block_size} | {result.num_warps} | "
            f"`{result.load_cache_modifier or 'default'}` | {result.correctness} | "
            f"{p20} | {result.median_ms:.6f} | {p80} | "
            f"{bandwidth} | {speedup} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Compilagent Triton GPU benchmark sweeps.")
    parser.add_argument("--n-elements", type=int, default=8_388_608)
    parser.add_argument("--block-sizes", default="128,256,512,1024,2048")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--load-cache-modifiers", default="none")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(".compilagent-triton/reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"vector-add-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    trace_store = TraceStore(args.output_dir.parent).ensure()
    start_payload = {
        "run_id": run_id,
        "family": "vector_add",
        "n_elements": args.n_elements,
        "block_sizes": args.block_sizes,
        "num_warps": args.num_warps,
        "load_cache_modifiers": args.load_cache_modifiers,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
    }
    trace_store.emit("run.started", payload=start_payload)
    trace_store.emit("benchmark.started", payload=start_payload)
    started = time.perf_counter()
    try:
        results = run_vector_add_sweep(
            n_elements=args.n_elements,
            block_sizes=_parse_int_tuple(args.block_sizes),
            num_warps_values=_parse_int_tuple(args.num_warps),
            load_cache_modifiers=_parse_cache_modifiers(args.load_cache_modifiers),
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
    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_vector_add_report(results), encoding="utf-8")
    best = min((result for result in results if result.correctness), key=lambda item: item.median_ms)
    trace_store.emit(
        "benchmark.completed",
        payload={
            "run_id": run_id,
            "family": "vector_add",
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
            "family": "vector_add",
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
            "family": "vector_add",
            "best_candidate_id": best.candidate_id,
            "best_median_ms": best.median_ms,
            "speedup_vs_baseline": best.speedup_vs_baseline,
            "elapsed_ms": elapsed_ms,
        },
        artifact_paths=[str(json_path), str(md_path)],
    )
    trace_store.emit("run.completed", payload={**start_payload, "elapsed_ms": elapsed_ms})
    print(render_vector_add_report(results))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_cache_modifiers(value: str) -> tuple[str, ...]:
    aliases = {"none": "", "default": "", "": ""}
    parsed = tuple(aliases.get(part.strip(), part.strip()) for part in value.split(","))
    if not parsed:
        raise ValueError("expected at least one cache modifier")
    invalid = [item for item in parsed if item not in {"", ".ca", ".cg"}]
    if invalid:
        raise ValueError(f"unsupported cache modifiers: {', '.join(invalid)}")
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
    bytes_moved = n_elements * 3 * 4
    return bytes_moved / (median_ms / 1000) / 1_000_000_000


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
