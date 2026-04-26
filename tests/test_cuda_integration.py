from __future__ import annotations

import pytest

from compilagent_triton.benchmarking import BenchmarkBudget, run_callable_benchmark

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vector_add_cuda_benchmark_smoke() -> None:
    @triton.jit
    def vector_add_kernel(x, y, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        out_vals = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
        tl.store(out + offsets, out_vals, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    expected = x + y

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def run_kernel() -> None:
        vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=128)
        torch.cuda.synchronize()

    def check_correctness() -> bool:
        run_kernel()
        return bool(torch.allclose(out, expected))

    result = run_callable_benchmark(
        kernel_id="cuda-vector-add",
        fn=run_kernel,
        correctness_check=check_correctness,
        budget=BenchmarkBudget(warmup=1, repetitions=2, max_seconds=30),
    )

    assert result.correctness == "passed"
    assert result.median_ms is not None
