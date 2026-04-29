"""Example workload: masked elementwise copy Triton kernel.

Bandwidth-bound — surfaces TTGIR coalescing decisions distinct from the
arithmetic in `vector_add`. Same module-top-level kernel pattern: the
Triton compile path reads the kernel symbol off the imported module, so
the `@triton.jit` MUST live at module scope (not inside `build_workload`).
"""

from __future__ import annotations

from compilagent.core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload_safely

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover — triton-less envs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:
    @triton.jit
    def vector_copy_kernel(
        x_ptr,
        out_ptr,
        n,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x_vals = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x_vals, mask=mask)

    def _compilagent_compile_vector_copy(meta: dict) -> object:
        """Hook the Triton harness invokes during baseline / candidate compile.

        See `vector_add._compilagent_compile_vector_add` for rationale —
        the harness must launch through `[grid]`, not call the JIT directly.
        """

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for vector_copy compile.")
        n = int(meta.get("n_elements", 8_388_608))
        block_size = int(meta.get("BLOCK_SIZE", 1024))
        num_warps = int(meta.get("num_warps", 4))
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        def grid(grid_meta):
            return (triton.cdiv(n, grid_meta["BLOCK_SIZE"]),)

        handle = vector_copy_kernel[grid](
            x, out, n,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return handle

    vector_copy_kernel.compilagent_compile = _compilagent_compile_vector_copy


_SPEC = WorkloadSpec(
    id="vector_copy",
    title="Vector Copy",
    description=(
        "Bandwidth-bound elementwise copy Triton kernel — exposes TTGIR "
        "coalescing / load-cache-modifier decisions."
    ),
    kind=WorkloadKind.KERNEL,
    backend_id="triton",
    dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
    shape_policy=ShapePolicy(extra={"n_elements": 8_388_608}),
    tolerance=ToleranceConfig(atol=1e-7, rtol=1e-7),
    budget=BenchmarkBudget(warmup=5, repetitions=20, max_seconds=120.0),
    metadata={
        "source_path": __file__,
        "kernel_symbol": "vector_copy_kernel",
        "block_size": 1024,
        "num_warps": 4,
    },
)


@register_workload_safely(_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch

    if triton is None:
        raise RuntimeError("triton is not installed; vector_copy cannot be materialised.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vector_copy.")

    n = int(spec.shape_policy.extra.get("n_elements", 1024 * 1024))
    block_size = int(spec.metadata.get("block_size", 1024))
    num_warps = int(spec.metadata.get("num_warps", 4))
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[spec.dtype_policy.activation_dtype]

    x = torch.randn(n, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    def grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def forward() -> torch.Tensor:
        vector_copy_kernel[grid](
            x, out, n,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        return out

    return WorkloadInstance(
        spec=spec,
        forward=forward,
        example_inputs=(x,),
        metadata={"output_buffer": out, "n_elements": n},
    )
