"""ViT-B/16 full forward pass on ImageNet-shape inputs.

Builds `torchvision.models.vit_b_16(weights=None)` with `torch.manual_seed(0)`
for deterministic random init, runs forward on `[B, 3, 224, 224]` bf16 inputs.
The inductor backend compiles the full model — Inductor lowers ~75 kernels for
a typical ViT-B/16 forward pass; the agent's per-kernel autotune lever (Phase 6)
operates on those.

Random weights are fine: numerical correctness is verified by comparing the
baseline-compiled vs. candidate-compiled forward outputs under a bf16 tolerance.
The fp32 reference is recorded for reporting but not gating.
"""

from __future__ import annotations

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


_VIT_B16_SPEC = WorkloadSpec(
    id="vit_b_16",
    title="ViT-B/16 full forward",
    description=(
        "torchvision.models.vit_b_16 forward pass on ImageNet-shape inputs "
        "(B=32, 3x224x224, bf16). Random weights, seeded. ~86M params, ~75 "
        "Inductor-generated kernels per forward."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.vit_b_16:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=32,
        image_size=(224, 224),
        sequence_length=197,
        extra={"channels": 3, "num_classes": 1000},
    ),
    tolerance=ToleranceConfig(
        atol=5e-4, rtol=5e-3,
        notes=(
            "bf16-vs-bf16 logits tolerance. fp32 reference is recorded "
            "(atol=2e-2, rtol=2e-2) but non-gating."
        ),
    ),
    budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=120.0),
    metadata={"seed": 0, "model_family": "vit", "variant": "b_16"},
)


@register_workload(_VIT_B16_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    from torchvision.models import vit_b_16

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vit_b_16.")
    seed = int(spec.metadata.get("seed", 0))
    torch.manual_seed(seed)

    activation_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[spec.dtype_policy.activation_dtype]
    param_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[spec.dtype_policy.param_dtype]

    model = vit_b_16(weights=None).to(device="cuda", dtype=param_dtype).eval()
    H, W = spec.shape_policy.image_size or (224, 224)
    inputs = (
        torch.randn(
            spec.shape_policy.batch_size or 1,
            int(spec.shape_policy.extra.get("channels", 3)),
            H, W,
            device="cuda",
            dtype=activation_dtype,
        ),
    )

    def forward():
        with torch.no_grad():
            return model(inputs[0])

    return WorkloadInstance(
        spec=spec,
        forward=forward,
        example_inputs=inputs,
        metadata={
            "module": model,
            "param_count": sum(p.numel() for p in model.parameters()),
            "seed": seed,
            "image_size": list(spec.shape_policy.image_size or (224, 224)),
        },
    )
