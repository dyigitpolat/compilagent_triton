"""ViT-L/16 full forward pass.

A larger sibling of vit_b_16 (~307M params, 24 encoder blocks, hidden_dim 1024,
mlp_dim 4096) for stress-testing the optimization loop on a substantially
heavier compile graph. Same dtype / shape policy shape as vit_b_16.
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


_VIT_L16_SPEC = WorkloadSpec(
    id="vit_l_16",
    title="ViT-L/16 full forward",
    description=(
        "torchvision.models.vit_l_16 forward pass on ImageNet-shape inputs. "
        "~307M params, 24 encoder blocks. Heavier compile/bench cycle than vit_b_16."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.vit_l_16:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=16,
        image_size=(224, 224),
        sequence_length=197,
        extra={"channels": 3, "num_classes": 1000},
    ),
    tolerance=ToleranceConfig(atol=5e-4, rtol=5e-3),
    budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=240.0),
    metadata={"seed": 0, "model_family": "vit", "variant": "l_16"},
)


@register_workload(_VIT_L16_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    from torchvision.models import vit_l_16

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vit_l_16.")
    seed = int(spec.metadata.get("seed", 0))
    torch.manual_seed(seed)

    activation_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
    }[spec.dtype_policy.activation_dtype]
    param_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
    }[spec.dtype_policy.param_dtype]

    model = vit_l_16(weights=None).to(device="cuda", dtype=param_dtype).eval()
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
        },
    )
