"""Single ViT-B/16 transformer encoder block (faster iteration than the full model).

ViT-B/16 has 12 identical encoder blocks; one block is the cheapest unit that
exercises the same op set Inductor will see for the full model (LayerNorm,
multi-head attention, MLP with GELU, residual adds). Compile-and-bench cycle
is roughly 10× shorter than `vit_b_16`, so this is the agent's preferred
sandbox during a long optimization episode.

Built fresh from torchvision's `EncoderBlock` with `torch.manual_seed(0)` for
reproducibility. Inputs are an embedded sequence `[B, 197, 768]` (197 = 196
patches + 1 CLS token), bf16 by default on Blackwell tensor cores.
"""

from __future__ import annotations

from typing import Any

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


_VIT_BLOCK_SPEC = WorkloadSpec(
    id="vit_block",
    title="ViT-B/16 single encoder block",
    description=(
        "One ViT-B/16 transformer encoder block (LayerNorm + MHA + LayerNorm + "
        "MLP) on an embedded sequence — bf16 on Blackwell tensor cores. Faster "
        "compile/bench cycle than the full model; ideal for agent iteration."
    ),
    kind=WorkloadKind.FULL_MODEL,
    backend_id="torch_inductor",
    entrypoint="compilagent_triton.workloads.pytorch_models.vit_block:build_workload",
    dtype_policy=DtypePolicy(activation_dtype="bf16", param_dtype="bf16"),
    shape_policy=ShapePolicy(
        batch_size=32,
        sequence_length=197,
        extra={"hidden_dim": 768, "num_heads": 12, "mlp_dim": 3072},
    ),
    tolerance=ToleranceConfig(
        atol=5e-4, rtol=5e-3,
        notes="bf16-vs-bf16 forward pass; baseline-vs-fp32-reference is recorded but non-gating.",
    ),
    budget=BenchmarkBudget(warmup=3, repetitions=20, max_seconds=60.0),
    metadata={"seed": 0},
)


@register_workload(_VIT_BLOCK_SPEC)
def build_workload(spec: WorkloadSpec) -> WorkloadInstance:
    import torch
    from torchvision.models.vision_transformer import EncoderBlock

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialise vit_block.")
    seed = int(spec.metadata.get("seed", 0))
    torch.manual_seed(seed)

    hidden_dim = int(spec.shape_policy.extra.get("hidden_dim", 768))
    num_heads = int(spec.shape_policy.extra.get("num_heads", 12))
    mlp_dim = int(spec.shape_policy.extra.get("mlp_dim", 3072))
    seq_len = int(spec.shape_policy.sequence_length or 197)
    batch = int(spec.shape_policy.batch_size or 1)

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

    block = EncoderBlock(
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=0.0,
        attention_dropout=0.0,
    ).to(device="cuda", dtype=param_dtype).eval()
    inputs = (
        torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=activation_dtype),
    )

    def forward():
        with torch.no_grad():
            return block(inputs[0])

    return WorkloadInstance(
        spec=spec,
        forward=forward,
        example_inputs=inputs,
        metadata={
            "module": block,
            "param_count": sum(p.numel() for p in block.parameters()),
            "seed": seed,
        },
    )
