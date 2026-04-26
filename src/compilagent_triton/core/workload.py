"""Backend-agnostic workload abstraction.

A `WorkloadSpec` describes one compile target — anything from a single
`@triton.jit` kernel to a full `nn.Module` forward pass — together with the
shape / dtype policy needed to materialise concrete inputs and the tolerance
budget used to validate numerical correctness.

A `WorkloadInstance` is the materialised result: a callable plus the example
inputs the backend will compile and time. Backends never construct workloads;
they only consume an instance produced by the workload's registered builder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkloadKind(StrEnum):
    KERNEL = "kernel"
    FUSED_SUBGRAPH = "fused_subgraph"
    FULL_MODEL = "full_model"


@dataclass(frozen=True, slots=True)
class DtypePolicy:
    """Activation / parameter dtype policy for a workload.

    `param_dtype` is the dtype the workload's parameters live in;
    `activation_dtype` is what flows through tensors at runtime.
    For PyTorch workloads under autocast, `activation_dtype` is the autocast
    dtype while `param_dtype` stays the original.
    """

    activation_dtype: str = "fp32"      # "fp32" | "bf16" | "fp16"
    param_dtype: str = "fp32"
    autocast: bool = False


@dataclass(frozen=True, slots=True)
class ShapePolicy:
    """Workload-level shape envelope (batch / image / sequence)."""

    batch_size: int = 1
    image_size: tuple[int, int] | None = None     # (H, W) for vision workloads
    sequence_length: int | None = None            # for transformer-style workloads
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    """Per-dtype absolute / relative tolerance budget for correctness checks."""

    atol: float = 1e-5
    rtol: float = 1e-4
    notes: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    """How much time/repetitions the runtime spends benchmarking each candidate."""

    warmup: int = 3
    repetitions: int = 10
    max_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Declarative description of a compile target.

    `entrypoint` is a dotted reference (`module:fn`) into the package; the
    builder is invoked once per workload-instance request to produce a fresh
    `WorkloadInstance`. Builders are registered via `@register_workload(id)` in
    `compilagent_triton.workloads.registry` — there is no global hand-coded
    workload list anywhere.
    """

    id: str
    title: str
    description: str
    kind: WorkloadKind
    backend_id: str
    entrypoint: str                             # "compilagent_triton.workloads.foo:build"
    dtype_policy: DtypePolicy = field(default_factory=DtypePolicy)
    shape_policy: ShapePolicy = field(default_factory=ShapePolicy)
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    budget: BenchmarkBudget = field(default_factory=BenchmarkBudget)
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "kind": self.kind.value,
            "backend_id": self.backend_id,
            "entrypoint": self.entrypoint,
            "dtype_policy": {
                "activation_dtype": self.dtype_policy.activation_dtype,
                "param_dtype": self.dtype_policy.param_dtype,
                "autocast": self.dtype_policy.autocast,
            },
            "shape_policy": {
                "batch_size": self.shape_policy.batch_size,
                "image_size": list(self.shape_policy.image_size) if self.shape_policy.image_size else None,
                "sequence_length": self.shape_policy.sequence_length,
                "extra": dict(self.shape_policy.extra),
            },
            "tolerance": {
                "atol": self.tolerance.atol,
                "rtol": self.tolerance.rtol,
                "notes": self.tolerance.notes,
            },
            "budget": {
                "warmup": self.budget.warmup,
                "repetitions": self.budget.repetitions,
                "max_seconds": self.budget.max_seconds,
            },
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class WorkloadInstance:
    """Materialised workload ready for a backend to compile + time + verify.

    `forward` runs the workload once and returns whatever the workload chooses
    as its "output" (a tensor for nn.Module workloads, a Triton kernel handle
    for kernel workloads, ...). Backends interpret it.
    """

    spec: WorkloadSpec
    forward: Callable[[], Any]
    example_inputs: tuple[Any, ...] = ()
    # Optional metadata the backend may consult — e.g., a list of
    # nn.Module child names for per-kernel attribution, a path to the kernel
    # `.py` file for source preview, ...
    metadata: dict[str, Any] = field(default_factory=dict)


WorkloadBuilder = Callable[[WorkloadSpec], WorkloadInstance]
