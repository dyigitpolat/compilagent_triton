"""Abstract `Backend` protocol shared by every compiler integration.

The runtime, agent tools, episode store, and observation UI all program against
this interface. Concrete implementations live in `backends/<name>/backend.py`.
A backend is responsible for:

  - Describing its target device.
  - Compiling a `WorkloadSpec` under a `Plan` of `Intervention` objects.
  - Capturing per-stage compile artifacts in a backend-defined set of
    `artifact_stages` (e.g., `("ttir","ttgir","llir","ptx")` for Triton,
    `("fx_graph","output_code","autotune_log")` for TorchInductor).
  - Timing a workload's end-to-end execution.
  - Validating numerical correctness against a reference compile.
  - Deriving a `SearchSpace` of typed `Lever`s from the workload's analysis —
    no hand-coded values.
  - Listing extra introspection tools to expose to the agent (e.g.,
    `list_triton_passes`, `list_inductor_knobs`).

The contract is intentionally narrow. Backend-specific concepts (MLIR passes,
inductor knobs, FX rewrites) are encapsulated in the backend's `Plan` and
`Intervention` objects; the runtime never inspects them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Device capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """Hardware envelope a backend will compile against."""

    arch: str                       # e.g. "cuda:sm_120", "cuda:sm_90", "cpu", "rocm:gfx942"
    capability_int: int | None      # numeric form (sm number) where applicable
    name: str                       # human-readable: "RTX PRO 6000 Blackwell"
    memory_total_bytes: int | None
    memory_peak_bandwidth_gbps: float | None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tolerance + correctness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    """Per-dtype absolute / relative tolerance budget."""

    atol: float
    rtol: float
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Outcome of comparing baseline vs. candidate output tensors."""

    ok: bool
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    p99_abs_diff: float | None = None
    failed_at: str | None = None        # "output[3]" / "logits" / etc.
    diagnostics: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plans / interventions / targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    """Where in the compile pipeline an intervention applies.

    A discriminated string `kind` keeps this serializable across the agent
    boundary; backends interpret the `selector` string in their own vocabulary.
    Common kinds (advisory, not enforced):

      - "launch"      — selector=kernel_id; payload is launch meta dict
      - "pass"        — selector="<stage>:<pass_name>"; payload is action+args
      - "knob"        — selector=dotted config path; payload is value
      - "lowering"    — selector=aten op qualname; payload is module:fn ref
      - "fx_node"     — selector=node target name; payload is rewrite spec
      - "kernel_src"  — selector=file:span; payload is replacement text
      - "scheduler"   — selector="pre_fusion"|"post_fusion"; payload is callable ref
    """

    kind: str
    selector: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind}({self.selector})" if self.selector else self.kind


@dataclass(frozen=True, slots=True)
class Intervention:
    """One concrete change an agent proposes for a candidate compile.

    `payload` is a backend-specific JSON-serialisable value (e.g., a knob value,
    a pass-action dict, a layout dict). Backends validate the shape.
    """

    target: Target
    payload: Any
    rationale: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "target": {"kind": self.target.kind, "selector": self.target.selector},
            "payload": self.payload,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered list of interventions making up a single candidate compile."""

    interventions: tuple[Intervention, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.interventions

    def serialize(self) -> list[dict[str, Any]]:
        return [iv.serialize() for iv in self.interventions]


# ---------------------------------------------------------------------------
# Per-pass / per-step streaming callback
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PassEvent:
    """A single executed compile step the backend wants to surface live.

    Streamed to the trace store so the UI can render a per-pass timeline
    regardless of the backend (Triton MLIR pass, Inductor scheduler pass,
    config-knob application, etc.).
    """

    stage: str
    name: str
    duration_ms: float
    action: str = "run"                 # "run" | "skip" | "replace"
    args: list[Any] = field(default_factory=list)
    ir_after_size: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PassCallback = Callable[[PassEvent], None]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Analysis:
    """Backend-specific structured introspection of a workload + baseline.

    Concrete fields are backend-defined (TritonAnalysis, InductorAnalysis); the
    `summary` dict is the generic surface every backend MUST populate so the
    runtime can show a backend-agnostic snapshot.

    Required `summary` keys (advisory; backends may add more):
      - "kind": workload kind ("kernel"|"fused_subgraph"|"full_model")
      - "tensor_shapes": dict[str, list[int]]   (representative shapes)
      - "dtypes": list[str]                     (e.g., ["bf16","fp32"])
      - "op_counts": dict[str, int]
    """

    summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Raw end-to-end timing samples from `Backend.time_workload`."""

    timings_ms: tuple[float, ...]
    median_ms: float | None
    p20_ms: float | None
    p80_ms: float | None
    profile_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: str | None = None


# ---------------------------------------------------------------------------
# Introspection tools (extra agent tools the backend wants registered)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntrospectionTool:
    """Description of an extra agent tool a backend exposes.

    `fn` must be a plain callable with typed args + a `str` return; the agent
    runner registers it with the same surface a built-in tool would have.
    """

    name: str
    description: str
    fn: Callable[..., str]


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Abstract compiler backend."""

    id: str
    artifact_stages: tuple[str, ...]

    def device_capability(self) -> DeviceCapability: ...

    def analyze(
        self,
        workload: Any,                  # WorkloadSpec — typed in core/workload.py
        *,
        baseline_artifacts: Sequence[Any],  # CompileArtifact list
    ) -> Analysis: ...

    def compile(
        self,
        workload: Any,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> Any:                           # CompileResult — defined in core/schemas
        ...

    def time_workload(
        self,
        workload: Any,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: float | None = None,
    ) -> TimingResult: ...

    def validate_correctness(
        self,
        workload: Any,
        baseline: Any,                   # CompileResult
        candidate: Any,                  # CompileResult
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult: ...

    def list_introspection_tools(self) -> Sequence[IntrospectionTool]: ...

    def derive_search_space(
        self,
        workload: Any,
        analysis: Analysis,
    ) -> Any:                            # SearchSpace — defined in core/search_space
        ...

    def apply_intervention(
        self,
        plan: Plan,
        intervention: Intervention,
    ) -> Plan: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BackendRegistry:
    """Process-wide map from backend id to a zero-arg factory."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Backend]] = {}

    def register(self, backend_id: str, factory: Callable[[], Backend]) -> None:
        if backend_id in self._factories:
            raise ValueError(f"Backend `{backend_id}` is already registered.")
        self._factories[backend_id] = factory

    def get(self, backend_id: str) -> Backend:
        if backend_id not in self._factories:
            known = sorted(self._factories.keys())
            raise KeyError(
                f"Unknown backend `{backend_id}`. Registered: {known or '(none)'}."
            )
        return self._factories[backend_id]()

    def ids(self) -> list[str]:
        return sorted(self._factories.keys())

    def clear(self) -> None:                # for tests
        self._factories.clear()


backend_registry = BackendRegistry()
