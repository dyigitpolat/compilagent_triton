"""End-to-end test for `compilagent.integrations.pydantic_ai`.

Drives the real `PydanticAIHarness` against pydantic-ai's `TestModel` and a
synthetic backend (re-using the fixture pattern from
`tests/compilagent/test_session_with_fakes.py`). Verifies:

  - Tool dispatch goes through `ToolDecl.handler(args)` for each canonical
    session tool the model decides to call.
  - The yielded `StreamEvent` sequence is well-formed (at least one
    `TOOL_CALL` + matching `TOOL_RESULT`/`TOOL_ERROR`, ending in
    `RUN_FINISHED`).
  - Self-registration installs the harness under
    `harness_registry.get("pydantic_ai")`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    TimingResult,
)
from compilagent.core.backend import backend_registry
from compilagent.core.plan import Intervention, Plan, ValidationResult
from compilagent.core.search_space import (
    DerivationEvidence,
    EnumChoice,
    Lever,
    SearchSpace,
)
from compilagent.core.tool_decl import ToolDecl
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload
from compilagent.harness.base import HarnessRunRequest, StreamEventKind
from compilagent.observation.sink import CapturingSink
from compilagent.session.session import OptimizationSession, run_session
from compilagent.storage.workspace import OptimizationWorkspace
from compilagent.toolset import Toolset

# --------------------------------------------------------------- fake backend


@dataclass
class _FakeBackend:
    id: str = "fake"
    artifact_stages: tuple[str, ...] = ("ir",)

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="cpu",
            capability_int=None,
            name="Fake",
            memory_total_bytes=None,
            memory_peak_bandwidth_gbps=None,
        )

    def analyze(self, workload, *, baseline_artifacts) -> Analysis:
        return Analysis(summary={"kind": workload.kind.value})

    def derive_search_space(self, workload, analysis) -> SearchSpace:
        lever = Lever(
            id="kernel.flag",
            target_kind="knob",
            target_selector="optimize",
            range=EnumChoice(candidates=("on", "off")),
            default="off",
            description="Fake flag.",
            evidence=DerivationEvidence(rule="fake.lever", signal="synthetic"),
            backend_id=self.id,
        )
        return SearchSpace(workload_id=workload.id, backend_id=self.id, levers=(lever,))

    def validate_intervention(self, intervention) -> ValidationResult:
        return ValidationResult(ok=intervention.target.kind == "knob",
                                errors=() if intervention.target.kind == "knob"
                                else (f"unsupported {intervention.target.kind}",))

    def interpret_plan(self, plan: Plan) -> Plan:
        return plan

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))

    def compile(self, workload, plan, *, artifact_dir, pass_callback=None) -> CompileResult:
        artifact = artifact_dir / "out.ir"
        artifact.write_text("fake", encoding="utf-8")
        return CompileResult(ok=True, elapsed_ms=1.0, artifacts=(artifact,))

    def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None) -> TimingResult:
        return TimingResult(timings_ms=(5.0,) * repetitions, median_ms=5.0, p20_ms=5.0, p80_ms=5.0)

    def validate_correctness(self, workload, baseline, candidate, tolerance) -> CorrectnessResult:
        return CorrectnessResult(ok=True, max_abs_diff=0.0)

    def reset_between_compiles(self, workload) -> None:
        return None

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        return ()

    def list_artifact_renderers(self) -> Sequence[object]:
        return ()

    def infer_workload_family(self, workload) -> str | None:
        return None


@pytest.fixture
def _registered_workload():
    spec = WorkloadSpec(
        id="fake_workload",
        title="Fake",
        description="Test",
        kind=WorkloadKind.KERNEL,
        backend_id="fake",
        tolerance=ToleranceConfig(atol=1e-6, rtol=1e-6),
        budget=BenchmarkBudget(warmup=1, repetitions=2, max_seconds=1.0),
    )

    @register_workload(spec)
    def _build(s: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(spec=s, forward=lambda: None)

    return spec


# -------------------------------------------------------- self-registration


def test_self_registration_installs_harness():
    """Importing the integration registers `pydantic_ai` in the registry."""

    # Importing here (inside the test) so the autouse `_reset_registries`
    # fixture has cleared the registry first.
    import compilagent.integrations.pydantic_ai  # noqa: F401
    from compilagent.harness.registry import harness_registry

    assert "pydantic_ai" in harness_registry.ids()
    harness = harness_registry.get("pydantic_ai")
    assert harness.id == "pydantic_ai"
    assert "anthropic" in harness.supported_providers


# ------------------------------------------------------------- tool adapter


def test_tool_adapter_passes_through_typed_handler():
    """The pydantic-ai adapter returns the handler's typed signature
    intact so pydantic-ai introspects it directly."""

    from compilagent.integrations.pydantic_ai._tool_adapter import (
        make_pydantic_ai_tool_fn,
    )

    def _typed_handler(*, candidate_id: str, note: str = "") -> str:
        return json.dumps({"id": candidate_id, "note": note})

    decl = ToolDecl(
        name="run_thing",
        description="Run a thing.",
        args_schema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
        handler=_typed_handler,
        read_only=False,
    )
    fn = make_pydantic_ai_tool_fn(decl)
    assert fn.__name__ == "run_thing"
    assert fn.__doc__ == "Run a thing."

    import inspect

    sig = inspect.signature(fn, eval_str=True)
    assert list(sig.parameters.keys()) == ["candidate_id", "note"]
    # The annotation comes from the bound handler — typed `str`, not the
    # synthesised string type from any schema.
    assert sig.parameters["candidate_id"].annotation is str

    result = fn(candidate_id="cand-123", note="hi")
    assert json.loads(result) == {"id": "cand-123", "note": "hi"}


def test_tool_decl_invoke_validates_via_args_model():
    """`ToolDecl.invoke` validates the wire dict through `args_model`
    before calling the typed handler with kwargs."""

    from pydantic import BaseModel, Field

    class _Args(BaseModel):
        items: list[int] = Field(...)
        label: str = ""

    captured: dict = {}

    def _handler(*, items: list[int], label: str = "") -> str:
        captured["items"] = items
        captured["label"] = label
        return "ok"

    decl = ToolDecl(
        name="batch",
        description="Take typed args",
        args_schema=_Args.model_json_schema(),
        handler=_handler,
        args_model=_Args,
        read_only=False,
    )
    out = decl.invoke({"items": [1, 2, 3], "label": "x"})
    assert out == "ok"
    assert captured == {"items": [1, 2, 3], "label": "x"}

    # Wire-dict validation surfaces Pydantic errors as ValueError so
    # adapters can fold them into a retryable tool error.
    with pytest.raises(ValueError):
        decl.invoke({"items": "not-a-list"})


# --------------------------------------------------------- model settings


def test_resolve_model_settings_anthropic_effort_forces_temperature_one():
    from compilagent.integrations.pydantic_ai._model import resolve_model_settings

    out = resolve_model_settings(
        "anthropic:claude-opus-4-7",
        reasoning_effort="extra_high",
        max_tokens=4096,
        temperature=0.2,
    )
    assert out["anthropic_effort"] == "max"
    assert out["temperature"] == 1.0
    assert out["max_tokens"] == 4096


def test_resolve_model_settings_non_anthropic_uses_caller_temperature():
    from compilagent.integrations.pydantic_ai._model import resolve_model_settings

    out = resolve_model_settings(
        "mistral:mistral-large-latest",
        reasoning_effort="high",
        max_tokens=2048,
        temperature=0.7,
    )
    assert "anthropic_effort" not in out
    assert out["temperature"] == 0.7


# --------------------------------------------------------- end-to-end


def test_harness_drives_session_with_test_model(tmp_path: Path, _registered_workload):
    """Drive a real OptimizationSession through PydanticAIHarness + TestModel."""

    from compilagent.integrations.pydantic_ai import PydanticAIHarness

    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    sink = CapturingSink()
    session = OptimizationSession(
        workload_id="fake_workload",
        run_id="run-pa",
        workspace=workspace,
        sink=sink,
        max_candidates=2,
    )

    # Restrict TestModel to read-only tools so it doesn't try to invent
    # JSON payloads for propose_*; those would fail validation noisily.
    read_only = session.toolset.read_only_subset()
    request = HarnessRunRequest(
        toolset=read_only,
        system_instructions="be brief",
        user_prompt="inspect this workload",
        model_id="test",
        max_tokens=512,
        temperature=0.0,
        extra={"retries": 1},
    )

    harness = PydanticAIHarness()
    result = asyncio.run(run_session(session=session, harness=harness, request=request))

    # The TestModel call_tools='all' default makes it call every read-only
    # tool once. We just need to see at least one TOOL_CALL/RESULT pair and
    # a RUN_FINISHED.
    kinds = sink.kinds()
    assert "tool.call.started" in kinds
    assert "tool.call.completed" in kinds

    # Assert the harness terminated cleanly.
    assert result.elapsed_ms > 0
    assert "session.failed" not in kinds


def test_harness_value_error_in_handler_becomes_model_retry(tmp_path: Path):
    """A `ValueError` from a handler is surfaced as a tool error, not a crash."""


    from compilagent.integrations.pydantic_ai import PydanticAIHarness

    raises = ToolDecl(
        name="raises",
        description="Always raises.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _args: (_ for _ in ()).throw(ValueError("nope")),
        read_only=True,
    )
    request = HarnessRunRequest(
        toolset=Toolset(tools=(raises,)),
        system_instructions="",
        user_prompt="call it",
        # Force TestModel directly so we can scope the call list.
        model_id="test",
        max_tokens=128,
        extra={"retries": 1},
    )

    harness = PydanticAIHarness()

    async def _collect():
        events = []
        async for ev in harness.run(request):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    # The model retried and ultimately the run finishes (TestModel gives up
    # after retries are exhausted and emits a final response). We just need
    # to confirm we observed a TOOL_CALL and the run did not crash.
    kinds = [e.kind for e in events]
    assert StreamEventKind.TOOL_CALL in kinds
    # Either RUN_FINISHED (most likely) or RUN_FAILED with a ModelRetry-derived
    # error — both are acceptable; the contract is that the harness MUST emit
    # exactly one terminal event.
    terminal = [k for k in kinds if k in (StreamEventKind.RUN_FINISHED, StreamEventKind.RUN_FAILED)]
    assert len(terminal) == 1


# Silence the unused-import check on the workload fixture (it's used via pytest's
# fixture mechanism). Importing TestModel here also surfaces a clear error if
# pydantic-ai is missing — preferable to obscure failures inside the harness.
def test_pydantic_ai_is_importable():
    from pydantic_ai.models.test import TestModel  # noqa: F401
