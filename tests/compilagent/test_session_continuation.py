"""End-to-end tests for the continuation orchestrator in `run_session`.

A `_StoppingFakeHarness` proposes + runs ONE candidate then yields
`RUN_FINISHED`. With `max_candidates > 1` the orchestrator should
re-invoke the harness; the fake records each snapshot it sees so we can
assert on the iteration progression.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from compilagent.core.backend import backend_registry
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload
from compilagent.harness.base import HarnessRunRequest, StreamEvent, StreamEventKind
from compilagent.observation.events import EventKind
from compilagent.observation.sink import CapturingSink
from compilagent.session.completion import RunSnapshot
from compilagent.session.session import OptimizationSession, run_session
from compilagent.storage.workspace import OptimizationWorkspace

from tests.compilagent.test_session_with_fakes import _FakeBackend


@dataclass
class _StoppingFakeHarness:
    """Runs ONE candidate per `run()` then stops; records continuation snapshots."""

    id: str = "stopping_fake"
    supported_providers: tuple[str, ...] = ("fake",)
    snapshots_seen: list[RunSnapshot] = field(default_factory=list)
    continuation_prompts: list[str] = field(default_factory=list)
    runs: int = 0

    def build_continuation_request(
        self, previous: HarnessRunRequest, snapshot: RunSnapshot
    ) -> HarnessRunRequest:
        self.snapshots_seen.append(snapshot)
        prompt = (
            f"<continuation iteration={snapshot.iteration} "
            f"successful={snapshot.successful_count}/{snapshot.max_candidates}>"
        )
        self.continuation_prompts.append(prompt)
        return replace(previous, user_prompt=prompt)

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        self.runs += 1
        toolset = request.toolset

        async def _yield_call(name: str, args: dict, call_id: str):
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                tool_name=name,
                tool_call_id=call_id,
                tool_args=args,
            )
            try:
                result = toolset.by_name(name).invoke(args)
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_RESULT,
                    tool_name=name,
                    tool_call_id=call_id,
                    tool_result=result,
                )
            except ValueError as exc:
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_ERROR,
                    tool_name=name,
                    tool_call_id=call_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        propose_args = {
            "interventions": [
                {
                    "target_kind": "knob",
                    "target_selector": "optimize",
                    "payload": "on",
                    "rationale": f"iteration {self.runs}",
                }
            ],
            "description": f"iteration {self.runs}",
        }
        registered_id: str | None = None
        async for ev in _yield_call("propose_candidate", propose_args, f"p{self.runs}"):
            if ev.kind is StreamEventKind.TOOL_RESULT and ev.tool_result:
                registered_id = json.loads(ev.tool_result)["id"]
            yield ev

        assert registered_id is not None
        async for ev in _yield_call(
            "run_candidate", {"candidate_id": registered_id}, f"r{self.runs}"
        ):
            yield ev

        yield StreamEvent(
            kind=StreamEventKind.RUN_FINISHED,
            text=f"stopped early after iteration {self.runs}",
        )


@pytest.fixture
def _registered_workload():
    spec = WorkloadSpec(
        id="cont_workload",
        title="Continuation workload",
        description="Synthetic test workload for continuation tests",
        kind=WorkloadKind.KERNEL,
        backend_id="fake",
        tolerance=ToleranceConfig(atol=1e-6, rtol=1e-6),
        budget=BenchmarkBudget(warmup=1, repetitions=3, max_seconds=1.0),
    )

    @register_workload(spec)
    def _build(s: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(spec=s, forward=lambda: None)

    return spec


def test_orchestrator_drives_continuations_until_budget_and_reflection_met(
    tmp_path: Path, _registered_workload
):
    """The agent stops after every single candidate; the orchestrator
    should re-engage it until budget is met AND reflection tools fire.

    The well-behaved continuation prompt would lead the agent to call the
    reflection tools; our stripped-down stopping fake doesn't. So the loop
    should exit via the `exhausted` ceiling once `max_continuations` is
    hit, but in the meantime it should produce a candidate per iteration.
    """

    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    sink = CapturingSink()
    session = OptimizationSession(
        workload_id="cont_workload",
        run_id="run-cont",
        workspace=workspace,
        sink=sink,
        max_candidates=3,
    )

    harness = _StoppingFakeHarness()
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="be brief",
        user_prompt="initial",
        model_id="fake:fake-1",
    )
    result = asyncio.run(
        run_session(
            session=session,
            harness=harness,
            request=request,
            max_continuations=2,
        )
    )

    # The harness was driven 3 times total: initial + 2 continuations.
    assert harness.runs == 3
    # Three candidates ran successfully — one per harness invocation.
    assert session.budget_state["successful_count"] == 3

    # Two continuation events were emitted (between the 3 harness runs).
    continuations = [
        e for e in sink.events if e.kind == EventKind.RUN_CONTINUATION.value
    ]
    assert len(continuations) == 2
    iterations = [e.payload["iteration"] for e in continuations]
    assert iterations == [1, 2]

    # The harness saw monotonic snapshots feeding into its prompt builder.
    assert [s.iteration for s in harness.snapshots_seen] == [0, 1]
    assert [s.successful_count for s in harness.snapshots_seen] == [1, 2]
    # The 2nd snapshot shows progress on best_speedup.
    assert harness.snapshots_seen[-1].best_speedup is not None
    assert harness.snapshots_seen[-1].best_speedup > 1.0

    # The metadata records why we stopped — `exhausted` because the
    # stopping fake never calls the reflection tools.
    assert result.metadata.get("completion_reason") == "exhausted"


def test_orchestrator_skips_continuation_when_budget_and_reflection_met(
    tmp_path: Path, _registered_workload
):
    """A well-behaved fake that meets budget + reflection on iteration 0
    must NOT trigger any continuation."""

    from tests.compilagent.test_session_with_fakes import _FakeHarness

    backend_registry.register("fake", _FakeBackend)
    workspace = OptimizationWorkspace(session_cwd=tmp_path)
    sink = CapturingSink()
    session = OptimizationSession(
        workload_id="cont_workload",
        run_id="run-noop",
        workspace=workspace,
        sink=sink,
        max_candidates=1,
    )

    harness = _FakeHarness()
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions="be brief",
        user_prompt="initial",
        model_id="fake:fake-1",
    )
    result = asyncio.run(
        run_session(
            session=session,
            harness=harness,
            request=request,
            max_continuations=4,
        )
    )

    # No continuation events.
    assert not any(
        e.kind == EventKind.RUN_CONTINUATION.value for e in sink.events
    )
    assert result.metadata.get("completion_reason") == "budget_met"
