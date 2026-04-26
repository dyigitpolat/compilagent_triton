"""In-process agent invocation for the observation server's optimize mode.

Drives the pydantic-ai agent against a registered example kernel and streams
agent events (thinking, text, tool calls/results) through the shared TraceStore
so the observer UI sees them in real time.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)

from .examples import RunRequest, get_example
from .llm import model_for_settings, model_settings_for_settings
from .optimizer_runtime import OptimizerRuntime, optimizer_instructions
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workspace import OptimizationWorkspace


def run_agent_optimization(
    *,
    request: RunRequest,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
    settings: CompilagentSettings | None = None,
) -> dict[str, Any]:
    """Synchronous entry point for FastAPI background tasks."""

    return asyncio.run(
        _run_agent_optimization_async(
            request=request,
            run_id=run_id,
            workspace_root=workspace_root,
            trace_store=trace_store,
            settings=settings or CompilagentSettings.from_env(project_root=workspace_root.parent),
        )
    )


async def _run_agent_optimization_async(
    *,
    request: RunRequest,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
    settings: CompilagentSettings,
) -> dict[str, Any]:
    workspace = OptimizationWorkspace(
        session_cwd=workspace_root.parent,
        root_name=workspace_root.name,
    ).ensure()

    runtime = OptimizerRuntime(
        settings=settings,
        workspace=workspace,
        harness_label="observer",
    )
    runtime.trace_store = trace_store

    example = get_example(request.example_id)
    episode_id: str | None = None
    if example.kernel_symbol is not None:
        runtime.register_kernel(
            kernel_id=example.id,
            name=example.kernel_family,
            path=str(example.source_path),
            entrypoint=example.kernel_symbol,
        )
        episode_payload = json.loads(
            runtime.start_episode(
                kernel_id=example.id,
                objective=f"Beat the Triton baseline on the `{example.kernel_family}` example.",
                budget_json=json.dumps(
                    {"max_candidates": settings.max_candidates},
                ),
            )
        )
        episode_id = episode_payload.get("id")

    agent = _build_observer_agent(runtime=runtime, settings=settings, workspace=workspace)
    prompt = _starter_prompt(example=example, episode_id=episode_id)

    trace_store.emit(
        "agent.run_started",
        episode_id=episode_id,
        payload={
            "run_id": run_id,
            "example_id": example.id,
            "model": settings.model_name,
            "reasoning_effort": settings.reasoning_effort,
            "max_candidates": settings.max_candidates,
        },
    )

    started_at = time.perf_counter()
    final_text: str | None = None
    try:
        async with agent.iter(
            prompt,
            model_settings=model_settings_for_settings(settings),
        ) as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        await _stream_model_response(
                            stream=stream,
                            trace_store=trace_store,
                            episode_id=episode_id,
                            run_id=run_id,
                        )
                elif Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        await _stream_tool_events(
                            stream=stream,
                            trace_store=trace_store,
                            episode_id=episode_id,
                            run_id=run_id,
                        )
        final_text = (
            str(agent_run.result.output) if agent_run.result is not None else None
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        trace_store.emit(
            "agent.run_failed",
            episode_id=episode_id,
            payload={
                "run_id": run_id,
                "elapsed_ms": elapsed_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    trace_store.emit(
        "agent.run_completed",
        episode_id=episode_id,
        payload={
            "run_id": run_id,
            "elapsed_ms": elapsed_ms,
            "final_text": final_text,
        },
    )
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "elapsed_ms": elapsed_ms,
        "final_text": final_text,
    }


async def _stream_model_response(
    *,
    stream,
    trace_store: TraceStore,
    episode_id: str | None,
    run_id: str,
) -> None:
    async for event in stream:
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, ThinkingPart):
                trace_store.emit(
                    "agent.thinking_started",
                    episode_id=episode_id,
                    payload={"run_id": run_id, "index": event.index},
                )
                if part.content:
                    trace_store.emit(
                        "agent.thinking_delta",
                        episode_id=episode_id,
                        payload={
                            "run_id": run_id,
                            "index": event.index,
                            "delta": part.content,
                        },
                    )
            elif isinstance(part, TextPart):
                trace_store.emit(
                    "agent.text_started",
                    episode_id=episode_id,
                    payload={"run_id": run_id, "index": event.index},
                )
                if part.content:
                    trace_store.emit(
                        "agent.text_delta",
                        episode_id=episode_id,
                        payload={
                            "run_id": run_id,
                            "index": event.index,
                            "delta": part.content,
                        },
                    )
            elif isinstance(part, ToolCallPart):
                # Tool call args arrive in deltas; the assembled call is surfaced
                # via FunctionToolCallEvent in _stream_tool_events instead.
                continue
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, ThinkingPartDelta):
                content = getattr(delta, "content_delta", None)
                if content:
                    trace_store.emit(
                        "agent.thinking_delta",
                        episode_id=episode_id,
                        payload={
                            "run_id": run_id,
                            "index": event.index,
                            "delta": content,
                        },
                    )
            elif isinstance(delta, TextPartDelta):
                content = getattr(delta, "content_delta", None)
                if content:
                    trace_store.emit(
                        "agent.text_delta",
                        episode_id=episode_id,
                        payload={
                            "run_id": run_id,
                            "index": event.index,
                            "delta": content,
                        },
                    )


async def _stream_tool_events(
    *,
    stream,
    trace_store: TraceStore,
    episode_id: str | None,
    run_id: str,
) -> None:
    async for event in stream:
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            args: Any
            try:
                args = part.args_as_dict()
            except Exception:
                args = getattr(part, "args", None)
            trace_store.emit(
                "agent.tool_call",
                episode_id=episode_id,
                payload={
                    "run_id": run_id,
                    "tool": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "args": args,
                },
            )
        elif isinstance(event, FunctionToolResultEvent):
            result_part = getattr(event, "result", None)
            tool_call_id = getattr(result_part, "tool_call_id", None)
            content = getattr(result_part, "content", None)
            preview: str | None = None
            if isinstance(content, str):
                preview = content[:600]
            trace_store.emit(
                "agent.tool_result",
                episode_id=episode_id,
                payload={
                    "run_id": run_id,
                    "tool_call_id": tool_call_id,
                    "preview": preview,
                },
            )
        elif isinstance(event, (BuiltinToolCallEvent, BuiltinToolResultEvent)):
            continue


def _retry_on_value_error(fn):
    """Decorator: turn ValueError into ModelRetry so the agent can self-correct.

    Preserves the wrapped function's signature with functools.wraps — pydantic-ai
    introspects the wrapper to build the tool's JSON schema, so a generic
    ``*args, **kwargs`` wrapper would erase all argument types.
    """

    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return wrapped


def _build_observer_agent(
    *,
    runtime: OptimizerRuntime,
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(
        model_for_settings(settings),
        name="triton-observer-optimizer",
        instructions=optimizer_instructions(
            settings, workspace, harness_label="observer"
        ),
    )

    @agent.tool_plain
    def describe_optimizer_surface() -> str:
        """Summarize the optimizer environment, constraints, and workflow."""

        return runtime.describe_optimizer_surface()

    @agent.tool_plain
    def list_benchmarks() -> str:
        """List registered kernel specs and built-in benchmark families."""

        return runtime.list_benchmarks()

    @agent.tool_plain
    def compile_baseline(kernel_id: str, meta_json: str = "{}") -> str:
        """Compile the registered kernel baseline and capture artifacts."""

        return runtime.compile_baseline(kernel_id=kernel_id, meta_json=meta_json)

    @agent.tool_plain
    @_retry_on_value_error
    def inspect_ir(run_id: str, stage: str = "ttgir", max_chars: int = 6000) -> str:
        """Read a captured IR artifact for a previous compile result.

        `run_id` accepts either the compile-result id (returned by
        `compile_baseline` / `run_candidate`) or the candidate id
        (returned by `propose_candidate_from_toolset`); we resolve to the
        most recent matching compile result.
        """

        return runtime.inspect_ir(run_id=run_id, stage=stage, max_chars=max_chars)

    @agent.tool_plain
    @_retry_on_value_error
    def summarize_decisions(run_id: str, stage: str = "ttgir") -> str:
        """Extract and summarize coalescing and matmul decisions from captured IR.

        `run_id` accepts the compile-result id or the candidate id (resolves
        to the most recent matching compile result).
        """

        return runtime.summarize_decisions(run_id=run_id, stage=stage)

    @agent.tool_plain
    def inspect_optimization_toolset(kernel_id: str) -> str:
        """Inspect the optimization toolset (levers, constraints, evidence) for a kernel."""

        return runtime.inspect_optimization_toolset(kernel_id=kernel_id)

    @agent.tool_plain
    @_retry_on_value_error
    def propose_candidate_from_toolset(
        kernel_id: str,
        kind: str,
        changes_json: str,
        description: str,
        expected_effect: str,
        hypothesis_id: str = "",
    ) -> str:
        """Propose a typed candidate using levers from the optimization toolset.

        `kind` MUST be a CandidateKind value (e.g. `"meta_parameters"`,
        `"memory_access_policy"`, `"coalescing_policy"`, `"matmul_policy"`),
        NOT the lever's `id`. Memory_access / coalescing / matmul candidates
        MAY also include launch-meta keys (BLOCK_SIZE, num_warps, num_stages,
        num_ctas, maxnreg) in the same `changes_json` payload, so a candidate
        can combine a cache modifier with a launch shape in one go.
        """

        return runtime.propose_candidate_from_toolset(
            kernel_id=kernel_id,
            kind=kind,
            changes_json=changes_json,
            description=description,
            expected_effect=expected_effect,
            hypothesis_id=hypothesis_id or None,
        )

    @agent.tool_plain
    @_retry_on_value_error
    def validate_candidate(candidate_json: str) -> str:
        """Validate a typed candidate before compilation or benchmarking."""

        return runtime.validate_candidate(candidate_json=candidate_json)

    @agent.tool_plain
    def record_hypothesis(
        episode_id: str,
        statement: str,
        expected_effect: str,
        evidence_refs_json: str = "[]",
    ) -> str:
        """Record a grounded optimization hypothesis before experiments."""

        return runtime.record_hypothesis(
            episode_id=episode_id,
            statement=statement,
            expected_effect=expected_effect,
            evidence_refs_json=evidence_refs_json,
        )

    @agent.tool_plain
    def record_reasoning_summary(
        episode_id: str,
        summary: str,
        next_step: str = "",
        evidence_refs_json: str = "[]",
        linked_hypothesis_id: str = "",
        linked_candidate_id: str = "",
    ) -> str:
        """Record a visible reasoning summary linked to evidence and next step."""

        return runtime.record_reasoning_summary(
            episode_id=episode_id,
            summary=summary,
            next_step=next_step,
            evidence_refs_json=evidence_refs_json,
            linked_hypothesis_id=linked_hypothesis_id or None,
            linked_candidate_id=linked_candidate_id or None,
        )

    @agent.tool_plain
    def run_baseline_benchmark(episode_id: str, kernel_id: str) -> str:
        """Run a baseline GPU benchmark for the registered kernel."""

        return runtime.run_baseline_benchmark(episode_id=episode_id, kernel_id=kernel_id)

    @agent.tool_plain
    def run_candidate_benchmark(episode_id: str, candidate_json: str) -> str:
        """Compile and benchmark a typed candidate against the registered kernel."""

        return runtime.run_candidate_benchmark(
            episode_id=episode_id,
            candidate_json=candidate_json,
        )

    @agent.tool_plain
    @_retry_on_value_error
    def compare_benchmarks(
        episode_id: str,
        baseline_run_id: str,
        candidate_run_ids_json: str,
    ) -> str:
        """Compare candidate benchmark results against the baseline run.

        `baseline_run_id` and each entry in `candidate_run_ids_json` may be a
        benchmark-result id (`bench-…`) or a candidate id (`cand-…`).
        """

        return runtime.compare_benchmarks(
            episode_id=episode_id,
            baseline_id=baseline_run_id,
            candidate_ids_json=candidate_run_ids_json,
        )

    @agent.tool_plain
    @_retry_on_value_error
    def accept_or_reject_candidate(
        episode_id: str,
        candidate_id: str,
        status: str,
        rationale: str,
        evidence_ids_json: str = "[]",
        compile_only: bool = False,
    ) -> str:
        """Record an explicit accept/reject decision for a candidate.

        `status` must be one of: "accepted", "rejected", "inconclusive",
        "failed". `evidence_ids_json` is a JSON list of benchmark-result ids
        (or candidate ids) supporting the decision; required unless
        `compile_only=true`.
        """

        return runtime.accept_or_reject_candidate(
            episode_id=episode_id,
            candidate_id=candidate_id,
            status=status,
            rationale=rationale,
            evidence_ids_json=evidence_ids_json,
            compile_only=compile_only,
        )

    @agent.tool_plain
    def write_report(episode_id: str) -> str:
        """Write a human-readable optimization report for the episode."""

        return runtime.write_report(episode_id=episode_id)

    @agent.tool_plain
    def list_compiler_passes(stage: str = "ttgir") -> str:
        """List MLIR passes available at a stage on this GPU, in default order.

        Use this BEFORE proposing a pass intervention so you only target real
        passes. Stage is `ttir` or `ttgir`.
        """

        return runtime.list_compiler_passes(stage=stage)

    @agent.tool_plain
    def describe_compiler_pass(pass_name: str) -> str:
        """Return purpose, parameter signature, and capability gate for a pass."""

        return runtime.describe_compiler_pass(pass_name=pass_name)

    @agent.tool_plain
    def read_pass_source(pass_name: str, max_chars: int = 8000) -> str:
        """Best-effort lookup of the upstream MLIR pass source for context."""

        return runtime.read_pass_source(pass_name=pass_name, max_chars=max_chars)

    @agent.tool_plain
    @_retry_on_value_error
    def propose_pass_intervention(
        kernel_id: str,
        interventions_json: str,
        description: str,
        expected_effect: str,
        hypothesis_id: str = "",
    ) -> str:
        """Propose a candidate that overrides specific MLIR passes (skip / param tweak).

        `interventions_json` is a JSON list of
        {pass_name, action ('run'|'skip'|'replace'), args?, rationale?} dicts.
        Use only pass names returned by `list_compiler_passes`.
        """

        return runtime.propose_pass_intervention(
            kernel_id=kernel_id,
            interventions_json=interventions_json,
            description=description,
            expected_effect=expected_effect,
            hypothesis_id=hypothesis_id or None,
        )

    return agent


def _starter_prompt(*, example: Any, episode_id: str | None) -> str:
    lines = [
        f"Optimize the Triton kernel for the `{example.kernel_family}` example.",
        f"Kernel id: `{example.id}` is already registered.",
    ]
    if episode_id:
        lines.append(f"Active episode id: `{episode_id}` is already started.")
    lines.append("")
    lines.append(
        "You have two complementary intervention surfaces:\n"
        "  (a) launch-meta candidates (BLOCK_SIZE, num_warps, num_stages, cache modifiers)\n"
        "      via `inspect_optimization_toolset` + `propose_candidate_from_toolset`;\n"
        "  (b) compiler-pass interventions (skip a pass, change pass parameters such as\n"
        "      tritongpu-pipeline num_stages) via `list_compiler_passes` +\n"
        "      `describe_compiler_pass` + `read_pass_source` + `propose_pass_intervention`.\n"
        "Surface (b) lets you actually change Triton's MLIR optimization decisions, not\n"
        "just sweep launch hyperparameters."
    )
    lines.append("")
    lines.append(
        "Workflow:\n"
        "1. Compile and benchmark the baseline.\n"
        "2. Inspect the IR (inspect_ir, summarize_decisions) and the available levers\n"
        "   (inspect_optimization_toolset, list_compiler_passes).\n"
        "3. Record a grounded hypothesis explaining the bottleneck.\n"
        "4. Propose 1-3 high-signal candidates — prefer pass interventions when the\n"
        "   bottleneck looks structural (layout, pipelining, coalescing).\n"
        "5. Run candidate benchmarks, compare, accept/reject with rationale.\n"
        "6. Record a reasoning summary, then write a report.\n"
    )
    lines.append(
        "Stop after the report is written. Do not exceed the candidate budget. "
        "Reasoning summaries must be grounded in measured evidence."
    )
    return "\n".join(lines)
