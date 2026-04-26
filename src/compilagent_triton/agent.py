from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic_acp import (
    AdapterConfig,
    FileSystemProjectionMap,
    MemorySessionStore,
    NativeApprovalBridge,
    PrepareToolsBridge,
    PrepareToolsMode,
    ThinkingBridge,
    run_acp,
    truncate_text,
)
from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition

from .candidates import validate_candidate as validate_candidate_config
from .claude_agent_harness import build_claude_sdk_proxy_agent
from .compiler import TritonCompileHarness
from .decision_traces import extract_decision_traces, summarize_decision_traces
from .episodes import EpisodeStore
from .events import ObservationEvent
from .experiment_memory import ExperimentMemory
from .harness_selection import (
    CLAUDE_AGENT_SDK_HARNESS,
    HarnessConfigOptionsProvider,
    selected_harness,
)
from .llm import model_for_settings
from .optimizer_runtime import (
    MUTATING_TOOLS,
    READ_IR_TOOL,
    WRITE_REPORT_TOOL,
    OptimizerRuntime,
    optimizer_instructions,
)
from .policy import CandidatePolicy
from .schemas import CandidateConfig, CompileRequest, CompileResult, KernelSpec
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workspace import OptimizationWorkspace

_READ_IR_TOOL = READ_IR_TOOL
_WRITE_REPORT_TOOL = WRITE_REPORT_TOOL
_MUTATING_TOOLS = MUTATING_TOOLS


def _read_only_tools(
    ctx: RunContext[None],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    del ctx
    return [tool_def for tool_def in tool_defs if tool_def.name not in _MUTATING_TOOLS]


def _all_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    del ctx
    return list(tool_defs)


def agent_factory_from_session(session: Any) -> Agent[None, str | DeferredToolRequests]:
    settings = CompilagentSettings.from_env(project_root=Path(session.cwd))
    workspace = OptimizationWorkspace(
        session_cwd=Path(session.cwd),
        root_name=settings.workspace_dir_name,
    ).ensure()
    harness = selected_harness(session, default=settings.harness)
    if harness == CLAUDE_AGENT_SDK_HARNESS:
        mode = str(getattr(session, "config_values", {}).get("mode", "inspect"))
        return build_claude_sdk_proxy_agent(settings=settings, workspace=workspace, mode=mode)
    return build_optimizer_agent(settings=settings, workspace=workspace)


def _legacy_build_optimizer_agent(
    *,
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
) -> Agent[None, str | DeferredToolRequests]:
    kernels: dict[str, KernelSpec] = {}
    compile_results: dict[str, CompileResult] = {}
    traces_by_run: dict[str, str] = {}
    episode_store = EpisodeStore(workspace)
    harness = TritonCompileHarness(workspace)
    trace_store = TraceStore(workspace.root).ensure()
    memory = ExperimentMemory(workspace.root)
    policy = CandidatePolicy(memory)

    def emit(
        kind: str,
        *,
        episode_id: str | None = None,
        payload: dict[str, Any] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> ObservationEvent:
        return trace_store.emit(
            kind,
            episode_id=episode_id,
            payload=payload,
            artifact_paths=artifact_paths,
        )

    @contextmanager
    def observe_tool(
        tool: str,
        *,
        episode_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ):
        started = time.perf_counter()
        emit("tool.started", episode_id=episode_id, payload={"tool": tool, **(payload or {})})
        try:
            yield
        except Exception as exc:
            emit(
                "tool.failed",
                episode_id=episode_id,
                payload={
                    "tool": tool,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        else:
            emit(
                "tool.completed",
                episode_id=episode_id,
                payload={"tool": tool, "duration_ms": (time.perf_counter() - started) * 1000},
            )

    emit("agent.session_started", payload=settings.public_metadata())

    agent = Agent(
        model_for_settings(settings),
        name="triton-acp-optimizer",
        output_type=[str, DeferredToolRequests],
        instructions=_instructions(settings, workspace),
    )

    @agent.tool_plain
    def describe_optimizer_surface() -> str:
        """Summarize the optimizer environment, constraints, and workflow."""

        with observe_tool("describe_optimizer_surface"):
            return "\n".join(
                (
                    "Triton ACP optimizer surface:",
                    "- pydantic-ai agent exposed through pydantic-acp",
                    "- inspect mode is read-only by default",
                    "- optimize mode allows approval-gated experiments and reports",
                    "- compiler workspaces are session-local",
                    "- API keys are read from environment only and never persisted",
                    f"- model: {settings.model_name}",
                    f"- reasoning effort: {settings.reasoning_effort}",
                )
            )

    @agent.tool_plain
    def list_benchmarks() -> str:
        """List registered kernel specs and built-in benchmark families."""

        with observe_tool("list_benchmarks"):
            registered = "\n".join(f"- {spec.id}: {spec.name}" for spec in kernels.values())
            builtins = "\n".join(
                (
                    "- vector_copy: contiguous and strided memory movement",
                    "- vector_add: masked elementwise addition",
                    "- reduction: layout-sensitive reductions",
                    "- matmul: simplified persistent matmul family",
                )
            )
            learned = memory.summarize_priors()
            return "\n\n".join(
                (
                    "Built-in benchmark families:",
                    builtins,
                    "Empirical priors:",
                    learned,
                    "Registered kernels:",
                    registered or "- none",
                )
            )

    @agent.tool_plain(name="register_kernel", requires_approval=True)
    def register_kernel(
        kernel_id: str,
        name: str,
        path: str,
        entrypoint: str,
        shapes_json: str = "[]",
        dtypes_json: str = "[]",
    ) -> str:
        """Register a Triton kernel module for compile and benchmark experiments."""

        with observe_tool("register_kernel", payload={"kernel_id": kernel_id}):
            kernel_path = Path(path).expanduser()
            if not kernel_path.is_absolute():
                kernel_path = (workspace.session_cwd / kernel_path).resolve()
            shapes = [tuple(shape) for shape in json.loads(shapes_json)]
            dtypes = list(json.loads(dtypes_json))
            spec = KernelSpec(
                id=kernel_id,
                name=name,
                path=kernel_path,
                entrypoint=entrypoint,
                shapes=shapes,
                dtypes=dtypes,
            )
            kernels[spec.id] = spec
            return f"Registered kernel `{spec.id}` from `{kernel_path}`."

    @agent.tool_plain(name="start_episode", requires_approval=True)
    def start_episode(kernel_id: str, objective: str, budget_json: str = "{}") -> str:
        """Start a file-backed optimization episode."""

        with observe_tool("start_episode", payload={"kernel_id": kernel_id}):
            budget = {
                "max_candidates": settings.max_candidates,
                "max_benchmark_seconds": settings.max_benchmark_seconds,
                **json.loads(budget_json),
            }
            episode = episode_store.create(
                kernel_id=kernel_id,
                objective=objective,
                budget=budget,
                model_metadata=settings.public_metadata(),
            )
            emit(
                "agent.episode_started",
                episode_id=episode.id,
                payload={"kernel_id": kernel_id, "objective": objective, "budget": budget},
                artifact_paths=[str(workspace.episode_path(episode.id))],
            )
            return episode.model_dump_json(indent=2, exclude_none=True)

    @agent.tool_plain
    def compile_baseline(kernel_id: str, meta_json: str = "{}") -> str:
        """Compile the registered kernel baseline and capture artifacts."""

        with observe_tool("compile_baseline", payload={"kernel_id": kernel_id}):
            spec = _require_kernel(kernels, kernel_id)
            request = CompileRequest(kernel_id=kernel_id, meta=json.loads(meta_json))
            result = harness.compile_kernel(
                spec,
                request,
                artifact_dir=workspace.baseline_dir(kernel_id),
                use_stage_hook=True,
            )
            compile_results[result.id] = result
            emit(
                "artifact.created",
                payload={"tool": "compile_baseline", "run_id": result.id, "ok": result.ok},
                artifact_paths=[str(artifact.path) for artifact in result.artifacts if artifact.path],
            )
            return result.model_dump_json(indent=2, exclude_none=True)

    @agent.tool_plain(name=_READ_IR_TOOL)
    def inspect_ir(run_id: str, stage: str = "ttgir", max_chars: int = 6000) -> str:
        """Read a captured IR artifact for a previous compile result."""

        with observe_tool("inspect_ir", payload={"run_id": run_id, "stage": stage}):
            result = _require_compile_result(compile_results, run_id)
            for artifact in result.artifacts:
                if artifact.stage == stage:
                    if artifact.inline_text is not None:
                        return truncate_text(artifact.inline_text, limit=max_chars)
                    if artifact.path is not None and artifact.path.exists():
                        return truncate_text(artifact.path.read_text(encoding="utf-8"), limit=max_chars)
            raise ValueError(f"No `{stage}` artifact found for run `{run_id}`.")

    @agent.tool_plain
    def summarize_decisions(run_id: str, stage: str = "ttgir") -> str:
        """Extract and summarize coalescing and matmul decisions from captured IR."""

        with observe_tool("summarize_decisions", payload={"run_id": run_id, "stage": stage}):
            ir_text = inspect_ir(run_id, stage=stage, max_chars=200_000)
            traces = extract_decision_traces(ir_text, run_id=run_id)
            traces_by_run[run_id] = json.dumps(
                [trace.model_dump(mode="json", exclude_none=True) for trace in traces],
                indent=2,
            )
            emit(
                "decision_trace.created",
                payload={"tool": "summarize_decisions", "run_id": run_id, "trace_count": len(traces)},
            )
            return summarize_decision_traces(traces)

    @agent.tool_plain(name="record_hypothesis", requires_approval=True)
    def record_hypothesis(
        episode_id: str,
        statement: str,
        expected_effect: str,
        evidence_refs_json: str = "[]",
    ) -> str:
        """Record a grounded optimization hypothesis before experiments."""

        with observe_tool("record_hypothesis", episode_id=episode_id):
            hypothesis = episode_store.record_hypothesis(
                episode_id,
                statement=statement,
                expected_effect=expected_effect,
                evidence_refs=json.loads(evidence_refs_json),
            )
            emit(
                "hypothesis.recorded",
                episode_id=episode_id,
                payload=hypothesis.model_dump(mode="json", exclude_none=True),
            )
            return hypothesis.model_dump_json(indent=2, exclude_none=True)

    @agent.tool_plain
    def propose_candidates(
        kernel_id: str,
        objective: str,
        budget: int = 2,
        hypothesis_id: str | None = None,
    ) -> str:
        """Return a small typed candidate set for the agent to inspect and validate."""

        with observe_tool("propose_candidates", payload={"kernel_id": kernel_id, "budget": budget}):
            count = max(1, min(budget, settings.max_candidates))
            candidates = policy.propose(
                kernel_id=kernel_id,
                objective=objective,
                budget=count,
                hypothesis_id=hypothesis_id,
            )
            emit(
                "candidate.proposed",
                payload={
                    "kernel_id": kernel_id,
                    "objective": objective,
                    "count": len(candidates),
                    "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                },
            )
            return json.dumps(
                [candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates],
                indent=2,
            )

    @agent.tool_plain
    def validate_candidate(candidate_json: str) -> str:
        """Validate a typed candidate before compilation or benchmarking."""

        with observe_tool("validate_candidate"):
            candidate = CandidateConfig.model_validate_json(candidate_json)
            validation = validate_candidate_config(candidate)
            emit(
                "candidate.validated",
                payload={
                    "candidate_id": candidate.id,
                    "ok": validation.ok,
                    "diagnostics": validation.diagnostics,
                },
            )
            return validation.summary()

    @agent.tool_plain(name="run_candidate", requires_approval=True)
    def run_candidate(candidate_json: str, meta_json: str = "{}") -> str:
        """Compile a validated candidate and capture diagnostics."""

        with observe_tool("run_candidate"):
            candidate = CandidateConfig.model_validate_json(candidate_json)
            validation = validate_candidate_config(candidate)
            if not validation.ok:
                return validation.summary()
            spec = _require_kernel(kernels, candidate.kernel_id)
            meta = {**json.loads(meta_json), **candidate.changes}
            request = CompileRequest(
                kernel_id=candidate.kernel_id,
                candidate_id=candidate.id,
                meta=meta,
                stage_hook_key=candidate.model_dump_json(exclude_none=True),
            )
            result = harness.compile_kernel(
                spec,
                request,
                artifact_dir=workspace.candidate_dir(candidate.kernel_id, candidate.id),
                use_stage_hook=True,
            )
            compile_results[result.id] = result
            emit(
                "artifact.created",
                payload={
                    "tool": "run_candidate",
                    "run_id": result.id,
                    "candidate_id": candidate.id,
                    "ok": result.ok,
                },
                artifact_paths=[str(artifact.path) for artifact in result.artifacts if artifact.path],
            )
            return result.model_dump_json(indent=2, exclude_none=True)

    @agent.tool_plain
    def compare_runs(baseline_id: str, candidate_ids_json: str) -> str:
        """Compare compile outcomes and artifact availability for candidate runs."""

        with observe_tool("compare_runs", payload={"baseline_id": baseline_id}):
            baseline = _require_compile_result(compile_results, baseline_id)
            candidates = [
                _require_compile_result(compile_results, run_id) for run_id in json.loads(candidate_ids_json)
            ]
            lines = [f"Baseline `{baseline.id}` ok={baseline.ok} artifacts={len(baseline.artifacts)}"]
            for candidate in candidates:
                lines.append(
                    f"Candidate `{candidate.id}` ok={candidate.ok} "
                    f"candidate_id={candidate.candidate_id} artifacts={len(candidate.artifacts)}"
                )
            emit(
                "comparison.created",
                payload={
                    "tool": "compare_runs",
                    "baseline_id": baseline_id,
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "run_id": candidate.id,
                            "candidate_id": candidate.candidate_id,
                            "ok": candidate.ok,
                            "artifact_count": len(candidate.artifacts),
                        }
                        for candidate in candidates
                    ],
                },
            )
            return "\n".join(lines)

    @agent.tool_plain(name="accept_or_reject_candidate", requires_approval=True)
    def accept_or_reject_candidate(
        episode_id: str,
        candidate_id: str,
        status: str,
        rationale: str,
    ) -> str:
        """Record the agent's judgment for a candidate."""

        with observe_tool("accept_or_reject_candidate", episode_id=episode_id):
            from .schemas import CandidateStatus

            episode = episode_store.set_candidate_status(
                episode_id,
                candidate_id,
                CandidateStatus(status),
                rationale,
            )
            emit(
                "candidate.judged",
                episode_id=episode_id,
                payload={"candidate_id": candidate_id, "status": status, "rationale": rationale},
            )
            return episode.model_dump_json(indent=2, exclude_none=True)

    @agent.tool_plain(name=_WRITE_REPORT_TOOL, requires_approval=True)
    def write_report(episode_id: str) -> str:
        """Write a human-readable optimization report for an episode."""

        with observe_tool("write_report", episode_id=episode_id):
            path = episode_store.write_report(episode_id)
            emit(
                "artifact.created",
                episode_id=episode_id,
                payload={"tool": "write_report", "path": str(path)},
                artifact_paths=[str(path)],
            )
            emit(
                "loop.summary",
                episode_id=episode_id,
                payload={"episode_id": episode_id, "report_path": str(path)},
                artifact_paths=[str(path)],
            )
            return f"Wrote report `{path.relative_to(workspace.root)}`."

    return agent


def build_optimizer_agent(
    *,
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
    harness_label: str = "pydantic-ai",
) -> Agent[None, str | DeferredToolRequests]:
    runtime = OptimizerRuntime(
        settings=settings,
        workspace=workspace,
        harness_label=harness_label,
    )
    agent = Agent(
        model_for_settings(settings),
        name="triton-acp-optimizer",
        output_type=[str, DeferredToolRequests],
        instructions=optimizer_instructions(
            settings,
            workspace,
            harness_label=harness_label,
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

    @agent.tool_plain(name="register_kernel", requires_approval=True)
    def register_kernel(
        kernel_id: str,
        name: str,
        path: str,
        entrypoint: str,
        shapes_json: str = "[]",
        dtypes_json: str = "[]",
    ) -> str:
        """Register a Triton kernel module for compile and benchmark experiments."""

        return runtime.register_kernel(
            kernel_id=kernel_id,
            name=name,
            path=path,
            entrypoint=entrypoint,
            shapes_json=shapes_json,
            dtypes_json=dtypes_json,
        )

    @agent.tool_plain(name="start_episode", requires_approval=True)
    def start_episode(kernel_id: str, objective: str, budget_json: str = "{}") -> str:
        """Start a file-backed optimization episode."""

        return runtime.start_episode(kernel_id=kernel_id, objective=objective, budget_json=budget_json)

    @agent.tool_plain
    def compile_baseline(kernel_id: str, meta_json: str = "{}") -> str:
        """Compile the registered kernel baseline and capture artifacts."""

        return runtime.compile_baseline(kernel_id=kernel_id, meta_json=meta_json)

    @agent.tool_plain(name=_READ_IR_TOOL)
    def inspect_ir(run_id: str, stage: str = "ttgir", max_chars: int = 6000) -> str:
        """Read a captured IR artifact for a previous compile result."""

        return runtime.inspect_ir(run_id=run_id, stage=stage, max_chars=max_chars)

    @agent.tool_plain
    def summarize_decisions(run_id: str, stage: str = "ttgir") -> str:
        """Extract and summarize coalescing and matmul decisions from captured IR."""

        return runtime.summarize_decisions(run_id=run_id, stage=stage)

    @agent.tool_plain(name="record_hypothesis", requires_approval=True)
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
    def propose_candidates(
        kernel_id: str,
        objective: str,
        budget: int = 2,
        hypothesis_id: str | None = None,
    ) -> str:
        """Return a small typed candidate set for the agent to inspect and validate."""

        return runtime.propose_candidates(
            kernel_id=kernel_id,
            objective=objective,
            budget=budget,
            hypothesis_id=hypothesis_id,
        )

    @agent.tool_plain
    def validate_candidate(candidate_json: str) -> str:
        """Validate a typed candidate before compilation or benchmarking."""

        return runtime.validate_candidate(candidate_json=candidate_json)

    @agent.tool_plain(name="run_candidate", requires_approval=True)
    def run_candidate(candidate_json: str, meta_json: str = "{}") -> str:
        """Compile a validated candidate and capture diagnostics."""

        return runtime.run_candidate(candidate_json=candidate_json, meta_json=meta_json)

    @agent.tool_plain
    def compare_runs(baseline_id: str, candidate_ids_json: str) -> str:
        """Compare compile outcomes and artifact availability for candidate runs."""

        return runtime.compare_runs(baseline_id=baseline_id, candidate_ids_json=candidate_ids_json)

    @agent.tool_plain(name="accept_or_reject_candidate", requires_approval=True)
    def accept_or_reject_candidate(
        episode_id: str,
        candidate_id: str,
        status: str,
        rationale: str,
    ) -> str:
        """Record the agent's judgment for a candidate."""

        return runtime.accept_or_reject_candidate(
            episode_id=episode_id,
            candidate_id=candidate_id,
            status=status,
            rationale=rationale,
        )

    @agent.tool_plain(name=_WRITE_REPORT_TOOL, requires_approval=True)
    def write_report(episode_id: str) -> str:
        """Write a human-readable optimization report for an episode."""

        return runtime.write_report(episode_id=episode_id)

    return agent


def build_config() -> AdapterConfig:
    return AdapterConfig(
        session_store=MemorySessionStore(),
        config_options_provider=HarnessConfigOptionsProvider(),
        capability_bridges=[
            ThinkingBridge(),
            PrepareToolsBridge(
                default_mode_id="inspect",
                default_plan_generation_type="structured",
                modes=[
                    PrepareToolsMode(
                        id="inspect",
                        name="Inspect",
                        description="Read-only compiler and benchmark inspection.",
                        prepare_func=_read_only_tools,
                    ),
                    PrepareToolsMode(
                        id="plan",
                        name="Plan",
                        description="Create structured optimization plans without mutations.",
                        prepare_func=_read_only_tools,
                        plan_mode=True,
                    ),
                    PrepareToolsMode(
                        id="optimize",
                        name="Optimize",
                        description="Run approval-gated compile experiments and reports.",
                        prepare_func=_all_tools,
                        plan_tools=True,
                    ),
                ],
            ),
        ],
        approval_bridge=NativeApprovalBridge(),
        projection_maps=[
            FileSystemProjectionMap(
                default_read_tool=_READ_IR_TOOL,
                default_write_tool=_WRITE_REPORT_TOOL,
            )
        ],
    )


def main() -> None:
    run_acp(agent_factory=agent_factory_from_session, config=build_config())


def _instructions(settings: CompilagentSettings, workspace: OptimizationWorkspace) -> str:
    return "\n".join(
        (
            "You are a compiler optimization researcher operating through ACP tools.",
            "Use a hypothesis-first workflow: observe, explain, propose, validate, run, judge.",
            "Never claim a performance win without correctness evidence and paired baseline comparison.",
            "Prefer small high-signal experiments over broad blind sweeps.",
            "Preserve failed candidates as evidence.",
            "Do not print, store, or reason about API keys.",
            f"Workspace root: {workspace.root}",
            f"Model: {settings.model_name}",
            f"Reasoning effort: {settings.reasoning_effort}",
            f"Max candidates per iteration: {settings.max_candidates}",
        )
    )


def _require_kernel(kernels: dict[str, KernelSpec], kernel_id: str) -> KernelSpec:
    try:
        return kernels[kernel_id]
    except KeyError as exc:
        raise ValueError(f"Unknown kernel id: {kernel_id}") from exc


def _require_compile_result(results: dict[str, CompileResult], run_id: str) -> CompileResult:
    try:
        return results[run_id]
    except KeyError as exc:
        raise ValueError(f"Unknown compile run id: {run_id}") from exc
