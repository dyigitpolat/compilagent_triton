from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .candidates import validate_candidate as validate_candidate_config
from .compiler import TritonCompileHarness
from .decision_traces import extract_decision_traces, summarize_decision_traces
from .episodes import EpisodeStore
from .events import ObservationEvent
from .experiment_memory import ExperimentMemory
from .policy import CandidatePolicy
from .schemas import CandidateConfig, CompileRequest, CompileResult, KernelSpec
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workspace import OptimizationWorkspace

READ_IR_TOOL = "inspect_ir"
WRITE_REPORT_TOOL = "write_report"
MUTATING_TOOLS = frozenset(
    {
        "register_kernel",
        "start_episode",
        "record_hypothesis",
        "run_candidate",
        "accept_or_reject_candidate",
        WRITE_REPORT_TOOL,
    }
)


class OptimizerRuntime:
    """Shared optimizer tool implementation for ACP-facing agent harnesses."""

    def __init__(
        self,
        *,
        settings: CompilagentSettings,
        workspace: OptimizationWorkspace,
        harness_label: str,
    ) -> None:
        self.settings = settings
        self.workspace = workspace
        self.harness_label = harness_label
        self.kernels: dict[str, KernelSpec] = {}
        self.compile_results: dict[str, CompileResult] = {}
        self.traces_by_run: dict[str, str] = {}
        self.episode_store = EpisodeStore(workspace)
        self.harness = TritonCompileHarness(workspace)
        self.trace_store = TraceStore(workspace.root).ensure()
        self.memory = ExperimentMemory(workspace.root)
        self.policy = CandidatePolicy(self.memory)
        self.emit(
            "agent.session_started",
            payload={**settings.public_metadata(), "harness": harness_label},
        )

    def emit(
        self,
        kind: str,
        *,
        episode_id: str | None = None,
        payload: dict[str, Any] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> ObservationEvent:
        return self.trace_store.emit(
            kind,
            episode_id=episode_id,
            payload=payload,
            artifact_paths=artifact_paths,
        )

    @contextmanager
    def observe_tool(
        self,
        tool: str,
        *,
        episode_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started = time.perf_counter()
        self.emit("tool.started", episode_id=episode_id, payload={"tool": tool, **(payload or {})})
        try:
            yield
        except Exception as exc:
            self.emit(
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
            self.emit(
                "tool.completed",
                episode_id=episode_id,
                payload={"tool": tool, "duration_ms": (time.perf_counter() - started) * 1000},
            )

    def describe_optimizer_surface(self) -> str:
        """Summarize the optimizer environment, constraints, and workflow."""

        with self.observe_tool("describe_optimizer_surface"):
            return "\n".join(
                (
                    "Triton ACP optimizer surface:",
                    f"- harness: {self.harness_label}",
                    "- inspect mode is read-only by default",
                    "- optimize mode allows approval-gated experiments and reports",
                    "- compiler workspaces are session-local",
                    "- API keys are read from environment only and never persisted",
                    f"- model: {self.settings.model_name}",
                    f"- reasoning effort: {self.settings.reasoning_effort}",
                )
            )

    def list_benchmarks(self) -> str:
        """List registered kernel specs and built-in benchmark families."""

        with self.observe_tool("list_benchmarks"):
            registered = "\n".join(f"- {spec.id}: {spec.name}" for spec in self.kernels.values())
            builtins = "\n".join(
                (
                    "- vector_copy: contiguous and strided memory movement",
                    "- vector_add: masked elementwise addition",
                    "- reduction: layout-sensitive reductions",
                    "- matmul: simplified persistent matmul family",
                )
            )
            learned = self.memory.summarize_priors()
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

    def register_kernel(
        self,
        kernel_id: str,
        name: str,
        path: str,
        entrypoint: str,
        shapes_json: str = "[]",
        dtypes_json: str = "[]",
    ) -> str:
        """Register a Triton kernel module for compile and benchmark experiments."""

        with self.observe_tool("register_kernel", payload={"kernel_id": kernel_id}):
            kernel_path = Path(path).expanduser()
            if not kernel_path.is_absolute():
                kernel_path = (self.workspace.session_cwd / kernel_path).resolve()
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
            self.kernels[spec.id] = spec
            return f"Registered kernel `{spec.id}` from `{kernel_path}`."

    def start_episode(self, kernel_id: str, objective: str, budget_json: str = "{}") -> str:
        """Start a file-backed optimization episode."""

        with self.observe_tool("start_episode", payload={"kernel_id": kernel_id}):
            budget = {
                "max_candidates": self.settings.max_candidates,
                "max_benchmark_seconds": self.settings.max_benchmark_seconds,
                **json.loads(budget_json),
            }
            episode = self.episode_store.create(
                kernel_id=kernel_id,
                objective=objective,
                budget=budget,
                model_metadata=self.settings.public_metadata(),
            )
            self.emit(
                "agent.episode_started",
                episode_id=episode.id,
                payload={"kernel_id": kernel_id, "objective": objective, "budget": budget},
                artifact_paths=[str(self.workspace.episode_path(episode.id))],
            )
            return episode.model_dump_json(indent=2, exclude_none=True)

    def compile_baseline(self, kernel_id: str, meta_json: str = "{}") -> str:
        """Compile the registered kernel baseline and capture artifacts."""

        with self.observe_tool("compile_baseline", payload={"kernel_id": kernel_id}):
            spec = self.require_kernel(kernel_id)
            request = CompileRequest(kernel_id=kernel_id, meta=json.loads(meta_json))
            result = self.harness.compile_kernel(
                spec,
                request,
                artifact_dir=self.workspace.baseline_dir(kernel_id),
                use_stage_hook=True,
            )
            self.compile_results[result.id] = result
            self.emit(
                "artifact.created",
                payload={"tool": "compile_baseline", "run_id": result.id, "ok": result.ok},
                artifact_paths=[str(artifact.path) for artifact in result.artifacts if artifact.path],
            )
            return result.model_dump_json(indent=2, exclude_none=True)

    def inspect_ir(self, run_id: str, stage: str = "ttgir", max_chars: int = 6000) -> str:
        """Read a captured IR artifact for a previous compile result."""

        with self.observe_tool("inspect_ir", payload={"run_id": run_id, "stage": stage}):
            result = self.require_compile_result(run_id)
            for artifact in result.artifacts:
                if artifact.stage == stage:
                    if artifact.inline_text is not None:
                        return truncate_text(artifact.inline_text, limit=max_chars)
                    if artifact.path is not None and artifact.path.exists():
                        return truncate_text(artifact.path.read_text(encoding="utf-8"), limit=max_chars)
            raise ValueError(f"No `{stage}` artifact found for run `{run_id}`.")

    def summarize_decisions(self, run_id: str, stage: str = "ttgir") -> str:
        """Extract and summarize coalescing and matmul decisions from captured IR."""

        with self.observe_tool("summarize_decisions", payload={"run_id": run_id, "stage": stage}):
            ir_text = self.inspect_ir(run_id, stage=stage, max_chars=200_000)
            traces = extract_decision_traces(ir_text, run_id=run_id)
            self.traces_by_run[run_id] = json.dumps(
                [trace.model_dump(mode="json", exclude_none=True) for trace in traces],
                indent=2,
            )
            self.emit(
                "decision_trace.created",
                payload={"tool": "summarize_decisions", "run_id": run_id, "trace_count": len(traces)},
            )
            return summarize_decision_traces(traces)

    def record_hypothesis(
        self,
        episode_id: str,
        statement: str,
        expected_effect: str,
        evidence_refs_json: str = "[]",
    ) -> str:
        """Record a grounded optimization hypothesis before experiments."""

        with self.observe_tool("record_hypothesis", episode_id=episode_id):
            hypothesis = self.episode_store.record_hypothesis(
                episode_id,
                statement=statement,
                expected_effect=expected_effect,
                evidence_refs=json.loads(evidence_refs_json),
            )
            self.emit(
                "hypothesis.recorded",
                episode_id=episode_id,
                payload=hypothesis.model_dump(mode="json", exclude_none=True),
            )
            return hypothesis.model_dump_json(indent=2, exclude_none=True)

    def propose_candidates(
        self,
        kernel_id: str,
        objective: str,
        budget: int = 2,
        hypothesis_id: str | None = None,
    ) -> str:
        """Return a small typed candidate set for the agent to inspect and validate."""

        with self.observe_tool("propose_candidates", payload={"kernel_id": kernel_id, "budget": budget}):
            count = max(1, min(budget, self.settings.max_candidates))
            candidates = self.policy.propose(
                kernel_id=kernel_id,
                objective=objective,
                budget=count,
                hypothesis_id=hypothesis_id,
            )
            self.emit(
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

    def validate_candidate(self, candidate_json: str) -> str:
        """Validate a typed candidate before compilation or benchmarking."""

        with self.observe_tool("validate_candidate"):
            candidate = CandidateConfig.model_validate_json(candidate_json)
            validation = validate_candidate_config(candidate)
            self.emit(
                "candidate.validated",
                payload={
                    "candidate_id": candidate.id,
                    "ok": validation.ok,
                    "diagnostics": validation.diagnostics,
                },
            )
            return validation.summary()

    def run_candidate(self, candidate_json: str, meta_json: str = "{}") -> str:
        """Compile a validated candidate and capture diagnostics."""

        with self.observe_tool("run_candidate"):
            candidate = CandidateConfig.model_validate_json(candidate_json)
            validation = validate_candidate_config(candidate)
            if not validation.ok:
                return validation.summary()
            spec = self.require_kernel(candidate.kernel_id)
            meta = {**json.loads(meta_json), **candidate.changes}
            request = CompileRequest(
                kernel_id=candidate.kernel_id,
                candidate_id=candidate.id,
                meta=meta,
                stage_hook_key=candidate.model_dump_json(exclude_none=True),
            )
            result = self.harness.compile_kernel(
                spec,
                request,
                artifact_dir=self.workspace.candidate_dir(candidate.kernel_id, candidate.id),
                use_stage_hook=True,
            )
            self.compile_results[result.id] = result
            self.emit(
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

    def compare_runs(self, baseline_id: str, candidate_ids_json: str) -> str:
        """Compare compile outcomes and artifact availability for candidate runs."""

        with self.observe_tool("compare_runs", payload={"baseline_id": baseline_id}):
            baseline = self.require_compile_result(baseline_id)
            candidates = [
                self.require_compile_result(run_id) for run_id in json.loads(candidate_ids_json)
            ]
            lines = [f"Baseline `{baseline.id}` ok={baseline.ok} artifacts={len(baseline.artifacts)}"]
            for candidate in candidates:
                lines.append(
                    f"Candidate `{candidate.id}` ok={candidate.ok} "
                    f"candidate_id={candidate.candidate_id} artifacts={len(candidate.artifacts)}"
                )
            self.emit(
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

    def accept_or_reject_candidate(
        self,
        episode_id: str,
        candidate_id: str,
        status: str,
        rationale: str,
    ) -> str:
        """Record the agent's judgment for a candidate."""

        with self.observe_tool("accept_or_reject_candidate", episode_id=episode_id):
            from .schemas import CandidateStatus

            episode = self.episode_store.set_candidate_status(
                episode_id,
                candidate_id,
                CandidateStatus(status),
                rationale,
            )
            self.emit(
                "candidate.judged",
                episode_id=episode_id,
                payload={"candidate_id": candidate_id, "status": status, "rationale": rationale},
            )
            return episode.model_dump_json(indent=2, exclude_none=True)

    def write_report(self, episode_id: str) -> str:
        """Write a human-readable optimization report for an episode."""

        with self.observe_tool("write_report", episode_id=episode_id):
            path = self.episode_store.write_report(episode_id)
            self.emit(
                "artifact.created",
                episode_id=episode_id,
                payload={"tool": "write_report", "path": str(path)},
                artifact_paths=[str(path)],
            )
            self.emit(
                "loop.summary",
                episode_id=episode_id,
                payload={"episode_id": episode_id, "report_path": str(path)},
                artifact_paths=[str(path)],
            )
            return f"Wrote report `{path.relative_to(self.workspace.root)}`."

    def require_kernel(self, kernel_id: str) -> KernelSpec:
        try:
            return self.kernels[kernel_id]
        except KeyError as exc:
            raise ValueError(f"Unknown kernel id: {kernel_id}") from exc

    def require_compile_result(self, run_id: str) -> CompileResult:
        try:
            return self.compile_results[run_id]
        except KeyError as exc:
            raise ValueError(f"Unknown compile run id: {run_id}") from exc


def optimizer_instructions(
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
    *,
    harness_label: str,
) -> str:
    return "\n".join(
        (
            "You are a compiler optimization researcher operating through ACP tools.",
            f"Active harness: {harness_label}.",
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


def truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    suffix = f"\n... truncated {len(text) - limit} chars ..."
    return text[: max(0, limit - len(suffix))] + suffix
