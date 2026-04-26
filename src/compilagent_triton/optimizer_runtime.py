from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .candidates import validate_candidate as validate_candidate_config
from .compiler import TritonCompileHarness
from .decision_traces import extract_decision_traces, summarize_decision_traces
from .episodes import EpisodeStore
from .events import ObservationEvent
from .experiment_memory import ExperimentMemory
from .optimization_toolset import build_optimization_toolset, search_space_from_toolset
from .policy import CandidatePolicy
from .triton_hooks.pipeline import PassIntervention, PassResult
from .schemas import (
    BenchmarkResult,
    CandidateConfig,
    CandidateKind,
    CompileRequest,
    CompileResult,
    KernelSpec,
    ReasoningSummary,
)
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
        "record_reasoning_summary",
        "propose_candidate_from_toolset",
        "run_candidate",
        "run_baseline_benchmark",
        "run_candidate_benchmark",
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
        self.proposed_candidates: dict[str, CandidateConfig] = {}
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
            trace_dicts = [trace.model_dump(mode="json", exclude_none=True) for trace in traces]
            self.traces_by_run[run_id] = json.dumps(trace_dicts, indent=2)
            for trace in trace_dicts:
                self.emit(
                    "decision_trace.created",
                    payload={
                        "tool": "summarize_decisions",
                        "run_id": run_id,
                        "stage": stage,
                        **trace,
                    },
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

    def record_reasoning_summary(
        self,
        episode_id: str,
        summary: str,
        linked_hypothesis_id: str | None = None,
        linked_candidate_id: str | None = None,
        evidence_refs_json: str = "[]",
        next_step: str | None = None,
    ) -> str:
        """Record a concise visible decision rationale, not hidden chain-of-thought."""

        with self.observe_tool("record_reasoning_summary", episode_id=episode_id):
            reasoning = ReasoningSummary(
                summary=summary,
                linked_hypothesis_id=linked_hypothesis_id,
                linked_candidate_id=linked_candidate_id,
                evidence_refs=json.loads(evidence_refs_json),
                next_step=next_step,
            )
            self.episode_store.add_reasoning_summary(episode_id, reasoning)
            payload = reasoning.model_dump(mode="json", exclude_none=True)
            self.emit("agent.reasoning_summary", episode_id=episode_id, payload=payload)
            if next_step:
                self.emit(
                    "experiment.next_step",
                    episode_id=episode_id,
                    payload={"reasoning_id": reasoning.id, "next_step": next_step},
                )
            return reasoning.model_dump_json(indent=2, exclude_none=True)

    def list_compiler_passes(self, stage: str = "ttgir") -> str:
        """List the MLIR passes available at a given stage on the active GPU.

        Stage names: `ttir` or `ttgir`. The list reflects the agent's actual
        leverage surface — only passes whose capability gate matches the local
        device are returned, plus the parameter signature for each.
        """

        from .triton_hooks.passes import describe_pass, list_passes_for_capability
        from .triton_hooks.pipeline import build_ttgir_plan, build_ttir_plan

        with self.observe_tool("list_compiler_passes", payload={"stage": stage}):
            stage = stage.lower()
            capability = self._device_capability()
            passes_for_cap = list_passes_for_capability(capability)
            origin_filter = {"common", "ttir"} if stage == "ttir" else {
                "common", "ttir", "ttgpuir", "ttnvgpuir", "hopper",
            }
            named = [d.name for d in passes_for_cap if d.origin in origin_filter]
            if stage == "ttir":
                plan = build_ttir_plan(capability=capability, num_warps=4, num_stages=3, num_ctas=1)
            else:
                plan = build_ttgir_plan(capability=capability, num_warps=4, num_stages=3, num_ctas=1)
            ordered = plan.names()
            payload = {
                "stage": stage,
                "capability": capability,
                "default_pipeline": ordered,
                "passes": [describe_pass(name) for name in named],
            }
            return json.dumps(payload, indent=2)

    def describe_compiler_pass(self, pass_name: str) -> str:
        """Return the catalog entry for a single compiler pass."""

        from .triton_hooks.passes import describe_pass

        with self.observe_tool("describe_compiler_pass", payload={"pass": pass_name}):
            try:
                desc = describe_pass(pass_name)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            return json.dumps(desc, indent=2)

    def read_pass_source(self, pass_name: str, max_chars: int = 8000) -> str:
        """Best-effort lookup of the upstream MLIR pass source text.

        We grep the vendored Triton submodule for the pass name. The agent uses
        this to ground its theories about what a pass actually does.
        """

        from .triton_hooks.passes import get_pass

        with self.observe_tool("read_pass_source", payload={"pass": pass_name}):
            try:
                desc = get_pass(pass_name)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            triton_root = Path(self.settings.triton_path)
            if not triton_root.exists():
                return json.dumps(
                    {
                        "pass": desc.name,
                        "source_text": None,
                        "error": f"triton submodule not at {triton_root}",
                    },
                    indent=2,
                )
            hits: list[dict[str, Any]] = []
            haystacks = [
                triton_root / "lib" / "Dialect",
                triton_root / "third_party" / "nvidia" / "lib",
            ]
            # Build a list of plausible needles to grep for.
            short = desc.name.split("-", 1)[-1]  # strip "tritongpu-" / "ttng-" prefix
            camel = "".join(part.capitalize() for part in short.split("-"))
            needles = [desc.name, camel, short.replace("-", "_")]
            for root in haystacks:
                if not root.exists():
                    continue
                for path in root.rglob("*.cpp"):
                    text = path.read_text(encoding="utf-8", errors="replace")
                    for needle in needles:
                        if needle and needle in text:
                            snippet = _extract_snippet(text, needle)
                            if snippet:
                                hits.append({
                                    "path": str(path.relative_to(triton_root)),
                                    "needle": needle,
                                    "snippet": snippet[:max_chars],
                                })
                                break
                    if len(hits) >= 3:
                        break
                if len(hits) >= 3:
                    break
            return json.dumps(
                {
                    "pass": desc.name,
                    "wrapper": f"{desc.origin}.{desc.pyname}",
                    "purpose": desc.purpose,
                    "params": list(desc.params),
                    "hits": hits,
                },
                indent=2,
            )

    def propose_pass_intervention(
        self,
        kernel_id: str,
        interventions_json: str,
        description: str,
        expected_effect: str,
        hypothesis_id: str | None = None,
    ) -> str:
        """Build a `PASS_INTERVENTIONS` candidate from agent-supplied interventions.

        `interventions_json` must be a JSON list of
        {pass_name, action, args?, rationale?} dicts.
        """

        with self.observe_tool(
            "propose_pass_intervention",
            payload={"kernel_id": kernel_id},
        ):
            self.require_kernel(kernel_id)
            interventions = json.loads(interventions_json)
            if not isinstance(interventions, list) or not interventions:
                raise ValueError("interventions_json must be a non-empty JSON list")
            candidate = CandidateConfig(
                kernel_id=kernel_id,
                kind=CandidateKind.PASS_INTERVENTIONS,
                description=description,
                expected_effect=expected_effect,
                hypothesis_id=hypothesis_id,
                changes={"pass_interventions": interventions},
                validation_constraints=[
                    "pass_name must be in the catalog from list_compiler_passes",
                    "action must be one of run, skip, replace",
                ],
            )
            self.proposed_candidates[candidate.id] = candidate
            self.emit(
                "candidate.proposed",
                payload={
                    "kernel_id": kernel_id,
                    "source": "pass_interventions",
                    "count": 1,
                    "candidates": [candidate.model_dump(mode="json")],
                },
            )
            return candidate.model_dump_json(indent=2, exclude_none=True)

    def _device_capability(self) -> int:
        """Return the active CUDA compute capability as an int (e.g. 120 for sm_120)."""

        try:
            import torch  # type: ignore[import-not-found]
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                return major * 10 + minor
        except Exception:
            pass
        return 80  # safe default

    def inspect_search_space(self, kernel_id: str, *, backend_id: str = "triton") -> str:
        """Return the *derived* lever catalog for the active backend.

        Unlike `inspect_optimization_toolset` (the legacy hand-coded surface),
        every lever returned here carries `range.candidates` automatically
        computed from the workload's analysis (tensor shapes, op counts, IR
        diffs) plus the device capability. The lever's `evidence` field cites
        the signal that produced it. **No hand-coded value lists are emitted.**
        """

        import json as _json
        from .backends import backend_registry
        from .backends.base import Analysis

        with self.observe_tool(
            "inspect_search_space",
            payload={"kernel_id": kernel_id, "backend_id": backend_id},
        ):
            spec = self.require_kernel(kernel_id)
            backend = backend_registry.get(backend_id)
            cap = backend.device_capability()
            compile_results = [r for r in self.compile_results.values() if r.kernel_id == kernel_id]
            artifacts: list = []
            for r in compile_results:
                artifacts.extend(r.artifacts)
            analysis = backend.analyze(spec, baseline_artifacts=artifacts)
            # Enrich with the device-capability + a (possibly empty) pass-impact log
            extra = dict(analysis.extra)
            extra.setdefault("device_capability_int", cap.capability_int)
            analysis = Analysis(summary=analysis.summary, extra=extra)
            search_space = backend.derive_search_space(spec, analysis)
            payload = {
                "workload_id": getattr(spec, "id", kernel_id),
                "backend_id": backend.id,
                "device": {
                    "arch": cap.arch,
                    "name": cap.name,
                    "capability_int": cap.capability_int,
                    "memory_total_bytes": cap.memory_total_bytes,
                    "peak_bandwidth_gbps": cap.memory_peak_bandwidth_gbps,
                },
                "levers": [lev.serialize() for lev in search_space.levers],
                "usage_hint": (
                    "Each lever's `range.candidates` is derived; pick from those values "
                    "or interpolate within the typed bounds. The `evidence` field cites "
                    "the signal that produced the lever."
                ),
            }
            return _json.dumps(payload, indent=2)

    def inspect_optimization_toolset(self, kernel_id: str) -> str:
        """Return available optimization levers, constraints, defaults, and evidence.

        The output explicitly echoes `kind_value` for each lever so the agent
        passes the right CandidateKind string (e.g. `"meta_parameters"`) to
        `propose_candidate_from_toolset`, not the lever id (e.g. `"launch_meta"`).
        """

        with self.observe_tool("inspect_optimization_toolset", payload={"kernel_id": kernel_id}):
            spec = self.require_kernel(kernel_id)
            compiler_evidence = {
                "compile_runs": [
                    result.model_dump(mode="json", exclude_none=True)
                    for result in self.compile_results.values()
                    if result.kernel_id == kernel_id
                ],
                "decision_traces": self.traces_by_run,
            }
            toolset = build_optimization_toolset(
                spec,
                compiler_evidence=compiler_evidence,
                prior_evidence=self.memory.summarize_priors(),
            )
            payload = toolset.model_dump(mode="json", exclude_none=True)
            payload["levers"] = [lever.with_hint() for lever in toolset.levers]
            payload["usage_hint"] = (
                "When calling `propose_candidate_from_toolset`, pass the lever's "
                "`kind_value` as the `kind` argument (NOT the `id`). Valid kinds: "
                + ", ".join(k.value for k in CandidateKind)
            )
            return json.dumps(payload, indent=2)

    def propose_candidate_from_toolset(
        self,
        kernel_id: str,
        kind: str,
        changes_json: str,
        description: str,
        expected_effect: str,
        hypothesis_id: str | None = None,
    ) -> str:
        """Create a validator-backed candidate from agent-selected toolset levers."""

        with self.observe_tool("propose_candidate_from_toolset", payload={"kernel_id": kernel_id, "kind": kind}):
            spec = self.require_kernel(kernel_id)
            toolset = build_optimization_toolset(spec, prior_evidence=self.memory.summarize_priors())
            # Accept either the CandidateKind value (e.g. "meta_parameters") or
            # the lever id (e.g. "launch_meta") for forgiveness.
            kind_value = kind
            try:
                resolved_kind = CandidateKind(kind_value)
            except ValueError:
                lever_match = next((lever for lever in toolset.levers if lever.id == kind), None)
                if lever_match is None:
                    valid = [k.value for k in CandidateKind] + [
                        lever.id for lever in toolset.levers
                    ]
                    raise ValueError(
                        f"`{kind}` is neither a CandidateKind nor a known lever id. "
                        f"Valid: {valid}"
                    ) from None
                resolved_kind = lever_match.kind
            search_space = search_space_from_toolset(toolset)
            candidate = search_space.candidate_from_toolset(
                kind=resolved_kind,
                description=description,
                changes=json.loads(changes_json),
                expected_effect=expected_effect,
                hypothesis_id=hypothesis_id,
            )
            self.proposed_candidates[candidate.id] = candidate
            self.emit(
                "candidate.proposed",
                payload={
                    "kernel_id": kernel_id,
                    "source": "toolset",
                    "count": 1,
                    "candidates": [candidate.model_dump(mode="json")],
                },
            )
            if hypothesis_id:
                self.emit(
                    "candidate.rationale",
                    payload={
                        "candidate_id": candidate.id,
                        "hypothesis_id": hypothesis_id,
                        "rationale": expected_effect,
                    },
                )
            return candidate.model_dump_json(indent=2, exclude_none=True)

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
            for candidate in candidates:
                self.proposed_candidates[candidate.id] = candidate
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
        """Validate a typed candidate before compilation or benchmarking.

        Accepts either a full CandidateConfig JSON or a stub `{"id": "cand-..."}`
        — in the latter case the candidate is resolved from the most recent
        episode that contains it.
        """

        with self.observe_tool("validate_candidate"):
            candidate = self._resolve_candidate(candidate_json)
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

    def _resolve_candidate(self, candidate_json: str) -> CandidateConfig:
        """Parse a CandidateConfig from JSON, resolving id-only stubs.

        If the JSON is just `{"id": "..."}` (a common agent shortcut), search
        recent episodes for a matching proposed candidate and return that.
        """

        try:
            data = json.loads(candidate_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate_json is not valid JSON: {exc}") from exc
        if isinstance(data, dict):
            cid = data.get("id")
            # If the agent passed an id that we already know about, prefer the
            # canonical cached/persisted CandidateConfig — even if extra
            # fields were supplied alongside, they're either a stale subset or
            # missing required fields. This forgives both `{"id": "..."}` and
            # `{"id": "...", "kind": "..."}` style stubs.
            if cid:
                cached = self.proposed_candidates.get(cid)
                if cached is not None:
                    return cached
                for episode in self.episode_store.list_recent():
                    for cand in getattr(episode, "candidates", []):
                        if cand.id == cid:
                            return cand
            # Either no id supplied, or the id is unknown. Try a normal validate
            # with the dict; if that fails AND we had an id, surface a helpful
            # message listing recently proposed ids.
            try:
                return CandidateConfig.model_validate(data)
            except Exception as exc:
                if cid:
                    recent_ids = list(self.proposed_candidates.keys())[-5:]
                    raise ValueError(
                        f"candidate id `{cid}` was not found in memory or recent "
                        f"episodes, and the supplied dict is incomplete: {exc}. "
                        f"Recently proposed: {recent_ids or '(none)'}. "
                        "Pass either {\"id\": \"cand-...\"} for an existing "
                        "candidate, or the full CandidateConfig JSON returned "
                        "by propose_candidate_from_toolset."
                    ) from exc
                raise ValueError(f"candidate_json is not a valid CandidateConfig: {exc}") from exc
        raise ValueError(
            "candidate_json must be a JSON object (dict). Got: "
            f"{type(data).__name__}."
        )

    def run_candidate(self, candidate_json: str, meta_json: str = "{}") -> str:
        """Compile a validated candidate and capture diagnostics."""

        with self.observe_tool("run_candidate"):
            candidate = self._resolve_candidate(candidate_json)
            validation = validate_candidate_config(candidate)
            if not validation.ok:
                return validation.summary()
            spec = self.require_kernel(candidate.kernel_id)
            interventions, launch_meta = self._split_pass_interventions(candidate.changes)
            meta = {**json.loads(meta_json), **launch_meta}
            request = CompileRequest(
                kernel_id=candidate.kernel_id,
                candidate_id=candidate.id,
                meta=meta,
                stage_hook_key=candidate.model_dump_json(exclude_none=True),
            )
            replace_stages: tuple[str, ...] = ("ttir", "ttgir") if interventions else ()
            result = self.harness.compile_kernel(
                spec,
                request,
                artifact_dir=self.workspace.candidate_dir(candidate.kernel_id, candidate.id),
                use_stage_hook=True,
                replace_stages=replace_stages,
                interventions=interventions,
                pass_callback=self._make_pass_callback(candidate_id=candidate.id),
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

    def run_baseline_benchmark(
        self,
        episode_id: str,
        kernel_id: str,
        workload_json: str = "{}",
    ) -> str:
        """Compile and record baseline evidence for an episode workload."""

        with self.observe_tool("run_baseline_benchmark", episode_id=episode_id, payload={"kernel_id": kernel_id}):
            workload = json.loads(workload_json)
            result = CompileResult.model_validate_json(self.compile_baseline(kernel_id, meta_json=workload_json))
            spec = self.require_kernel(kernel_id)
            if not workload.get("timings_ms") and workload.get("median_ms") is None:
                workload = {**workload, **self._time_kernel(spec, workload)}
            benchmark = self._benchmark_from_compile(
                result,
                workload=workload,
                candidate_id=None,
            )
            self.episode_store.add_benchmark_result(episode_id, benchmark)
            self.emit(
                "benchmark.completed",
                episode_id=episode_id,
                payload={
                    "episode_id": episode_id,
                    "kernel_id": kernel_id,
                    "baseline_id": benchmark.id,
                    "best": benchmark.model_dump(mode="json", exclude_none=True),
                    "results": [benchmark.model_dump(mode="json", exclude_none=True)],
                    "candidate_count": 1,
                },
                artifact_paths=list(benchmark.artifacts.values()),
            )
            return benchmark.model_dump_json(indent=2, exclude_none=True)

    def run_candidate_benchmark(
        self,
        episode_id: str,
        candidate_json: str,
        workload_json: str = "{}",
    ) -> str:
        """Compile, validate, and record candidate benchmark evidence for an episode."""

        with self.observe_tool("run_candidate_benchmark", episode_id=episode_id):
            candidate = self._resolve_candidate(candidate_json)
            validation = validate_candidate_config(candidate)
            if not validation.ok:
                return validation.summary()
            self.episode_store.add_candidate(episode_id, candidate)
            result = CompileResult.model_validate_json(
                self.run_candidate(candidate_json, meta_json=workload_json)
            )
            episode = self.episode_store.load(episode_id)
            baseline = next(
                (item for item in episode.benchmark_results if item.candidate_id is None and item.median_ms),
                None,
            )
            workload = json.loads(workload_json)
            spec = self.require_kernel(candidate.kernel_id)
            _, launch_meta = self._split_pass_interventions(candidate.changes)
            workload = {**workload, **launch_meta}
            if not workload.get("timings_ms") and workload.get("median_ms") is None:
                workload = {**workload, **self._time_kernel(spec, workload)}
            benchmark = self._benchmark_from_compile(
                result,
                workload=workload,
                candidate_id=candidate.id,
                baseline=baseline,
            )
            self.episode_store.add_benchmark_result(episode_id, benchmark)
            self.emit(
                "benchmark.completed",
                episode_id=episode_id,
                payload={
                    "episode_id": episode_id,
                    "kernel_id": candidate.kernel_id,
                    "candidate_id": candidate.id,
                    "best": benchmark.model_dump(mode="json", exclude_none=True),
                    "results": [benchmark.model_dump(mode="json", exclude_none=True)],
                    "candidate_count": 1,
                },
                artifact_paths=list(benchmark.artifacts.values()),
            )
            return benchmark.model_dump_json(indent=2, exclude_none=True)

    def compare_benchmarks(self, episode_id: str, baseline_id: str, candidate_ids_json: str) -> str:
        """Compare benchmark evidence records by id and emit an auditable judgment.

        Each id may be a `bench-…` benchmark-result id OR a `cand-…` candidate
        id (we resolve to the most recent benchmark for that candidate). Empty
        / unknown ids are reported with a clear list of available ones rather
        than a bare KeyError.
        """

        with self.observe_tool("compare_benchmarks", episode_id=episode_id, payload={"baseline_id": baseline_id}):
            episode = self.episode_store.load(episode_id)
            by_id = {result.id: result for result in episode.benchmark_results}
            by_candidate: dict[str, Any] = {}
            for result in episode.benchmark_results:
                cid = getattr(result, "candidate_id", None)
                if cid:
                    by_candidate[cid] = result  # most-recent wins (later overrides)

            def lookup(token: str, *, role: str):
                if token in by_id:
                    return by_id[token]
                if token in by_candidate:
                    return by_candidate[token]
                bench_ids = list(by_id.keys())
                cand_ids = list(by_candidate.keys())
                raise ValueError(
                    f"{role} id `{token}` is not a known benchmark or candidate. "
                    f"Known benchmark ids: {bench_ids[-5:] or '(none)'}; "
                    f"known candidate ids: {cand_ids[-5:] or '(none)'}. "
                    "Run run_baseline_benchmark / run_candidate_benchmark first to "
                    "produce evidence, then pass the returned `id` (or candidate id)."
                )

            baseline = lookup(baseline_id, role="baseline")
            candidate_ids = json.loads(candidate_ids_json) or []
            candidates = [lookup(cid, role="candidate") for cid in candidate_ids]
            comparisons = []
            for candidate in candidates:
                speedup = candidate.speedup_vs_baseline
                delta = (speedup - 1) * 100 if isinstance(speedup, int | float) else None
                comparisons.append(
                    {
                        "benchmark_id": candidate.id,
                        "candidate_id": candidate.candidate_id,
                        "median_ms": candidate.median_ms,
                        "baseline_median_ms": baseline.median_ms,
                        "speedup_vs_baseline": speedup,
                        "delta_percent": delta,
                        "confidence": candidate.speedup_confidence,
                    }
                )
            payload = {
                "episode_id": episode_id,
                "baseline_id": baseline_id,
                "candidate_count": len(comparisons),
                "candidates": comparisons,
                "conclusion": self._benchmark_conclusion(comparisons),
            }
            self.emit("comparison.created", episode_id=episode_id, payload=payload)
            return json.dumps(payload, indent=2)

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
        evidence_ids_json: str = "[]",
        compile_only: bool = False,
    ) -> str:
        """Record the agent's judgment for a candidate."""

        with self.observe_tool("accept_or_reject_candidate", episode_id=episode_id):
            from .schemas import CandidateStatus

            evidence_ids = json.loads(evidence_ids_json)
            episode_before = self.episode_store.load(episode_id)
            # If the candidate isn't yet attached to the episode (agent skipped
            # run_candidate_benchmark or used a proposal that we have only in
            # memory), auto-attach it from the in-memory proposal cache so the
            # judgment lands on a real record instead of erroring out.
            episode_candidate_ids = {cand.id for cand in episode_before.candidates}
            if candidate_id not in episode_candidate_ids:
                cached = self.proposed_candidates.get(candidate_id)
                if cached is not None:
                    self.episode_store.add_candidate(episode_id, cached)
                    episode_before = self.episode_store.load(episode_id)
            benchmark_ids = {result.id for result in episode_before.benchmark_results}
            has_evidence = any(evidence_id in benchmark_ids for evidence_id in evidence_ids) or any(
                result.candidate_id == candidate_id for result in episode_before.benchmark_results
            )
            if not compile_only and not has_evidence:
                raise ValueError("candidate judgment requires benchmark evidence unless compile_only=true")
            episode = self.episode_store.set_candidate_status(
                episode_id,
                candidate_id,
                CandidateStatus(status),
                rationale,
            )
            self.emit(
                "candidate.judged",
                episode_id=episode_id,
                payload={
                    "candidate_id": candidate_id,
                    "status": status,
                    "rationale": rationale,
                    "evidence_ids": evidence_ids,
                    "compile_only": compile_only,
                },
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
        # Direct lookup by compile-result id.
        if run_id in self.compile_results:
            return self.compile_results[run_id]
        # Forgiveness layer: accept a candidate id (cand-…) or kernel id and
        # return the most recent matching compile result. The agent often
        # confuses these because run_id is opaque and candidate_id is what
        # the agent just produced.
        matches = [
            r for r in self.compile_results.values()
            if r.candidate_id == run_id or r.kernel_id == run_id
        ]
        if matches:
            return matches[-1]
        known_runs = list(self.compile_results.keys())
        known_cands = sorted({r.candidate_id for r in self.compile_results.values() if r.candidate_id})
        raise ValueError(
            f"Unknown compile run id: `{run_id}`. "
            f"Known run ids: {known_runs[-5:] or '(none)'}; "
            f"known candidate ids: {known_cands[-5:] or '(none)'}."
        )

    @staticmethod
    def _split_pass_interventions(
        changes: dict[str, Any],
    ) -> tuple[tuple[PassIntervention, ...], dict[str, Any]]:
        """Pull out `pass_interventions` from a candidate's changes dict.

        Returns (interventions tuple, remaining launch-meta dict). The remaining
        keys are forwarded to the kernel launch as constexpr/runtime args.
        """

        raw = changes.get("pass_interventions")
        if not raw:
            return (), changes
        interventions: list[PassIntervention] = []
        for item in raw:
            if isinstance(item, PassIntervention):
                interventions.append(item)
                continue
            if not isinstance(item, dict):
                raise ValueError(
                    "pass_interventions entries must be dicts or PassIntervention objects"
                )
            interventions.append(
                PassIntervention(
                    pass_name=str(item["pass_name"]),
                    action=item.get("action", "run"),
                    args=dict(item.get("args", {}) or {}),
                    rationale=str(item.get("rationale", "")),
                )
            )
        leftover = {k: v for k, v in changes.items() if k != "pass_interventions"}
        return tuple(interventions), leftover

    def _time_kernel(
        self,
        spec: KernelSpec,
        meta: dict[str, Any],
        *,
        warmup: int = 3,
        repetitions: int = 12,
    ) -> dict[str, Any]:
        """Time a kernel by re-invoking its `compilagent_compile` hook.

        Returns a workload-shaped dict with `timings_ms`, `median_ms`, `p20_ms`,
        `p80_ms`, and (for memory-bound families) an `bandwidth_gbps` estimate.
        Returns an empty dict on failure rather than raising — the agent's
        loop survives a timing miss.
        """

        from .benchmarking import time_with_cuda_events

        try:
            from .compiler import _execute_compile  # type: ignore[attr-defined]
            # warmup
            for _ in range(max(0, warmup)):
                _execute_compile(spec, meta)
            timings: list[float] = []
            for _ in range(max(1, repetitions)):
                ms = time_with_cuda_events(lambda: _execute_compile(spec, meta))
                if ms > 0 and ms < 60_000:
                    timings.append(float(ms))
            if not timings:
                return {}
            srt = sorted(timings)
            median = srt[len(srt) // 2]
            p20 = srt[max(0, int(len(srt) * 0.2) - 1)]
            p80 = srt[min(len(srt) - 1, int(len(srt) * 0.8))]
            out: dict[str, Any] = {
                "timings_ms": timings,
                "median_ms": median,
                "p20_ms": p20,
                "p80_ms": p80,
            }
            n = meta.get("n_elements")
            if isinstance(n, int) and n > 0:
                # Best-effort bytes-moved estimate per family.
                family = str(spec.metadata.get("family") or spec.id).lower()
                bytes_per = 12 if "vector_add" in family else 8 if "vector_copy" in family else 4
                bytes_moved = n * bytes_per
                out["bytes_moved"] = bytes_moved
                out["bandwidth_gbps"] = bytes_moved / 1e9 / (median / 1000.0)
            return out
        except Exception as exc:  # noqa: BLE001
            self.emit(
                "log.line",
                payload={"level": "warn", "message": f"timing failed: {exc!r}"},
            )
            return {}

    def _make_pass_callback(
        self, *, candidate_id: str | None
    ) -> Callable[[str, PassResult], None]:
        def on_pass(stage: str, result: PassResult) -> None:
            self.emit(
                "compiler.pass",
                payload={
                    "candidate_id": candidate_id,
                    "stage": stage,
                    "name": result.name,
                    "pass": result.name,
                    "action": result.action,
                    "duration_ms": result.duration_ms,
                    "args": result.args,
                    "ir_after_size": (
                        len(result.ir_after) if result.ir_after is not None else None
                    ),
                    "error": result.error,
                },
            )

        return on_pass

    def _benchmark_from_compile(
        self,
        result: CompileResult,
        *,
        workload: dict[str, Any],
        candidate_id: str | None,
        baseline: BenchmarkResult | None = None,
    ) -> BenchmarkResult:
        from .benchmarking import compute_profile_metrics

        timings = [float(item) for item in workload.get("timings_ms", []) if isinstance(item, int | float)]
        median = float(workload["median_ms"]) if isinstance(workload.get("median_ms"), int | float) else None
        p20 = float(workload["p20_ms"]) if isinstance(workload.get("p20_ms"), int | float) else median
        p80 = float(workload["p80_ms"]) if isinstance(workload.get("p80_ms"), int | float) else median
        if median is None and timings:
            sorted_timings = sorted(timings)
            median = sorted_timings[len(sorted_timings) // 2]
            p20 = sorted_timings[max(0, int(len(sorted_timings) * 0.2) - 1)]
            p80 = sorted_timings[min(len(sorted_timings) - 1, int(len(sorted_timings) * 0.8))]
        speedup = None
        if baseline and baseline.median_ms and median and median > 0:
            speedup = baseline.median_ms / median
        bytes_moved = workload.get("bytes_moved")
        flops = workload.get("flops")
        profile_metrics = compute_profile_metrics(
            median_ms=median,
            bytes_moved=int(bytes_moved) if isinstance(bytes_moved, int | float) else None,
            flops=int(flops) if isinstance(flops, int | float) else None,
        )
        if isinstance(workload.get("bandwidth_gbps"), int | float):
            profile_metrics.setdefault("achieved_bandwidth_gbps", float(workload["bandwidth_gbps"]))
        return BenchmarkResult(
            kernel_id=result.kernel_id,
            candidate_id=candidate_id,
            correctness="skipped" if result.ok else "failed",
            compile_ok=result.ok,
            timings_ms=timings,
            median_ms=median,
            p20_ms=p20,
            p80_ms=p80,
            sample_count=len(timings),
            speedup_vs_baseline=speedup,
            speedup_confidence=self._speedup_confidence(speedup),
            noise_threshold_pct=self.settings.noise_threshold_pct,
            diagnostics=result.diagnostics,
            artifacts={artifact.stage: str(artifact.path) for artifact in result.artifacts if artifact.path},
            workload=workload,
            profile_metrics=profile_metrics,
        )

    def _speedup_confidence(self, speedup: float | None) -> str:
        if speedup is None:
            return "none"
        delta_pct = abs(speedup - 1.0) * 100
        if delta_pct >= self.settings.noise_threshold_pct * 2:
            return "high"
        if delta_pct >= self.settings.noise_threshold_pct:
            return "medium"
        return "low"

    @staticmethod
    def _benchmark_conclusion(comparisons: list[dict[str, Any]]) -> str:
        if not comparisons:
            return "no candidates"
        best = max(
            comparisons,
            key=lambda item: item.get("speedup_vs_baseline")
            if isinstance(item.get("speedup_vs_baseline"), int | float)
            else float("-inf"),
        )
        speedup = best.get("speedup_vs_baseline")
        if not isinstance(speedup, int | float):
            return "no timing evidence"
        if speedup > 1.02:
            return "candidate improved baseline"
        if speedup < 0.98:
            return "candidate regressed baseline"
        return "within noise band"


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
            "Inspect the optimization toolset before proposing candidates; use toolset-derived candidates as the main path.",
            "Record concise reasoning summaries after important candidate decisions without exposing hidden chain-of-thought.",
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


def _extract_snippet(text: str, needle: str, *, before: int = 6, after: int = 30) -> str:
    """Return ~`before+after` lines around the first occurrence of `needle`."""

    lower = text.lower()
    nlower = needle.lower()
    idx = lower.find(nlower)
    if idx < 0:
        # try without dashes
        nlower2 = nlower.replace("-", "")
        # build a normalized text only as needed
        idx = lower.replace("-", "").find(nlower2)
        if idx < 0:
            return ""
    line_start = text.count("\n", 0, idx)
    lines = text.splitlines()
    a = max(0, line_start - before)
    b = min(len(lines), line_start + after)
    return "\n".join(lines[a:b])


def truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    suffix = f"\n... truncated {len(text) - limit} chars ..."
    return text[: max(0, limit - len(suffix))] + suffix
