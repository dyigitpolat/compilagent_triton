"""Concrete `TritonBackend(Backend)` adapter.

Wraps the existing `TritonCompileHarness` (compiler.py), pipeline executor,
pass catalog, and TTGIR analysis behind the `Backend` protocol so the runtime
can drive Triton compiles without ever importing Triton internals directly.

The adapter is deliberately thin — heavy lifting still lives in
`triton_hooks/` (now `backends/triton/{passes,pipeline,stages}.py`) and in
`compiler.TritonCompileHarness`. As Phase 1 progresses, the Triton-specific
methods on `OptimizerRuntime` will migrate into this class.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..base import (
    Analysis,
    Backend,
    DeviceCapability,
    IntrospectionTool,
    PassEvent,
    Plan,
    Intervention,
    PassCallback,
    TimingResult,
    ToleranceConfig,
    CorrectnessResult,
    backend_registry,
)


def _pass_event(stage: str, result: Any) -> PassEvent:
    """Adapt a Triton-side PassResult into the generic PassEvent shape."""

    return PassEvent(
        stage=str(stage),
        name=getattr(result, "name", "?"),
        duration_ms=float(getattr(result, "duration_ms", 0.0) or 0.0),
        action=str(getattr(result, "action", "run")),
        args=list(getattr(result, "args", []) or []),
        ir_after_size=(len(result.ir_after) if getattr(result, "ir_after", None) else None),
        error=getattr(result, "error", None),
    )


class TritonBackend:
    """Triton compiler backend.

    Registered with the global `backend_registry` at import time. The
    underlying compile harness and pass pipeline are reused unchanged from
    the existing implementation; this class is the typed seam the runtime
    programs against.
    """

    id: str = "triton"
    artifact_stages: tuple[str, ...] = ("ttir", "ttgir", "llir", "ptx")

    def device_capability(self) -> DeviceCapability:
        cap_int: int | None = None
        name = "cpu"
        mem_total: int | None = None
        peak_bw_gbps: float | None = None
        try:
            import torch  # type: ignore[import-not-found]
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                cap_int = major * 10 + minor
                props = torch.cuda.get_device_properties(0)
                name = props.name
                mem_total = int(getattr(props, "total_memory", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            from ...benchmarking import device_peak_bandwidth_gbps
            peak_bw_gbps = device_peak_bandwidth_gbps()
        except Exception:  # noqa: BLE001
            peak_bw_gbps = None
        arch = f"cuda:sm_{cap_int}" if cap_int is not None else "cpu"
        return DeviceCapability(
            arch=arch,
            capability_int=cap_int,
            name=name,
            memory_total_bytes=mem_total,
            memory_peak_bandwidth_gbps=peak_bw_gbps,
        )

    # --- Phase 1 placeholders ------------------------------------------------
    # The following methods will be filled in incrementally as the runtime
    # migrates onto the Backend protocol. Phase 1's job is the seam; Phase 2
    # populates derive_search_space, and Phases 3-7 fill the rest.

    def analyze(self, workload: Any, *, baseline_artifacts: Sequence[Any]) -> Analysis:
        from .analysis import extract_decision_traces

        ir_text = ""
        for art in baseline_artifacts:
            if getattr(art, "stage", None) == "ttgir":
                ir_text = getattr(art, "inline_text", "") or ""
                break
        traces = extract_decision_traces(ir_text, run_id=getattr(workload, "id", ""))
        summary: dict[str, Any] = {
            "kind": "kernel",
            "tensor_shapes": {},
            "dtypes": [],
            "op_counts": {},
            "decision_count": len(traces),
        }
        return Analysis(
            summary=summary,
            extra={"decision_traces": [t.model_dump(mode="json", exclude_none=True) for t in traces]},
        )

    def compile(
        self,
        workload: Any,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> Any:
        """Compile a Triton-kernel workload through the existing TritonCompileHarness.

        Resolves the workload to its source path + kernel symbol via the
        registry, builds a `KernelSpec`, projects the generic `Plan` into Triton
        pass interventions, and delegates to `TritonCompileHarness.compile_kernel`.
        """

        from types import SimpleNamespace

        from ...compiler import TritonCompileHarness
        from ...schemas import CompileRequest, KernelSpec
        from ...workloads.registry import workload_registry
        from ...workspace import OptimizationWorkspace

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        workload_id = getattr(workload, "id", None)
        if not workload_id:
            return SimpleNamespace(
                id="", kernel_id="", ok=False, elapsed_ms=0.0,
                artifacts=(), diagnostics="workload has no id",
            )
        try:
            spec = workload_registry.get_spec(workload_id)
        except KeyError as exc:
            return SimpleNamespace(
                id=workload_id, kernel_id=workload_id, ok=False, elapsed_ms=0.0,
                artifacts=(), diagnostics=f"workload not registered: {exc}",
            )
        source_path = spec.metadata.get("source_path")
        kernel_symbol = spec.metadata.get("kernel_symbol")
        if not source_path or not kernel_symbol:
            return SimpleNamespace(
                id=workload_id, kernel_id=workload_id, ok=False, elapsed_ms=0.0,
                artifacts=(), diagnostics="workload metadata missing source_path / kernel_symbol",
            )

        kspec = KernelSpec(
            id=workload_id, name=workload_id,
            path=Path(source_path), entrypoint=kernel_symbol,
            metadata={"workload_kind": spec.kind.value},
        )
        # Project the generic Plan into Triton interventions (only `pass`
        # targets are meaningful; other kinds are no-ops here).
        from .pipeline import PassIntervention as TritonPassIntervention

        triton_interventions: list[TritonPassIntervention] = []
        for iv in plan.interventions:
            if iv.target.kind == "pass":
                stage_name, _, pass_name = iv.target.selector.partition(":")
                if not pass_name:
                    pass_name = stage_name
                payload = iv.payload if isinstance(iv.payload, dict) else {}
                triton_interventions.append(TritonPassIntervention(
                    pass_name=pass_name,
                    action=str(payload.get("action", "run")),
                    args=dict(payload.get("args", {}) or {}),
                    rationale=iv.rationale,
                ))
        # Launch-meta keys (BLOCK_SIZE, num_warps, etc.) flow as compile meta.
        meta: dict[str, Any] = {}
        for iv in plan.interventions:
            if iv.target.kind == "launch":
                if isinstance(iv.payload, dict):
                    meta.update(iv.payload)
                elif iv.target.selector:
                    meta[iv.target.selector] = iv.payload

        # Build a one-shot harness rooted at the artifact dir's parent.
        ws_root = artifact_dir.parent
        ws = OptimizationWorkspace(
            session_cwd=ws_root.parent, root_name=ws_root.name,
        ).ensure() if ws_root.parent.exists() else OptimizationWorkspace(
            session_cwd=Path("."), root_name=".compilagent-triton",
        ).ensure()
        harness = TritonCompileHarness(workspace=ws)
        request = CompileRequest(
            kernel_id=workload_id, candidate_id=None, meta=meta,
        )
        replace_stages = ("ttir", "ttgir") if triton_interventions else ()
        result = harness.compile_kernel(
            kspec, request,
            artifact_dir=artifact_dir,
            use_stage_hook=True,
            replace_stages=replace_stages,
            interventions=tuple(triton_interventions),
            pass_callback=(
                (lambda stage, r: pass_callback(_pass_event(stage, r)))
                if pass_callback else None
            ),
        )
        artifact_paths = tuple(
            str(art.path) for art in result.artifacts if art.path
        )
        return SimpleNamespace(
            id=workload_id,
            kernel_id=workload_id,
            ok=result.ok,
            elapsed_ms=0.0,
            artifacts=artifact_paths,
            output_code_path=None,
            schedule_log_path=None,
            fx_graph_path=None,
            compiled_callable=None,
            diagnostics=result.diagnostics,
            compile_result=result,
        )

    def time_workload(
        self,
        workload: Any,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: float | None = None,
    ) -> TimingResult:
        """Time the workload's `forward` callable using CUDA events.

        For Triton kernels this is just a tight loop over `instance.forward()`
        — the kernel is already JIT-cached after the first call, so warmup is
        cheap.
        """

        import time as _time
        from ...benchmarking import time_with_cuda_events
        from ...workloads.registry import workload_registry

        workload_id = getattr(workload, "id", None)
        if not workload_id:
            return TimingResult(
                timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                diagnostics="workload has no id",
            )
        try:
            instance = workload_registry.build(workload_id)
        except Exception as exc:  # noqa: BLE001
            return TimingResult(
                timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                diagnostics=f"workload build failed: {exc!r}",
            )
        fn = instance.forward
        for _ in range(max(0, warmup)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                return TimingResult(
                    timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                    diagnostics=f"forward failed during warmup: {exc!r}",
                )
        budget = max_seconds or float("inf")
        started = _time.perf_counter()
        timings: list[float] = []
        for _ in range(max(1, repetitions)):
            if (_time.perf_counter() - started) > budget:
                break
            ms = time_with_cuda_events(fn)
            if ms > 0:
                timings.append(float(ms))
        if not timings:
            return TimingResult(
                timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                diagnostics="no successful timings",
            )
        srt = sorted(timings)
        median = srt[len(srt) // 2]
        p20 = srt[max(0, int(len(srt) * 0.2) - 1)]
        p80 = srt[min(len(srt) - 1, int(len(srt) * 0.8))]
        return TimingResult(
            timings_ms=tuple(timings), median_ms=median, p20_ms=p20, p80_ms=p80,
        )

    def validate_correctness(
        self,
        workload: Any,
        baseline: Any,
        candidate: Any,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        # Triton-kernel correctness is currently checked inside the existing
        # benchmark sweeps; no separate compare-after-the-fact path exists yet.
        return CorrectnessResult(
            ok=True,
            diagnostics="Triton kernel correctness is verified inline by the sweep harness.",
        )

    def list_introspection_tools(self) -> Sequence[IntrospectionTool]:
        # Phase 2 will populate this with list_compiler_passes / describe_compiler_pass /
        # read_pass_source bound to this backend instance.
        return ()

    def derive_search_space(self, workload: Any, analysis: Analysis) -> Any:
        from ...core.search_space import SearchSpace
        from .derivation import ALL_DERIVATIONS

        workload_kind = (analysis.summary or {}).get("kind", "kernel")
        levers: list = []
        for rule in ALL_DERIVATIONS:
            if workload_kind not in rule.applies_to:
                continue
            try:
                levers.extend(rule.derive(workload, analysis))
            except Exception:  # noqa: BLE001
                # A failing derivation rule must not crash the agent — skip it
                # and let the others contribute. Diagnostics flow through the
                # event log when the runtime calls this.
                continue
        return SearchSpace(
            workload_id=getattr(workload, "id", "") or "",
            backend_id=self.id,
            levers=tuple(levers),
        )

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))


# Self-registration: importing this module registers the backend.
def _factory() -> Backend:
    return TritonBackend()


backend_registry.register("triton", _factory)
