"""TorchInductor compiler backend (concrete `Backend` adapter)."""

from __future__ import annotations

import json as _json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..base import (
    Analysis,
    Backend,
    CorrectnessResult,
    DeviceCapability,
    Intervention,
    IntrospectionTool,
    PassCallback,
    PassEvent,
    Plan,
    TimingResult,
    ToleranceConfig,
    backend_registry,
)
from .analysis import parse_inductor_artifacts, to_generic_analysis
from .correctness import compare_forward
from .harness import InductorPlan, drive_compile
from .knobs import build_knob_catalog


_KNOB_CATALOG_CACHE: Any = None


def _knob_catalog() -> Any:
    global _KNOB_CATALOG_CACHE
    if _KNOB_CATALOG_CACHE is None:
        _KNOB_CATALOG_CACHE = build_knob_catalog()
    return _KNOB_CATALOG_CACHE


class TorchInductorBackend:
    """`torch.compile` driven by a custom Dynamo backend."""

    id: str = "torch_inductor"
    artifact_stages: tuple[str, ...] = (
        "fx_graph",
        "output_code",
        "schedule_log",
        "fusion_log",
    )

    # ---------- device --------------------------------------------------------

    def device_capability(self) -> DeviceCapability:
        cap_int: int | None = None
        name = "cpu"
        mem_total: int | None = None
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
            peak = device_peak_bandwidth_gbps()
        except Exception:  # noqa: BLE001
            peak = None
        arch = f"cuda:sm_{cap_int}" if cap_int is not None else "cpu"
        return DeviceCapability(
            arch=arch, capability_int=cap_int, name=name,
            memory_total_bytes=mem_total, memory_peak_bandwidth_gbps=peak,
        )

    # ---------- plan projection ----------------------------------------------

    def _project_plan(self, plan: Plan) -> InductorPlan:
        """Translate a generic `Plan` into an `InductorPlan`."""

        ip = InductorPlan()
        for iv in plan.interventions:
            kind = iv.target.kind
            if kind == "knob":
                # selector is `inductor.<leaf>` or `dynamo.<leaf>`
                if iv.target.selector.startswith("dynamo."):
                    ip.dynamo_config[iv.target.selector] = iv.payload
                else:
                    ip.inductor_config[iv.target.selector] = iv.payload
            elif kind == "lowering":
                # payload is expected to be a callable already; agents that pass
                # a "module:fn" string can pre-resolve in their tool wrapper.
                ip.lowering_overrides[iv.target.selector] = iv.payload
            elif kind == "fx_node":
                ip.fx_rewriters.append(iv.payload)
            elif kind == "scheduler":
                if iv.target.selector == "pre_fusion":
                    ip.pre_fusion_pass = iv.payload
                elif iv.target.selector == "post_fusion":
                    ip.post_fusion_pass = iv.payload
            elif kind == "choices":
                ip.choices_handler = iv.payload
            else:
                # Unknown kinds are silently dropped here; the runtime warns.
                pass
        return ip

    # ---------- compile / time / validate ------------------------------------

    def compile(
        self,
        workload: Any,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> Any:
        from ...workloads.registry import workload_registry
        instance = workload_registry.build(workload.id)
        outcome = drive_compile(instance, self._project_plan(plan), artifact_dir=artifact_dir)
        # Inductor's compile pipeline doesn't expose discrete named "passes" the
        # way Triton MLIR does — surface one synthetic event per compile so the
        # UI's compiler tab still populates with the elapsed time + ok flag.
        if pass_callback is not None:
            try:
                pass_callback(PassEvent(
                    stage="inductor",
                    name="compile_fx",
                    duration_ms=float(outcome.elapsed_ms or 0.0),
                    action="run",
                    args=[],
                    error=outcome.diagnostics if not outcome.ok else None,
                ))
            except Exception:  # noqa: BLE001
                pass
        # Minimal CompileResult-shaped namespace; the runtime imports the real
        # CompileResult only when it needs to round-trip through pydantic.
        from types import SimpleNamespace
        return SimpleNamespace(
            id=workload.id,
            kernel_id=workload.id,
            ok=outcome.ok,
            elapsed_ms=outcome.elapsed_ms,
            artifacts=tuple(outcome.captured_logs.values()),
            output_code_path=outcome.output_code_path,
            schedule_log_path=outcome.schedule_log_path,
            fx_graph_path=outcome.fx_graph_path,
            compiled_callable=outcome.compiled_callable,
            diagnostics=outcome.diagnostics,
            warnings=list(getattr(outcome, "warnings", []) or []),
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
        from ...benchmarking import time_with_cuda_events
        from ...workloads.registry import workload_registry

        instance = workload_registry.build(workload.id)
        # Compile once; the timer reuses the compiled callable.
        with _scratch_dir() as tmp:
            outcome = drive_compile(instance, self._project_plan(plan), artifact_dir=tmp)
        if not outcome.ok or outcome.compiled_callable is None:
            return TimingResult(
                timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                diagnostics=outcome.diagnostics or "compile failed",
            )
        compiled = outcome.compiled_callable
        ex = instance.example_inputs

        def _run():
            return compiled(*ex) if ex else compiled()

        for _ in range(max(0, warmup)):
            try:
                _run()
            except Exception:  # noqa: BLE001
                break
        timings: list[float] = []
        budget = max_seconds or float("inf")
        started = time.perf_counter()
        for _ in range(max(1, repetitions)):
            if (time.perf_counter() - started) > budget:
                break
            ms = time_with_cuda_events(_run)
            if ms > 0:
                timings.append(float(ms))
        if not timings:
            return TimingResult(timings_ms=(), median_ms=None, p20_ms=None, p80_ms=None,
                                diagnostics="no successful timings")
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
        from ...workloads.registry import workload_registry

        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return CorrectnessResult(ok=False, diagnostics=f"torch import failed: {exc!r}")
        spec = workload
        for fn in (baseline.compiled_callable, candidate.compiled_callable):
            if fn is None:
                return CorrectnessResult(ok=False, diagnostics="missing compiled callable")
        instance = workload_registry.build(spec.id)
        ex = instance.example_inputs

        def _run(fn):
            torch.manual_seed(int(spec.metadata.get("seed", 0)))
            return fn(*ex) if ex else fn()

        b_out = _run(baseline.compiled_callable)
        c_out = _run(candidate.compiled_callable)
        return compare_forward(baseline_run=b_out, candidate_run=c_out, tolerance=tolerance)

    # ---------- analysis / search-space --------------------------------------

    def analyze(self, workload: Any, *, baseline_artifacts: Sequence[Any]) -> Analysis:
        """Project captured inductor artifacts into the generic Analysis shape.

        `baseline_artifacts` is the iterable of paths produced by
        `compile()`; we filter by suffix. The FX-graph text is folded into
        `extra["fx_text"]` so derivation plugins can grep for aten ops.
        """

        def _find(suffix: str) -> Path | None:
            for p in baseline_artifacts:
                if str(p).endswith(suffix):
                    return Path(p)
            return None

        output_code = _find("output_code.py") or _find("output_code.log")
        schedule = _find("schedule.log")
        fx_graph = _find("fx_graph.py")
        data = parse_inductor_artifacts(
            output_code=output_code, schedule_log=schedule, fx_graph=fx_graph,
        )
        analysis = to_generic_analysis(data, workload_kind="full_model")
        # Carry the FX text through so derivation plugins can introspect ops.
        if fx_graph and fx_graph.exists():
            try:
                analysis.extra["fx_text"] = fx_graph.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                analysis.extra["fx_text"] = ""
        return analysis

    def derive_search_space(self, workload: Any, analysis: Analysis) -> Any:
        from ...core.search_space import SearchSpace
        from .derivation import ALL_DERIVATIONS

        kind = (analysis.summary or {}).get("kind", "full_model")
        levers: list = []
        for rule in ALL_DERIVATIONS:
            if kind not in rule.applies_to:
                continue
            try:
                levers.extend(rule.derive(workload, analysis))
            except Exception:  # noqa: BLE001
                # Refuse to crash search-space construction on a single rule.
                continue
        return SearchSpace(
            workload_id=getattr(workload, "id", "") or "",
            backend_id=self.id,
            levers=tuple(levers),
        )

    def apply_intervention(self, plan: Plan, intervention: Intervention) -> Plan:
        return Plan(interventions=plan.interventions + (intervention,))

    # ---------- introspection tools the agent gets ---------------------------

    def list_introspection_tools(self) -> Sequence[IntrospectionTool]:
        catalog = _knob_catalog()

        def list_inductor_knobs(namespace: str = "inductor") -> str:
            ns = catalog.in_namespace(namespace)
            return _json.dumps(
                {"namespace": namespace, "count": len(ns), "knobs": [k.serialize() for k in ns]},
                indent=2,
                default=str,
            )

        def describe_inductor_knob(name: str) -> str:
            knob = catalog.by_name(name)
            if knob is None:
                return _json.dumps({"name": name, "error": "not_found"}, indent=2)
            return _json.dumps(knob.serialize(), indent=2, default=str)

        return (
            IntrospectionTool(
                name="list_inductor_knobs",
                description=(
                    "List torch._inductor.config / torch._dynamo.config knobs the agent can "
                    "override per candidate. Each entry includes default + heuristic candidates."
                ),
                fn=list_inductor_knobs,
            ),
            IntrospectionTool(
                name="describe_inductor_knob",
                description="Describe one inductor / dynamo knob by name (dotted or leaf).",
                fn=describe_inductor_knob,
            ),
        )


# Scratch-dir helper -------------------------------------------------------


from contextlib import contextmanager
import tempfile


@contextmanager
def _scratch_dir():
    import shutil
    p = Path(tempfile.mkdtemp(prefix="compilagent-inductor-"))
    try:
        yield p
    finally:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# Self-register


def _factory() -> Backend:
    return TorchInductorBackend()


backend_registry.register("torch_inductor", _factory)
