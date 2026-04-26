from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .trace_store import TraceStore

PACKAGE_DIR = Path(__file__).resolve().parent
MAX_CANDIDATES_PER_RUN = 80


class RunConfig(BaseModel):
    n_elements: int = Field(default=8_388_608, ge=1, le=536_870_912)
    block_sizes: list[int] = Field(default_factory=lambda: [256, 512, 1024, 2048])
    num_warps: list[int] = Field(default_factory=lambda: [4, 8])
    load_cache_modifiers: list[str] = Field(default_factory=lambda: [""])
    repetitions: int = Field(default=20, ge=1, le=200)
    warmup: int = Field(default=5, ge=0, le=50)
    max_benchmark_seconds: int = Field(default=120, ge=1, le=900)
    gpu_index: int | None = Field(default=None, ge=0)

    @field_validator("block_sizes", "num_warps", mode="before")
    @classmethod
    def _parse_int_list(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return list(value)

    @field_validator("load_cache_modifiers", mode="before")
    @classmethod
    def _parse_cache_modifiers(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = [part.strip() for part in value.split(",")]
        else:
            raw_values = [str(part).strip() for part in value]
        aliases = {"none": "", "default": "", "": ""}
        normalized = [aliases.get(item, item) for item in raw_values]
        invalid = [item for item in normalized if item not in {"", ".ca", ".cg"}]
        if invalid:
            raise ValueError(f"unsupported cache modifiers: {', '.join(invalid)}")
        return normalized or [""]

    def bounded_for(self, example: ExampleSpec) -> RunConfig:
        fields = example.supported_knobs
        config = self.model_copy(deep=True)
        if "load_cache_modifiers" not in fields:
            config.load_cache_modifiers = [""]
        candidate_count = len(config.block_sizes) * len(config.num_warps) * len(config.load_cache_modifiers)
        if candidate_count > MAX_CANDIDATES_PER_RUN:
            raise ValueError(
                f"run would create {candidate_count} candidates; limit is {MAX_CANDIDATES_PER_RUN}"
            )
        return config

    @property
    def device(self) -> str:
        if self.gpu_index is None:
            return "cuda"
        return f"cuda:{self.gpu_index}"


class ExampleSpec(BaseModel):
    id: str
    title: str
    description: str
    kernel_family: str
    source_path: Path
    entrypoint: str
    supported_knobs: list[str]
    default_config: RunConfig
    enabled: bool = True
    disabled_reason: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "kernel_family": self.kernel_family,
            "entrypoint": self.entrypoint,
            "supported_knobs": self.supported_knobs,
            "default_config": self.default_config.model_dump(mode="json"),
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "source_path": str(self.source_path),
        }


class RunRequest(BaseModel):
    example_id: str
    config: RunConfig = Field(default_factory=RunConfig)
    mode: Literal["benchmark", "optimize"] = "benchmark"


def list_examples() -> list[dict[str, Any]]:
    return [example.public_metadata() for example in _registry().values()]


def get_example(example_id: str) -> ExampleSpec:
    try:
        return _registry()[example_id]
    except KeyError as exc:
        raise ValueError(f"unknown example id: {example_id}") from exc


def preview_source(example_id: str, *, max_chars: int = 40_000) -> dict[str, Any]:
    example = get_example(example_id)
    source_path = example.source_path.resolve()
    if not source_path.is_file():
        raise ValueError(f"source for example `{example_id}` is missing")
    text = source_path.read_text(encoding="utf-8")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return {
        **example.public_metadata(),
        "language": "python",
        "source": text,
        "line_count": text.count("\n") + (1 if text else 0),
        "truncated": truncated,
    }


def create_run_id(example_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{example_id.replace('_', '-')}-{timestamp}-{uuid4().hex[:8]}"


def run_registered_example(
    *,
    request: RunRequest,
    run_id: str,
    workspace_root: Path,
    trace_store: TraceStore,
) -> dict[str, Any]:
    example = get_example(request.example_id)
    if not example.enabled:
        raise ValueError(example.disabled_reason or f"example `{example.id}` is disabled")
    config = request.config.bounded_for(example)
    payload = {
        "run_id": run_id,
        "example_id": example.id,
        "family": example.kernel_family,
        "mode": request.mode,
        "config": config.model_dump(mode="json"),
    }
    trace_store.emit("run.started", payload=payload)
    trace_store.emit("benchmark.started", payload=payload)
    started = datetime.now(UTC)
    try:
        result = _runner_for(example.id)(config, workspace_root, run_id)
    except Exception as exc:
        trace_store.emit(
            "run.failed",
            payload={**payload, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000
    best = result["best"]
    artifacts = result["artifacts"]
    trace_store.emit(
        "benchmark.completed",
        payload={
            **payload,
            "best": best,
            "candidate_count": len(result["results"]),
            "results": result["results"],
            "elapsed_ms": elapsed_ms,
        },
        artifact_paths=artifacts,
    )
    trace_store.emit(
        "comparison.created",
        payload=_comparison_payload(run_id=run_id, family=example.kernel_family, best=best),
    )
    for artifact in artifacts:
        trace_store.emit("artifact.created", payload={"run_id": run_id, "path": artifact}, artifact_paths=[artifact])
    trace_store.emit(
        "loop.summary",
        payload={
            "run_id": run_id,
            "example_id": example.id,
            "best_candidate_id": best.get("candidate_id"),
            "best_median_ms": best.get("median_ms"),
            "speedup_vs_baseline": best.get("speedup_vs_baseline"),
            "elapsed_ms": elapsed_ms,
        },
        artifact_paths=artifacts,
    )
    trace_store.emit("run.completed", payload={**payload, "elapsed_ms": elapsed_ms}, artifact_paths=artifacts)
    return {"run_id": run_id, **result}


def _registry() -> dict[str, ExampleSpec]:
    return {
        "vector_add": ExampleSpec(
            id="vector_add",
            title="Vector Add",
            description="Masked elementwise add with block size, warp count, and load cache modifier sweeps.",
            kernel_family="vector_add",
            source_path=PACKAGE_DIR / "gpu_benchmarks.py",
            entrypoint="run_vector_add_sweep",
            supported_knobs=["block_sizes", "num_warps", "load_cache_modifiers"],
            default_config=RunConfig(
                n_elements=8_388_608,
                block_sizes=[256, 512, 1024, 2048],
                num_warps=[4, 8],
                load_cache_modifiers=["", ".ca", ".cg"],
                repetitions=20,
                warmup=5,
            ),
        ),
        "vector_copy": ExampleSpec(
            id="vector_copy",
            title="Vector Copy",
            description="Contiguous vector copy sweep focused on memory throughput and launch configuration.",
            kernel_family="vector_copy",
            source_path=PACKAGE_DIR / "gpu_copy_benchmarks.py",
            entrypoint="run_copy_sweep",
            supported_knobs=["block_sizes", "num_warps"],
            default_config=RunConfig(
                n_elements=8_388_608,
                block_sizes=[256, 512, 1024, 2048, 4096],
                num_warps=[1, 2, 4, 8, 16],
                repetitions=100,
                warmup=20,
            ),
        ),
        "matmul_stub": ExampleSpec(
            id="matmul_stub",
            title="Matmul",
            description="Reserved for a reliable matmul optimization harness.",
            kernel_family="matmul",
            source_path=PACKAGE_DIR / "gpu_benchmarks.py",
            entrypoint="not_implemented",
            supported_knobs=[],
            default_config=RunConfig(),
            enabled=False,
            disabled_reason="Matmul harness is not enabled yet.",
        ),
    }


def _runner_for(example_id: str) -> Callable[[RunConfig, Path, str], dict[str, Any]]:
    if example_id == "vector_add":
        return _run_vector_add
    if example_id == "vector_copy":
        return _run_vector_copy
    raise ValueError(f"example `{example_id}` has no runner")


def _run_vector_add(config: RunConfig, workspace_root: Path, run_id: str) -> dict[str, Any]:
    from .gpu_benchmarks import render_vector_add_report, run_vector_add_sweep

    results = run_vector_add_sweep(
        n_elements=config.n_elements,
        block_sizes=tuple(config.block_sizes),
        num_warps_values=tuple(config.num_warps),
        load_cache_modifiers=tuple(config.load_cache_modifiers),
        repetitions=config.repetitions,
        warmup=config.warmup,
        device=config.device,
    )
    return _write_result_artifacts(
        workspace_root=workspace_root,
        run_id=run_id,
        n_elements=config.n_elements,
        family="vector_add",
        results=[asdict(result) for result in results],
        markdown=render_vector_add_report(results),
    )


def _run_vector_copy(config: RunConfig, workspace_root: Path, run_id: str) -> dict[str, Any]:
    from .gpu_copy_benchmarks import render_copy_report, run_copy_sweep

    results = run_copy_sweep(
        n_elements=config.n_elements,
        block_sizes=tuple(config.block_sizes),
        num_warps_values=tuple(config.num_warps),
        repetitions=config.repetitions,
        warmup=config.warmup,
        device=config.device,
    )
    return _write_result_artifacts(
        workspace_root=workspace_root,
        run_id=run_id,
        n_elements=config.n_elements,
        family="vector_copy",
        results=[asdict(result) for result in results],
        markdown=render_copy_report(results),
    )


def _write_result_artifacts(
    *,
    workspace_root: Path,
    run_id: str,
    n_elements: int,
    family: str,
    results: list[dict[str, Any]],
    markdown: str,
) -> dict[str, Any]:
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_n{n_elements}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    correct_results = [result for result in results if result.get("correctness")]
    if not correct_results:
        raise RuntimeError(f"{family} run completed without a correct candidate")
    best = min(correct_results, key=lambda item: item.get("median_ms", float("inf")))
    return {
        "family": family,
        "results": results,
        "best": best,
        "artifacts": [str(json_path), str(md_path)],
    }


def _comparison_payload(*, run_id: str, family: str, best: dict[str, Any]) -> dict[str, Any]:
    speedup = best.get("speedup_vs_baseline")
    if speedup is None:
        conclusion = "no baseline"
    elif speedup > 1.02:
        conclusion = "candidate improved baseline"
    elif speedup < 0.98:
        conclusion = "candidate regressed baseline"
    else:
        conclusion = "within noise band"
    return {
        "run_id": run_id,
        "family": family,
        "candidate_id": best.get("candidate_id"),
        "speedup_vs_baseline": speedup,
        "delta_percent": (speedup - 1) * 100 if isinstance(speedup, int | float) else None,
        "conclusion": conclusion,
    }
