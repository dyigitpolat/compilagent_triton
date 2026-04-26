from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateKind(StrEnum):
    META_PARAMETERS = "meta_parameters"
    COALESCING_POLICY = "coalescing_policy"
    MATMUL_POLICY = "matmul_policy"


class CandidateStatus(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ACCEPTED = "accepted"


class DecisionKind(StrEnum):
    COALESCING = "coalescing"
    MATMUL = "matmul"
    UNKNOWN = "unknown"


class KernelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    path: Path | None = None
    entrypoint: str
    shapes: list[tuple[int, ...]] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "name", "entrypoint")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kernel_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    force_recompile: bool = True
    dump_ir: bool = True
    stage_hook_key: str | None = None
    candidate_id: str | None = None


class CompileArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    path: Path | None = None
    inline_text: str | None = None


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"compile-{uuid4().hex[:12]}")
    kernel_id: str
    candidate_id: str | None = None
    ok: bool
    cache_hit: bool | None = None
    target: str | None = None
    source_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[CompileArtifact] = Field(default_factory=list)
    diagnostics: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"decision-{uuid4().hex[:12]}")
    run_id: str | None = None
    kind: DecisionKind = DecisionKind.UNKNOWN
    op_name: str | None = None
    op_location: str | None = None
    tensor_shape: tuple[int, ...] | None = None
    element_bitwidth: int | None = None
    chosen_order: tuple[int, ...] | None = None
    size_per_thread: tuple[int, ...] | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    shape_per_cta: tuple[int, ...] | None = None
    mma_version: int | None = None
    warps_per_tile: tuple[int, ...] | None = None
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"hyp-{uuid4().hex[:10]}")
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    expected_effect: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("statement", "expected_effect")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"cand-{uuid4().hex[:10]}")
    kernel_id: str
    kind: CandidateKind
    hypothesis_id: str | None = None
    description: str
    changes: dict[str, Any]
    expected_effect: str
    validation_constraints: list[str] = Field(default_factory=list)
    status: CandidateStatus = CandidateStatus.PROPOSED

    @field_validator("description", "expected_effect")
    @classmethod
    def _text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"bench-{uuid4().hex[:12]}")
    kernel_id: str
    candidate_id: str | None = None
    correctness: Literal["passed", "failed", "skipped"]
    compile_ok: bool
    timings_ms: list[float] = Field(default_factory=list)
    median_ms: float | None = None
    speedup_vs_baseline: float | None = None
    diagnostics: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OptimizationEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"episode-{uuid4().hex[:12]}")
    kernel_id: str
    objective: str
    budget: dict[str, Any]
    model_metadata: dict[str, Any]
    baseline_run_id: str | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    candidates: list[CandidateConfig] = Field(default_factory=list)
    benchmark_results: list[BenchmarkResult] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    report_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def touch(self) -> OptimizationEpisode:
        return self.model_copy(update={"updated_at": datetime.now(UTC)})
