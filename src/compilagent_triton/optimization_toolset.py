from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas import CandidateConfig, CandidateKind, KernelSpec


class OptimizationLever(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: CandidateKind
    description: str
    allowed_keys: list[str]
    constraints: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)

    def with_hint(self) -> dict[str, Any]:
        """Serialize with an explicit `kind_value` echo so agents pass the right
        enum string (not the lever id) to `propose_candidate_from_toolset`."""

        data = self.model_dump(mode="json")
        data["kind_value"] = self.kind.value
        return data


class OptimizationToolset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kernel_id: str
    kernel_name: str
    family: str
    levers: list[OptimizationLever]
    compiler_evidence: dict[str, Any] = Field(default_factory=dict)
    prior_evidence: str | None = None


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kernel_id: str
    levers: list[OptimizationLever]
    constraints: list[str] = Field(default_factory=list)

    def candidate_from_toolset(
        self,
        *,
        kind: CandidateKind,
        description: str,
        changes: dict[str, Any],
        expected_effect: str,
        hypothesis_id: str | None = None,
    ) -> CandidateConfig:
        lever = next((item for item in self.levers if item.kind == kind), None)
        if lever is None:
            raise ValueError(f"`{kind}` is not available for kernel `{self.kernel_id}`")
        unknown = sorted(set(changes) - set(lever.allowed_keys))
        if unknown:
            raise ValueError(f"changes use keys outside `{lever.id}`: {', '.join(unknown)}")
        return CandidateConfig(
            kernel_id=self.kernel_id,
            kind=kind,
            hypothesis_id=hypothesis_id,
            description=description,
            changes=changes,
            expected_effect=expected_effect,
            validation_constraints=[*self.constraints, *lever.constraints],
        )


_LAUNCH_META_KEYS = ["BLOCK_SIZE", "num_warps", "num_stages", "num_ctas", "maxnreg"]


def build_optimization_toolset(
    spec: KernelSpec,
    *,
    compiler_evidence: dict[str, Any] | None = None,
    prior_evidence: str | None = None,
) -> OptimizationToolset:
    family = str(spec.metadata.get("family") or spec.id)
    levers = [
        OptimizationLever(
            id="launch_meta",
            kind=CandidateKind.META_PARAMETERS,
            description="Launch shape and backend meta-parameters that affect occupancy and tiling.",
            allowed_keys=list(_LAUNCH_META_KEYS),
            constraints=[
                "num_warps must be a positive power of two",
                "candidate must keep correctness measurable against the baseline",
            ],
            defaults={"num_warps": 4, "BLOCK_SIZE": 1024},
        ),
        OptimizationLever(
            id="memory_access",
            kind=CandidateKind.MEMORY_ACCESS_POLICY,
            description=(
                "Load/cache policy and vectorization levers for memory-bound kernels. "
                "May be combined with launch-meta keys (BLOCK_SIZE / num_warps / etc.) "
                "in the same candidate."
            ),
            allowed_keys=[
                "LOAD_CACHE_MODIFIER", "eviction_policy", "vector_width",
                "mask_strategy", "reason", *_LAUNCH_META_KEYS,
            ],
            constraints=["cache modifier must be '', '.ca', or '.cg'"],
            defaults={"LOAD_CACHE_MODIFIER": "", "vector_width": 1},
        ),
        OptimizationLever(
            id="coalescing_layout",
            kind=CandidateKind.COALESCING_POLICY,
            description=(
                "Layout/coalescing decisions inferred from TTIR/TTGIR memory operations. "
                "May be combined with launch-meta keys."
            ),
            allowed_keys=[
                "op_name", "order", "per_thread", "vector_width", "reason",
                *_LAUNCH_META_KEYS,
            ],
            constraints=["order must describe an observed tensor rank"],
        ),
    ]
    if "matmul" in family.lower() or "matmul" in spec.name.lower():
        levers.append(
            OptimizationLever(
                id="mma_policy",
                kind=CandidateKind.MATMUL_POLICY,
                description=(
                    "MMA tile and warp policy for matmul kernels. "
                    "May be combined with launch-meta keys."
                ),
                allowed_keys=[
                    "mma_version", "warps_per_tile", "reason", *_LAUNCH_META_KEYS,
                ],
                constraints=["MMA version must be one of 1, 2, 3, or 5"],
            )
        )
    return OptimizationToolset(
        kernel_id=spec.id,
        kernel_name=spec.name,
        family=family,
        levers=levers,
        compiler_evidence=compiler_evidence or {},
        prior_evidence=prior_evidence,
    )


def search_space_from_toolset(toolset: OptimizationToolset) -> SearchSpace:
    return SearchSpace(
        kernel_id=toolset.kernel_id,
        levers=toolset.levers,
        constraints=[
            "record a hypothesis or reasoning summary before running expensive candidates",
            "benchmark evidence is required before accepting a candidate",
        ],
    )
