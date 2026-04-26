from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import CandidateConfig, CandidateKind, CandidateStatus

_VALID_META_KEYS = {
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "BLOCK_SIZE",
    "GROUP_SIZE_M",
    "LOAD_CACHE_MODIFIER",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
    "ir_override",
}


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    ok: bool
    diagnostics: list[str]
    candidate: CandidateConfig

    def summary(self) -> str:
        status = "valid" if self.ok else "invalid"
        details = "\n".join(f"- {item}" for item in self.diagnostics)
        return f"Candidate `{self.candidate.id}` is {status}.\n{details}".rstrip()


def validate_candidate(candidate: CandidateConfig) -> CandidateValidation:
    diagnostics: list[str] = []
    if candidate.kind == CandidateKind.META_PARAMETERS:
        _validate_meta_candidate(candidate.changes, diagnostics)
    elif candidate.kind == CandidateKind.COALESCING_POLICY:
        _validate_coalescing_candidate(candidate.changes, diagnostics)
    elif candidate.kind == CandidateKind.MATMUL_POLICY:
        _validate_matmul_candidate(candidate.changes, diagnostics)
    else:
        diagnostics.append(f"Unsupported candidate kind: {candidate.kind}")

    ok = not diagnostics
    status = CandidateStatus.VALIDATED if ok else CandidateStatus.REJECTED
    return CandidateValidation(
        ok=ok,
        diagnostics=diagnostics or ["All validation checks passed."],
        candidate=candidate.model_copy(update={"status": status}),
    )


def _validate_meta_candidate(changes: dict[str, Any], diagnostics: list[str]) -> None:
    if not changes:
        diagnostics.append("Meta-parameter candidate must include at least one change.")
        return
    unknown = sorted(set(changes) - _VALID_META_KEYS)
    if unknown:
        diagnostics.append(f"Unsupported meta-parameter keys: {', '.join(unknown)}")
    for key in ("num_warps", "num_stages", "num_ctas"):
        if key in changes and (not isinstance(changes[key], int) or changes[key] <= 0):
            diagnostics.append(f"`{key}` must be a positive integer.")
    if "num_warps" in changes and changes["num_warps"] & (changes["num_warps"] - 1):
        diagnostics.append("`num_warps` must be a power of two.")
    if (
        "maxnreg" in changes
        and changes["maxnreg"] is not None
        and (not isinstance(changes["maxnreg"], int) or changes["maxnreg"] <= 0)
    ):
        diagnostics.append("`maxnreg` must be a positive integer or null.")


def _validate_coalescing_candidate(changes: dict[str, Any], diagnostics: list[str]) -> None:
    allowed = {"op_name", "order", "per_thread", "vector_width", "reason"}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        diagnostics.append(f"Unsupported coalescing keys: {', '.join(unknown)}")
    if "order" in changes:
        order = changes["order"]
        if not isinstance(order, list) or not order or not all(isinstance(i, int) for i in order):
            diagnostics.append("`order` must be a non-empty list of integers.")
    for key in ("per_thread", "vector_width"):
        if key in changes and (not isinstance(changes[key], int) or changes[key] <= 0):
            diagnostics.append(f"`{key}` must be a positive integer.")


def _validate_matmul_candidate(changes: dict[str, Any], diagnostics: list[str]) -> None:
    allowed = {"mma_version", "warps_per_tile", "reason"}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        diagnostics.append(f"Unsupported matmul keys: {', '.join(unknown)}")
    if "mma_version" in changes and changes["mma_version"] not in {1, 2, 3, 5}:
        diagnostics.append("`mma_version` must be one of 1, 2, 3, or 5.")
    if "warps_per_tile" in changes:
        warps = changes["warps_per_tile"]
        if not isinstance(warps, list) or not warps or not all(isinstance(i, int) and i > 0 for i in warps):
            diagnostics.append("`warps_per_tile` must be a non-empty list of positive integers.")
