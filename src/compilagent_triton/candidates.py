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
    elif candidate.kind == CandidateKind.MEMORY_ACCESS_POLICY:
        _validate_memory_access_candidate(candidate.changes, diagnostics)
    elif candidate.kind == CandidateKind.MATMUL_POLICY:
        _validate_matmul_candidate(candidate.changes, diagnostics)
    elif candidate.kind == CandidateKind.PASS_INTERVENTIONS:
        _validate_pass_interventions_candidate(candidate.changes, diagnostics)
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
    allowed = {"op_name", "order", "per_thread", "vector_width", "reason"} | _VALID_META_KEYS
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


def _validate_memory_access_candidate(changes: dict[str, Any], diagnostics: list[str]) -> None:
    allowed = {
        "LOAD_CACHE_MODIFIER", "eviction_policy", "vector_width",
        "mask_strategy", "reason",
    } | _VALID_META_KEYS
    unknown = sorted(set(changes) - allowed)
    if unknown:
        diagnostics.append(f"Unsupported memory-access keys: {', '.join(unknown)}")
    if "LOAD_CACHE_MODIFIER" in changes and changes["LOAD_CACHE_MODIFIER"] not in {"", ".ca", ".cg"}:
        diagnostics.append("`LOAD_CACHE_MODIFIER` must be one of '', '.ca', or '.cg'.")
    if "eviction_policy" in changes and changes["eviction_policy"] not in {"", "evict_last", "evict_first"}:
        diagnostics.append("`eviction_policy` must be '', 'evict_last', or 'evict_first'.")
    if "vector_width" in changes and (not isinstance(changes["vector_width"], int) or changes["vector_width"] <= 0):
        diagnostics.append("`vector_width` must be a positive integer.")


def _validate_pass_interventions_candidate(
    changes: dict[str, Any], diagnostics: list[str]
) -> None:
    from .triton_hooks.passes import all_pass_names

    allowed = {"pass_interventions", "reason"} | _VALID_META_KEYS
    unknown = sorted(set(changes) - allowed)
    if unknown:
        diagnostics.append(f"Unsupported pass-intervention keys: {', '.join(unknown)}")
    items = changes.get("pass_interventions") or []
    if not isinstance(items, list) or not items:
        diagnostics.append(
            "`pass_interventions` must be a non-empty list of {pass_name, action[, args, rationale]} dicts."
        )
        return
    valid_passes = set(all_pass_names())
    valid_actions = {"run", "skip", "replace"}
    for entry in items:
        if not isinstance(entry, dict):
            diagnostics.append("each pass_interventions entry must be a dict")
            continue
        pass_name = entry.get("pass_name")
        if pass_name not in valid_passes:
            diagnostics.append(f"unknown pass `{pass_name}`")
        action = entry.get("action", "run")
        if action not in valid_actions:
            diagnostics.append(
                f"action must be one of {sorted(valid_actions)}, got `{action}`"
            )


def _validate_matmul_candidate(changes: dict[str, Any], diagnostics: list[str]) -> None:
    allowed = {"mma_version", "warps_per_tile", "reason"} | _VALID_META_KEYS
    unknown = sorted(set(changes) - allowed)
    if unknown:
        diagnostics.append(f"Unsupported matmul keys: {', '.join(unknown)}")
    if "mma_version" in changes and changes["mma_version"] not in {1, 2, 3, 5}:
        diagnostics.append("`mma_version` must be one of 1, 2, 3, or 5.")
    if "warps_per_tile" in changes:
        warps = changes["warps_per_tile"]
        if not isinstance(warps, list) or not warps or not all(isinstance(i, int) and i > 0 for i in warps):
            diagnostics.append("`warps_per_tile` must be a non-empty list of positive integers.")
