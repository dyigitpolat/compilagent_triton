from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PriorResult:
    source_path: Path
    candidate_id: str
    kernel_family: str
    n_elements: int | None
    changes: dict[str, Any]
    median_ms: float
    speedup_vs_baseline: float | None
    bandwidth_gbps: float | None
    correctness: bool

    @property
    def score(self) -> float:
        if not self.correctness:
            return 0.0
        return self.speedup_vs_baseline or 1.0


@dataclass(slots=True)
class ExperimentMemory:
    root: Path

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def load_prior_results(
        self,
        *,
        kernel_family: str | None = None,
        capability: int | None = None,
        min_speedup: float = 1.0,
    ) -> list[PriorResult]:
        results: list[PriorResult] = []
        for path in sorted(self.reports_dir.glob("*.json")):
            try:
                raw_results = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in raw_results:
                prior = _prior_from_item(path, item)
                if prior is None:
                    continue
                if kernel_family is not None and prior.kernel_family != kernel_family:
                    continue
                if capability is not None:
                    item_cap = item.get("compute_capability") if isinstance(item, dict) else None
                    if isinstance(item_cap, int) and item_cap != capability:
                        continue
                if prior.score < min_speedup:
                    continue
                results.append(prior)
        return sorted(results, key=lambda result: result.score, reverse=True)

    def summarize_priors(self, *, limit: int = 8) -> str:
        priors = self.load_prior_results(min_speedup=0.0)[:limit]
        if not priors:
            return "No benchmark priors have been learned yet."
        lines = ["Learned benchmark priors:"]
        for prior in priors:
            speedup = (
                f"{prior.speedup_vs_baseline:.4f}x"
                if prior.speedup_vs_baseline is not None
                else "n/a"
            )
            bandwidth = f"{prior.bandwidth_gbps:.1f} GB/s" if prior.bandwidth_gbps else "n/a"
            lines.append(
                f"- `{prior.candidate_id}` family={prior.kernel_family} "
                f"median={prior.median_ms:.6f} ms speedup={speedup} bandwidth={bandwidth}"
            )
        return "\n".join(lines)


def infer_kernel_family(kernel_id: str, objective: str = "") -> str | None:
    text = f"{kernel_id} {objective}".lower().replace("-", "_")
    if "copy" in text:
        return "vector_copy"
    if "add" in text or "elementwise" in text:
        return "vector_add"
    if "matmul" in text or "dot" in text:
        return "matmul"
    if "reduction" in text or "reduce" in text:
        return "reduction"
    return None


def _prior_from_item(path: Path, item: dict[str, Any]) -> PriorResult | None:
    if not isinstance(item, dict) or not item.get("correctness", False):
        return None
    candidate_id = str(item.get("candidate_id") or "")
    if not candidate_id:
        return None
    median = item.get("median_ms")
    if not isinstance(median, int | float):
        return None
    family = _family_from_path_or_candidate(path, candidate_id)
    changes: dict[str, Any] = {}
    if "block_size" in item:
        changes["BLOCK_SIZE"] = item["block_size"]
    if "num_warps" in item:
        changes["num_warps"] = item["num_warps"]
    if item.get("load_cache_modifier"):
        changes["LOAD_CACHE_MODIFIER"] = item["load_cache_modifier"]
    return PriorResult(
        source_path=path,
        candidate_id=candidate_id,
        kernel_family=family,
        n_elements=item.get("n_elements") if isinstance(item.get("n_elements"), int) else None,
        changes=changes,
        median_ms=float(median),
        speedup_vs_baseline=float(item["speedup_vs_baseline"])
        if isinstance(item.get("speedup_vs_baseline"), int | float)
        else None,
        bandwidth_gbps=float(item["bandwidth_gbps"])
        if isinstance(item.get("bandwidth_gbps"), int | float)
        else None,
        correctness=True,
    )


def _family_from_path_or_candidate(path: Path, candidate_id: str) -> str:
    text = f"{path.stem} {candidate_id}".lower().replace("-", "_")
    if "vector_copy" in text:
        return "vector_copy"
    if "vector_add" in text:
        return "vector_add"
    if "matmul" in text:
        return "matmul"
    return "unknown"
