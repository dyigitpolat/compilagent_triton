from __future__ import annotations

from dataclasses import dataclass

from .experiment_memory import ExperimentMemory, infer_kernel_family
from .schemas import CandidateConfig, CandidateKind


@dataclass(slots=True)
class CandidatePolicy:
    memory: ExperimentMemory

    def propose(
        self,
        *,
        kernel_id: str,
        objective: str,
        budget: int,
        hypothesis_id: str | None = None,
    ) -> list[CandidateConfig]:
        family = infer_kernel_family(kernel_id, objective)
        priors = self.memory.load_prior_results(kernel_family=family, min_speedup=1.0) if family else []
        candidates: list[CandidateConfig] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for prior in priors:
            key = _changes_key(prior.changes)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateConfig(
                    kernel_id=kernel_id,
                    kind=CandidateKind.META_PARAMETERS,
                    hypothesis_id=hypothesis_id,
                    description=(
                        "Evidence-ranked meta-parameter candidate from experiment memory "
                        f"({prior.candidate_id})"
                    ),
                    changes=prior.changes,
                    expected_effect=(
                        f"Retest prior result with observed speedup "
                        f"{prior.speedup_vs_baseline or 1.0:.4f}x on related workload."
                    ),
                    validation_constraints=[
                        "candidate changes must be accepted by the target Triton kernel",
                        "prior must be retested on the current kernel before acceptance",
                    ],
                )
            )
            if len(candidates) >= budget:
                return candidates

        # No hand-coded fallback: when prior experience is empty, the agent
        # discovers candidates through `inspect_search_space()` (the typed
        # derivation registry) and proposes them via the intervention tools.
        return candidates


def _changes_key(changes: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in changes.items()))
