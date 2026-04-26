from __future__ import annotations

import json

from compilagent_triton.experiment_memory import ExperimentMemory
from compilagent_triton.policy import CandidatePolicy


def test_experiment_memory_reads_benchmark_priors(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "run.json").write_text(
        json.dumps(
            [
                {
                    "candidate_id": "vector-add-n8-b512-w8-loadcg",
                    "n_elements": 8,
                    "block_size": 512,
                    "num_warps": 8,
                    "load_cache_modifier": ".cg",
                    "correctness": True,
                    "median_ms": 0.1,
                    "speedup_vs_baseline": 1.2,
                    "bandwidth_gbps": 100.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    priors = ExperimentMemory(tmp_path).load_prior_results(kernel_family="vector_add")

    assert len(priors) == 1
    assert priors[0].changes == {
        "BLOCK_SIZE": 512,
        "num_warps": 8,
        "LOAD_CACHE_MODIFIER": ".cg",
    }


def test_candidate_policy_uses_memory_then_generic_fallback(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "run.json").write_text(
        json.dumps(
            [
                {
                    "candidate_id": "vector-add-n8-b512-w8-loadcg",
                    "n_elements": 8,
                    "block_size": 512,
                    "num_warps": 8,
                    "load_cache_modifier": ".cg",
                    "correctness": True,
                    "median_ms": 0.1,
                    "speedup_vs_baseline": 1.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    policy = CandidatePolicy(ExperimentMemory(tmp_path))

    candidates = policy.propose(kernel_id="vector_add", objective="optimize add", budget=2)

    # Only the prior is returned; the legacy hand-coded fallback is gone.
    # Fresh exploration goes through `inspect_search_space` (the typed
    # derivation registry), not a static list embedded in the policy.
    assert candidates[0].changes["LOAD_CACHE_MODIFIER"] == ".cg"
    assert len(candidates) == 1


def test_candidate_policy_returns_empty_without_evidence(tmp_path) -> None:
    """No experiment memory + no static fallback ⇒ no candidates from the policy.

    Discovery now happens through `inspect_search_space` (the typed derivation
    registry on the active backend), which is invoked by the agent — not by
    the policy.
    """

    candidates = CandidatePolicy(ExperimentMemory(tmp_path)).propose(
        kernel_id="unknown",
        objective="explore",
        budget=2,
    )

    assert candidates == []
