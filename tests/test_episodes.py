from __future__ import annotations

from pathlib import Path

from compilagent_triton.episodes import EpisodeStore
from compilagent_triton.schemas import CandidateConfig, CandidateKind
from compilagent_triton.workspace import OptimizationWorkspace


def test_episode_store_persists_hypothesis_candidate_and_report(tmp_path: Path) -> None:
    workspace = OptimizationWorkspace(tmp_path).ensure()
    store = EpisodeStore(workspace)
    episode = store.create(
        kernel_id="k",
        objective="beat baseline",
        budget={"max_candidates": 1},
        model_metadata={"model_name": "test", "reasoning_effort": "extra_high"},
    )

    hypothesis = store.record_hypothesis(
        episode.id,
        statement="Coalescing is too conservative.",
        expected_effect="Improve memory bandwidth.",
    )
    candidate = CandidateConfig(
        kernel_id="k",
        kind=CandidateKind.META_PARAMETERS,
        hypothesis_id=hypothesis.id,
        description="Try fewer warps",
        changes={"num_warps": 4},
        expected_effect="Reduce register pressure.",
    )
    store.add_candidate(episode.id, candidate)
    report_path = store.write_report(episode.id)

    loaded = store.load(episode.id)
    assert len(loaded.hypotheses) == 1
    assert len(loaded.candidates) == 1
    assert report_path.exists()
