from __future__ import annotations

from compilagent_triton.candidates import validate_candidate
from compilagent_triton.schemas import CandidateConfig, CandidateKind, CandidateStatus


def test_meta_candidate_validation_accepts_power_of_two_warps() -> None:
    candidate = CandidateConfig(
        kernel_id="k",
        kind=CandidateKind.META_PARAMETERS,
        description="Try more warps",
        changes={"num_warps": 8},
        expected_effect="Increase parallelism.",
    )

    validation = validate_candidate(candidate)

    assert validation.ok
    assert validation.candidate.status == CandidateStatus.VALIDATED


def test_meta_candidate_validation_rejects_unknown_key() -> None:
    candidate = CandidateConfig(
        kernel_id="k",
        kind=CandidateKind.META_PARAMETERS,
        description="Try unsupported knob",
        changes={"unsafe": True},
        expected_effect="Unknown.",
    )

    validation = validate_candidate(candidate)

    assert not validation.ok
    assert validation.candidate.status == CandidateStatus.REJECTED
    assert "Unsupported meta-parameter keys" in validation.summary()
