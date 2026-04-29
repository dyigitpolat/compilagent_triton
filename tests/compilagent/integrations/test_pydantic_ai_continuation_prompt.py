"""Unit tests for the pydantic-ai harness's continuation prompt builder."""

from __future__ import annotations

from compilagent.integrations.pydantic_ai.prompts import continuation_user_prompt
from compilagent.session.completion import RunSnapshot


def _snap(**overrides):
    base = dict(
        successful_count=1,
        failed_attempts=0,
        max_candidates=8,
        max_failed_attempts=24,
        tools_called=frozenset(),
        iteration=0,
        max_continuations=4,
        harness_failed=False,
        best_speedup=None,
    )
    base.update(overrides)
    return RunSnapshot(**base)


def test_continuation_lists_both_missing_reflection_tools():
    text = continuation_user_prompt(_snap(tools_called=frozenset()))
    assert "synthesize_findings" in text
    assert "compare_runs" in text


def test_continuation_omits_reflection_step_when_already_called():
    text = continuation_user_prompt(
        _snap(tools_called=frozenset({"synthesize_findings", "compare_runs"}))
    )
    assert "to reflect" not in text


def test_continuation_quotes_progress_counts_and_iteration():
    text = continuation_user_prompt(
        _snap(
            successful_count=3,
            failed_attempts=2,
            max_candidates=8,
            iteration=1,
            best_speedup=1.84,
        )
    )
    assert "successful_count=3 / 8" in text
    assert "slots_remaining=5" in text
    assert "failed_attempts=2" in text
    assert "1.84x" in text
    assert "continuation #2" in text


def test_continuation_handles_missing_speedup():
    text = continuation_user_prompt(_snap(best_speedup=None))
    assert "no validated win yet" in text


def test_continuation_caps_propose_count_to_three():
    text = continuation_user_prompt(_snap(successful_count=0, max_candidates=20))
    assert "Propose 3 more candidate" in text
