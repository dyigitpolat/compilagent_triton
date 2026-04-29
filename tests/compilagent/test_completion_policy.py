"""Unit tests for `DefaultCompletionPolicy`.

The policy is a pure function of a snapshot — these tests cover the
decision matrix at every interesting boundary.
"""

from __future__ import annotations

import pytest

from compilagent.session.completion import (
    DefaultCompletionPolicy,
    RunSnapshot,
)

REFLECTED = frozenset({"synthesize_findings", "compare_runs"})


def _snap(**overrides):
    base = dict(
        successful_count=0,
        failed_attempts=0,
        max_candidates=4,
        max_failed_attempts=12,
        tools_called=frozenset(),
        iteration=0,
        max_continuations=4,
        harness_failed=False,
        best_speedup=None,
    )
    base.update(overrides)
    return RunSnapshot(**base)


def test_budget_met_with_reflection_terminates():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(successful_count=4, tools_called=REFLECTED)
    )
    assert decision.done is True
    assert decision.reason == "budget_met"


def test_budget_met_without_reflection_continues():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(_snap(successful_count=4, tools_called=frozenset()))
    assert decision.done is False
    assert decision.reason == "continue"


def test_budget_unmet_continues_when_iterations_remain():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(successful_count=1, iteration=0, max_continuations=4)
    )
    assert decision.done is False
    assert decision.reason == "continue"


def test_continuation_ceiling_terminates():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(successful_count=1, iteration=4, max_continuations=4)
    )
    assert decision.done is True
    assert decision.reason == "exhausted"


def test_failure_cap_terminates():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(failed_attempts=12, max_failed_attempts=12)
    )
    assert decision.done is True
    assert decision.reason == "failure_cap"


def test_harness_failure_short_circuits():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(successful_count=4, tools_called=REFLECTED, harness_failed=True)
    )
    assert decision.done is True
    assert decision.reason == "harness_failed"


def test_max_candidates_zero_with_reflection_terminates():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(max_candidates=0, tools_called=REFLECTED)
    )
    assert decision.done is True
    assert decision.reason == "budget_met"


def test_max_candidates_zero_without_reflection_continues():
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(_snap(max_candidates=0, tools_called=frozenset()))
    assert decision.done is False


@pytest.mark.parametrize("partial", [{"synthesize_findings"}, {"compare_runs"}])
def test_partial_reflection_does_not_satisfy_budget_met(partial):
    policy = DefaultCompletionPolicy()
    decision = policy.evaluate(
        _snap(successful_count=4, tools_called=frozenset(partial))
    )
    assert decision.done is False
    assert decision.reason == "continue"
