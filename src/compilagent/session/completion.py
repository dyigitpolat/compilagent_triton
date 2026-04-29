"""Run-completion policy and snapshot used by the continuation orchestrator.

`run_session()` drives the agent in a loop: harness.run -> evaluate the
snapshot -> ask the harness for a continuation request -> harness.run again.
This module owns the *evaluation* half of that loop. It is pure: no I/O, no
prompt text, no LLM. The harness owns continuation prompt construction.

`DefaultCompletionPolicy` only declares the run done when the candidate
budget has been met **and** both reflection tools (`synthesize_findings`,
`compare_runs`) have fired at least once across the iterations. That is what
guarantees we always get reflections from successful candidates and
benchmarks before the session terminates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Read-only view of session state at one continuation decision point."""

    successful_count: int
    failed_attempts: int
    max_candidates: int
    max_failed_attempts: int
    tools_called: frozenset[str]
    """Union of tool names called across all iterations seen so far."""
    iteration: int
    """0 = first run, 1 = first continuation, ..."""
    max_continuations: int
    harness_failed: bool
    """True iff the most recent `harness.run` ended with `RUN_FAILED`."""
    best_speedup: float | None
    """Best validated speedup achieved so far; None when no validated win."""


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    done: bool
    reason: str
    """One of: `budget_met`, `exhausted`, `harness_failed`, `failure_cap`,
    `continue`."""


@runtime_checkable
class RunCompletionPolicy(Protocol):
    """Decides whether the orchestrator should terminate the session."""

    def evaluate(self, snap: RunSnapshot) -> CompletionDecision: ...


class DefaultCompletionPolicy:
    """Stops the run only when:

      (a) `successful_count >= max_candidates` AND both reflection tools
          have been called, OR
      (b) `iteration >= max_continuations` (hard ceiling), OR
      (c) `failed_attempts >= max_failed_attempts` (failure cap), OR
      (d) the previous harness run yielded `RUN_FAILED`.

    `max_candidates == 0` is treated as "no candidate budget — only
    reflection required"; the run ends as soon as both reflection tools
    have been called.
    """

    REFLECTION_TOOLS = frozenset({"synthesize_findings", "compare_runs"})

    def evaluate(self, snap: RunSnapshot) -> CompletionDecision:
        if snap.harness_failed:
            return CompletionDecision(True, "harness_failed")
        if snap.failed_attempts >= snap.max_failed_attempts:
            return CompletionDecision(True, "failure_cap")
        budget_met = snap.successful_count >= snap.max_candidates
        reflected = self.REFLECTION_TOOLS.issubset(snap.tools_called)
        if (budget_met or snap.max_candidates == 0) and reflected:
            return CompletionDecision(True, "budget_met")
        if snap.iteration >= snap.max_continuations:
            return CompletionDecision(True, "exhausted")
        return CompletionDecision(False, "continue")
