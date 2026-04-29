"""Prompt builders local to the pydantic-ai harness.

The session-level continuation orchestrator delegates prompt construction
to each harness via `Harness.build_continuation_request`. This module owns
the pydantic-ai-flavoured continuation prompt — kept here (not in
`session/`) so each harness can evolve its prompting independently.
"""

from __future__ import annotations

from compilagent.session.completion import RunSnapshot

REFLECTION_TOOLS = ("synthesize_findings", "compare_runs")


def continuation_user_prompt(snap: RunSnapshot) -> str:
    """Build the next user-prompt the agent sees after stopping early.

    The prompt names every gap explicitly: missing reflection tools,
    remaining candidate slots, the best speedup so far. Tone is directive
    ("You stopped …", "Required next actions") because the agent's
    discretionary stop is exactly what we're overriding.
    """

    missing = sorted(set(REFLECTION_TOOLS) - snap.tools_called)
    slots = max(0, snap.max_candidates - snap.successful_count)
    best = (
        f"{snap.best_speedup:.2f}x"
        if snap.best_speedup
        else "no validated win yet"
    )

    lines = [
        "You stopped before the budget was met. Resume the session.",
        (
            f"- successful_count={snap.successful_count} / "
            f"{snap.max_candidates} (slots_remaining={slots})"
        ),
        f"- failed_attempts={snap.failed_attempts}",
        f"- best so far: {best}",
        f"- continuation #{snap.iteration + 1}",
        "",
        "Required next actions:",
    ]
    step = 1
    if missing:
        lines.append(
            f"{step}. Call {' and '.join(missing)} to reflect on what you've "
            "learned."
        )
        step += 1
    propose_count = min(slots, 3) if slots else 1
    lines.append(
        f"{step}. Propose {propose_count} more candidate(s) targeting lever "
        "combinations you have NOT yet tried, informed by the "
        "synthesize_findings output. Run them. Do not stop until "
        "slots_remaining == 0 or the orchestrator closes the session."
    )
    return "\n".join(lines)
