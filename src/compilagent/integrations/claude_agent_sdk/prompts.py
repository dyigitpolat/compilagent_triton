"""Prompt builders local to the Claude-Agent-SDK harness.

The session orchestrator delegates continuation prompt construction to
the harness via `Harness.build_continuation_request`. This module owns
the SDK-flavoured continuation prompt — Markdown headers play well with
the `claude_code` system preset, so we lean on that style here.
"""

from __future__ import annotations

from compilagent.session.completion import RunSnapshot

REFLECTION_TOOLS = ("synthesize_findings", "compare_runs")


def continuation_user_prompt(snap: RunSnapshot) -> str:
    """Build the next user-prompt the SDK agent sees after stopping early.

    The Claude SDK's `claude_code` preset is trained to respond well to
    Markdown-headed instructions; we use a `## Resume` block to make the
    re-engagement explicit, then enumerate required actions.
    """

    missing = sorted(set(REFLECTION_TOOLS) - snap.tools_called)
    slots = max(0, snap.max_candidates - snap.successful_count)
    best = (
        f"{snap.best_speedup:.2f}x"
        if snap.best_speedup
        else "no validated win yet"
    )

    body: list[str] = [
        "## Resume optimization session",
        "",
        "You stopped before the candidate budget was met. Resume now.",
        "",
        "**State:**",
        (
            f"- successful_count: {snap.successful_count} / "
            f"{snap.max_candidates}  (slots_remaining={slots})"
        ),
        f"- failed_attempts: {snap.failed_attempts}",
        f"- best so far: {best}",
        f"- continuation: #{snap.iteration + 1}",
        "",
        "**Required next actions:**",
    ]
    step = 1
    if missing:
        body.append(
            f"{step}. Call `{'` and `'.join(missing)}` to reflect on what you "
            "have learned so far."
        )
        step += 1
    propose_count = min(slots, 3) if slots else 1
    body.append(
        f"{step}. Propose {propose_count} more candidate(s) targeting lever "
        "combinations you have NOT yet tried, informed by the "
        "`synthesize_findings` output. Run them via `run_candidate` or "
        "`run_candidates`. Do not stop until `slots_remaining == 0` or the "
        "orchestrator closes the session."
    )
    return "\n".join(body)
