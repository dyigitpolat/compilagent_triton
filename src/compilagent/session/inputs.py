"""Pydantic input models for the canonical session toolset.

These models are the source of truth for the agent-facing tool schemas.
pydantic-ai introspects the typed parameters of the session methods and
emits a JSON Schema that mirrors these models — the agent sees structured
arrays / objects and never has to emit string-encoded JSON.

Other harness adapters (e.g. the Claude Agent SDK MCP bridge) generate the
same JSON Schema by calling `model_json_schema()` on the wrapper model
returned by `tool_args_model_for(method)` (defined in `session.tools`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InterventionInput(BaseModel):
    """One backend intervention the agent proposes for a candidate compile.

    `target_kind` and `target_selector` together identify what the
    intervention modifies (the backend interprets the kind string).
    `payload` is opaque to the core — the backend validates it via
    `Backend.validate_intervention`.
    """

    model_config = ConfigDict(extra="forbid")

    target_kind: str = Field(
        description=(
            "Backend-defined intervention kind. Discover the valid kinds for "
            "the current backend via `inspect_search_space`; common values "
            "include `pass`, `launch`, `knob`, `lowering`, `fx_node`, "
            "`scheduler`."
        )
    )
    target_selector: str = Field(
        default="",
        description=(
            "Locator within the kind. For `pass` interventions this is "
            "`<stage>:<pass_name>`; for `knob` it is the dotted config path; "
            "for `launch` it is the kernel symbol."
        ),
    )
    payload: Any = Field(
        default=None,
        description=(
            "Opaque value the backend interprets. For pass interventions: "
            "`{\"action\": \"skip\"|\"insert\"|\"reorder\"|\"run\", "
            "\"args\": {...}}`. For knob interventions: the value to set."
        ),
    )
    rationale: str = Field(
        default="",
        description="Optional one-line explanation surfaced in the trace UI.",
    )


class PlanInput(BaseModel):
    """One candidate plan: a description plus the interventions it bundles."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        description="Short human description of what this candidate tries."
    )
    expected_effect: str = Field(
        default="",
        description="Optional hypothesis: how this should beat the baseline.",
    )
    interventions: list[InterventionInput] = Field(
        description=(
            "Inline list of interventions applied together to a single "
            "compile + benchmark. MUST be a JSON array — never a string."
        ),
    )
