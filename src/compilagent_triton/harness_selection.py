from __future__ import annotations

from typing import Any

from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

from .settings import DEFAULT_HARNESS, HarnessName

HARNESS_CONFIG_ID = "harness"
PYDANTIC_HARNESS: HarnessName = "pydantic_ai"
CLAUDE_AGENT_SDK_HARNESS: HarnessName = "claude_agent_sdk"
HARNESS_OPTIONS: tuple[HarnessName, ...] = (PYDANTIC_HARNESS, CLAUDE_AGENT_SDK_HARNESS)


class HarnessConfigOptionsProvider:
    """Expose the optimizer harness as a session-local ACP config option."""

    def get_config_options(
        self,
        session: Any,
        agent: Any,
    ) -> list[SessionConfigOptionSelect]:
        del agent
        return [
            SessionConfigOptionSelect(
                id=HARNESS_CONFIG_ID,
                name="Harness",
                category="agent",
                description="Select the optimizer agentic harness for this ACP session.",
                type="select",
                current_value=selected_harness(session, default=DEFAULT_HARNESS),
                options=[
                    SessionConfigSelectOption(value=PYDANTIC_HARNESS, name="Current"),
                    SessionConfigSelectOption(
                        value=CLAUDE_AGENT_SDK_HARNESS,
                        name="Claude Agent SDK",
                    ),
                ],
            )
        ]

    def set_config_option(
        self,
        session: Any,
        agent: Any,
        config_id: str,
        value: str | bool,
    ) -> list[SessionConfigOptionSelect] | None:
        if config_id != HARNESS_CONFIG_ID or not isinstance(value, str):
            return None
        normalized = normalize_harness(value)
        if normalized not in HARNESS_OPTIONS:
            return None
        session.config_values[HARNESS_CONFIG_ID] = normalized
        return self.get_config_options(session, agent)


def selected_harness(session: Any, *, default: str) -> HarnessName:
    configured = normalize_harness(getattr(session, "config_values", {}).get(HARNESS_CONFIG_ID, default))
    if configured == CLAUDE_AGENT_SDK_HARNESS:
        return CLAUDE_AGENT_SDK_HARNESS
    return PYDANTIC_HARNESS


def normalize_harness(value: str) -> str:
    return value.strip().replace("-", "_")
