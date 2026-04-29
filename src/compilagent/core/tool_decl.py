"""Canonical agent-tool declaration.

Tools are declared once as `ToolDecl` records. Harness adapters
(`pydantic_ai`, `claude_agent_sdk`, ...) bind these into their native tool
surfaces — pydantic-ai's `@agent.tool_plain` decorators, the Claude Agent
SDK's MCP `@tool` registrations — without diverging on description, schema,
or read-only / destructive flags.

`handler` is the **typed callable** the session exposes (the bound session
method). Adapters that introspect typed signatures (pydantic-ai) call it
directly with kwargs; adapters that receive a JSON-schema-shaped dict over
the wire (Claude Agent SDK MCP) go through `ToolDecl.invoke(args_dict)`,
which validates the dict against `args_model` (if set) before calling the
handler with typed kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

ToolHandler = Callable[..., str]
ReturnsKind = Literal["json", "text"]


@dataclass(frozen=True, slots=True)
class ToolDecl:
    """One agent-facing tool, declared once and bound by harness adapters.

    `handler` is a typed callable (a bound session method); harness
    adapters that introspect Python signatures consume it directly.

    `args_schema` is the precomputed JSON Schema (derived from
    `args_model`'s `model_json_schema()` when `args_model` is set, or
    hand-written for tools that take no args). Adapters that drive their
    SDK off a static JSON schema (e.g. Claude Agent SDK MCP) read this.

    `args_model` is the Pydantic model that mirrors the handler's typed
    parameters. `invoke(args_dict)` validates a wire-shaped dict against
    it and calls `handler` with the resulting typed kwargs.

    Handlers may raise `ValueError` to signal bad input; adapters
    translate that into the harness's native retry / error response.
    """

    name: str
    description: str
    args_schema: dict[str, Any]
    handler: ToolHandler
    read_only: bool
    args_model: type[BaseModel] | None = None
    returns_kind: ReturnsKind = "json"
    metadata: dict[str, Any] = field(default_factory=dict)

    def invoke(self, args: dict[str, Any]) -> str:
        """Validate `args` and call the handler with typed kwargs.

        When `args_model` is set the dict is validated through it first
        (turning nested dicts into typed Pydantic instances). Validation
        errors are surfaced as `ValueError` so adapters fold them into a
        retryable tool error consistent with handler-emitted errors.
        """

        if self.args_model is None:
            return self.handler(**(args or {}))
        try:
            parsed = self.args_model.model_validate(args or {})
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        kwargs = {
            name: getattr(parsed, name) for name in self.args_model.model_fields
        }
        return self.handler(**kwargs)
