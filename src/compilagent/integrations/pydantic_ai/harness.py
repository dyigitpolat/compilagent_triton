"""`PydanticAIHarness` — implements the core `Harness` protocol on pydantic-ai.

The harness:
  1. Resolves a pydantic-ai `Model` from `request.model_id` + `request.extra`.
  2. Builds an `Agent` whose tools come from `request.toolset.tools`.
  3. Drives `agent.iter(prompt, ...)` and translates the streaming events
     into core `StreamEvent`s, yielding `RUN_FINISHED` (or `RUN_FAILED`) at
     the end.

Tool dispatch is owned by pydantic-ai itself: each `ToolDecl` is wrapped in
a synthetic function whose signature matches the decl's `args_schema`, so
the model-side tool schema is correct. The function calls
`decl.handler(args_dict)` and returns its string result; `ValueError`s are
re-raised as `ModelRetry` so pydantic-ai surfaces them to the model with a
chance to self-correct (matches the legacy contract).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping
from typing import Any

from compilagent.harness.base import (
    HarnessRunRequest,
    StreamEvent,
    StreamEventKind,
)

from ._model import resolve_model, resolve_model_settings
from ._tool_adapter import make_pydantic_ai_tool_fn
from ._translate import translate_model_event, translate_tool_event


def _retry_on_value_error(fn):
    """Wrap a tool fn so `ValueError` → pydantic-ai `ModelRetry` (lazy import)."""

    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from pydantic_ai.exceptions import ModelRetry

        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return wrapped


class PydanticAIHarness:
    """Harness adapter that drives a pydantic-ai `Agent`."""

    id: str = "pydantic_ai"
    supported_providers: tuple[str, ...] = ("anthropic", "mistral", "openai")
    # Known-good model strings for this harness. Surfaced in the
    # observation UI's model dropdown when the user picks `pydantic_ai`.
    # Models that need adaptive thinking should support pydantic-ai's
    # `anthropic_effort` setting; haiku-class Anthropic models don't
    # accept `output_config.effort` and are deliberately excluded.
    example_models: tuple[str, ...] = (
        "anthropic:claude-opus-4-7",
        "mistral:mistral-large-latest",
    )

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        """Drive one agent run; yield `StreamEvent`s until completion."""

        from pydantic_ai import Agent

        model = resolve_model(request.model_id, request.extra)
        model_settings = resolve_model_settings(
            request.model_id,
            reasoning_effort=request.reasoning_effort,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        agent: Agent[None, str] = Agent(
            model,
            instructions=request.system_instructions,
        )

        retries = int(request.extra.get("retries", 5))
        for decl in request.toolset.tools:
            fn = make_pydantic_ai_tool_fn(decl)
            agent.tool_plain(retries=retries)(_retry_on_value_error(fn))

        async for event in _drive_agent(agent, request, model_settings):
            yield event


async def _drive_agent(
    agent: Any,
    request: HarnessRunRequest,
    model_settings: Mapping[str, Any],
) -> AsyncIterator[StreamEvent]:
    from pydantic_ai import Agent

    final_text: str | None = None
    metadata: dict[str, Any] = {}
    try:
        async with agent.iter(
            request.user_prompt,
            model_settings=dict(model_settings),
        ) as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        async for ev in stream:
                            for translated in translate_model_event(ev):
                                yield translated
                elif Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        async for ev in stream:
                            for translated in translate_tool_event(ev):
                                yield translated
            result = getattr(agent_run, "result", None)
            if result is not None:
                output = getattr(result, "output", None)
                final_text = str(output) if output is not None else None
                usage = getattr(result, "usage", None)
                if usage is not None:
                    with contextlib.suppress(Exception):
                        metadata["usage"] = {
                            "request_tokens": getattr(usage, "request_tokens", None),
                            "response_tokens": getattr(usage, "response_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
    except Exception as exc:  # noqa: BLE001
        yield StreamEvent(
            kind=StreamEventKind.RUN_FAILED,
            text=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            extra=metadata,
        )
        return

    yield StreamEvent(
        kind=StreamEventKind.RUN_FINISHED,
        text=final_text,
        extra=metadata,
    )
