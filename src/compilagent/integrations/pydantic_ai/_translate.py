"""Translate pydantic-ai message stream events into core `StreamEvent`s.

The harness drives `agent.iter(...)` and gets per-node streams of pydantic-ai
events: `PartStartEvent` / `PartDeltaEvent` carrying `ThinkingPart` /
`TextPart` / `ToolCallPart`, and `FunctionToolCallEvent` /
`FunctionToolResultEvent`. We map them into the harness-agnostic
`StreamEvent` vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from compilagent.harness.base import StreamEvent, StreamEventKind


def translate_model_event(event: Any) -> Iterator[StreamEvent]:
    """Translate one model-stream event into zero or more `StreamEvent`s."""

    from pydantic_ai.messages import (
        PartDeltaEvent,
        PartStartEvent,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
    )

    if isinstance(event, PartStartEvent):
        part = event.part
        if isinstance(part, ThinkingPart):
            yield StreamEvent(
                kind=StreamEventKind.THINKING_STARTED,
                part_index=event.index,
            )
            if part.content:
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    part_index=event.index,
                    text=part.content,
                )
        elif isinstance(part, TextPart):
            yield StreamEvent(
                kind=StreamEventKind.TEXT_STARTED,
                part_index=event.index,
            )
            if part.content:
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA,
                    part_index=event.index,
                    text=part.content,
                )
        # ToolCallPart is surfaced via FunctionToolCallEvent on the call_tools node.
    elif isinstance(event, PartDeltaEvent):
        delta = event.delta
        if isinstance(delta, ThinkingPartDelta):
            content = getattr(delta, "content_delta", None)
            if content:
                yield StreamEvent(
                    kind=StreamEventKind.THINKING_DELTA,
                    part_index=event.index,
                    text=content,
                )
        elif isinstance(delta, TextPartDelta):
            content = getattr(delta, "content_delta", None)
            if content:
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA,
                    part_index=event.index,
                    text=content,
                )


def translate_tool_event(event: Any) -> Iterator[StreamEvent]:
    """Translate one tool-stream event into zero or more `StreamEvent`s."""

    from pydantic_ai.messages import (
        BuiltinToolCallEvent,
        BuiltinToolResultEvent,
        FunctionToolCallEvent,
        FunctionToolResultEvent,
    )

    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        try:
            args = part.args_as_dict()
        except Exception:  # noqa: BLE001
            args = getattr(part, "args", None)
        yield StreamEvent(
            kind=StreamEventKind.TOOL_CALL,
            tool_name=getattr(part, "tool_name", None),
            tool_call_id=getattr(part, "tool_call_id", None),
            tool_args=args if isinstance(args, dict) else None,
        )
    elif isinstance(event, FunctionToolResultEvent):
        result_part = getattr(event, "result", None)
        tool_call_id = getattr(result_part, "tool_call_id", None)
        tool_name = getattr(result_part, "tool_name", None)
        content = getattr(result_part, "content", None)
        if isinstance(content, str):
            yield StreamEvent(
                kind=StreamEventKind.TOOL_RESULT,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_result=content,
            )
        else:
            # Pydantic-ai wraps tool errors in a non-string content; surface
            # them as TOOL_ERROR so the session distinguishes from successes.
            error_message = str(content) if content is not None else ""
            yield StreamEvent(
                kind=StreamEventKind.TOOL_ERROR,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                error_type="ToolError",
                error_message=error_message,
            )
    elif isinstance(event, (BuiltinToolCallEvent, BuiltinToolResultEvent)):
        # Built-in tools (e.g. retrieval, web search) — not relevant to the
        # optimizer's tool surface; ignore.
        return
