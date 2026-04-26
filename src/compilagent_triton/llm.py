from __future__ import annotations

import asyncio
import os
import random
from contextlib import asynccontextmanager
from typing import Any

from .settings import CompilagentSettings


_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a Retry-After hint from a ModelHTTPError, if the provider gave one."""

    headers = getattr(exc, "headers", None) or {}
    if isinstance(headers, dict):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except (TypeError, ValueError):
                return None
    return None


def _build_retrying_model(wrapped: Any, *, max_attempts: int = 6,
                          base_seconds: float = 1.0, max_seconds: float = 30.0) -> Any:
    """Build a pydantic-ai Model subclass that retries 4xx/5xx HTTP errors.

    Constructed dynamically so we can inherit from `WrapperModel` (which is
    only importable at runtime) and remain a real `Model` instance — pydantic-ai
    enforces that with isinstance checks during agent construction.
    """

    from pydantic_ai.exceptions import ModelHTTPError  # lazy
    from pydantic_ai.models.wrapper import WrapperModel  # lazy

    def _is_retryable(exc: Exception) -> bool:
        return isinstance(exc, ModelHTTPError) and getattr(exc, "status_code", None) in _RETRY_STATUS

    async def _sleep(attempt: int, exc: Exception) -> None:
        hint = _retry_after_seconds(exc)
        if hint is not None and hint > 0:
            wait = min(hint, max_seconds)
        else:
            cap = min(max_seconds, base_seconds * (2 ** attempt))
            wait = random.uniform(base_seconds, max(base_seconds, cap))
        await asyncio.sleep(wait)

    class RetryingModel(WrapperModel):
        async def request(self, messages, model_settings, model_request_parameters):
            for attempt in range(max_attempts):
                try:
                    return await self.wrapped.request(messages, model_settings, model_request_parameters)
                except Exception as exc:
                    if not _is_retryable(exc) or attempt == max_attempts - 1:
                        raise
                    await _sleep(attempt, exc)

        @asynccontextmanager
        async def request_stream(
            self, messages, model_settings, model_request_parameters, run_context=None,
        ):
            for attempt in range(max_attempts):
                try:
                    async with self.wrapped.request_stream(
                        messages, model_settings, model_request_parameters, run_context,
                    ) as resp:
                        yield resp
                        return
                except Exception as exc:
                    if not _is_retryable(exc) or attempt == max_attempts - 1:
                        raise
                    await _sleep(attempt, exc)

    return RetryingModel(wrapped)


def model_for_settings(settings: CompilagentSettings) -> Any:
    """Return a pydantic-ai model spec, wiring per-provider API keys from .env."""

    name = settings.model_name
    if name in {"test", "test-model"}:
        from pydantic_ai.models.test import TestModel

        return TestModel()

    base: Any
    if name.startswith("anthropic:") and settings.anthropic_api_key is not None:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        base = AnthropicModel(
            name.removeprefix("anthropic:"),
            provider=AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value()),
        )
    elif name.startswith("mistral:") and settings.mistral_api_key is not None:
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.providers.mistral import MistralProvider

        base = MistralModel(
            name.removeprefix("mistral:"),
            provider=MistralProvider(api_key=settings.mistral_api_key.get_secret_value()),
        )
    elif name.startswith("openai:") and settings.openai_api_key is not None:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base = OpenAIChatModel(
            name.removeprefix("openai:"),
            provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
        )
    else:
        base = name

    max_attempts = int(os.environ.get("COMPILAGENT_RETRY_MAX_ATTEMPTS", "8"))
    base_seconds = float(os.environ.get("COMPILAGENT_RETRY_BASE_SECONDS", "2.0"))
    max_seconds = float(os.environ.get("COMPILAGENT_RETRY_MAX_SECONDS", "60.0"))
    return _build_retrying_model(
        base,
        max_attempts=max_attempts,
        base_seconds=base_seconds,
        max_seconds=max_seconds,
    )


# Map our internal reasoning effort to Anthropic's adaptive-thinking effort
# levels. Newer Anthropic models (Opus 4.7+) require `output_config.effort`
# rather than the legacy `thinking.type=enabled` form; pydantic-ai exposes
# this as the `anthropic_effort` model setting (Literal['low','medium','high','max']).
_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "extra_high": "max",
    "extra-high": "max",
    "max": "max",
}


def model_settings_for_settings(settings: CompilagentSettings) -> dict[str, Any]:
    """Build provider settings used by pydantic-ai.

    Translates our internal `reasoning_effort` to Anthropic's adaptive-thinking
    `output_config.effort` (`anthropic_effort` in pydantic-ai). When effort is
    set, temperature must be 1.0 (Anthropic requirement for thinking-enabled
    requests).
    """

    raw = (settings.reasoning_effort or "").strip().lower().replace("-", "_")
    effort = _EFFORT_MAP.get(raw)
    out: dict[str, Any] = {"max_tokens": settings.max_tokens}
    is_anthropic = settings.model_name.startswith("anthropic:")
    if is_anthropic and effort is not None:
        # Anthropic adaptive thinking — temperature must be 1.0 when enabled.
        out["anthropic_effort"] = effort
        out["temperature"] = 1.0
    else:
        out["temperature"] = settings.temperature
    return out
