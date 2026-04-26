from __future__ import annotations

from typing import Any

from .settings import CompilagentSettings


def model_for_settings(settings: CompilagentSettings) -> Any:
    """Return a pydantic-ai model spec without exposing secrets."""

    if settings.model_name in {"test", "test-model"}:
        from pydantic_ai.models.test import TestModel

        return TestModel()
    return settings.model_name


def model_settings_for_settings(settings: CompilagentSettings) -> dict[str, Any]:
    """Build provider settings used by pydantic-ai.

    Pydantic AI provider support evolves quickly, so the settings are kept as a
    plain dict and filtered to non-secret values.
    """

    return {
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "thinking": settings.reasoning_effort,
    }
