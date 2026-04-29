"""pydantic-ai harness adapter for the compilagent core.

Self-registers under `harness_registry` at import time. Bringing this
integration online is `import compilagent.integrations.pydantic_ai` (or a
matching entry point in pyproject — see `docs/integration_guide.md`).
"""

from __future__ import annotations

from compilagent.harness.registry import harness_registry

from .harness import PydanticAIHarness

if "pydantic_ai" not in harness_registry.ids():
    harness_registry.register("pydantic_ai", PydanticAIHarness)

__all__ = ["PydanticAIHarness"]
