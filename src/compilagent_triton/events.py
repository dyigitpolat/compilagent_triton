from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

_SECRET_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
_REDACTED = "[redacted]"


class ObservationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt-{uuid4().hex[:16]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: str
    session_id: str | None = None
    episode_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
        payload: dict[str, Any] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> ObservationEvent:
        return cls(
            kind=kind,
            session_id=session_id,
            episode_id=episode_id,
            payload=redact(payload or {}),
            artifact_paths=artifact_paths or [],
        )


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = _REDACTED
            else:
                redacted[key_text] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str) and _looks_like_secret_value(value):
        return _REDACTED
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SECRET_FRAGMENTS)


def _looks_like_secret_value(value: str) -> bool:
    lowered = value.lower()
    if "anthropic_api_key" in lowered or "authorization:" in lowered:
        return True
    return value.startswith(("sk-", "sk-ant-"))
