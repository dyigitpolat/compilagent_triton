from __future__ import annotations

from compilagent_triton.events import ObservationEvent, redact
from compilagent_triton.trace_store import TraceStore


def test_observation_event_redacts_secret_keys() -> None:
    event = ObservationEvent.create(
        "tool.completed",
        payload={
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "nested": {"token": "sk-secret", "safe": "value"},
        },
    )

    assert event.payload["ANTHROPIC_API_KEY"] == "[redacted]"
    assert event.payload["nested"]["token"] == "[redacted]"
    assert event.payload["nested"]["safe"] == "value"


def test_redact_secret_like_values() -> None:
    assert redact({"value": "sk-ant-abc"}) == {"value": "[redacted]"}


def test_trace_store_append_and_filter(tmp_path) -> None:
    store = TraceStore(tmp_path).ensure()
    first = store.emit("tool.started", session_id="s1", payload={"tool": "a"})
    second = store.emit("tool.completed", session_id="s2", payload={"tool": "b"})

    assert [event.event_id for event in store.read_events()] == [first.event_id, second.event_id]
    assert [event.session_id for event in store.read_events(session_id="s2")] == ["s2"]
    assert [event.event_id for event in store.read_events(after=first.event_id)] == [second.event_id]
