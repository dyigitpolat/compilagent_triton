from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from compilagent_triton.events import ObservationEvent
from compilagent_triton.examples import RunRequest
from compilagent_triton.observation_server import _event_stream, create_app, read_gpu_telemetry
from compilagent_triton.trace_store import TraceStore


def test_observation_server_serves_dashboard_and_events(tmp_path) -> None:
    workspace = tmp_path / ".compilagent-triton"
    store = TraceStore(workspace).ensure()
    event = store.emit("tool.completed", session_id="s1", payload={"tool": "x"})
    app = create_app(workspace_root=workspace)
    client = TestClient(app)

    index = client.get("/")
    events = client.get("/api/events")
    sessions = client.get("/api/sessions")

    assert index.status_code == 200
    assert "Optimization Cockpit" in index.text
    assert events.json()["events"][0]["event_id"] == event.event_id
    assert sessions.json()["sessions"] == ["s1"]


def test_observation_server_serves_safe_artifact(tmp_path) -> None:
    workspace = tmp_path / ".compilagent-triton"
    artifact = workspace / "reports" / "report.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("# report", encoding="utf-8")
    client = TestClient(create_app(workspace_root=workspace))

    assert client.get("/api/artifacts/reports/report.md").status_code == 200
    assert client.get("/api/artifacts/../secret").status_code in {403, 404}


def test_observation_server_serves_examples_and_source(tmp_path) -> None:
    workspace = tmp_path / ".compilagent-triton"
    client = TestClient(create_app(workspace_root=workspace))

    examples = client.get("/api/examples")
    source = client.get("/api/examples/vector_add")

    assert examples.status_code == 200
    assert any(example["id"] == "vector_add" for example in examples.json()["examples"])
    assert source.status_code == 200
    assert "run_vector_add_sweep" in source.json()["source"]


def test_observation_server_aggregates_loops_benchmarks_and_previews(tmp_path) -> None:
    workspace = tmp_path / ".compilagent-triton"
    report = workspace / "reports" / "run-1.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        """
[
  {"candidate_id": "baseline", "correctness": true, "median_ms": 2.0, "p20_ms": 1.9, "p80_ms": 2.1, "bandwidth_gbps": 10.0, "speedup_vs_baseline": 1.0},
  {"candidate_id": "best", "correctness": true, "median_ms": 1.0, "p20_ms": 0.9, "p80_ms": 1.1, "bandwidth_gbps": 20.0, "speedup_vs_baseline": 2.0}
]
""",
        encoding="utf-8",
    )
    store = TraceStore(workspace).ensure()
    store.emit("run.started", payload={"run_id": "run-1", "family": "vector_add"})
    store.emit(
        "benchmark.completed",
        payload={
            "run_id": "run-1",
            "family": "vector_add",
            "best": {"candidate_id": "best", "median_ms": 1.0, "speedup_vs_baseline": 2.0},
            "candidate_count": 2,
        },
        artifact_paths=[str(report)],
    )
    store.emit(
        "comparison.created",
        payload={
            "run_id": "run-1",
            "candidate_id": "best",
            "speedup_vs_baseline": 2.0,
            "delta_percent": 100.0,
            "conclusion": "candidate improved baseline",
        },
    )
    client = TestClient(create_app(workspace_root=workspace))

    loops = client.get("/api/loops").json()["loops"]
    benchmarks = client.get("/api/benchmarks").json()["benchmarks"]
    series = client.get("/api/benchmarks/run-1/series").json()["series"]
    comparisons = client.get("/api/comparisons").json()["comparisons"]
    preview = client.get("/api/artifacts/reports/run-1.json/preview").json()

    assert loops[0]["id"] == "run-1"
    assert {row["candidate_id"] for row in benchmarks} == {"baseline", "best"}
    assert series[0]["candidate_id"] == "best"
    assert comparisons[0]["conclusion"] == "candidate improved baseline"
    assert preview["render_mode"] == "json"
    assert '"candidate_id": "baseline"' in preview["text"]


def test_observation_server_run_endpoint_uses_registered_runner(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".compilagent-triton"

    def fake_run_registered_example(
        *,
        request: RunRequest,
        run_id: str,
        workspace_root: Path,
        trace_store: TraceStore,
    ) -> dict:
        trace_store.emit("run.started", payload={"run_id": run_id, "example_id": request.example_id})
        trace_store.emit("run.completed", payload={"run_id": run_id, "example_id": request.example_id})
        return {"run_id": run_id}

    monkeypatch.setattr(
        "compilagent_triton.observation_server.run_registered_example",
        fake_run_registered_example,
    )
    client = TestClient(create_app(workspace_root=workspace))

    response = client.post(
        "/api/runs",
        json={
            "example_id": "vector_add",
            "config": {"block_sizes": [256], "num_warps": [4], "load_cache_modifiers": ["none"]},
        },
    )

    assert response.status_code == 200
    events = client.get("/api/events").json()["events"]
    assert [event["kind"] for event in events] == ["run.requested", "run.started", "run.completed"]


def test_observation_server_telemetry_fallback(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("subprocess.check_output", fail)

    assert read_gpu_telemetry() == []


def test_event_stream_yields_existing_event(tmp_path) -> None:
    workspace = tmp_path / ".compilagent-triton"
    store = TraceStore(workspace)
    event = store.append(ObservationEvent.create("log.line", payload={"line": "hello"}))

    async def read_one() -> str:
        stream = _event_stream(store)
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    chunk = asyncio.run(read_one())

    assert f"id: {event.event_id}" in chunk
    assert "event: log.line" in chunk
