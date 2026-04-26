from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_acp")
pytest.importorskip("pydantic_ai")

from compilagent_triton.agent import agent_factory_from_session, build_config, build_optimizer_agent
from compilagent_triton.claude_agent_harness import ClaudeAgentSdkHarness
from compilagent_triton.harness_selection import CLAUDE_AGENT_SDK_HARNESS, HARNESS_CONFIG_ID
from compilagent_triton.optimizer_runtime import OptimizerRuntime
from compilagent_triton.settings import CompilagentSettings
from compilagent_triton.workspace import OptimizationWorkspace


def test_agent_and_config_build_with_test_model(tmp_path: Path) -> None:
    settings = CompilagentSettings(model_name="test")
    workspace = OptimizationWorkspace(tmp_path).ensure()

    agent = build_optimizer_agent(settings=settings, workspace=workspace)
    config = build_config()

    assert agent.name == "triton-acp-optimizer"
    assert config.projection_maps


def test_harness_config_option_switches_session_agent(tmp_path: Path) -> None:
    settings = CompilagentSettings(model_name="test")
    session = SimpleNamespace(
        cwd=tmp_path,
        config_values={HARNESS_CONFIG_ID: CLAUDE_AGENT_SDK_HARNESS},
    )

    agent = agent_factory_from_session(session)
    config = build_config()
    option = config.config_options_provider.get_config_options(session, agent)[0]

    assert option.id == HARNESS_CONFIG_ID
    assert option.current_value == CLAUDE_AGENT_SDK_HARNESS
    assert agent.name == "triton-claude-sdk-optimizer"
    assert settings.harness == "pydantic_ai"


def test_harness_config_option_is_visible_over_acp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_acp import BlackBoxHarness

    monkeypatch.setenv("COMPILAGENT_MODEL", "test")
    harness = BlackBoxHarness.create(
        agent_factory=agent_factory_from_session,
        config=build_config(),
    )

    session = asyncio.run(harness.new_session(cwd=str(tmp_path)))
    assert session.config_options is not None
    option_ids = [option.id for option in session.config_options]
    assert HARNESS_CONFIG_ID in option_ids

    response = asyncio.run(
        harness.adapter.set_config_option(
            config_id=HARNESS_CONFIG_ID,
            session_id=session.session_id,
            value=CLAUDE_AGENT_SDK_HARNESS,
        )
    )

    assert response is not None
    harness_option = next(
        option for option in response.config_options if option.id == HARNESS_CONFIG_ID
    )
    assert harness_option.current_value.replace("-", "_") == CLAUDE_AGENT_SDK_HARNESS


def test_claude_sdk_harness_uses_mocked_sdk_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = OptimizationWorkspace(tmp_path).ensure()
    settings = CompilagentSettings(model_name="anthropic:claude-opus-4-7")
    runtime = OptimizerRuntime(
        settings=settings,
        workspace=workspace,
        harness_label="Claude Agent SDK",
    )

    class FakeResult:
        result = "mock sdk result"
        subtype = "success"
        session_id = "sdk-session"
        total_cost_usd = 0.01

    class FakeClient:
        def __init__(self, *, options):
            self.options = options
            self.prompt = None

        async def connect(self):
            return None

        async def query(self, prompt):
            self.prompt = prompt

        async def receive_response(self):
            yield FakeResult()

    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeToolAnnotations:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSdk:
        ClaudeSDKClient = FakeClient
        ClaudeAgentOptions = FakeOptions
        ToolAnnotations = FakeToolAnnotations

        @staticmethod
        def tool(name, description, input_schema, annotations=None):
            def decorate(func):
                func.tool_name = name
                func.tool_description = description
                func.input_schema = input_schema
                func.annotations = annotations
                return func

            return decorate

        @staticmethod
        def create_sdk_mcp_server(name, version, tools):
            return {"name": name, "version": version, "tools": tools}

    monkeypatch.setattr("compilagent_triton.claude_agent_harness._import_sdk", lambda: FakeSdk)

    sdk_harness = ClaudeAgentSdkHarness(
        runtime=runtime,
        settings=settings,
        mode="optimize",
    )
    result = asyncio.run(sdk_harness.run("optimize this kernel"))

    assert result == "mock sdk result"
    assert sdk_harness._client.options.kwargs["model"] == "claude-opus-4-7"
    assert "mcp__triton_optimizer__run_candidate" in sdk_harness._client.options.kwargs[
        "allowed_tools"
    ]
