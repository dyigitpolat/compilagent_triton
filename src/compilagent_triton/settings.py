from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_MODEL = "mistral:mistral-large-latest"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_HARNESS = "pydantic_ai"
HarnessName = Literal["pydantic_ai", "claude_agent_sdk"]


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _env_value(name: str, dotenv: dict[str, str], default: str | None = None) -> str | None:
    return os.environ.get(name) or dotenv.get(name) or default


def _env_int(name: str, dotenv: dict[str, str], default: int) -> int:
    value = _env_value(name, dotenv)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, dotenv: dict[str, str], default: float) -> float:
    value = _env_value(name, dotenv)
    if value is None or value == "":
        return default
    return float(value)


def _env_float_optional(name: str, dotenv: dict[str, str]) -> float | None:
    value = _env_value(name, dotenv)
    if value is None or value == "":
        return None
    return float(value)


class CompilagentSettings(BaseModel):
    """Runtime settings for the optimizer agent.

    Secret values are intentionally excluded from repr and serialized artifacts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anthropic_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    mistral_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    model_name: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_tokens: int = 8192
    temperature: float = 0.2
    max_candidates: int = 4
    max_benchmark_seconds: int = 120
    noise_threshold_pct: float = 2.0
    workspace_dir_name: str = ".compilagent-triton"
    triton_path: Path = Path("triton")
    acpkit_path: Path = Path("acpkit")
    harness: HarnessName = DEFAULT_HARNESS
    claude_sdk_max_turns: int = 24
    claude_sdk_max_budget_usd: float | None = None
    claude_sdk_permission_mode: str = "dontAsk"

    @field_validator("max_tokens", "max_candidates", "max_benchmark_seconds")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("noise_threshold_pct")
    @classmethod
    def _valid_noise_threshold(cls, value: float) -> float:
        if value < 0:
            raise ValueError("noise threshold must be non-negative")
        return value

    @field_validator("harness")
    @classmethod
    def _valid_harness(cls, value: str) -> str:
        value = value.strip().replace("-", "_")
        if value not in {"pydantic_ai", "claude_agent_sdk"}:
            raise ValueError("harness must be `pydantic_ai` or `claude_agent_sdk`")
        return value

    @classmethod
    def from_env(
        cls,
        *,
        dotenv_path: Path | str | None = None,
        project_root: Path | str | None = None,
    ) -> CompilagentSettings:
        root = Path.cwd() if project_root is None else Path(project_root)
        env_path = root / ".env" if dotenv_path is None else Path(dotenv_path)
        dotenv = _parse_env_file(env_path)

        api_key = _env_value("ANTHROPIC_API_KEY", dotenv)
        mistral_key = _env_value("MISTRAL_API_KEY", dotenv)
        openai_key = _env_value("OPENAI_API_KEY", dotenv)
        data: dict[str, Any] = {
            "anthropic_api_key": SecretStr(api_key) if api_key else None,
            "mistral_api_key": SecretStr(mistral_key) if mistral_key else None,
            "openai_api_key": SecretStr(openai_key) if openai_key else None,
            "model_name": _env_value("COMPILAGENT_MODEL", dotenv, DEFAULT_MODEL),
            "reasoning_effort": _env_value(
                "COMPILAGENT_REASONING_EFFORT", dotenv, DEFAULT_REASONING_EFFORT
            ),
            "max_tokens": _env_int("COMPILAGENT_MAX_TOKENS", dotenv, 8192),
            "temperature": _env_float("COMPILAGENT_TEMPERATURE", dotenv, 0.2),
            "max_candidates": _env_int("COMPILAGENT_MAX_CANDIDATES", dotenv, 4),
            "max_benchmark_seconds": _env_int(
                "COMPILAGENT_MAX_BENCHMARK_SECONDS", dotenv, 120
            ),
            "noise_threshold_pct": _env_float("COMPILAGENT_NOISE_THRESHOLD_PCT", dotenv, 2.0),
            "workspace_dir_name": _env_value(
                "COMPILAGENT_WORKSPACE_DIR", dotenv, ".compilagent-triton"
            ),
            "triton_path": Path(_env_value("COMPILAGENT_TRITON_PATH", dotenv, str(root / "triton"))),
            "acpkit_path": Path(_env_value("COMPILAGENT_ACPKIT_PATH", dotenv, str(root / "acpkit"))),
            "harness": _env_value("COMPILAGENT_HARNESS", dotenv, DEFAULT_HARNESS),
            "claude_sdk_max_turns": _env_int("COMPILAGENT_CLAUDE_SDK_MAX_TURNS", dotenv, 24),
            "claude_sdk_max_budget_usd": _env_float_optional(
                "COMPILAGENT_CLAUDE_SDK_MAX_BUDGET_USD", dotenv
            ),
            "claude_sdk_permission_mode": _env_value(
                "COMPILAGENT_CLAUDE_SDK_PERMISSION_MODE", dotenv, "dontAsk"
            ),
        }
        return cls(**data)

    def public_metadata(self) -> dict[str, Any]:
        """Return non-secret settings suitable for episode artifacts."""

        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_candidates": self.max_candidates,
            "max_benchmark_seconds": self.max_benchmark_seconds,
            "noise_threshold_pct": self.noise_threshold_pct,
            "workspace_dir_name": self.workspace_dir_name,
            "triton_path": str(self.triton_path),
            "harness": self.harness,
            "claude_sdk_max_turns": self.claude_sdk_max_turns,
            "claude_sdk_max_budget_usd": self.claude_sdk_max_budget_usd,
            "claude_sdk_permission_mode": self.claude_sdk_permission_mode,
        }

    def claude_sdk_model_name(self) -> str:
        """Return the model id expected by the Claude Agent SDK."""

        return self.model_name.removeprefix("anthropic:")

    def claude_sdk_effort_value(self) -> Literal["low", "medium", "high", "max"] | None:
        effort = self.reasoning_effort.strip().lower().replace("-", "_")
        if effort in {"low", "medium", "high"}:
            return effort  # type: ignore[return-value]
        if effort in {"extra_high", "xhigh", "max"}:
            return "max"
        return None
