from __future__ import annotations

from pathlib import Path

from compilagent_triton.settings import CompilagentSettings


def test_settings_loads_dotenv_without_serializing_secret(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            (
                "ANTHROPIC_API_KEY=secret-value",
                "COMPILAGENT_MODEL=anthropic:claude-opus-4-7",
                "COMPILAGENT_REASONING_EFFORT=extra_high",
                "COMPILAGENT_MAX_CANDIDATES=3",
            )
        ),
        encoding="utf-8",
    )

    settings = CompilagentSettings.from_env(project_root=tmp_path)

    assert settings.anthropic_api_key is not None
    assert settings.model_name == "anthropic:claude-opus-4-7"
    assert settings.reasoning_effort == "extra_high"
    assert settings.max_candidates == 3
    assert "secret-value" not in settings.model_dump_json()
    assert "secret-value" not in str(settings.public_metadata())
