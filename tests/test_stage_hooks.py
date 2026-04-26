from __future__ import annotations

from pathlib import Path

from compilagent_triton.triton_hooks.stages import StageHookConfig, make_stage_inspection_hook


def test_stage_hook_no_arg_returns_key_and_hash(tmp_path: Path) -> None:
    config = StageHookConfig(key_material="abc", artifact_dir=tmp_path)
    hook = make_stage_inspection_hook(config)

    key, digest = hook()

    assert key == "compilagent-stage-hook:abc"
    assert len(digest) == 64


def test_stage_hook_wraps_ttgir_and_writes_artifact(tmp_path: Path) -> None:
    config = StageHookConfig(key_material="abc", artifact_dir=tmp_path)
    records = []
    hook = make_stage_inspection_hook(config, records=records)
    stages = {"ttgir": lambda src, metadata: "module { tt.func @kernel }"}

    hook(object(), stages, object(), "triton", 90)
    result = stages["ttgir"]("src", {})

    assert result == "module { tt.func @kernel }"
    assert len(records) == 1
    assert records[0].artifact_path is not None
    assert records[0].artifact_path.exists()
