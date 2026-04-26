from __future__ import annotations

from pathlib import Path

import pytest

from compilagent_triton.workspace import OptimizationWorkspace


def test_workspace_creates_expected_directories(tmp_path: Path) -> None:
    workspace = OptimizationWorkspace(tmp_path).ensure()

    assert workspace.root.exists()
    assert workspace.kernels_dir.exists()
    assert workspace.runs_dir.exists()
    assert workspace.reports_dir.exists()
    assert workspace.benchmarks_dir.exists()


def test_workspace_rejects_escape_paths(tmp_path: Path) -> None:
    workspace = OptimizationWorkspace(tmp_path).ensure()

    with pytest.raises(ValueError, match="inside the optimization workspace"):
        workspace.resolve("../outside.txt")


def test_workspace_candidate_path_is_safe(tmp_path: Path) -> None:
    workspace = OptimizationWorkspace(tmp_path).ensure()

    path = workspace.candidate_dir("episode/one", "candidate:two")

    assert path.relative_to(workspace.root)
    assert "episode-one" in str(path)
    assert "candidate-two" in str(path)
