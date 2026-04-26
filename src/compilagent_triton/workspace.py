from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OptimizationWorkspace:
    """Path-safe workspace rooted inside an ACP session cwd."""

    session_cwd: Path
    root_name: str = ".compilagent-triton"

    @property
    def root(self) -> Path:
        return (self.session_cwd / self.root_name).resolve()

    @property
    def kernels_dir(self) -> Path:
        return self.root / "kernels"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def benchmarks_dir(self) -> Path:
        return self.root / "benchmarks"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    def ensure(self) -> OptimizationWorkspace:
        for path in (
            self.root,
            self.kernels_dir,
            self.runs_dir,
            self.reports_dir,
            self.benchmarks_dir,
            self.traces_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path must stay inside the optimization workspace.") from exc
        return candidate

    def run_dir(self, episode_id: str, *parts: str) -> Path:
        safe_episode = _safe_name(episode_id)
        return self.resolve(Path("runs") / safe_episode / Path(*parts))

    def baseline_dir(self, episode_id: str) -> Path:
        return self.run_dir(episode_id, "baseline")

    def candidate_dir(self, episode_id: str, candidate_id: str) -> Path:
        return self.run_dir(episode_id, "candidates", _safe_name(candidate_id))

    def report_path(self, episode_id: str) -> Path:
        return self.resolve(Path("reports") / f"{_safe_name(episode_id)}.md")

    def episode_path(self, episode_id: str) -> Path:
        return self.run_dir(episode_id, "episode.json")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    safe = safe.strip(".-")
    if not safe:
        raise ValueError("identifier must contain at least one safe character")
    return safe
