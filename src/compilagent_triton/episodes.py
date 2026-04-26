from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import (
    BenchmarkResult,
    CandidateConfig,
    CandidateStatus,
    Hypothesis,
    OptimizationEpisode,
    ReasoningSummary,
)
from .workspace import OptimizationWorkspace


@dataclass(slots=True)
class EpisodeStore:
    workspace: OptimizationWorkspace

    def create(
        self,
        *,
        kernel_id: str,
        objective: str,
        budget: dict[str, Any],
        model_metadata: dict[str, Any],
    ) -> OptimizationEpisode:
        self.workspace.ensure()
        episode = OptimizationEpisode(
            kernel_id=kernel_id,
            objective=objective,
            budget=budget,
            model_metadata=model_metadata,
        )
        self.save(episode)
        return episode

    def load(self, episode_id: str) -> OptimizationEpisode:
        path = self.workspace.episode_path(episode_id)
        if not path.exists():
            raise ValueError(f"Episode not found: {episode_id}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            # Self-heal: a previous crashed save truncated the file. Remove the
            # empty stub so subsequent save()s start clean, then surface this
            # as a normal "not found" so callers can recreate or skip.
            try:
                path.unlink()
            except OSError:
                pass
            raise ValueError(
                f"Episode `{episode_id}` was empty (likely a crashed prior save) "
                "and has been cleared; recreate the episode with start_episode."
            )
        try:
            return OptimizationEpisode.model_validate_json(text)
        except Exception as exc:
            raise ValueError(
                f"Episode `{episode_id}` is unreadable: {exc}"
            ) from exc

    def list_recent(self, *, limit: int = 16) -> list[OptimizationEpisode]:
        """Return the most recently modified episodes (best-effort, error-tolerant)."""

        runs_dir = self.workspace.runs_dir
        if not runs_dir.exists():
            return []
        candidates: list[tuple[float, Path]] = []
        for episode_dir in runs_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            episode_path = episode_dir / "episode.json"
            if not episode_path.exists():
                continue
            try:
                candidates.append((episode_path.stat().st_mtime, episode_path))
            except OSError:
                continue
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        episodes: list[OptimizationEpisode] = []
        for _, path in candidates[:limit]:
            try:
                episodes.append(
                    OptimizationEpisode.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return episodes

    def save(self, episode: OptimizationEpisode) -> Path:
        """Atomically persist an episode.

        write_text truncates the destination on open(), so a crash between
        truncate and write leaves a 0-byte file. Use a tmp-file + os.replace
        instead — the destination only exists when fully written.
        """

        import os
        import tempfile

        path = self.workspace.episode_path(episode.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = episode.touch().model_dump_json(indent=2, exclude_none=True)
        # NamedTemporaryFile in the same dir guarantees os.replace is atomic.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return path

    def record_hypothesis(
        self,
        episode_id: str,
        *,
        statement: str,
        expected_effect: str,
        evidence_refs: list[str] | None = None,
    ) -> Hypothesis:
        episode = self.load(episode_id)
        hypothesis = Hypothesis(
            statement=statement,
            expected_effect=expected_effect,
            evidence_refs=evidence_refs or [],
        )
        self.save(episode.model_copy(update={"hypotheses": [*episode.hypotheses, hypothesis]}))
        return hypothesis

    def add_candidate(self, episode_id: str, candidate: CandidateConfig) -> CandidateConfig:
        episode = self.load(episode_id)
        self.save(episode.model_copy(update={"candidates": [*episode.candidates, candidate]}))
        return candidate

    def add_benchmark_result(
        self,
        episode_id: str,
        result: BenchmarkResult,
    ) -> BenchmarkResult:
        episode = self.load(episode_id)
        self.save(
            episode.model_copy(
                update={"benchmark_results": [*episode.benchmark_results, result]}
            )
        )
        return result

    def add_reasoning_summary(
        self,
        episode_id: str,
        summary: ReasoningSummary,
    ) -> ReasoningSummary:
        episode = self.load(episode_id)
        self.save(
            episode.model_copy(
                update={"reasoning_summaries": [*episode.reasoning_summaries, summary]}
            )
        )
        return summary

    def set_candidate_status(
        self,
        episode_id: str,
        candidate_id: str,
        status: CandidateStatus,
        rationale: str,
    ) -> OptimizationEpisode:
        episode = self.load(episode_id)
        candidates = [
            candidate.model_copy(update={"status": status})
            if candidate.id == candidate_id
            else candidate
            for candidate in episode.candidates
        ]
        if all(candidate.id != candidate_id for candidate in episode.candidates):
            raise ValueError(f"Candidate not found: {candidate_id}")
        conclusions = [*episode.conclusions, f"{candidate_id}: {status.value} - {rationale}"]
        updated = episode.model_copy(update={"candidates": candidates, "conclusions": conclusions})
        self.save(updated)
        return updated

    def write_report(self, episode_id: str) -> Path:
        episode = self.load(episode_id)
        path = self.workspace.report_path(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_episode_report(episode), encoding="utf-8")
        self.save(episode.model_copy(update={"report_path": path}))
        return path


def render_episode_report(episode: OptimizationEpisode) -> str:
    lines = [
        f"# Optimization Episode `{episode.id}`",
        "",
        f"- kernel: `{episode.kernel_id}`",
        f"- objective: {episode.objective}",
        f"- model: `{episode.model_metadata.get('model_name', 'unknown')}`",
        f"- reasoning effort: `{episode.model_metadata.get('reasoning_effort', 'unknown')}`",
        "",
        "## Hypotheses",
    ]
    if episode.hypotheses:
        for hypothesis in episode.hypotheses:
            lines.append(f"- `{hypothesis.id}`: {hypothesis.statement}")
            lines.append(f"  Expected effect: {hypothesis.expected_effect}")
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Candidates"])
    if episode.candidates:
        for candidate in episode.candidates:
            lines.append(f"- `{candidate.id}` [{candidate.status.value}]: {candidate.description}")
            lines.append(f"  Changes: `{json.dumps(candidate.changes, sort_keys=True)}`")
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Reasoning Summaries"])
    if episode.reasoning_summaries:
        for summary in episode.reasoning_summaries:
            linked = ", ".join(
                item
                for item in (summary.linked_hypothesis_id, summary.linked_candidate_id)
                if item
            )
            suffix = f" ({linked})" if linked else ""
            lines.append(f"- `{summary.id}`{suffix}: {summary.summary}")
            if summary.next_step:
                lines.append(f"  Next: {summary.next_step}")
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Benchmarks"])
    if episode.benchmark_results:
        for result in episode.benchmark_results:
            speedup = (
                f"{result.speedup_vs_baseline:.4f}x"
                if result.speedup_vs_baseline is not None
                else "n/a"
            )
            lines.append(
                f"- `{result.id}` candidate `{result.candidate_id or 'baseline'}`: "
                f"correctness={result.correctness}, median={result.median_ms}, speedup={speedup}"
            )
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Conclusions"])
    if episode.conclusions:
        lines.extend(f"- {conclusion}" for conclusion in episode.conclusions)
    else:
        lines.append("- No conclusion yet.")

    return "\n".join(lines) + "\n"
