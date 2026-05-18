"""Leaderboard / best-validated-candidate helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    candidate_id: str
    median_ms: float | None
    speedup_vs_baseline: float | None
    correctness_ok: bool | None
    rationale: str = ""
    # Optional multi-objective surface populated when the backend implements
    # `Backend.objectives_for_candidate(...)`. Each entry is the serialized
    # form of an `Objective` dataclass: `{name, value, goal, unit}`. Empty
    # for single-axis backends (the default).
    objectives: dict[str, dict[str, Any]] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "median_ms": self.median_ms,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "correctness_ok": self.correctness_ok,
            "rationale": self.rationale,
            "objectives": dict(self.objectives),
        }


def build_leaderboard(
    *,
    baseline_median_ms: float | None,
    candidates: Iterable[Mapping[str, Any]],
) -> list[LeaderboardRow]:
    """Sort baseline + judged candidates by `median_ms` (lower is better).

    Candidates without a `median_ms` are placed at the bottom. Each candidate
    mapping may carry an `objectives` key (a `dict[str, dict]` of serialized
    `Objective`s); it is round-tripped onto the row unchanged.
    """

    rows: list[LeaderboardRow] = [
        LeaderboardRow(
            candidate_id="baseline",
            median_ms=baseline_median_ms,
            speedup_vs_baseline=1.0 if baseline_median_ms else None,
            correctness_ok=True,
        )
    ]
    for c in candidates:
        objectives_raw = c.get("objectives") or {}
        objectives = {
            str(k): dict(v) if isinstance(v, Mapping) else v
            for k, v in objectives_raw.items()
        }
        rows.append(
            LeaderboardRow(
                candidate_id=str(c.get("id") or ""),
                median_ms=c.get("median_ms"),
                speedup_vs_baseline=c.get("speedup_vs_baseline"),
                correctness_ok=c.get("correctness_ok"),
                rationale=str(c.get("rationale") or ""),
                objectives=objectives,
            )
        )

    def _sort_key(row: LeaderboardRow) -> float:
        return row.median_ms if row.median_ms is not None else float("inf")

    rows.sort(key=_sort_key)
    return rows


def best_validated_candidate(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the highest-speedup candidate that passed correctness.

    Returns None if no candidate cleared the bar.
    """

    qualifying = [
        c
        for c in candidates
        if c.get("compile_ok")
        and c.get("speedup_vs_baseline") is not None
        and (c.get("speedup_vs_baseline") or 0) > 1.0
        and (c.get("correctness_ok") is None or c.get("correctness_ok"))
    ]
    if not qualifying:
        return None
    qualifying.sort(key=lambda c: c.get("speedup_vs_baseline") or 0.0, reverse=True)
    return qualifying[0]
