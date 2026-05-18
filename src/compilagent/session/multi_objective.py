"""Multi-objective leaderboard primitives.

Pure functions over the ``LeaderboardRow.objectives`` shape (
``dict[str, dict[str, Any]]`` mapping objective name to its serialized
``Objective`` payload — ``{name, value, goal, unit}``). They are used by
``OptimizationSession`` to expose tools the agent can call when a backend
opts into the multi-objective surface via
``Backend.objectives_for_candidate(...)``.

Single-axis backends never populate the ``objectives`` field, so every
helper here returns an empty / no-op result for those sessions — the
existing ``median_ms`` / ``speedup_vs_baseline`` flow stays untouched.

Each function takes a ``rows`` iterable of mappings shaped like::

    {
        "candidate_id": "cand-...",
        "objectives": {
            "accuracy": {"name": "accuracy", "value": 0.91, "goal": "max", "unit": ""},
            "latency_ms": {"name": "latency_ms", "value": 12.3, "goal": "min", "unit": "ms"},
            ...
        },
        # other LeaderboardRow fields (median_ms, speedup, ...) are
        # ignored by the helpers below; they only look at `objectives`.
    }

Goal direction is read from each axis' ``goal`` (``"min"`` or ``"max"``);
when goals disagree across rows on the same metric we trust the first
one we see (the backend should be consistent).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #


def _objective_value(row: Mapping[str, Any], metric: str) -> float | None:
    objs = row.get("objectives") or {}
    entry = objs.get(metric)
    if isinstance(entry, Mapping) and "value" in entry:
        try:
            return float(entry["value"])
        except (TypeError, ValueError):
            return None
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


def _goal_for(rows: Iterable[Mapping[str, Any]], metric: str) -> str:
    for r in rows:
        objs = r.get("objectives") or {}
        entry = objs.get(metric)
        if isinstance(entry, Mapping) and entry.get("goal") in ("min", "max"):
            return str(entry["goal"])
    return "min"


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("id") or "")


def _common_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for r in rows:
        for k in (r.get("objectives") or {}):
            seen.setdefault(str(k), None)
    return tuple(seen.keys())


def _has_objectives(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(bool(r.get("objectives")) for r in rows)


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


def rank_by_metric(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Sort rows by one metric, honouring its goal direction.

    Rows missing the metric (or with a non-numeric value) are placed at
    the bottom in their existing order. Returns a list of dicts shaped
    like ``{candidate_id, value, rank, goal}`` with ``rank`` starting at
    1 (best). Ties get the same rank (dense ranking).
    """

    rows_list = list(rows)
    if not rows_list:
        return []
    goal = _goal_for(rows_list, metric)
    indexed: list[tuple[int, Mapping[str, Any], float | None]] = [
        (i, r, _objective_value(r, metric)) for i, r in enumerate(rows_list)
    ]
    have_value = [t for t in indexed if t[2] is not None]
    missing = [t for t in indexed if t[2] is None]
    reverse = goal == "max"
    have_value.sort(key=lambda t: (t[2], t[0]), reverse=reverse)

    out: list[dict[str, Any]] = []
    current_rank = 0
    last_val: float | None = None
    for pos, (_, row, val) in enumerate(have_value):
        if pos == 0 or val != last_val:
            current_rank = pos + 1
            last_val = val
        out.append({
            "candidate_id": _candidate_id(row),
            "value": val,
            "rank": current_rank,
            "goal": goal,
        })
    for _, row, _val in missing:
        out.append({
            "candidate_id": _candidate_id(row),
            "value": None,
            "rank": None,
            "goal": goal,
        })
    if top_k is not None:
        return out[: int(top_k)]
    return out


def _dominates(
    a_objs: Mapping[str, Any],
    b_objs: Mapping[str, Any],
    metrics: Sequence[str],
    goal_by_metric: Mapping[str, str],
) -> bool:
    """Return True iff `a` Pareto-dominates `b` over `metrics`.

    `a` dominates `b` iff `a` is no worse than `b` on every metric and
    strictly better on at least one.
    """

    better_in_one = False
    for m in metrics:
        a_entry = a_objs.get(m)
        b_entry = b_objs.get(m)
        if not isinstance(a_entry, Mapping) or "value" not in a_entry:
            return False
        if not isinstance(b_entry, Mapping) or "value" not in b_entry:
            return False
        try:
            av = float(a_entry["value"])
            bv = float(b_entry["value"])
        except (TypeError, ValueError):
            return False
        if goal_by_metric.get(m, "min") == "max":
            if av < bv:
                return False
            if av > bv:
                better_in_one = True
        else:  # min
            if av > bv:
                return False
            if av < bv:
                better_in_one = True
    return better_in_one


def pareto_front(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the non-dominated subset of ``rows`` across all common metrics.

    Rows missing any of the common metrics are skipped (a row that does
    not score on every active axis cannot be ranked). For an empty or
    single-row input the input itself is returned (degenerate Pareto
    front).
    """

    rows_list = [r for r in rows if r.get("objectives")]
    if len(rows_list) <= 1:
        return list(rows_list)
    metrics = _common_metrics(rows_list)
    if not metrics:
        return list(rows_list)
    goal_by_metric = {m: _goal_for(rows_list, m) for m in metrics}
    objs_list = [r.get("objectives") or {} for r in rows_list]

    front: list[Mapping[str, Any]] = []
    for i, row in enumerate(rows_list):
        dominated = False
        for j, other in enumerate(rows_list):
            if i == j:
                continue
            if _dominates(objs_list[j], objs_list[i], metrics, goal_by_metric):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def domination_count(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """For each candidate, count how many other rows Pareto-dominate it.

    0 means the candidate is on the Pareto front. Rows without objectives
    are skipped. Returned keys are candidate ids.
    """

    rows_list = [r for r in rows if r.get("objectives")]
    if not rows_list:
        return {}
    metrics = _common_metrics(rows_list)
    if not metrics:
        return {_candidate_id(r): 0 for r in rows_list}
    goal_by_metric = {m: _goal_for(rows_list, m) for m in metrics}
    objs_list = [r.get("objectives") or {} for r in rows_list]
    out: dict[str, int] = {}
    for i, row in enumerate(rows_list):
        count = 0
        for j in range(len(rows_list)):
            if i == j:
                continue
            if _dominates(objs_list[j], objs_list[i], metrics, goal_by_metric):
                count += 1
        out[_candidate_id(row)] = count
    return out


def metric_summary(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-metric best/worst/median table.

    Returns ``{metric_name: {goal, unit, best: {candidate_id, value},
    worst: {candidate_id, value}, median: float, count: int}}``. Metrics
    that no row scores on are omitted.
    """

    rows_list = list(rows)
    metrics = _common_metrics(rows_list)
    out: dict[str, dict[str, Any]] = {}
    for m in metrics:
        scored = [
            (r, _objective_value(r, m)) for r in rows_list if _objective_value(r, m) is not None
        ]
        if not scored:
            continue
        goal = _goal_for(rows_list, m)
        # Pick unit from any row that has it
        unit = ""
        for r in rows_list:
            entry = (r.get("objectives") or {}).get(m)
            if isinstance(entry, Mapping) and entry.get("unit"):
                unit = str(entry["unit"])
                break
        values = [v for _, v in scored]
        if goal == "max":
            best = max(scored, key=lambda t: t[1])
            worst = min(scored, key=lambda t: t[1])
        else:
            best = min(scored, key=lambda t: t[1])
            worst = max(scored, key=lambda t: t[1])
        sorted_values = sorted(values)
        n = len(sorted_values)
        median = (
            sorted_values[n // 2]
            if n % 2 == 1
            else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
        )
        out[m] = {
            "goal": goal,
            "unit": unit,
            "best": {"candidate_id": _candidate_id(best[0]), "value": float(best[1])},
            "worst": {"candidate_id": _candidate_id(worst[0]), "value": float(worst[1])},
            "median": float(median),
            "count": int(n),
        }
    return out


__all__ = [
    "domination_count",
    "metric_summary",
    "pareto_front",
    "rank_by_metric",
]
