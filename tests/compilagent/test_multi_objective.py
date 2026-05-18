"""Unit tests for the multi-objective leaderboard primitives + IntFreeform.

These pin the contract used by ``OptimizationSession.query_top_candidates``,
``pareto_front``, ``metric_summary``, and ``compare_candidates`` (the
tools added for multi-objective backends).
"""

from __future__ import annotations

import pytest

from compilagent import IntFreeform
from compilagent.session.multi_objective import (
    domination_count,
    metric_summary,
    pareto_front,
    rank_by_metric,
)


def _row(cid: str, **objs: tuple[float, str]) -> dict:
    """Helper that builds a candidate row in the shape multi_objective consumes.

    ``objs`` is keyword-style ``{metric_name: (value, goal)}``; each
    value is serialized to ``{name, value, goal, unit}``.
    """

    return {
        "candidate_id": cid,
        "objectives": {
            name: {"name": name, "value": float(v), "goal": g, "unit": ""}
            for name, (v, g) in objs.items()
        },
    }


# Synthetic fixture: 4 candidates, 3 objectives (accuracy max, params min,
# fragmentation min). The Pareto front for this fixture is {A, C, D}: B
# is strictly dominated by A (same accuracy, more params, more frag).
@pytest.fixture
def rows():
    return [
        _row("A", accuracy=(0.9, "max"), params=(10000, "min"), frag=(20, "min")),
        _row("B", accuracy=(0.9, "max"), params=(20000, "min"), frag=(40, "min")),
        _row("C", accuracy=(0.7, "max"), params=(5000, "min"),  frag=(30, "min")),
        _row("D", accuracy=(0.95, "max"), params=(30000, "min"), frag=(50, "min")),
    ]


# ---------------------------------------------------------------- IntFreeform


class TestIntFreeform:
    def test_serialize_round_trip(self):
        r = IntFreeform(min=8, max=2048, step=8, units="axons")
        d = r.serialize()
        assert d == {
            "kind": "int_freeform",
            "min": 8,
            "max": 2048,
            "step": 8,
            "units": "axons",
        }

    def test_default_step_is_one(self):
        r = IntFreeform(min=1, max=100)
        assert r.step == 1
        assert r.serialize()["step"] == 1

    def test_immutable(self):
        r = IntFreeform(min=8, max=2048)
        with pytest.raises(AttributeError):
            r.min = 0  # type: ignore[misc]


# ---------------------------------------------------------------- rank_by_metric


class TestRankByMetric:
    def test_max_goal_descending(self, rows):
        ranked = rank_by_metric(rows, "accuracy", top_k=10)
        ids = [r["candidate_id"] for r in ranked]
        assert ids[0] == "D"  # 0.95 is highest
        # A and B tie at 0.9 — they should share rank 2
        assert {ranked[1]["candidate_id"], ranked[2]["candidate_id"]} == {"A", "B"}
        assert ranked[1]["rank"] == ranked[2]["rank"] == 2
        # C trails at 0.7
        assert ranked[3]["candidate_id"] == "C"

    def test_min_goal_ascending(self, rows):
        ranked = rank_by_metric(rows, "params")
        assert ranked[0]["candidate_id"] == "C"  # 5000 smallest
        assert ranked[0]["rank"] == 1

    def test_top_k_truncates(self, rows):
        ranked = rank_by_metric(rows, "frag", top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["candidate_id"] == "A"  # 20% smallest frag

    def test_missing_metric_returns_no_ranks(self, rows):
        ranked = rank_by_metric(rows, "nonexistent")
        # All rows missing the metric → all rank=None at the bottom
        assert all(r["rank"] is None for r in ranked)


# ---------------------------------------------------------------- pareto_front


class TestParetoFront:
    def test_finds_non_dominated_set(self, rows):
        front = pareto_front(rows)
        ids = sorted(r["candidate_id"] for r in front)
        assert ids == ["A", "C", "D"]

    def test_single_row_is_degenerate_front(self):
        front = pareto_front([_row("X", x=(1.0, "min"))])
        assert len(front) == 1

    def test_empty_input(self):
        assert pareto_front([]) == []

    def test_skips_rows_without_objectives(self):
        rs = [
            {"candidate_id": "noobj"},
            _row("A", x=(1.0, "min")),
        ]
        front = pareto_front(rs)
        ids = [r["candidate_id"] for r in front]
        assert ids == ["A"]


# ---------------------------------------------------------------- domination_count


class TestDominationCount:
    def test_front_members_are_zero(self, rows):
        counts = domination_count(rows)
        # A, C, D on front
        assert counts["A"] == 0
        assert counts["C"] == 0
        assert counts["D"] == 0
        # B dominated only by A (same accuracy, A has lower params + lower frag)
        assert counts["B"] == 1


# ---------------------------------------------------------------- metric_summary


class TestMetricSummary:
    def test_best_worst_per_metric(self, rows):
        s = metric_summary(rows)
        assert set(s) == {"accuracy", "params", "frag"}
        assert s["accuracy"]["best"]["candidate_id"] == "D"
        assert s["accuracy"]["worst"]["candidate_id"] == "C"
        assert s["params"]["best"]["candidate_id"] == "C"
        assert s["params"]["worst"]["candidate_id"] == "D"
        assert s["frag"]["best"]["candidate_id"] == "A"

    def test_median_is_computed(self, rows):
        s = metric_summary(rows)
        # accuracy values sorted: 0.7, 0.9, 0.9, 0.95 → median = 0.9
        assert s["accuracy"]["median"] == pytest.approx(0.9)

    def test_empty_for_no_rows(self):
        assert metric_summary([]) == {}

    def test_unit_propagated_when_present(self):
        rows = [{
            "candidate_id": "A",
            "objectives": {"latency_ms": {"name": "latency_ms", "value": 1.0, "goal": "min", "unit": "ms"}},
        }]
        s = metric_summary(rows)
        assert s["latency_ms"]["unit"] == "ms"
