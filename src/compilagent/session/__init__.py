"""Optimization session — the canonical loop + canonical toolset."""

from .completion import (
    CompletionDecision,
    DefaultCompletionPolicy,
    RunCompletionPolicy,
    RunSnapshot,
)
from .leaderboard import LeaderboardRow, best_validated_candidate, build_leaderboard
from .session import OptimizationSession, run_session
from .tools import build_session_toolset

__all__ = [
    "CompletionDecision",
    "DefaultCompletionPolicy",
    "LeaderboardRow",
    "OptimizationSession",
    "RunCompletionPolicy",
    "RunSnapshot",
    "best_validated_candidate",
    "build_leaderboard",
    "build_session_toolset",
    "run_session",
]
