"""Compatibility shim — moved to `backends/triton/analysis.py`."""

from __future__ import annotations

from .backends.triton.analysis import *  # noqa: F401,F403
from .backends.triton.analysis import (  # explicit re-exports for grep-discoverability
    extract_decision_traces,
    summarize_decision_traces,
)

__all__ = ["extract_decision_traces", "summarize_decision_traces"]
