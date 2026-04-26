"""Triton compiler backend.

Wraps Triton's stage-replacement pipeline (`pipeline.py`), MLIR pass catalog
(`passes.py`), and `add_stages_inspection_hook` integration (`stages.py`) into
a `Backend` that the runtime can drive without importing Triton directly.

The concrete `TritonBackend(Backend)` adapter will land in `backend.py` next
to the existing helpers.
"""

from __future__ import annotations

from .backend import TritonBackend
from .stages import StageHookConfig, StageHookRecord, scoped_stage_hook

__all__ = [
    "StageHookConfig",
    "StageHookRecord",
    "TritonBackend",
    "scoped_stage_hook",
]
