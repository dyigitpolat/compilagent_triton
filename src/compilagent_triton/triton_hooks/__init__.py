"""Compatibility shim — the real implementation lives in `backends/triton/`.

External callers should migrate to `compilagent_triton.backends.triton`.
This shim re-exports the public surface so existing imports continue to work
during the re-engineering. It is an explicit deprecation surface.
"""

from __future__ import annotations

import sys as _sys

from ..backends.triton import passes as _passes_mod
from ..backends.triton import pipeline as _pipeline_mod
from ..backends.triton import stages as _stages_mod
from ..backends.triton.stages import (
    StageHookConfig,
    StageHookRecord,
    scoped_stage_hook,
)

# Make `compilagent_triton.triton_hooks.passes` resolve to the relocated module
# so dotted imports keep working during the migration window.
_sys.modules[__name__ + ".passes"] = _passes_mod
_sys.modules[__name__ + ".pipeline"] = _pipeline_mod
_sys.modules[__name__ + ".stages"] = _stages_mod

__all__ = ["StageHookConfig", "StageHookRecord", "scoped_stage_hook"]
