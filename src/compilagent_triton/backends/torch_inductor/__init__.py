"""TorchInductor compiler backend.

Drives `torch.compile(...)` via a custom Dynamo backend (interception.py),
captures Inductor artifacts (output code, schedule logs, autotune logs) and
exposes a typed `KnobCatalog` (knobs.py) that the agent can target with knob
overrides. The concrete `TorchInductorBackend(Backend)` adapter lives in
`backend.py` and is registered with the global `backend_registry` at import
time.
"""

from __future__ import annotations

# Import the backend module to trigger registration when this package is
# imported by the runtime / observation server.
from . import backend as _backend_module  # noqa: F401

from .knobs import KnobCatalog, KnobDescriptor, build_knob_catalog

__all__ = ["KnobCatalog", "KnobDescriptor", "build_knob_catalog"]
