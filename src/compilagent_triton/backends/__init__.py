"""Pluggable compiler-backend implementations.

Each backend provides a concrete `Backend` (see `.base`) that the agent's
runtime drives without ever importing the backend's compiler library directly.
The active backend is chosen at episode-start time from `settings.backend`.
"""

from .base import (
    Analysis,
    Backend,
    BackendRegistry,
    DeviceCapability,
    IntrospectionTool,
    Plan,
    Intervention,
    PassCallback,
    Target,
    TimingResult,
    ToleranceConfig,
    CorrectnessResult,
    backend_registry,
)


def import_backend_packages() -> list[str]:
    """Discover and import every concrete backend subpackage.

    Walks `compilagent_triton.backends/*` and imports each subpackage so its
    `Backend` adapter self-registers. New backends are picked up by dropping a
    `backends/<id>/__init__.py` (or `backend.py`) — no edit required here.

    Returns the names of backends successfully imported (for diagnostics).
    Per-backend failures are logged and skipped; the rest of the sweep
    continues so a missing optional dependency doesn't take the whole runtime
    down.
    """

    import importlib
    import logging as _logging
    import pkgutil

    log = _logging.getLogger("compilagent.backends")
    imported: list[str] = []

    for mod_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if not mod_info.ispkg:
            continue
        leaf = mod_info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or leaf == "base":
            continue
        try:
            importlib.import_module(mod_info.name)
            imported.append(leaf)
        except Exception as exc:  # noqa: BLE001
            log.warning("backend `%s` failed to register: %r", leaf, exc)
            continue
    return imported


__all__ = [
    "Analysis",
    "Backend",
    "BackendRegistry",
    "DeviceCapability",
    "IntrospectionTool",
    "Plan",
    "Intervention",
    "PassCallback",
    "Target",
    "TimingResult",
    "ToleranceConfig",
    "CorrectnessResult",
    "backend_registry",
    "import_backend_packages",
]
