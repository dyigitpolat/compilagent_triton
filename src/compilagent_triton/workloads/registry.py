"""Workload registry — decorator-based, no hand-coded list anywhere.

Modules under `workloads/` register their builders by importing this module
and applying `@register_workload(spec)`. The runtime calls
`workload_registry.get(workload_id)` to obtain a fresh `WorkloadInstance`.

Discovery is import-driven: importing `workloads.triton_kernels` or
`workloads.pytorch_models` triggers their `__init__` which imports the
individual workload modules; each module registers itself at import time.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

from ..core.workload import WorkloadBuilder, WorkloadInstance, WorkloadSpec


class WorkloadRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, WorkloadSpec] = {}
        self._builders: dict[str, WorkloadBuilder] = {}

    def register(self, spec: WorkloadSpec, builder: WorkloadBuilder) -> None:
        if spec.id in self._specs:
            raise ValueError(f"Workload `{spec.id}` is already registered.")
        self._specs[spec.id] = spec
        self._builders[spec.id] = builder

    def get_spec(self, workload_id: str) -> WorkloadSpec:
        if workload_id not in self._specs:
            known = sorted(self._specs.keys())
            raise KeyError(
                f"Unknown workload `{workload_id}`. Registered: {known or '(none)'}."
            )
        return self._specs[workload_id]

    def build(self, workload_id: str) -> WorkloadInstance:
        spec = self.get_spec(workload_id)
        builder = self._builders[workload_id]
        instance = builder(spec)
        if not isinstance(instance, WorkloadInstance):
            raise TypeError(
                f"Workload builder for `{workload_id}` returned {type(instance).__name__}, "
                "expected WorkloadInstance."
            )
        return instance

    def ids(self) -> list[str]:
        return sorted(self._specs.keys())

    def specs(self) -> Iterable[WorkloadSpec]:
        return list(self._specs.values())

    def clear(self) -> None:                # for tests
        self._specs.clear()
        self._builders.clear()


workload_registry = WorkloadRegistry()


def register_workload(spec: WorkloadSpec):
    """Decorator binding a builder to a workload spec.

    The decorated function MUST accept a single `WorkloadSpec` argument and
    return a `WorkloadInstance`. Registration happens at import time, so
    populating the registry is just a matter of importing the workload
    module.
    """

    def _decorator(fn: WorkloadBuilder) -> WorkloadBuilder:
        workload_registry.register(spec, fn)
        return fn

    return _decorator


def import_workload_packages() -> None:
    """Eagerly import every bundled workload module so it self-registers.

    Walks the `compilagent_triton.workloads` package recursively with
    `pkgutil.walk_packages` and imports every non-underscore module (skipping
    `registry`, `__init__`, and `_*`). New workloads are picked up just by
    dropping a file under `workloads/<family>/<name>.py` — no edit to this
    function required.

    Per-module import failures are logged but do not abort the rest of the
    sweep, so a workload that fails (e.g. missing `torchvision`) doesn't take
    the whole catalog down with it.
    """

    import logging as _logging
    import pkgutil

    log = _logging.getLogger("compilagent.workloads")

    try:
        from .. import workloads as _pkg
    except Exception as exc:  # noqa: BLE001
        log.warning("workloads package import failed: %r", exc)
        return

    for mod_info in pkgutil.walk_packages(_pkg.__path__, prefix=f"{_pkg.__name__}."):
        if mod_info.ispkg:
            continue
        leaf = mod_info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or leaf == "registry":
            continue
        try:
            importlib.import_module(mod_info.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to import workload `%s`: %r", mod_info.name, exc)
            continue
