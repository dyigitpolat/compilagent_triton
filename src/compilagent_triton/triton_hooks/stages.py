from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class StageHookConfig:
    key_material: str
    artifact_dir: Path | None = None
    wrap_ttgir: bool = True
    label: str = "compilagent-stage-hook"

    @property
    def key(self) -> str:
        return f"{self.label}:{self.key_material}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class StageHookRecord:
    stage_name: str
    artifact_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


StageCallback = Callable[[str, Any, dict[str, Any]], None]


def make_stage_inspection_hook(
    config: StageHookConfig,
    *,
    callback: StageCallback | None = None,
    records: list[StageHookRecord] | None = None,
):
    """Build a Triton-compatible `add_stages_inspection_hook` callable."""

    records = records if records is not None else []

    def inspect_stages_hook(
        self=None,
        stages=None,
        options=None,
        language=None,
        capability=None,
    ):
        if all(arg is None for arg in (self, stages, options, language, capability)):
            return config.key, config.digest
        if stages is None or self is None:
            return config.key, config.digest
        if not config.wrap_ttgir or "ttgir" not in stages:
            return config.key, config.digest

        original_ttgir = stages["ttgir"]

        def make_ttgir_wrapper(src, metadata):
            module = original_ttgir(src, metadata)
            artifact_path = _write_stage_artifact(config, "ttgir", module)
            record = StageHookRecord(
                stage_name="ttgir",
                artifact_path=artifact_path,
                metadata={
                    "language": str(language),
                    "capability": capability,
                    "options": _safe_options(options),
                },
            )
            records.append(record)
            if callback is not None:
                callback("ttgir", module, metadata)
            return module

        stages["ttgir"] = make_ttgir_wrapper
        return config.key, config.digest

    return inspect_stages_hook


@contextmanager
def scoped_stage_hook(
    config: StageHookConfig,
    *,
    callback: StageCallback | None = None,
    records: list[StageHookRecord] | None = None,
) -> Iterator[list[StageHookRecord]]:
    """Install a Triton stage hook for one experiment and restore previous state."""

    try:
        from triton import knobs
    except ImportError as exc:
        raise RuntimeError("Triton is not importable; install it into the top-level env.") from exc

    record_list = records if records is not None else []
    previous_hook = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = make_stage_inspection_hook(
        config,
        callback=callback,
        records=record_list,
    )
    try:
        yield record_list
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


def _write_stage_artifact(config: StageHookConfig, stage_name: str, module: Any) -> Path | None:
    if config.artifact_dir is None:
        return None
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = config.artifact_dir / f"{stage_name}-{config.digest[:12]}.mlir"
    path.write_text(str(module), encoding="utf-8")
    return path


def _safe_options(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if hasattr(options, "__dict__"):
        return {
            key: _safe_value(value)
            for key, value in vars(options).items()
            if not key.lower().endswith("key")
        }
    return {"repr": repr(options)}


def _safe_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return repr(value)
