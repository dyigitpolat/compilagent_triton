from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import CompileArtifact, CompileRequest, CompileResult, KernelSpec
from .triton_hooks import StageHookConfig, scoped_stage_hook
from .workspace import OptimizationWorkspace


@dataclass(slots=True)
class TritonCompileHarness:
    workspace: OptimizationWorkspace

    def compile_kernel(
        self,
        spec: KernelSpec,
        request: CompileRequest,
        *,
        artifact_dir: Path | None = None,
        use_stage_hook: bool = False,
    ) -> CompileResult:
        """Compile or warm up a Triton kernel and capture available artifacts.

        The harness supports kernels that expose a module-level function named
        by `KernelSpec.entrypoint`. If the function provides a custom
        `compilagent_compile(meta: dict[str, Any])` method, that hook is used.
        Otherwise, the function is called with no positional arguments and the
        request metadata as keyword arguments. GPU-heavy benchmark definitions
        can therefore hide launch/input setup inside their own compile hook.
        """

        if spec.path is None:
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=False,
                diagnostics="KernelSpec.path is required for compile execution.",
            )
        artifact_dir = artifact_dir or self.workspace.baseline_dir(request.kernel_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "TRITON_ALWAYS_COMPILE": "1" if request.force_recompile else "0",
            "TRITON_KERNEL_DUMP": "1" if request.dump_ir else "0",
            "TRITON_DUMP_DIR": str(artifact_dir),
        }
        hook_config = (
            StageHookConfig(
                key_material=request.stage_hook_key or request.model_dump_json(exclude_none=True),
                artifact_dir=artifact_dir,
            )
            if use_stage_hook
            else None
        )
        try:
            with patched_environ(env):
                if hook_config is None:
                    handle = _execute_compile(spec, request.meta)
                else:
                    with scoped_stage_hook(hook_config):
                        handle = _execute_compile(spec, request.meta)
            artifacts = _collect_artifacts(handle, artifact_dir)
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=True,
                source_hash=_source_hash(spec.path),
                metadata=_extract_metadata(handle),
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - diagnostics path depends on Triton/GPU
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=False,
                source_hash=_source_hash(spec.path),
                diagnostics=f"{exc.__class__.__name__}: {exc}",
                artifacts=_collect_artifacts(None, artifact_dir),
            )


def _execute_compile(spec: KernelSpec, meta: dict[str, Any]) -> Any:
    module = _load_module(spec.path)
    entrypoint = getattr(module, spec.entrypoint)
    compile_hook = getattr(entrypoint, "compilagent_compile", None)
    if callable(compile_hook):
        return compile_hook(meta)
    return entrypoint(**meta)


def _load_module(path: Path) -> Any:
    module_name = f"compilagent_kernel_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _collect_artifacts(handle: Any, artifact_dir: Path) -> list[CompileArtifact]:
    artifacts: list[CompileArtifact] = []
    asm = getattr(handle, "asm", None)
    if isinstance(asm, dict):
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            if stage in asm:
                artifacts.append(CompileArtifact(stage=stage, inline_text=str(asm[stage])))
    if artifact_dir.exists():
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_file() and path.suffix in {".ttir", ".ttgir", ".llir", ".ptx", ".mlir"}:
                artifacts.append(CompileArtifact(stage=path.suffix.lstrip("."), path=path))
    return artifacts


def _extract_metadata(handle: Any) -> dict[str, Any]:
    metadata = getattr(handle, "metadata", None)
    if metadata is None:
        return {}
    if hasattr(metadata, "_asdict"):
        return {key: _safe_metadata_value(value) for key, value in metadata._asdict().items()}
    if isinstance(metadata, dict):
        return {str(key): _safe_metadata_value(value) for key, value in metadata.items()}
    return {"repr": repr(metadata)}


def _safe_metadata_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_safe_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_metadata_value(item) for key, item in value.items()}
    return repr(value)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def patched_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
