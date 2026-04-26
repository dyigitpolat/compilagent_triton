from __future__ import annotations

from pathlib import Path

from compilagent_triton.compiler import TritonCompileHarness
from compilagent_triton.schemas import CompileRequest, KernelSpec
from compilagent_triton.workspace import OptimizationWorkspace


def test_compile_harness_uses_custom_compile_hook(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        """
class FakeHandle:
    asm = {"ttgir": "module { %0 = tt.load %ptr : tensor<128xf32> }"}
    metadata = {"target": "fake"}

def kernel(**meta):
    raise AssertionError("custom hook should be used")

def _compile(meta):
    return FakeHandle()

kernel.compilagent_compile = _compile
""",
        encoding="utf-8",
    )
    workspace = OptimizationWorkspace(tmp_path).ensure()
    harness = TritonCompileHarness(workspace)
    spec = KernelSpec(id="k", name="kernel", path=kernel_path, entrypoint="kernel")

    result = harness.compile_kernel(spec, CompileRequest(kernel_id="k"))

    assert result.ok
    assert result.source_hash is not None
    assert result.metadata == {"target": "fake"}
    assert result.artifacts[0].stage == "ttgir"
