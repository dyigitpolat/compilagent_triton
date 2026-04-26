from __future__ import annotations

import pytest

from compilagent_triton.examples import (
    RunConfig,
    RunRequest,
    get_example,
    list_examples,
    preview_kernel_source,
    preview_source,
)


def test_example_registry_lists_runnable_sources() -> None:
    examples = {example["id"]: example for example in list_examples()}

    assert examples["vector_add"]["enabled"] is True
    assert examples["vector_copy"]["enabled"] is True
    assert examples["reduction_sum"]["enabled"] is True
    assert examples["matmul_stub"]["enabled"] is False
    assert "block_sizes" in examples["vector_add"]["supported_knobs"]


def test_source_preview_returns_registered_source_only() -> None:
    preview = preview_source("vector_add")

    assert preview["language"] == "python"
    assert "run_vector_add_sweep" in preview["source"]


def test_kernel_preview_returns_jit_kernel_slice() -> None:
    preview = preview_kernel_source("vector_add")

    assert preview["source_kind"] == "kernel"
    assert preview["symbol"] == "vector_add_kernel"
    assert "@triton.jit" in preview["source"]
    assert "run_vector_add_sweep" not in preview["source"]


def test_run_config_bounds_candidate_count() -> None:
    example = get_example("vector_add")
    request = RunRequest(
        example_id="vector_add",
        config=RunConfig(
            block_sizes=list(range(1, 30)),
            num_warps=[1, 2, 4],
            load_cache_modifiers=["", ".ca", ".cg"],
        ),
    )

    with pytest.raises(ValueError, match="limit"):
        request.config.bounded_for(example)
