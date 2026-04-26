from __future__ import annotations

from compilagent_triton.decision_traces import extract_decision_traces, summarize_decision_traces
from compilagent_triton.schemas import DecisionKind


def test_extracts_memory_and_dot_traces() -> None:
    ir = """
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  %0 = tt.load %ptr : tensor<128xf32>
  %1 = tt.dot %a, %b : tensor<16x16xf32>
}
"""

    traces = extract_decision_traces(ir, run_id="run-1")

    assert [trace.kind for trace in traces] == [DecisionKind.COALESCING, DecisionKind.MATMUL]
    assert traces[0].num_warps == 4
    assert traces[0].threads_per_warp == 32
    assert traces[0].tensor_shape == (128,)
    assert traces[1].tensor_shape == (16, 16)


def test_summary_handles_empty_traces() -> None:
    assert "No coalescing" in summarize_decision_traces([])
