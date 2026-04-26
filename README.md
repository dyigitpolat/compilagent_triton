# Compilagent Triton

Compilagent Triton is a top-level integration package for building an ACP environment around Triton compiler optimization experiments.

The project keeps upstream submodules (`acpkit`, `triton`, and `code-mode`) as source references by default. New integration code lives under `src/compilagent_triton`; only edit submodules when a public hook, adapter, or plugin surface is insufficient.

## Environment

Use the top-level virtual environment:

```bash
source env/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -e acpkit/packages/adapters/pydantic-acp
```

For GPU benchmark experiments, install the optional GPU dependencies into the same `env`:

```bash
python -m pip install -e ".[gpu]"
```

Local runs load model configuration from `.env`. The API key is never printed or persisted by the package.

Useful settings:

- `ANTHROPIC_API_KEY`: Anthropic credential.
- `COMPILAGENT_MODEL`: defaults to `anthropic:claude-opus-4-7`.
- `COMPILAGENT_REASONING_EFFORT`: defaults to `extra_high`.
- `COMPILAGENT_MAX_TOKENS`: defaults to `8192`.
- `COMPILAGENT_TEMPERATURE`: defaults to `0.2`.
- `COMPILAGENT_MAX_CANDIDATES`: defaults to `4`.
- `COMPILAGENT_MAX_BENCHMARK_SECONDS`: defaults to `120`.
- `COMPILAGENT_HARNESS`: defaults to `pydantic_ai`. Set to `claude_agent_sdk` to start new ACP sessions on the Claude Agent SDK harness.
- `COMPILAGENT_CLAUDE_SDK_MAX_TURNS`: defaults to `24`.
- `COMPILAGENT_CLAUDE_SDK_MAX_BUDGET_USD`: optional Agent SDK cost cap.
- `COMPILAGENT_CLAUDE_SDK_PERMISSION_MODE`: defaults to `dontAsk`.

## ACP Server

After installing into `env`, launch the ACP server:

```bash
compilagent-triton
```

The server exposes optimizer agents through `pydantic-acp`. It creates per-session workspaces under `.compilagent-triton/`, records optimization episodes, captures Triton compile artifacts, and compares candidate configurations against baseline runs.

ACP clients receive a session-local `Harness` selector:

- `Current`: the existing `pydantic-ai` optimizer harness.
- `Claude Agent SDK`: routes prompts through the Claude Agent SDK while exposing the same Triton optimizer tools as an in-process MCP server.

The Claude Agent SDK harness uses the same `ANTHROPIC_API_KEY` and strips the `anthropic:` prefix from `COMPILAGENT_MODEL` before passing the model id to the SDK.

## Development Shape

Important modules:

- `settings.py`: typed environment and model settings.
- `agent.py`: `pydantic-ai` agent plus ACP adapter configuration.
- `workspace.py`: path-safe session workspace and artifact paths.
- `schemas.py`: Pydantic models for kernels, candidates, decisions, benchmarks, and episodes.
- `compiler.py`: controlled Triton compile harness.
- `triton_hooks/stages.py`: scoped `add_stages_inspection_hook` management.
- `decision_traces.py`: observe-only TTGIR decision extraction.
- `benchmarking.py`: correctness-checked benchmark helpers.
- `episodes.py`: file-backed episode state.

## Testing

Run top-level tests from the activated `env`:

```bash
python -m pytest
```

GPU-dependent Triton integration tests should be added separately and guarded so the core package remains testable on machines without CUDA.

## Benchmark Loop

Run the current vector-add candidate sweep on a selected GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python -m compilagent_triton.gpu_benchmarks \
  --n-elements 8388608 \
  --block-sizes 256,512,1024,2048 \
  --num-warps 4,8,16 \
  --load-cache-modifiers none,.ca,.cg
```

Reports are written under `.compilagent-triton/reports/`, which is ignored by git.
