"""Workload registry — backend-agnostic catalog of compile targets.

Each subpackage hosts workloads compiled by a specific backend:

  - `triton_kernels/` — single Triton `@triton.jit` kernels (existing).
  - `pytorch_models/` — full nn.Module forward passes driven through
    TorchInductor (vit_b_16, vit_l_16, vit_block).

`workload_registry` keeps the dotted-entrypoint discovery decoupled from any
hard-coded list. New workloads register themselves with
`@register_workload(...)` (see `registry.py`).
"""
