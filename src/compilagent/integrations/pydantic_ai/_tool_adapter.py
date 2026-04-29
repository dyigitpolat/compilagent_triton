"""Bind `ToolDecl`s into pydantic-ai tools.

The core declares each agent tool exactly once as a `ToolDecl` whose
`handler` is a typed Python callable (a bound `OptimizationSession`
method). pydantic-ai introspects the callable's signature with `inspect`
+ Pydantic's `TypeAdapter` and produces a JSON Schema that mirrors the
typed parameters — including nested Pydantic models, lists of models,
and so on. The agent therefore emits structured arrays and objects
directly; no string-encoded JSON, no escape pyramids.

This module is the thin glue: it takes a `ToolDecl`, sets the function's
`__name__` and `__doc__` to the decl's name and description (so the model
sees both in its tool list), and returns the typed callable for
`agent.tool_plain(...)` to introspect.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from compilagent.core.tool_decl import ToolDecl


def make_pydantic_ai_tool_fn(decl: ToolDecl) -> Callable[..., str]:
    """Return a typed callable pydantic-ai can introspect for `decl`.

    The returned function is the bound method itself, lightly re-tagged
    with the decl's name + description so pydantic-ai's tool registration
    picks them up. The signature (and therefore the model-facing JSON
    Schema) is whatever the bound method declares — pydantic-ai handles
    nested Pydantic models natively.
    """

    handler = decl.handler

    @functools.wraps(handler)
    def _tool(*args, **kwargs):
        return handler(*args, **kwargs)

    _tool.__name__ = decl.name
    _tool.__doc__ = decl.description
    # `wraps` copies `__wrapped__` so pydantic-ai still resolves the
    # original signature via `inspect.signature(_tool)` (which follows
    # `__wrapped__`). The cosmetic name/doc rebinding overrides
    # `wraps`'s own `__name__`/`__doc__` copy so the agent sees decl
    # values instead of the underlying method's defaults.
    return _tool
