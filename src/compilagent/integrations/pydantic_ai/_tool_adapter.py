"""Translate `ToolDecl` records into pydantic-ai tool functions.

The core declares each agent tool exactly once as a `ToolDecl` carrying a
JSON schema for its arguments. pydantic-ai derives its own tool schema from
a Python function's signature, so we synthesise a function whose parameters
match the JSON schema's `properties` and forward the call to
`decl.handler(args_dict)`.

Only `string`-typed properties are supported here — the canonical session
toolset uses string args (workloads/JSON payloads passed as strings) and
backend introspection tools that need richer types should provide their own
bound function in `Backend.list_introspection_tools()` rather than relying
on this synthesiser. (We can extend later when we have a concrete need.)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compilagent.core.tool_decl import ToolDecl


def make_pydantic_ai_tool_fn(decl: ToolDecl) -> Callable[..., str]:
    """Build a function whose signature matches `decl.args_schema.properties`.

    Calling the returned function dispatches into `decl.handler(args_dict)`
    where `args_dict` is built from the keyword arguments. Raises
    `ValueError` if the schema contains a non-`string` property type — the
    decl should bring its own handler in that case.
    """

    schema = decl.args_schema or {}
    props: dict[str, Any] = dict(schema.get("properties", {}) or {})
    required: set[str] = set(schema.get("required", []) or [])

    for name, spec in props.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"tool `{decl.name}` property `{name!r}` must be a valid Python identifier"
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"tool `{decl.name}` property `{name}` is not an object schema"
            )
        if spec.get("type") not in (None, "string"):
            raise ValueError(
                f"tool `{decl.name}` property `{name}` has unsupported type "
                f"`{spec.get('type')}`; only string is supported by the "
                f"pydantic-ai tool adapter."
            )

    handler = decl.handler

    if not props:
        def __no_args_tool() -> str:
            return handler({})

        __no_args_tool.__name__ = decl.name
        __no_args_tool.__doc__ = decl.description
        return __no_args_tool

    arg_lines: list[str] = []
    for name in props:
        if name in required:
            arg_lines.append(f"{name}: str")
        else:
            arg_lines.append(f"{name}: str = ''")
    sig = ", ".join(arg_lines)
    args_dict_expr = ", ".join(f"{n!r}: {n}" for n in props)
    src = (
        f"def __tool({sig}) -> str:\n"
        f"    return __handler({{{args_dict_expr}}})\n"
    )
    ns: dict[str, Any] = {"__handler": handler}
    exec(  # noqa: S102 — schema-driven, names sanitised below
        compile(src, f"<pydantic_ai_tool:{decl.name}>", "exec"),
        ns,
    )
    fn = ns["__tool"]
    fn.__name__ = decl.name
    fn.__doc__ = decl.description
    return fn
