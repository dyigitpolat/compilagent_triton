"""Canonical agent toolset for an `OptimizationSession`.

The 8 tools an agent uses to drive a session are declared here once. Both
pydantic-ai and Claude Agent SDK adapters consume the same `ToolDecl`s and
bind them into their native tool surfaces.

Tool argument schemas are derived directly from the typed signatures of
the bound session methods via Pydantic. The agent sees structured arrays
and nested objects (no string-encoded JSON), and the model emits
well-formed JSON because pydantic-ai introspects the typed function
signature when registering each tool.

Backend-supplied introspection tools are appended via
`Toolset.with_extra(backend.list_introspection_tools())` outside this file.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, create_model

from compilagent.core.tool_decl import ToolDecl
from compilagent.toolset import Toolset

if TYPE_CHECKING:
    from .session import OptimizationSession


def _args_model_for(method: Callable[..., Any], *, model_name: str) -> type[BaseModel]:
    """Build a Pydantic model that mirrors `method`'s keyword arguments.

    Skips `self` and any positional-only params. Forward-reference
    annotations (which `from __future__ import annotations` produces)
    are resolved via `eval_str=True` against the function's own globals.
    The resulting model's `model_json_schema()` is what we hand to
    harness adapters that drive their SDK off a static JSON Schema
    (e.g. the Claude SDK MCP bridge).
    """

    sig = inspect.signature(method, eval_str=True)
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = (
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else Any
        )
        default = (
            param.default if param.default is not inspect.Parameter.empty else ...
        )
        fields[name] = (annotation, default)
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def _decl_from_method(
    method: Callable[..., str],
    *,
    name: str,
    description: str,
    read_only: bool,
) -> ToolDecl:
    """Produce a `ToolDecl` whose schema mirrors the bound method's signature."""

    args_model = _args_model_for(method, model_name=f"{name}__Args")
    schema = args_model.model_json_schema()
    # Pydantic emits an open schema by default; lock it down for the wire.
    schema.setdefault("additionalProperties", False)
    return ToolDecl(
        name=name,
        description=description,
        args_schema=schema,
        handler=method,
        args_model=args_model,
        read_only=read_only,
    )


_TOOL_DESCRIPTIONS: dict[str, tuple[str, bool]] = {
    "inspect_workload": (
        "Return the workload spec, baseline timing, analysis summary, "
        "device capability, and any cross-run hints.",
        True,
    ),
    "inspect_search_space": (
        "Return the derived lever catalog with per-lever evidence so the "
        "agent can reason about which axes matter for this workload.",
        True,
    ),
    "propose_candidate": (
        "Register a multi-intervention candidate. Combine 2-4 levers per "
        "candidate when forming non-trivial hypotheses. Returns the new "
        "candidate id which run_candidate can then run.",
        False,
    ),
    "propose_candidates": (
        "Register several candidates at once. Useful for setting up a small "
        "batch of hypotheses (e.g. 3) before running them. The `plans` arg "
        "is an inline JSON array of plan objects — never a string.",
        False,
    ),
    "run_candidate": (
        "Compile + time + correctness-check a previously proposed candidate. "
        "Returns median_ms, speedup_vs_baseline, correctness_ok, plus a "
        "hint when the run failed. Multi-objective backends additionally "
        "populate `objectives` (a dict of named axes with goal directions); "
        "single-axis backends leave it empty.",
        False,
    ),
    "run_candidates": (
        "Run a batch of previously proposed candidates in sequence and "
        "return per-candidate results plus an aggregate summary. Each "
        "per-candidate entry mirrors `run_candidate`'s response, "
        "including the multi-objective `objectives` dict when the backend "
        "populates it.",
        False,
    ),
    "synthesize_findings": (
        "Aggregate the current run's results: per-target_kind speedup "
        "distributions, co-occurring lever pairs in winners, and "
        "interventions that appear in failures.",
        True,
    ),
    "compare_runs": (
        "Return a leaderboard of (baseline + judged candidates) sorted by "
        "median_ms ascending. Each row also carries the multi-objective "
        "`objectives` dict (empty for single-axis backends), so the agent "
        "can see all axes at a glance.",
        True,
    ),
    "query_top_candidates": (
        "Multi-objective only: return the top-k candidates sorted by one "
        "named objective, honouring its goal direction (max/min). Useful "
        "for finding the leader in a single axis without enumerating "
        "every row.",
        True,
    ),
    "pareto_front": (
        "Multi-objective only: return the non-dominated subset of "
        "candidates across every active objective. Each row includes the "
        "candidate's `objectives` dict and the registered plan "
        "description so the agent can reason about trade-offs.",
        True,
    ),
    "metric_summary": (
        "Multi-objective only: return per-objective best/worst/median "
        "across all candidates with their corresponding candidate ids. "
        "A quick way to see where the search has covered ground and "
        "where it has not.",
        True,
    ),
    "compare_candidates": (
        "Multi-objective only: return side-by-side objectives + "
        "descriptions for an explicit list of candidate ids. Use to "
        "compare the top-k of one metric against the top-k of another, "
        "or to inspect specific Pareto front members.",
        True,
    ),
}


def build_session_toolset(session: OptimizationSession) -> Toolset:
    """Build the canonical 8-tool toolset bound to this session instance.

    Each `ToolDecl.handler` is the bound session method itself, with its
    typed Pydantic-aware signature. pydantic-ai introspects this directly;
    the Claude Agent SDK adapter goes through `decl.invoke(args_dict)`,
    which validates the wire-shaped dict against the auto-derived
    `args_model` before calling the handler with typed kwargs.
    """

    decls: list[ToolDecl] = []
    for tool_name, (description, read_only) in _TOOL_DESCRIPTIONS.items():
        method = getattr(session, tool_name)
        decls.append(
            _decl_from_method(
                method,
                name=tool_name,
                description=description,
                read_only=read_only,
            )
        )
    return Toolset(tools=tuple(decls))
