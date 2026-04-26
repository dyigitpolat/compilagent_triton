"""Typed levers + a backend-pluggable derivation registry.

A `SearchSpace` is the catalog of candidate axes the agent can pull. Each
`Lever` declares a typed range, defaults, and `derivation_evidence` linking it
back to the signal that produced it (tensor shape, IR diff, op count, ...).
**No lever's range is hand-coded in user-facing code** — every range is
either a function of the workload analysis or the device capability.

A backend declares zero or more `SearchSpaceDerivation` plugins. The runtime
runs every applicable derivation against the workload's `Analysis` and
concatenates the resulting levers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


# ---------------------------------------------------------------------------
# Range types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntRange:
    """A bounded integer axis with explicit candidate values.

    `candidates` is the **derived** list of values; never hand-coded by the
    user. Backends populate it from workload analysis or device capability.
    """

    candidates: tuple[int, ...]
    units: str = ""              # e.g. "elements", "warps", "stages"

    def serialize(self) -> dict[str, Any]:
        return {"kind": "int_range", "candidates": list(self.candidates), "units": self.units}


@dataclass(frozen=True, slots=True)
class FloatRange:
    candidates: tuple[float, ...]
    units: str = ""

    def serialize(self) -> dict[str, Any]:
        return {"kind": "float_range", "candidates": list(self.candidates), "units": self.units}


@dataclass(frozen=True, slots=True)
class EnumChoice:
    """A finite categorical axis with derived members."""

    candidates: tuple[str, ...]

    def serialize(self) -> dict[str, Any]:
        return {"kind": "enum", "candidates": list(self.candidates)}


@dataclass(frozen=True, slots=True)
class BooleanFlag:
    def serialize(self) -> dict[str, Any]:
        return {"kind": "bool", "candidates": [True, False]}


@dataclass(frozen=True, slots=True)
class StructuredJsonRange:
    """A complex-payload axis (e.g. a layout dict, an FX rewrite spec).

    `examples` is a derived sample of valid payload shapes the agent may copy
    or mutate; the actual validity is checked by the backend.
    """

    examples: tuple[dict[str, Any], ...]
    schema_hint: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "structured_json",
            "examples": list(self.examples),
            "schema_hint": self.schema_hint,
        }


LeverRange = IntRange | FloatRange | EnumChoice | BooleanFlag | StructuredJsonRange


# ---------------------------------------------------------------------------
# Lever
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DerivationEvidence:
    """Tying a lever back to the signal that produced its range.

    The agent uses this to reason about why a lever exists and which signal
    might be the bottleneck.
    """

    rule: str                    # plugin name: "triton.meta_parameters.block_size"
    signal: str                  # short human description
    citations: tuple[str, ...] = ()   # opaque pointers (file:line, ir_op, knob_name)


@dataclass(frozen=True, slots=True)
class Lever:
    """One typed candidate axis with derived bounds."""

    id: str                                        # "block_size", "max_fusion_size", etc.
    target_kind: str                               # discriminator into Target.kind
    target_selector: str                           # the Target.selector
    range: LeverRange
    default: Any
    description: str
    evidence: DerivationEvidence
    backend_id: str                                # which backend owns this lever

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend_id": self.backend_id,
            "target": {"kind": self.target_kind, "selector": self.target_selector},
            "range": self.range.serialize(),
            "default": self.default,
            "description": self.description,
            "evidence": {
                "rule": self.evidence.rule,
                "signal": self.evidence.signal,
                "citations": list(self.evidence.citations),
            },
        }


# ---------------------------------------------------------------------------
# Search space + derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Concatenated lever list for one workload."""

    workload_id: str
    backend_id: str
    levers: tuple[Lever, ...]

    def serialize(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "backend_id": self.backend_id,
            "levers": [lever.serialize() for lever in self.levers],
        }


class SearchSpaceDerivation(Protocol):
    """A pluggable rule that maps (workload, analysis) → some levers."""

    name: str
    applies_to: tuple[str, ...]      # workload kinds this rule applies to

    def derive(self, workload: Any, analysis: Any) -> Iterable[Lever]: ...


# ---------------------------------------------------------------------------
# Helpers used by derivation plugins
# ---------------------------------------------------------------------------


def pow2_around(value: int, *, lo: int = 32, hi: int = 4096, count: int = 5) -> tuple[int, ...]:
    """Return up to `count` power-of-two values centered on `value`, clamped to [lo,hi]."""

    if value <= 0:
        return tuple(sorted({lo, lo * 2, lo * 4, lo * 8})[:count])
    # Round to nearest power of two
    import math
    centre = 1 << max(0, int(round(math.log2(value))))
    half = count // 2
    seq = []
    for shift in range(-half, count - half):
        v = max(lo, min(hi, centre << shift if shift >= 0 else centre >> -shift))
        if v not in seq:
            seq.append(v)
    return tuple(sorted(seq))


def pow2_range(lo: int, hi: int) -> tuple[int, ...]:
    """All powers of two in `[lo, hi]`."""

    out: list[int] = []
    v = 1
    while v < lo:
        v <<= 1
    while v <= hi:
        out.append(v)
        v <<= 1
    return tuple(out)
