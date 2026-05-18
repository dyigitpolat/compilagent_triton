"""Tiny stdlib compatibility shims.

Compilagent targets Python >=3.10. ``enum.StrEnum`` only landed in 3.11; on
older interpreters we polyfill the same `str`-mixin pattern so the rest of
the codebase can ``from compilagent._compat import StrEnum`` unconditionally.
"""

from __future__ import annotations

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum  # noqa: F401  (re-export)
else:
    class StrEnum(str, Enum):
        """Backport of ``enum.StrEnum`` for Python 3.10.

        Mirrors the 3.11 behaviour: members compare and serialise as plain
        strings; ``str(member)`` returns the value (not ``EnumName.MEMBER``).
        """

        def __new__(cls, value: str) -> "StrEnum":
            if not isinstance(value, str):
                raise TypeError(
                    f"StrEnum value must be str, got {type(value).__name__}"
                )
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj

        def __str__(self) -> str:  # match 3.11 StrEnum
            return str(self.value)


__all__ = ["StrEnum"]
