"""Persistent experiment memory for cross-run learning.

Every successful candidate (compile OK + valid timing + correctness within
tolerance) is appended as one JSON line to
`<workspace>/memory/experiments.jsonl`. Each row records:

  - run_id, workload_id, backend_id, arch
  - the candidate's interventions (target_kind / target_selector / payload)
  - speedup_vs_baseline, median_ms, correctness max-abs/rel-diff
  - timestamp

`ExperimentLog.recall(workload_id, backend_id, arch, ...)` reads the log,
filters to the requested context, and returns the top-N rows by speedup so
the agent can study what worked on the same kernel/model family before.

Failures are also persisted (with `successful=false`) so the agent can avoid
re-proposing combinations that previously OOM'd or drifted out of tolerance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentLog:
    """Append-only JSONL log under `<workspace>/memory/experiments.jsonl`."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / "memory" / "experiments.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = {**row}
            entry.setdefault("timestamp", time.time())
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:  # noqa: BLE001 — log failure must not break the run
            pass

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return rows

    def recall(
        self,
        *,
        workload_id: str | None = None,
        backend_id: str | None = None,
        arch: str | None = None,
        successful_only: bool = True,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        """Return the highest-speedup rows that match the given filters."""

        rows = self.read_all()
        out: list[dict[str, Any]] = []
        for r in rows:
            if successful_only and not r.get("successful", False):
                continue
            if workload_id is not None and r.get("workload_id") != workload_id:
                continue
            if backend_id is not None and r.get("backend_id") != backend_id:
                continue
            if arch is not None and r.get("arch") != arch:
                continue
            out.append(r)
        out.sort(key=lambda r: r.get("speedup", 0.0) or 0.0, reverse=True)
        return out[:top_n]

    def recall_failures(
        self,
        *,
        workload_id: str | None = None,
        backend_id: str | None = None,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        """Return recent failures so the agent can avoid known-bad combos."""

        rows = self.read_all()
        out = [
            r for r in rows
            if not r.get("successful", True)
            and (workload_id is None or r.get("workload_id") == workload_id)
            and (backend_id is None or r.get("backend_id") == backend_id)
        ]
        out.sort(key=lambda r: r.get("timestamp", 0.0) or 0.0, reverse=True)
        return out[:top_n]
