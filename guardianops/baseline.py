"""Per-server invocation baseline.

Novelty is a behavioral fact, not a discovery fact: a tool the server has always
advertised but the agent has never actually called is still a first-time
invocation, and that is the signal worth scoring. This keeps invocation counts
across runs so the baseline sharpens instead of resetting every process.

A brand-new agent has no baseline at all -- cold start is real, and until it
fills in, the deterministic controls (entitlement, pinning, tiers) are doing all
of the work.
"""

from __future__ import annotations

import json
from pathlib import Path


class InvocationBaseline:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data: dict[str, dict[str, int]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def count(self, server: str, tool: str) -> int:
        return self.data.get(server, {}).get(tool, 0)

    def is_novel(self, server: str, tool: str) -> bool:
        return self.count(server, tool) == 0

    def observe(self, server: str, tool: str) -> None:
        """Record an invocation that actually executed."""
        bucket = self.data.setdefault(server, {})
        bucket[tool] = bucket.get(tool, 0) + 1

    def total(self, server: str) -> int:
        return sum(self.data.get(server, {}).values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
