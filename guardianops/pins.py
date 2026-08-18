"""Trust-on-first-use pinning for MCP tool definitions.

An MCP server can change what a tool *does* without changing its name, simply by
rewriting the description the model reads or the schema it fills in. That is the
rug-pull / tool-poisoning shape of attack, and it is invisible to anything that
only inspects call arguments. We hash the definition on first sight and refuse
to let it change quietly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

NEW = "new"
OK = "ok"
CHANGED = "changed"


def digest(tool: dict[str, Any]) -> str:
    """Hash the parts of a tool definition that steer model behavior."""
    material = {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "inputSchema": tool.get("inputSchema", {}),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PinStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data: dict[str, dict[str, str]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def known(self, server: str, name: str) -> bool:
        return name in self.data.get(server, {})

    def check(self, server: str, tools: list[dict[str, Any]]) -> dict[str, str]:
        """Classify each advertised tool as new, unchanged, or changed.

        New tools are pinned immediately (trust on first use). Changed tools are
        reported and deliberately *not* re-pinned -- re-pinning on sight would
        make the control self-defeating. An operator approves the new definition
        explicitly via ``approve``.
        """
        pinned = self.data.setdefault(server, {})
        status: dict[str, str] = {}
        dirty = False
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            current = digest(tool)
            previous = pinned.get(name)
            if previous is None:
                pinned[name] = current
                status[name] = NEW
                dirty = True
            elif previous == current:
                status[name] = OK
            else:
                status[name] = CHANGED
        if dirty:
            self.save()
        return status

    def approve(self, server: str, tool: dict[str, Any]) -> None:
        name = tool.get("name")
        if not name:
            return
        self.data.setdefault(server, {})[name] = digest(tool)
        self.save()
