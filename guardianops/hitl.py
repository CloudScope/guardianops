"""Human-in-the-loop intercept, rendered on the controlling terminal.

The proxy's own stdin and stdout are the MCP transport -- writing a prompt there
would corrupt the protocol stream. So the intercept panel is drawn on /dev/tty
directly, which is also what lets an operator answer a prompt for an agent whose
stdio is wired to something else entirely.

If there is no controlling terminal (CI, a daemon, a container), approval is not
available and the caller falls back to the configured timeout policy. Silence is
never treated as consent.
"""

from __future__ import annotations

import asyncio
import select
import time
from typing import Any

ALLOW_ONCE = "allow"
BLOCK = "block"
WHITELIST = "whitelist"
# End the run, not just this call. Declining one call leaves an agent free to
# retry a variant immediately; this is the answer to that.
KILL_RUN = "kill_run"
TIMEOUT = "timeout"
UNAVAILABLE = "unavailable"

_RULE = "─" * 74


class TtyApprover:
    """Draws the intercept panel and reads one keystroke-ish answer."""

    def __init__(self) -> None:
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is None:
            try:
                with open("/dev/tty", "r"):
                    pass
                self._available = True
            except OSError:
                self._available = False
        return self._available

    async def request(self, panel: dict[str, Any], timeout: float) -> str:
        if not self.available():
            return UNAVAILABLE
        return await asyncio.to_thread(self._prompt, panel, timeout)

    def _prompt(self, panel: dict[str, Any], timeout: float) -> str:
        try:
            tty_in = open("/dev/tty", "r")
            tty_out = open("/dev/tty", "w")
        except OSError:
            return UNAVAILABLE

        with tty_in, tty_out:
            tty_out.write(self._render(panel, timeout))
            tty_out.flush()
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tty_out.write("\n  no answer within the approval window\n\n")
                    tty_out.flush()
                    return TIMEOUT
                # Poll rather than block so the deadline is honored even when the
                # operator never touches the keyboard.
                ready, _, _ = select.select([tty_in], [], [], min(0.25, remaining))
                if not ready:
                    continue
                line = tty_in.readline()
                if not line:
                    return TIMEOUT
                answer = line.strip().lower()
                if answer in ("a", "allow"):
                    return ALLOW_ONCE
                if answer in ("b", "block", "d", "deny", "n", "no"):
                    return BLOCK
                if answer in ("w", "whitelist"):
                    return WHITELIST
                if answer in ("k", "kill"):
                    return KILL_RUN
                tty_out.write(
                    "  expected [a]llow once, [b]lock, [w]hitelist, or [k]ill run: "
                )
                tty_out.flush()

    @staticmethod
    def _render(panel: dict[str, Any], timeout: float) -> str:
        signals = panel.get("signals", {})
        firing = {k: v for k, v in signals.items() if v}
        lines = [
            "",
            _RULE,
            f"  GUARDIANOPS · HUMAN-IN-THE-LOOP INTERCEPT      risk {panel.get('risk', 0):.2f}",
            _RULE,
            f"  run        {panel.get('run_id', '-')}",
            f"  server     {panel.get('server', '-')}",
            f"  tool       {panel.get('tool', '-')}  (tier: {panel.get('tier', '-')})",
            f"  reason     {panel.get('reason', '-')}",
        ]
        if firing:
            detail = "  ".join(f"{k}={v:.2f}" for k, v in sorted(firing.items()))
            lines.append(f"  signals    {detail}")
        args = panel.get("arguments")
        if args:
            lines.append("  arguments")
            for line in str(args).splitlines():
                lines.append(f"    {line[:68]}")
        lines += [
            _RULE,
            f"  [a] allow once   [b] block   [w] whitelist for run   [k] KILL RUN"
            f"   ({int(timeout)}s)",
            "  > ",
        ]
        return "\n".join(lines)
