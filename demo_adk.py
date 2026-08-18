#!/usr/bin/env python3
"""The ADK governance layer, demonstrated without ADK installed.

This replays an ADK-shaped agent run against the real callbacks -- the same
GuardianOpsGuard code path an ADK Runner drives -- using stand-ins for the
objects ADK passes in. It is here so the decisions are inspectable before you
wire the guard into a live agent; examples/adk_agent.py is the real thing.

    python3 demo_adk.py
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from guardianops import policy
from guardianops.adk import GuardianOpsGuard

ROOT = Path(__file__).parent
STATE = ROOT / ".guardianops"

OBJECTIVE = "Set the request timeout to 30 seconds for the docs server in mcp.json"

# A plausible ADK run: two agents, one handoff, and a tail that drifts.
SESSION = [
    ("planner", "read_config", {"path": "/workspace/mcp.json"}),
    ("planner", "transfer_to_agent", {"agent_name": "executor"}),
    ("executor", "read_config", {"path": "/workspace/mcp.json"}),
    ("executor", "set_timeout", {"path": "/workspace/mcp.json", "seconds": 30}),
    ("executor", "attach_role_policy", {"role": "agent-runner", "policy": "AdministratorAccess"}),
    ("executor", "query_customers", {"table": "customer_payment_records"}),
]


class Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class Ctx:
    """Shaped like ADK's ToolContext / CallbackContext."""

    def __init__(self, invocation_id: str, agent_name: str) -> None:
        self.invocation_id = invocation_id
        self.agent_name = agent_name


class Part:
    def __init__(self, text: str) -> None:
        self.text = text


class Content:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.parts = [Part(text)]


class LlmRequest:
    def __init__(self, text: str) -> None:
        self.contents = [Content("user", text)]
        self.config = None


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 74}\n  {title}\n{'─' * 74}\033[0m", flush=True)


async def run_session(mode: str, run_id: str, context_threshold: float | None) -> None:
    cfg = policy.Config.load(str(ROOT / "config.adk.json"))
    cfg.mode = mode
    cfg.approval.context_threshold = context_threshold
    guard = GuardianOpsGuard(
        cfg=cfg,
        approver=None,  # no tty in this demo, so onTimeout=block applies
        baseline_path=str(STATE / "adk-baseline.json"),
        app_name="mcp-copilot",
    )
    guard.audit.run_id = run_id

    ctx = Ctx(run_id, "planner")
    await guard.before_agent_callback(ctx)
    await guard.before_model_callback(ctx, LlmRequest(OBJECTIVE))
    print(f"\n  objective: {OBJECTIVE}\n")

    for agent, tool, args in SESSION:
        call_ctx = Ctx(run_id, agent)
        blocked = await guard.before_tool_callback(Tool(tool), args, call_ctx)
        mark = "\033[31m✕\033[0m" if blocked else "\033[32m✓\033[0m"
        note = blocked["reason"] if blocked else ""
        print(f"  {mark} {agent:9} {tool:20} {note[:44]}")
        await guard.after_tool_callback(Tool(tool), args, call_ctx, blocked or {"ok": True})

    await guard.after_agent_callback(Ctx(run_id, "executor"))


async def main() -> int:
    if STATE.exists():
        shutil.rmtree(STATE)

    rule("1 · SHADOW — context drift scored, nothing enforced")
    print("  The objective is captured from the first user turn. Every call is")
    print("  scored against it; the run is never interrupted.")
    await run_session(policy.SHADOW, "adk-shadow", context_threshold=None)

    rule("2 · ENFORCE — entitlement holds, context drift stays advisory")
    print("  attach_role_policy is outside the granted entitlement. The customer")
    print("  query is off-objective but still allowed: scoring on, acting off.")
    await run_session(policy.ENFORCE, "adk-enforce", context_threshold=None)

    rule("3 · ENFORCE + contextThreshold=0.8 — drift becomes actionable")
    print("  Same run, now holding calls that sit far from the stated objective.")
    print("  Only turn this on after tuning against real traffic.")
    await run_session(policy.ENFORCE, "adk-context", context_threshold=0.8)

    rule("4 · LEDGER")
    subprocess.run([sys.executable, "-m", "guardianops", "verify"], cwd=ROOT)
    subprocess.run([sys.executable, "-m", "guardianops", "report"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
