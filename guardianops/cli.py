"""Command line entry point.

    guardianops run  --config c.json -- python3 -m guardianops.mock_server
    guardianops run  --config c.json --upstream-url https://host/mcp
    guardianops verify --audit .guardianops/audit.jsonl
    guardianops report --audit .guardianops/audit.jsonl
    guardianops pins   --show
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

from . import policy
from .audit import read_all, verify
from .pins import PinStore
from .proxy import Proxy
from .upstream import HttpUpstream, StdioUpstream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardianops",
        description="Runtime governance proxy for MCP servers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="proxy an MCP server with governance applied")
    run.add_argument("--config", help="path to a JSON policy config")
    run.add_argument("--mode", choices=[policy.SHADOW, policy.ENFORCE],
                     help="override the configured mode")
    run.add_argument("--upstream-url", help="Streamable HTTP endpoint of the upstream server")
    run.add_argument("--header", action="append", default=[],
                     metavar="K:V", help="extra HTTP header (repeatable)")
    run.add_argument("--run-id", help="identifier recorded on every audit record")
    run.add_argument("--audit", help="override the audit ledger path")
    run.add_argument("--baseline",
                     help="override the invocation baseline path from the policy")
    run.add_argument("upstream_cmd", nargs=argparse.REMAINDER,
                     help="-- followed by the command that launches a stdio MCP server")

    ver = sub.add_parser("verify", help="verify the audit ledger hash chain")
    ver.add_argument("--audit", default=".guardianops/audit.jsonl",
                     help="a ledger file, or a directory of them (one per writer)")

    rep = sub.add_parser("report", help="summarize runs recorded in the ledger")
    rep.add_argument("--audit", default=".guardianops/audit.jsonl",
                     help="a ledger file, or a directory of them (one per writer)")
    rep.add_argument("--run-id", help="limit the report to one run")

    pins = sub.add_parser("pins", help="inspect pinned tool definitions")
    pins.add_argument("--path", default=".guardianops/pins.json")
    pins.add_argument("--show", action="store_true")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = policy.Config.load(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.audit:
        cfg.audit_path = args.audit

    argv = [a for a in args.upstream_cmd if a != "--"]
    if args.upstream_url and argv:
        sys.stderr.write("error: give either --upstream-url or a stdio command, not both\n")
        return 2
    if args.upstream_url:
        headers = {}
        for item in args.header:
            key, _, value = item.partition(":")
            if value:
                headers[key.strip()] = value.strip()
        up = HttpUpstream(args.upstream_url, headers)
    elif argv:
        up = StdioUpstream(argv)
    else:
        sys.stderr.write(
            "error: no upstream. Pass '-- <command>' for stdio or --upstream-url for HTTP\n"
        )
        return 2

    proxy = Proxy(cfg, up, run_id=args.run_id,
                  baseline_path=args.baseline or cfg.baseline_path)
    try:
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        pass
    return 0


def _ledgers(target: str) -> list[Path]:
    """Resolve one ledger path, or every ledger in a directory.

    A hash chain has exactly one writer: each process chains from the record it
    last wrote, so two of them appending to one file interleave into a chain
    that cannot verify. The answer is a file per writer and correlation at read
    time -- which is why these commands take a directory.
    """
    path = Path(target)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path] if path.exists() else []


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = _ledgers(args.audit)
    if not paths:
        sys.stderr.write(f"no ledger at {args.audit}\n")
        return 1

    failed = 0
    for path in paths:
        ok, count, error = verify(str(path))
        label = path.name if len(paths) > 1 else str(path)
        if ok:
            print(f"ledger intact · {count} records · chain verified · {label}")
        else:
            print(f"LEDGER TAMPERED · {error} · {label}")
            failed += 1
    return 1 if failed else 0


def _cmd_report(args: argparse.Namespace) -> int:
    paths = _ledgers(args.audit)
    if not paths:
        sys.stderr.write(f"no ledger at {args.audit}\n")
        return 1

    runs: dict[str, dict] = {}
    records = (r for p in paths for r in read_all(str(p)))
    for record in records:
        run_id = record.get("run_id", "?")
        if args.run_id and run_id != args.run_id:
            continue
        run = runs.setdefault(
            run_id,
            {
                "mode": "-",
                "server": "-",
                "calls": 0,
                "outcomes": collections.Counter(),
                "tools": collections.Counter(),
                "max_risk": 0.0,
                "riskiest": None,
                "withheld": set(),
                "changed": set(),
                "shadowed": 0,
                "held": 0,
                "approved": 0,
                "decision_us": [],
            },
        )
        event = record.get("event")
        if event == "run.start":
            run["mode"] = record.get("mode", "-")
        elif event == "agent.start":
            run["mode"] = record.get("mode", run["mode"])
            run["server"] = record.get("app", run["server"])
        elif event == "session.initialize":
            run["server"] = record.get("server", "-")
        elif event == "tools.list":
            run["withheld"].update(record.get("withheld", []))
            run["changed"].update(record.get("definition_changed", []))
        elif event == "tool.call":
            run["calls"] += 1
            run["outcomes"][record.get("outcome", "?")] += 1
            run["tools"][record.get("tool", "?")] += 1
            if record.get("shadowed"):
                run["shadowed"] += 1
            if record.get("held"):
                run["held"] += 1
                if record.get("approval") in ("allow", "whitelist"):
                    run["approved"] += 1
            if record.get("decision_us") is not None:
                run["decision_us"].append(record["decision_us"])
            risk = float(record.get("risk", 0) or 0)
            if risk >= run["max_risk"]:
                run["max_risk"] = risk
                run["riskiest"] = (record.get("tool"), record.get("reason"))

    if not runs:
        print("no matching runs")
        return 0

    for run_id, run in runs.items():
        out = run["outcomes"]
        print(f"\nrun {run_id}   mode={run['mode']}   server={run['server']}")
        print(f"  tool calls        {run['calls']}")
        print(f"  outcomes          allow={out['allow']} block={out['block']}")
        if run["held"]:
            print(
                f"  held for human    {run['held']} "
                f"({run['approved']} approved, {run['held'] - run['approved']} not)"
            )
        if run["shadowed"]:
            print(f"  would have acted  {run['shadowed']} (shadow mode, not enforced)")
        if run["decision_us"]:
            ordered = sorted(run["decision_us"])
            p50 = ordered[len(ordered) // 2]
            p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
            print(f"  policy eval       p50 {p50:.0f}µs   p99 {p99:.0f}µs")
        print(f"  peak risk         {run['max_risk']:.2f}")
        if run["riskiest"] and run["riskiest"][0]:
            print(f"                    {run['riskiest'][0]} — {run['riskiest'][1]}")
        if run["withheld"]:
            print(f"  withheld tools    {', '.join(sorted(run['withheld']))}")
        if run["changed"]:
            print(f"  MCP TRUST DRIFT   {', '.join(sorted(run['changed']))}")
        top = ", ".join(f"{name}×{n}" for name, n in run["tools"].most_common(5))
        if top:
            print(f"  most called       {top}")
    print()
    return 0


def _cmd_pins(args: argparse.Namespace) -> int:
    store = PinStore(args.path)
    if not store.data:
        print(f"no pins recorded at {args.path}")
        return 0
    print(json.dumps(store.data, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "pins":
        return _cmd_pins(args)
    return 2
