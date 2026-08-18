"""Command line entry point.

    guardianops run  --config c.json -- python3 -m guardianops.mock_server
    guardianops run  --config c.json --upstream-url https://host/mcp
    guardianops verify --audit .guardianops/audit.jsonl
    guardianops report --audit .guardianops/audit.jsonl
    guardianops pins   --show
    guardianops validate --config c.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

from . import policy
from .audit import KeyRequired, load_key as audit_load_key, read_all, verify
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
    run.add_argument("--skip-validation", action="store_true",
                     help="start even if the policy has errors (not recommended)")
    run.add_argument("upstream_cmd", nargs=argparse.REMAINDER,
                     help="-- followed by the command that launches a stdio MCP server")

    ver = sub.add_parser("verify", help="verify the audit ledger hash chain")
    ver.add_argument("--audit", default=".guardianops/audit.jsonl",
                     help="a ledger file, or a directory of them (one per writer)")
    ver.add_argument("--key", help="path to the ledger signing key (for signed ledgers)")
    ver.add_argument("--config", help="take the signing key from a policy's auditKey")

    rep = sub.add_parser("report", help="summarize runs recorded in the ledger")
    rep.add_argument("--audit", default=".guardianops/audit.jsonl",
                     help="a ledger file, or a directory of them (one per writer)")
    rep.add_argument("--run-id", help="limit the report to one run")

    pins = sub.add_parser("pins", help="inspect pinned tool definitions")
    pins.add_argument("--path", default=".guardianops/pins.json")
    pins.add_argument("--show", action="store_true")

    val = sub.add_parser("validate", help="check a policy config for errors")
    val.add_argument("--config", required=True, help="path to a JSON policy config")
    val.add_argument("--json", action="store_true", dest="as_json",
                     help="emit findings as JSON for machine consumption")
    val.add_argument("--strict", action="store_true",
                     help="treat warnings as errors")

    return parser


def _load_config(path: str | None) -> policy.Config:
    """Load a policy, turning every malformed-input failure into a clean message.

    Raises ValueError with text fit for stderr; the alternative is a traceback
    at startup, which tells an operator nothing about which line to fix.
    """
    try:
        return policy.Config.load(path)
    except FileNotFoundError:
        raise ValueError(f"no such config: {path}") from None
    except IsADirectoryError:
        raise ValueError(f"not a file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from None
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from None


def _report(findings: list[policy.Finding], stream) -> None:
    for finding in findings:
        stream.write(f"  [{finding.severity}] {finding}\n")


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.mode:
        cfg.mode = args.mode
    if args.audit:
        cfg.audit_path = args.audit

    # Validated after the overrides, so --mode is judged as it will actually
    # run. A proxy that boots on a policy it knows is broken is governance
    # theatre: in enforce mode the errors decide what traffic is refused, so
    # starting anyway is the one failure the operator cannot see.
    findings = cfg.validate()
    errors = [f for f in findings if f.severity == policy.ERROR]
    if findings:
        sys.stderr.write(f"policy {cfg.source or '<defaults>'}:\n")
        _report(findings, sys.stderr)
    if errors and not args.skip_validation:
        if cfg.mode == policy.ENFORCE:
            sys.stderr.write(
                f"error: refusing to enforce a policy with {len(errors)} error(s). "
                "Fix them, or pass --skip-validation to start anyway.\n"
            )
            return 2
        sys.stderr.write(
            "warning: policy has errors; continuing because mode is shadow\n"
        )

    # Fail here rather than inside the proxy: an operator who configured a
    # signing key must not end up with an unsigned ledger they believe is signed.
    try:
        cfg.audit_key()
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

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


def _verify_key(args: argparse.Namespace) -> bytes | None:
    """The signing key for verification, from --key or a policy's auditKey."""
    if args.key:
        return audit_load_key(args.key)
    if args.config:
        return _load_config(args.config).audit_key()
    return None


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = _ledgers(args.audit)
    if not paths:
        sys.stderr.write(f"no ledger at {args.audit}\n")
        return 1

    try:
        key = _verify_key(args)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    failed = 0
    for path in paths:
        label = path.name if len(paths) > 1 else str(path)
        try:
            ok, count, error = verify(str(path), key)
        except KeyRequired as exc:
            # Not a chain failure: the ledger may be perfectly intact and we
            # simply cannot tell. Saying "tampered" would start a false hunt.
            sys.stderr.write(f"cannot verify · {exc} · {label}\n")
            failed += 1
            continue
        signed = " · signed" if key else ""
        if ok:
            print(f"ledger intact · {count} records · chain verified{signed} · {label}")
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


def _cmd_validate(args: argparse.Namespace) -> int:
    """Report every problem in a policy, rather than dying on the first one.

    Exit codes are the contract for CI: 0 clean, 1 the policy is wrong, 2 the
    file could not be read at all. Warnings alone do not fail a build unless
    --strict says they should.
    """
    try:
        cfg = _load_config(args.config)
    except ValueError as exc:
        if args.as_json:
            print(json.dumps({"config": args.config, "readable": False,
                              "error": str(exc)}, indent=2))
        else:
            sys.stderr.write(f"error: {exc}\n")
        return 2

    findings = cfg.validate()
    errors = [f for f in findings if f.severity == policy.ERROR]
    warnings = [f for f in findings if f.severity == policy.WARNING]
    failed = bool(errors) or (bool(warnings) and args.strict)

    if args.as_json:
        print(json.dumps({
            "config": args.config,
            "readable": True,
            "ok": not failed,
            "mode": cfg.mode,
            "errors": len(errors),
            "warnings": len(warnings),
            "findings": [f.as_dict() for f in findings],
        }, indent=2))
        return 1 if failed else 0

    if not findings:
        print(f"{args.config}: ok ({cfg.mode} mode, {len(cfg.allow)} tools allowed)")
        return 0

    counts = ", ".join(
        f"{n} {label}" for n, label in
        ((len(errors), "error(s)"), (len(warnings), "warning(s)")) if n
    )
    stream = sys.stderr if failed else sys.stdout
    stream.write(f"{args.config}: {counts}\n")
    _report(findings, stream)
    if failed:
        return 1
    print(f"{args.config}: ok ({cfg.mode} mode, {len(cfg.allow)} tools allowed)")
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
    if args.command == "validate":
        return _cmd_validate(args)
    return 2
