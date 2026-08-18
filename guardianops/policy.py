"""Policy configuration and the deterministic decision engine.

Everything in this module runs on the hot path, so it is deliberately free of
model inference, network calls and I/O: set membership, integer counters and a
weighted sum. Four of the five drift signatures can be computed this way. The
fifth (context drift) needs an embedding model and is left at zero here -- see
``Signals.context``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import scan
from typing import Any

# Decisions
ALLOW = "allow"
ESCALATE = "escalate"
BLOCK = "block"

# Modes
SHADOW = "shadow"
ENFORCE = "enforce"

# Blast-radius tiers. Tools that are not classified land in UNCLASSIFIED, which
# is treated as approval-worthy rather than safe -- unknown blast radius is a
# risk, not an absence of one.
READ = "read"
WRITE = "write"
DESTRUCTIVE = "destructive"
UNCLASSIFIED = "unclassified"
# Control-plane calls -- agent handoff, checkpointing, session management. They
# carry agent names rather than subject matter, which is why they are exempt
# from context scoring by default; see ``ApprovalConfig.context_exempt_tiers``.
CONTROL = "control"

BUILTIN_TIERS = frozenset({READ, WRITE, DESTRUCTIVE, UNCLASSIFIED, CONTROL})

# Composite weights, as published in the concept brief.
WEIGHTS = {
    "context": 0.20,
    "tool": 0.20,
    "privilege": 0.25,
    "autonomy": 0.15,
    "mcp_trust": 0.20,
}

DEFAULT_REDACT = [
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "credential", "credentials", "private_key",
]


@dataclass
class ApprovalConfig:
    require_for: list[str] = field(default_factory=lambda: [DESTRUCTIVE, UNCLASSIFIED])
    timeout_seconds: int = 120
    # What happens when nobody answers. Defaulting to anything but "block" means
    # an unattended agent can outlast its own supervision.
    on_timeout: str = BLOCK
    # Calls allowed since the last human checkpoint before autonomy drift fires.
    autonomy_threshold: int = 25
    # Context drift score at which a call is held. None means score it but never
    # act on it -- the right default while the scorer is still lexical.
    context_threshold: float | None = None
    # Tiers exempt from context scoring. Control-plane calls (agent handoff,
    # checkpointing) carry agent names rather than subject matter, so they score
    # as maximally off-objective no matter how routine they are. Scoring them
    # produces false positives that would break every multi-agent run.
    context_exempt_tiers: list[str] = field(default_factory=lambda: ["control"])


# "scheme://" -- anything with an authority component, so the "//" is load
# bearing and must survive normalization.
_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# What a tool name may look like. Deliberately permissive: MCP servers publish
# dotted and hyphenated names ("github.create_issue", "brave-search") and the
# engine never treats a tool name as an identifier, so anything but whitespace
# and empty is workable. This catches typos and stray punctuation, not style.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")


# Severity of a validation finding. The split matters at startup: an ERROR is a
# policy that cannot mean what it says and stops an enforcing proxy, while a
# WARNING is suspicious but survivable and never blocks a deployment.
ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One problem with a policy, located and explained."""

    severity: str
    path: str      # dotted location in the policy document, e.g. "approval.onTimeout"
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "path": self.path, "message": self.message}


# Every key the loader understands, by the section it lives in. Anything else in
# a policy document is a key nobody reads -- which is how a typo silently
# deletes an allowlist and leaves a config that still validates clean. Reported
# as a warning rather than an error so a config written for a newer GuardianOps
# still runs here.
_KNOWN_KEYS: dict[str, frozenset[str]] = {
    "": frozenset({
        "name", "mode", "defaultDecision", "auditPath", "baselinePath",
        "auditSync", "filterToolList", "redactKeys",
        "tools", "approval", "pin", "scope", "scan",
    }),
    "tools": frozenset({"allow", "deny", "tiers", "constraints"}),
    "approval": frozenset({
        "requireFor", "timeoutSeconds", "onTimeout", "autonomyThreshold",
        "contextThreshold", "contextExemptTiers",
    }),
    "pin": frozenset({"path", "onChange"}),
    "scope": frozenset({
        "terms", "threshold", "denyPatterns", "onOutOfScope", "message",
    }),
    "scan": frozenset({
        "definitions", "responses", "definitionPatterns", "responsePatterns",
        "maxResponseBytes",
    }),
}

# Keys understood inside tools.constraints.<tool>.<argument>.
_KNOWN_CONSTRAINT_KEYS = frozenset({
    "prefix", "denyPrefix", "in", "pattern", "denyPattern", "required",
    "normalizePath",
})


def _unknown_keys(data: dict[str, Any]) -> list[str]:
    """Dotted paths of every key in a policy document that nothing reads."""
    found: list[str] = []

    for section, known in _KNOWN_KEYS.items():
        blob = data if section == "" else data.get(section)
        if not isinstance(blob, dict):
            continue
        prefix = f"{section}." if section else ""
        found.extend(f"{prefix}{key}" for key in blob if key not in known)

    constraints = (data.get("tools") or {}).get("constraints")
    if isinstance(constraints, dict):
        for tool, args in constraints.items():
            if not isinstance(args, dict):
                continue
            for arg, spec in args.items():
                if not isinstance(spec, dict):
                    continue
                found.extend(
                    f"tools.constraints.{tool}.{arg}.{key}"
                    for key in spec
                    if key not in _KNOWN_CONSTRAINT_KEYS
                )

    return sorted(found)


@dataclass
class ArgumentConstraint:
    """Bounds on one argument of one tool.

    Entitlement by tool name cannot tell ``delete_path("/tmp/scratch")`` from
    ``delete_path("/")`` -- same tool, same permission, wildly different blast
    radius. Constraints are where the resource half of the decision lives.
    """

    prefix: list[str] | None = None          # value must start with one of these
    deny_prefix: list[str] | None = None     # ...but never one of these
    allowed: list[str] | None = None         # exact-match allowlist
    pattern: str | None = None               # must fully match
    deny_pattern: str | None = None          # must not match anywhere
    required: bool = False                   # absence is itself a violation
    normalize_path: bool = True              # collapse ../ before prefix matching

    @staticmethod
    def parse(raw: dict[str, Any]) -> "ArgumentConstraint":
        return ArgumentConstraint(
            prefix=raw.get("prefix"),
            deny_prefix=raw.get("denyPrefix"),
            allowed=raw.get("in"),
            pattern=raw.get("pattern"),
            deny_pattern=raw.get("denyPattern"),
            required=raw.get("required", False),
            normalize_path=raw.get("normalizePath", True),
        )

    def violation(self, name: str, value: Any) -> str | None:
        """Return a human-readable reason, or None if the value is acceptable."""
        text = value if isinstance(value, str) else str(value)
        if self.normalize_path and (self.prefix or self.deny_prefix) and not _URL.match(text):
            # Without this, "/workspace/../../etc/passwd" satisfies a
            # "/workspace/" prefix rule. Symlinks are not resolved -- the target
            # may not exist on this host -- so prefix rules bound the path that
            # was asked for, not necessarily the inode it reaches.
            #
            # URLs are exempt because normpath collapses the "//" after the
            # scheme: "https://169.254.169.254" becomes "https:/169.254.169.254"
            # and a denyPrefix of "https://169.254." silently stops matching.
            # A path rule that mangles the value it is judging is worse than no
            # rule, because it looks like it is working.
            text = os.path.normpath(text)

        if self.deny_prefix and any(text.startswith(p) for p in self.deny_prefix):
            return f"{name}={text!r} is under a denied prefix"
        if self.prefix and not any(text.startswith(p) for p in self.prefix):
            allowed = ", ".join(repr(p) for p in self.prefix)
            return f"{name}={text!r} is outside the permitted prefixes ({allowed})"
        if self.allowed is not None and value not in self.allowed and text not in self.allowed:
            return f"{name}={text!r} is not in the permitted values"
        if self.deny_pattern and re.search(self.deny_pattern, text):
            return f"{name}={text!r} matches a denied pattern"
        if self.pattern and not re.fullmatch(self.pattern, text):
            return f"{name}={text!r} does not match the required pattern"
        return None


@dataclass
class PinConfig:
    path: str = ".guardianops/pins.json"
    # "block" | "escalate" | "warn" -- what a changed tool schema does.
    on_change: str = ESCALATE


@dataclass
class ScopeConfig:
    """What this agent is *for*, checked against what it was asked to do.

    Distinct from context drift, which asks whether a tool call matches the
    request. This asks whether the request itself belongs to the agent -- an
    off-topic question is perfectly self-consistent, so drift scoring can never
    catch it. Both are needed and neither substitutes for the other.

    Empty ``terms`` disables the check, which is the default: an agent with no
    declared purpose has nothing to be out of scope of.
    """

    terms: list[str] = field(default_factory=list)
    # Fraction of the request's vocabulary that may be unrelated to the terms
    # before it counts as out of scope. None disables enforcement while still
    # scoring, which is where any new scope config should start.
    threshold: float | None = None
    # Regexes matched against the request text. A hit is refused outright, no
    # scoring involved -- for categorical exclusions the topic score is too
    # blunt to express ("never give medical advice"). Matched case-insensitively
    # and anywhere in the text.
    deny_patterns: list[str] = field(default_factory=list)
    # "block" | "warn"
    on_out_of_scope: str = BLOCK
    # Shown to the user when a request is turned away.
    message: str = (
        "That request is outside what this agent is configured to handle."
    )

    @property
    def enabled(self) -> bool:
        return bool(self.terms or self.deny_patterns)

    def denied(self, text: str) -> str | None:
        """First deny pattern this request matches, or None.

        Note what this can and cannot do: it reads the words the user typed, so
        it catches a category named directly and misses the same request
        paraphrased. It is a coarse pre-filter, not a substitute for the
        argument constraints that bound what a tool may actually be called
        with.
        """
        for pattern in self.deny_patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    return pattern
            except re.error:
                continue
        return None


def _anchor(path: str, base: Path) -> str:
    """Resolve a state path against the policy's own directory."""
    return path if Path(path).is_absolute() else str((base / path).resolve())


@dataclass
class Config:
    # Who this policy is for. Used as the app/agent name when nothing overrides
    # it, so a new agent is a new file rather than a new argument.
    name: str = ""
    mode: str = SHADOW
    # Decision for a tool that matches no other rule.
    default_decision: str = ALLOW
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    tiers: dict[str, list[str]] = field(default_factory=dict)
    # tool name -> argument name -> constraint
    constraints: dict[str, dict[str, ArgumentConstraint]] = field(default_factory=dict)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    pin: PinConfig = field(default_factory=PinConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    # Content scanning for tool definitions and tool results -- the two places
    # MCP mixes instructions with data. Pinning covers neither.
    scan: scan.ScanConfig = field(default_factory=scan.ScanConfig)
    audit_path: str = ".guardianops/audit.jsonl"
    baseline_path: str = ".guardianops/baseline.json"
    # The file this was loaded from, if any. Relative state paths are resolved
    # against its directory, so a policy carries its own storage with it.
    source: str = ""
    # "always" | "critical" | "interval" -- see guardianops.audit. Governance
    # decisions are worth a real sync; routine allows are not.
    audit_sync: str = "critical"
    redact_keys: list[str] = field(default_factory=lambda: list(DEFAULT_REDACT))
    # Drop non-entitled tools from tools/list so the model never sees them.
    filter_tool_list: bool = True
    # Keys in the source document that no loader reads, recorded at parse time
    # because by the time validate() runs they have already been discarded.
    unknown_keys: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | None) -> "Config":
        if not path:
            return Config()
        raw = json.loads(Path(path).read_text())
        return Config.from_dict(
            raw, source=str(Path(path).resolve()), default_name=Path(path).stem
        )

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        source: str = "",
        default_name: str = "programmatic",
    ) -> "Config":
        """Build a Config from an already-parsed policy document.

        ``source`` is the file this policy came from, if any. A relative state
        path means "next to this policy", never "next to whatever launched the
        process": the proxy inherits its client's working directory, which is
        rarely the one anybody had in mind, and a ledger that lands somewhere
        different each run is not a ledger. With no source there is nothing to
        sit next to, so the working directory is all that is left.
        """
        if not isinstance(data, dict):
            raise TypeError(
                "a policy must be a JSON object, got "
                f"{type(data).__name__}"
            )

        cfg = Config()
        cfg.source = source
        cfg.unknown_keys = _unknown_keys(data)
        cfg.name = data.get("name", default_name)
        cfg.mode = data.get("mode", cfg.mode)
        cfg.default_decision = data.get("defaultDecision", cfg.default_decision)
        cfg.audit_path = data.get("auditPath", cfg.audit_path)
        cfg.baseline_path = data.get("baselinePath", cfg.baseline_path)
        cfg.audit_sync = data.get("auditSync", cfg.audit_sync)
        cfg.filter_tool_list = data.get("filterToolList", cfg.filter_tool_list)
        cfg.redact_keys = data.get("redactKeys", cfg.redact_keys)

        tools = data.get("tools", {})
        cfg.allow = tools.get("allow", [])
        cfg.deny = tools.get("deny", [])
        cfg.tiers = tools.get("tiers", {})
        cfg.constraints = {
            tool: {arg: ArgumentConstraint.parse(spec) for arg, spec in args.items()}
            for tool, args in (tools.get("constraints") or {}).items()
        }

        ap = data.get("approval", {})
        cfg.approval = ApprovalConfig(
            require_for=ap.get("requireFor", cfg.approval.require_for),
            timeout_seconds=ap.get("timeoutSeconds", cfg.approval.timeout_seconds),
            on_timeout=ap.get("onTimeout", cfg.approval.on_timeout),
            autonomy_threshold=ap.get("autonomyThreshold", cfg.approval.autonomy_threshold),
            context_threshold=ap.get("contextThreshold", cfg.approval.context_threshold),
            context_exempt_tiers=ap.get(
                "contextExemptTiers", cfg.approval.context_exempt_tiers
            ),
        )

        pin = data.get("pin", {})
        cfg.pin = PinConfig(
            path=pin.get("path", cfg.pin.path),
            on_change=pin.get("onChange", cfg.pin.on_change),
        )

        scope = data.get("scope", {})
        cfg.scope = ScopeConfig(
            terms=scope.get("terms", cfg.scope.terms),
            threshold=scope.get("threshold", cfg.scope.threshold),
            deny_patterns=scope.get("denyPatterns", cfg.scope.deny_patterns),
            on_out_of_scope=scope.get("onOutOfScope", cfg.scope.on_out_of_scope),
            message=scope.get("message", cfg.scope.message),
        )

        cfg.scan = scan.ScanConfig.parse(data.get("scan", {}))

        base = Path(source).resolve().parent if source else Path.cwd()
        cfg.audit_path = _anchor(cfg.audit_path, base)
        cfg.baseline_path = _anchor(cfg.baseline_path, base)
        cfg.pin.path = _anchor(cfg.pin.path, base)
        return cfg

    @staticmethod
    def from_json_string(text: str, source: str = "") -> "Config":
        """Build a Config from a JSON policy document."""
        return Config.from_dict(json.loads(text), source=source)

    def validate(self) -> list["Finding"]:
        """Every way this policy is malformed or suspicious.

        Empty means clean. Errors are policies that cannot mean what they say --
        an unparseable regex, a tier nothing will ever match. Warnings are
        survivable but usually mistakes, chiefly keys nothing reads.

        This checks the shape of a policy, not its judgement: an empty allowlist
        in enforce mode is a legitimate thing to want and is not reported.
        """
        found: list[Finding] = []

        def err(path: str, message: str) -> None:
            found.append(Finding(ERROR, path, message))

        def warn(path: str, message: str) -> None:
            found.append(Finding(WARNING, path, message))

        # A key nothing reads is the failure that hides best: the section it
        # belongs to silently keeps its defaults, so a misspelled "tools" drops
        # an entire allowlist and leaves a policy that looks perfectly healthy.
        for key in self.unknown_keys:
            warn(key, "unknown key -- nothing reads this, so the setting has no effect")

        if self.mode not in (SHADOW, ENFORCE):
            err("mode", f"must be 'shadow' or 'enforce', got {self.mode!r}")

        if self.default_decision not in (ALLOW, BLOCK, ESCALATE):
            err(
                "defaultDecision",
                f"must be 'allow', 'block' or 'escalate', got {self.default_decision!r}",
            )

        # A policy may declare tiers of its own; the built-ins are the ones the
        # engine attaches meaning to. Both are legal targets for requireFor and
        # contextExemptTiers, and a name in neither is a typo that would
        # silently disable whatever it was meant to cover.
        known_tiers = BUILTIN_TIERS | set(self.tiers)

        for tier, tools in self.tiers.items():
            for tool in tools:
                if not _TOOL_NAME.fullmatch(tool):
                    err(f"tools.tiers.{tier}", f"invalid tool name {tool!r}")

        for listing in ("allow", "deny"):
            for tool in getattr(self, listing):
                if not _TOOL_NAME.fullmatch(tool):
                    err(f"tools.{listing}", f"invalid tool name {tool!r}")

        overlap = set(self.allow) & set(self.deny)
        if overlap:
            err("tools", f"allow and deny overlap: {sorted(overlap)}")

        for tool, args in self.constraints.items():
            # Constraints apply to any tool that gets called, so an unlisted
            # tool is only a mistake when the allowlist makes it unreachable.
            if self.allow and tool not in self.allow:
                err(
                    "tools.constraints",
                    f"{tool!r} is not in tools.allow, so its constraints can never apply",
                )
            for arg_name, constraint in args.items():
                where = f"tools.constraints.{tool}.{arg_name}"
                if constraint.prefix and constraint.deny_prefix:
                    overlap_pfx = set(constraint.prefix) & set(constraint.deny_prefix)
                    if overlap_pfx:
                        err(where, f"prefix and denyPrefix overlap: {sorted(overlap_pfx)}")
                for field_name, pattern in (
                    ("pattern", constraint.pattern),
                    ("denyPattern", constraint.deny_pattern),
                ):
                    if not pattern:
                        continue
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        err(where, f"invalid {field_name} regex: {exc}")

        ap = self.approval
        if ap.timeout_seconds <= 0:
            err("approval.timeoutSeconds", "must be positive")
        if ap.on_timeout not in (ALLOW, BLOCK):
            err("approval.onTimeout", f"must be 'allow' or 'block', got {ap.on_timeout!r}")
        if ap.autonomy_threshold <= 0:
            err("approval.autonomyThreshold", "must be positive")
        if ap.context_threshold is not None and not 0.0 <= ap.context_threshold <= 1.0:
            err("approval.contextThreshold", "must be in [0.0, 1.0]")
        for tier in ap.require_for:
            if tier not in known_tiers:
                err(
                    "approval.requireFor",
                    f"unknown tier {tier!r} (known: {sorted(known_tiers)})",
                )
        for tier in ap.context_exempt_tiers:
            if tier not in known_tiers:
                err(
                    "approval.contextExemptTiers",
                    f"unknown tier {tier!r} (known: {sorted(known_tiers)})",
                )

        if self.pin.on_change not in (BLOCK, ESCALATE, "warn"):
            err(
                "pin.onChange",
                f"must be 'block', 'escalate' or 'warn', got {self.pin.on_change!r}",
            )

        if self.scope.threshold is not None and not 0.0 <= self.scope.threshold <= 1.0:
            err("scope.threshold", "must be in [0.0, 1.0]")
        if self.scope.on_out_of_scope not in (BLOCK, "warn"):
            err(
                "scope.onOutOfScope",
                f"must be 'block' or 'warn', got {self.scope.on_out_of_scope!r}",
            )
        for pattern in self.scope.deny_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                err("scope.denyPatterns", f"invalid regex {pattern!r}: {exc}")

        if self.scan.definitions not in ("", scan.BLOCK, scan.WARN, scan.LOG):
            err(
                "scan.definitions",
                f"must be 'block', 'warn', 'log' or '', got {self.scan.definitions!r}",
            )
        if self.scan.responses not in ("", scan.BLOCK, scan.SANITIZE, scan.LOG):
            err(
                "scan.responses",
                f"must be 'block', 'sanitize', 'log' or '', got {self.scan.responses!r}",
            )
        if self.scan.max_response_bytes <= 0:
            err("scan.maxResponseBytes", "must be positive")

        if self.audit_sync not in ("always", "critical", "interval"):
            err(
                "auditSync",
                f"must be 'always', 'critical' or 'interval', got {self.audit_sync!r}",
            )

        # Arguments are written to the ledger, and the ledger is append-only --
        # a secret that lands there cannot be taken back out. An empty list is
        # legal (some deployments redact upstream) but is rarely intended.
        if not self.redact_keys:
            warn("redactKeys", "empty -- credentials in tool arguments will be logged verbatim")

        return found

    def errors(self) -> list["Finding"]:
        """Only the findings that make a policy unfit to enforce."""
        return [f for f in self.validate() if f.severity == ERROR]

    def tier_of(self, tool: str) -> str:
        for tier, names in self.tiers.items():
            if tool in names:
                return tier
        return UNCLASSIFIED

    def constraint_violation(self, tool: str, arguments: dict[str, Any] | None) -> str | None:
        """First constraint this call breaks, or None."""
        specs = self.constraints.get(tool)
        if not specs:
            return None
        arguments = arguments or {}
        for name, constraint in specs.items():
            if name not in arguments:
                if constraint.required:
                    return f"required argument {name!r} is missing"
                continue
            reason = constraint.violation(name, arguments[name])
            if reason:
                return reason
        return None

    def entitled(self, tool: str) -> bool:
        if tool in self.deny:
            return False
        if self.allow:
            return tool in self.allow
        return True


@dataclass
class Signals:
    """Per-call drift scores, each in [0, 1]."""

    context: float = 0.0
    tool: float = 0.0
    privilege: float = 0.0
    autonomy: float = 0.0
    mcp_trust: float = 0.0

    def composite(self) -> float:
        total = sum(WEIGHTS[k] * getattr(self, k) for k in WEIGHTS)
        return round(total, 3)

    def as_dict(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 3) for k in WEIGHTS}


@dataclass
class Verdict:
    decision: str          # what policy concluded
    effective: str         # what we actually did (shadow mode downgrades to allow)
    reason: str
    tier: str
    signals: Signals
    risk: float

    @property
    def shadowed(self) -> bool:
        return self.decision != self.effective


def evaluate(
    cfg: Config,
    tool: str,
    *,
    novel: bool,
    pin_changed: bool,
    calls_since_approval: int,
    session_allowed: bool = False,
    context: float = 0.0,
    arguments: dict[str, Any] | None = None,
) -> Verdict:
    """Score one tool call and decide what to do with it.

    ``novel`` means this tool has never been invoked for this agent before -- the
    strongest single behavioral signal available without a trained baseline.

    ``context`` is the drift of this call from the agent's stated objective. It
    is only available where the objective is observable (the ADK integration
    sees it; the MCP proxy does not) and defaults to 0.0 everywhere else.
    """
    tier = cfg.tier_of(tool)
    sig = Signals()

    violation = cfg.constraint_violation(tool, arguments)

    sig.tool = 1.0 if novel else 0.0
    sig.mcp_trust = 1.0 if pin_changed else 0.0
    sig.autonomy = min(1.0, calls_since_approval / max(1, cfg.approval.autonomy_threshold))
    if not cfg.entitled(tool) or violation:
        sig.privilege = 1.0
    elif tier == UNCLASSIFIED:
        sig.privilege = 0.5
    sig.context = (
        0.0 if tier in cfg.approval.context_exempt_tiers else min(1.0, max(0.0, context))
    )

    risk = sig.composite()

    def verdict(decision: str, reason: str) -> Verdict:
        effective = ALLOW if cfg.mode == SHADOW else decision
        return Verdict(decision, effective, reason, tier, sig, risk)

    # Constraints are checked before the session whitelist on purpose. An
    # operator approving "delete_path for this run" is approving the tool, not
    # every path it could be pointed at.
    if violation:
        return verdict(BLOCK, f"argument constraint: {violation}")

    # An operator who whitelisted this path during the run has already made the
    # call; do not ask them again.
    if session_allowed:
        return verdict(ALLOW, "operator whitelisted this path for the run")

    if tool in cfg.deny:
        return verdict(BLOCK, "explicit deny rule")

    if cfg.allow and tool not in cfg.allow:
        return verdict(BLOCK, f"privilege drift: {tool!r} is outside the granted entitlement")

    if pin_changed:
        if cfg.pin.on_change == BLOCK:
            return verdict(BLOCK, "MCP trust drift: tool definition changed since it was pinned")
        if cfg.pin.on_change == ESCALATE:
            return verdict(ESCALATE, "MCP trust drift: tool definition changed since it was pinned")

    if tier in cfg.approval.require_for:
        return verdict(ESCALATE, f"tier {tier!r} requires human approval")

    threshold = cfg.approval.context_threshold
    if threshold is not None and sig.context >= threshold:
        return verdict(
            ESCALATE,
            f"context drift: this call sits {sig.context:.2f} away from the stated objective",
        )

    if sig.autonomy >= 1.0:
        return verdict(
            ESCALATE,
            f"autonomy drift: {calls_since_approval} calls since the last human checkpoint",
        )

    if novel:
        return verdict(cfg.default_decision, "tool drift: first observed use of this tool")

    return verdict(cfg.default_decision, "no rule matched; default decision")


def redact(value: Any, keys: list[str], _depth: int = 0) -> Any:
    """Recursively mask values whose key looks like a credential.

    Arguments are written to the audit ledger, and the ledger is WORM in
    production -- a secret that lands there cannot be taken back out.
    """
    if _depth > 12:
        return value
    lowered = [k.lower() for k in keys]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(term in str(k).lower() for term in lowered):
                out[k] = "***redacted***"
            else:
                out[k] = redact(v, keys, _depth + 1)
        return out
    if isinstance(value, list):
        return [redact(v, keys, _depth + 1) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + f"...<truncated {len(value) - 2000} chars>"
    return value
