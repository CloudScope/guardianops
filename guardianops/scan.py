"""Content scanning for the two places MCP mixes instructions with data.

MCP hands the model two kinds of text that look like data and read like
instructions: a tool's *description*, which enters the context window verbatim,
and a tool's *result*, which comes back from outside and is trusted the same
way. Pinning covers neither -- it detects that a description *changed*, so a
server hostile on first contact is pinned silently, and it never looks at
results at all.

This is pattern matching, and pattern matching against natural language is a
tripwire, not a barrier. It catches the careless and the reused, misses the
novel and the paraphrased, and will occasionally flag a legitimate string that
happens to read like an instruction. It is worth having anyway because the
alternative is nothing, but a finding is evidence, not proof, and the default
action for both scanners is deliberately not "block".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Actions, mirroring the ones policy already uses where they overlap.
BLOCK = "block"
SANITIZE = "sanitize"
WARN = "warn"
LOG = "log"

REDACTION = "[redacted by guardianops]"

# Text that tries to steer the model rather than describe a tool. Kept short and
# defensible on purpose: every pattern here has been seen in a published
# tool-poisoning writeup, and a long speculative list would only trade misses
# for false positives.
DEFAULT_DEFINITION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(the\s+)?(previous|prior|above|system)",
    r"(do\s?n[o']?t|never)\s+(tell|inform|mention\s+to|reveal\s+to)\s+the\s+user",
    r"before\s+(using|calling|invoking)\s+this\s+tool[^.]{0,60}\bread\b",
    r"\.(aws/credentials|ssh/id_rsa|env)\b",
    # Exfiltration reads as "<something secret> ... <move it somewhere>", in
    # either order. Written as two patterns rather than one because the halves
    # appear both ways round and a single alternation would match neither well.
    r"\b(credential|credentials|secret|secrets|token|api[_\s-]?key|private[_\s-]?key|password)\b"
    r"[^.]{0,60}\b(include|send|return|append|attach|add|forward|reveal)\b",
    r"\b(include|append|attach|forward|send)\b[^.]{0,60}"
    r"\b(credential|credentials|secret|secrets|token|api[_\s-]?key|password)\b",
    r"you\s+are\s+now\s+(a|an|the)\b",
    r"<\s*(system|important|secret)\s*>",
]

# Applied to what a server sends back. Injection arriving in a tool result is
# the same attack as injection in a description, one hop later.
DEFAULT_RESPONSE_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(the\s+)?(previous|prior|above|system)",
    r"new\s+(system\s+)?instructions?\s*:",
    r"(do\s?n[o']?t|never)\s+(tell|inform)\s+the\s+user",
    r"</?\s*(system|assistant)\s*>",
    r"\.(aws/credentials|ssh/id_rsa)\b",
]

# Characters with no business in a tool description: zero-width and directional
# overrides are how an instruction is hidden from a human reviewing the same
# text the model reads.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")


@dataclass
class ScanConfig:
    """Two scanners, configured separately because they differ in blast radius.

    Blocking a suspicious *definition* costs one tool. Blocking a suspicious
    *response* can break a working agent on a false positive, which is why
    ``responses`` defaults to sanitising rather than blocking.
    """

    definitions: str = WARN            # "block" | "warn" | "log" | ""
    responses: str = SANITIZE          # "block" | "sanitize" | "log" | ""
    definition_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_DEFINITION_PATTERNS)
    )
    response_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_RESPONSE_PATTERNS)
    )
    # Responses can be large. Scanning the whole of a 10 MB file read costs more
    # than it finds; injection that works has to appear where the model reads.
    max_response_bytes: int = 262_144

    @staticmethod
    def parse(raw: dict[str, Any]) -> "ScanConfig":
        cfg = ScanConfig()
        if not raw:
            return cfg
        cfg.definitions = raw.get("definitions", cfg.definitions)
        cfg.responses = raw.get("responses", cfg.responses)
        # An explicit null is a placeholder, not an instruction to drop the
        # defaults and leave the scanner with nothing to match.
        if raw.get("definitionPatterns") is not None:
            cfg.definition_patterns = raw["definitionPatterns"]
        if raw.get("responsePatterns") is not None:
            cfg.response_patterns = raw["responsePatterns"]
        cfg.max_response_bytes = raw.get("maxResponseBytes", cfg.max_response_bytes)
        return cfg


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE | re.DOTALL))
        except re.error:
            # A typo in config must not take the proxy down. The pattern is
            # dropped and the rest still apply.
            continue
    return out


def scan_text(text: str, patterns: list[str]) -> list[str]:
    """Every pattern the text matches, as human-readable findings."""
    findings: list[str] = []
    for rx in _compile(patterns):
        m = rx.search(text)
        if m:
            findings.append(f"matched {rx.pattern!r} at {m.start()}")
    if _INVISIBLE.search(text):
        findings.append("contains zero-width or bidirectional control characters")
    return findings


def scan_definition(tool: dict[str, Any], patterns: list[str]) -> list[str]:
    """Scan the parts of a tool definition the model actually reads.

    Argument descriptions count: a clean top-level description with an
    instruction buried in a property description reaches the model just the
    same.
    """
    parts = [str(tool.get("description", ""))]
    schema = tool.get("inputSchema") or {}
    for prop in (schema.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("description"):
            parts.append(str(prop["description"]))
    return scan_text("\n".join(parts), patterns)


def sanitize(text: str, patterns: list[str]) -> tuple[str, int]:
    """Blank out matching spans, leaving the rest of the result usable.

    Returns the cleaned text and how many spans were removed. Sanitising beats
    blocking for responses: the agent still gets its file listing, minus the
    sentence telling it to email your credentials somewhere.
    """
    removed = 0
    for rx in _compile(patterns):
        text, n = rx.subn(REDACTION, text)
        removed += n
    text, n = _INVISIBLE.subn("", text)
    return text, removed + n
