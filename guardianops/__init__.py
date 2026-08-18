"""GuardianOps -- runtime governance for autonomous AI agents and MCP ecosystems.

P0 scope: an inline MCP proxy that traces every message, pins tool definitions,
enforces entitlements, and holds high-risk calls for human approval.
"""

__version__ = "0.1.0"

# The policy engine is the part worth importing directly: everything else is
# either transport or a CLI concern. Re-exported here so a library user writes
# `from guardianops import Config` rather than reaching into a submodule.
from .policy import (  # noqa: E402
    ALLOW,
    BLOCK,
    ERROR,
    ESCALATE,
    ENFORCE,
    SHADOW,
    WARNING,
    Config,
    Finding,
    Signals,
    Verdict,
    evaluate,
    redact,
)

__all__ = [
    "__version__",
    "ALLOW",
    "BLOCK",
    "ERROR",
    "ESCALATE",
    "ENFORCE",
    "SHADOW",
    "WARNING",
    "Config",
    "Finding",
    "Signals",
    "Verdict",
    "evaluate",
    "redact",
]
