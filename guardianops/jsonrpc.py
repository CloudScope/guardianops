"""JSON-RPC 2.0 helpers for the MCP wire format.

MCP stdio framing is newline-delimited JSON: exactly one message per line, with
no embedded newlines. Messages move through the proxy as plain dicts so that any
field we do not understand survives the round trip untouched -- a governance
layer that silently drops protocol fields is a governance layer that breaks
every server it has not been tested against.
"""

from __future__ import annotations

import json
from typing import Any

Message = dict[str, Any]

# MCP payloads (file contents, tool results) routinely exceed the 64 KiB default
# that asyncio's StreamReader will accept on a single line.
MAX_LINE = 32 * 1024 * 1024

# JSON-RPC reserved error codes we use.
INVALID_REQUEST = -32600
INTERNAL_ERROR = -32603


def is_request(m: Message) -> bool:
    return "method" in m and "id" in m


def is_notification(m: Message) -> bool:
    return "method" in m and "id" not in m


def is_response(m: Message) -> bool:
    return "method" not in m and ("result" in m or "error" in m)


def encode(m: Message) -> bytes:
    return (json.dumps(m, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> Message | None:
    """Parse one framed line. Returns None for blank lines and malformed JSON."""
    line = line.strip()
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def error_response(mid: Any, code: int, message: str) -> Message:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def tool_error_result(mid: Any, text: str) -> Message:
    """A blocked tool call comes back as a tool-level error, not a transport error.

    The distinction matters: a JSON-RPC error is a broken call the agent will
    likely retry, while an ``isError`` result is feedback the model can read and
    reason about ("I am not entitled to do this, try another approach").
    """
    return {
        "jsonrpc": "2.0",
        "id": mid,
        "result": {"content": [{"type": "text", "text": text}], "isError": True},
    }
