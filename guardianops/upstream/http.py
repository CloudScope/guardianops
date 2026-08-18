"""Upstream MCP server reached over the Streamable HTTP transport.

Each client message is POSTed to the endpoint. The server may answer with a
single JSON object, with an SSE stream carrying several messages, or with 202
and no body at all (the normal reply to a notification). All three are
normalized onto ``inbox`` so the proxy sees one uniform message stream.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from ..jsonrpc import Message

PROTOCOL_VERSION = "2025-06-18"


class HttpUpstream:
    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.extra_headers = headers or {}
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.session_id: str | None = None
        self._tasks: set[asyncio.Task] = set()

    def describe(self) -> str:
        return f"http:{self.url}"

    async def start(self) -> None:  # nothing to open ahead of the first POST
        return None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        headers.update(self.extra_headers)
        return headers

    async def send(self, message: Message) -> None:
        task = asyncio.create_task(self._post(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _post(self, message: Message) -> None:
        loop = asyncio.get_running_loop()
        try:
            messages = await asyncio.to_thread(self._post_blocking, message)
        except urllib.error.HTTPError as exc:
            await self.inbox.put(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": f"upstream HTTP {exc.code}"},
                }
            )
            return
        except OSError as exc:
            await self.inbox.put(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": f"upstream unreachable: {exc}"},
                }
            )
            return
        for item in messages:
            await self.inbox.put(item)
        del loop

    def _post_blocking(self, message: Message) -> list[Message]:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers=self._headers(), method="POST"
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            # The server assigns a session on initialize; every later request
            # must carry it back.
            session = response.headers.get("Mcp-Session-Id")
            if session:
                self.session_id = session
            if response.status == 202:
                return []
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
            if content_type == "text/event-stream":
                return list(_read_sse(response))
            raw = response.read()
            if not raw.strip():
                return []
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [parsed]

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()


def _read_sse(stream: Any) -> list[Message]:
    """Collect JSON-RPC messages from an SSE body until the stream closes."""
    messages: list[Message] = []
    data_lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    messages.append(payload)
                elif isinstance(payload, list):
                    messages.extend(m for m in payload if isinstance(m, dict))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        try:
            payload = json.loads("\n".join(data_lines))
            if isinstance(payload, dict):
                messages.append(payload)
        except json.JSONDecodeError:
            pass
    return messages
