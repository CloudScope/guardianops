"""Upstream MCP server launched as a child process, spoken to over stdio."""

from __future__ import annotations

import asyncio
import shlex
import sys

from .. import jsonrpc
from ..jsonrpc import Message


class StdioUpstream:
    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.proc: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task] = []

    def describe(self) -> str:
        return "stdio:" + shlex.join(self.argv)

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=jsonrpc.MAX_LINE,
        )
        self._tasks.append(asyncio.create_task(self._pump_stdout()))
        self._tasks.append(asyncio.create_task(self._pump_stderr()))

    async def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                message = jsonrpc.decode(line)
                if message is not None:
                    await self.inbox.put(message)
        except (asyncio.CancelledError, ValueError):
            pass
        finally:
            # Sentinel: the server is gone, so the proxy should wind down too.
            await self.inbox.put(None)

    async def _pump_stderr(self) -> None:
        """MCP servers log on stderr. Relay it so operators keep their diagnostics."""
        assert self.proc and self.proc.stderr
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
                sys.stderr.buffer.write(b"[upstream] " + line)
                sys.stderr.buffer.flush()
        except (asyncio.CancelledError, ValueError):
            pass

    async def send(self, message: Message) -> None:
        if not self.proc or not self.proc.stdin:
            return
        self.proc.stdin.write(jsonrpc.encode(message))
        await self.proc.stdin.drain()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
