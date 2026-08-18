"""Transport interface shared by the stdio and Streamable HTTP upstreams.

Both transports are modeled as a full-duplex message pipe: ``send`` pushes one
message toward the server, and everything the server produces arrives on
``inbox``. HTTP is request/response underneath, but normalizing it to a stream
here keeps the proxy free of transport special cases.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from ..jsonrpc import Message


class Upstream(Protocol):
    inbox: asyncio.Queue

    async def start(self) -> None: ...

    async def send(self, message: Message) -> None: ...

    async def close(self) -> None: ...

    def describe(self) -> str: ...
