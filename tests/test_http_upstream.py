"""Tests for the Streamable HTTP upstream.

No network: a fake ``urlopen`` returns canned responses, which is enough because
everything interesting here is parsing. The SSE reader is the reason this file
exists -- it is the fiddliest code in the project and had no coverage at all.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardianops.upstream.http import (  # noqa: E402
    PROTOCOL_VERSION,
    HttpUpstream,
    _read_sse,
)


def sse(*chunks: str) -> list[Message]:  # type: ignore[name-defined]
    """Feed a raw SSE body to the reader the way urllib would: as byte lines."""
    body = "".join(chunks)
    return _read_sse(io.BytesIO(body.encode("utf-8")))


class FakeResponse:
    """The subset of http.client.HTTPResponse that HttpUpstream touches."""

    def __init__(self, body: bytes = b"", status: int = 200,
                 content_type: str = "application/json",
                 session_id: str | None = None):
        self.status = status
        self._body = body
        self.headers = {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if session_id:
            self.headers["Mcp-Session-Id"] = session_id
        self._stream = io.BytesIO(body)

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._stream)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestSseReader(unittest.TestCase):
    """One message per event, delimited by a blank line."""

    def test_single_event(self):
        self.assertEqual(
            sse('data: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'),
            [{"jsonrpc": "2.0", "id": 1, "result": {}}],
        )

    def test_several_events(self):
        out = sse(
            'data: {"id":1}\n\n',
            'data: {"id":2}\n\n',
            'data: {"id":3}\n\n',
        )
        self.assertEqual([m["id"] for m in out], [1, 2, 3])

    def test_data_without_a_leading_space(self):
        """"data:{...}" is as legal as "data: {...}"."""
        self.assertEqual(sse('data:{"id":1}\n\n'), [{"id": 1}])

    def test_only_one_leading_space_is_stripped(self):
        """The space after the colon is a delimiter; a second one is content."""
        out = sse('data:  {"id":1}\n\n')
        self.assertEqual(out, [{"id": 1}])

    def test_multi_line_data_is_joined_with_newlines(self):
        out = sse('data: {"id":\n', 'data: 7}\n\n')
        self.assertEqual(out, [{"id": 7}])

    def test_crlf_line_endings(self):
        self.assertEqual(sse('data: {"id":1}\r\n\r\n'), [{"id": 1}])

    def test_comments_are_ignored(self):
        """A bare ":" line is an SSE keep-alive, not data."""
        out = sse(': ping\n', 'data: {"id":1}\n\n', ': ping\n')
        self.assertEqual(out, [{"id": 1}])

    def test_other_fields_are_ignored(self):
        out = sse('event: message\n', 'id: 42\n', 'retry: 100\n',
                  'data: {"id":1}\n\n')
        self.assertEqual(out, [{"id": 1}])

    def test_final_event_without_a_trailing_blank_line(self):
        """Servers that close the stream without a final delimiter are common."""
        self.assertEqual(sse('data: {"id":1}\n'), [{"id": 1}])

    def test_array_payload_is_flattened(self):
        out = sse('data: [{"id":1},{"id":2}]\n\n')
        self.assertEqual([m["id"] for m in out], [1, 2])

    def test_array_payload_in_a_final_unterminated_event(self):
        """The tail path must agree with the delimited path about arrays."""
        out = sse('data: [{"id":1},{"id":2}]\n')
        self.assertEqual([m["id"] for m in out], [1, 2])

    def test_non_dict_members_of_an_array_are_dropped(self):
        out = sse('data: [{"id":1},"garbage",null]\n\n')
        self.assertEqual(out, [{"id": 1}])

    def test_scalar_payload_is_dropped(self):
        self.assertEqual(sse('data: 42\n\n'), [])
        self.assertEqual(sse('data: "hello"\n\n'), [])

    def test_malformed_json_is_skipped_without_losing_later_events(self):
        """A bad event must not poison the stream behind it."""
        out = sse('data: {not json\n\n', 'data: {"id":2}\n\n')
        self.assertEqual(out, [{"id": 2}])

    def test_malformed_json_in_the_tail_is_skipped(self):
        self.assertEqual(sse('data: {"id":1}\n\n', 'data: {broken\n'), [{"id": 1}])

    def test_empty_stream(self):
        self.assertEqual(sse(''), [])

    def test_blank_lines_only(self):
        self.assertEqual(sse('\n\n\n'), [])

    def test_repeated_blank_lines_do_not_emit_empty_messages(self):
        out = sse('data: {"id":1}\n\n\n\n', 'data: {"id":2}\n\n')
        self.assertEqual([m["id"] for m in out], [1, 2])


class TestHeaders(unittest.TestCase):

    def test_defaults_advertise_both_response_shapes(self):
        headers = HttpUpstream("http://x/mcp")._headers()
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("application/json", headers["Accept"])
        self.assertIn("text/event-stream", headers["Accept"])
        self.assertEqual(headers["MCP-Protocol-Version"], PROTOCOL_VERSION)

    def test_session_id_is_absent_until_the_server_assigns_one(self):
        up = HttpUpstream("http://x/mcp")
        self.assertNotIn("Mcp-Session-Id", up._headers())
        up.session_id = "abc"
        self.assertEqual(up._headers()["Mcp-Session-Id"], "abc")

    def test_extra_headers_are_merged_and_win(self):
        up = HttpUpstream("http://x/mcp", {"Authorization": "Bearer t",
                                          "Content-Type": "application/custom"})
        headers = up._headers()
        self.assertEqual(headers["Authorization"], "Bearer t")
        self.assertEqual(headers["Content-Type"], "application/custom")

    def test_describe_names_the_transport_and_endpoint(self):
        self.assertEqual(
            HttpUpstream("http://x/mcp").describe(), "http:http://x/mcp"
        )


class TestPostBlocking(unittest.TestCase):
    """Three legal reply shapes, normalized onto one list of messages."""

    def setUp(self):
        self.up = HttpUpstream("http://x/mcp")

    def _respond(self, response):
        return mock.patch("urllib.request.urlopen", return_value=response)

    def test_single_json_object(self):
        with self._respond(FakeResponse(b'{"jsonrpc":"2.0","id":1,"result":{}}')):
            out = self.up._post_blocking({"id": 1})
        self.assertEqual(out, [{"jsonrpc": "2.0", "id": 1, "result": {}}])

    def test_json_array_is_returned_as_many_messages(self):
        with self._respond(FakeResponse(b'[{"id":1},{"id":2}]')):
            out = self.up._post_blocking({"id": 1})
        self.assertEqual([m["id"] for m in out], [1, 2])

    def test_202_with_no_body_is_the_reply_to_a_notification(self):
        with self._respond(FakeResponse(b"", status=202)):
            self.assertEqual(self.up._post_blocking({"method": "notify"}), [])

    def test_empty_body_yields_nothing(self):
        with self._respond(FakeResponse(b"   \n")):
            self.assertEqual(self.up._post_blocking({"id": 1}), [])

    def test_event_stream_is_parsed_as_sse(self):
        body = b'data: {"id":1}\n\ndata: {"id":2}\n\n'
        with self._respond(FakeResponse(body, content_type="text/event-stream")):
            out = self.up._post_blocking({"id": 1})
        self.assertEqual([m["id"] for m in out], [1, 2])

    def test_content_type_parameters_are_ignored(self):
        """"text/event-stream; charset=utf-8" is still an event stream."""
        body = b'data: {"id":1}\n\n'
        with self._respond(
            FakeResponse(body, content_type="text/event-stream; charset=utf-8")
        ):
            self.assertEqual(self.up._post_blocking({"id": 1}), [{"id": 1}])

    def test_missing_content_type_falls_back_to_json(self):
        with self._respond(FakeResponse(b'{"id":1}', content_type=None)):
            self.assertEqual(self.up._post_blocking({"id": 1}), [{"id": 1}])

    def test_session_id_is_captured_and_then_sent_back(self):
        with self._respond(FakeResponse(b'{"id":1}', session_id="sess-9")):
            self.up._post_blocking({"id": 1})
        self.assertEqual(self.up.session_id, "sess-9")
        self.assertEqual(self.up._headers()["Mcp-Session-Id"], "sess-9")

    def test_a_response_without_a_session_header_keeps_the_existing_one(self):
        self.up.session_id = "sess-1"
        with self._respond(FakeResponse(b'{"id":1}')):
            self.up._post_blocking({"id": 1})
        self.assertEqual(self.up.session_id, "sess-1")

    def test_the_request_carries_the_message_as_compact_json(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["data"] = request.data
            captured["method"] = request.method
            captured["url"] = request.full_url
            return FakeResponse(b'{"id":1}')

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.up._post_blocking({"jsonrpc": "2.0", "id": 1, "method": "ping"})

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "http://x/mcp")
        self.assertNotIn(b", ", captured["data"])  # separators=(",", ":")
        self.assertEqual(json.loads(captured["data"])["method"], "ping")


class TestPostErrorMapping(unittest.IsolatedAsyncioTestCase):
    """A transport failure must reach the client as JSON-RPC, not as silence."""

    def setUp(self):
        self.up = HttpUpstream("http://x/mcp")

    async def test_http_error_becomes_a_jsonrpc_error(self):
        err = urllib.error.HTTPError("http://x/mcp", 503, "no", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            await self.up._post({"id": 7})
        message = self.up.inbox.get_nowait()
        self.assertEqual(message["id"], 7)
        self.assertEqual(message["error"]["code"], -32000)
        self.assertIn("503", message["error"]["message"])

    async def test_unreachable_host_becomes_a_jsonrpc_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            await self.up._post({"id": 8})
        message = self.up.inbox.get_nowait()
        self.assertEqual(message["id"], 8)
        self.assertIn("unreachable", message["error"]["message"])

    async def test_the_error_carries_the_id_so_the_client_can_correlate(self):
        """A notification has no id, and the error must still be well formed."""
        with mock.patch("urllib.request.urlopen", side_effect=OSError("x")):
            await self.up._post({"method": "notify"})
        message = self.up.inbox.get_nowait()
        self.assertIsNone(message["id"])
        self.assertEqual(message["jsonrpc"], "2.0")

    async def test_every_parsed_message_reaches_the_inbox(self):
        body = b'data: {"id":1}\n\ndata: {"id":2}\n\n'
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(body, content_type="text/event-stream")):
            await self.up._post({"id": 1})
        self.assertEqual(self.up.inbox.qsize(), 2)


class TestTaskLifecycle(unittest.IsolatedAsyncioTestCase):

    async def test_send_does_not_block_on_the_response(self):
        """send() hands off to a task; the reply arrives on the inbox later."""
        up = HttpUpstream("http://x/mcp")
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b'{"id":1}')):
            await up.send({"id": 1})
            self.assertEqual(up.inbox.qsize(), 0, "send must not wait for a reply")
            message = await asyncio.wait_for(up.inbox.get(), timeout=5)
        self.assertEqual(message["id"], 1)

    async def test_finished_tasks_are_not_retained(self):
        up = HttpUpstream("http://x/mcp")
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b'{"id":1}')):
            await up.send({"id": 1})
            await asyncio.wait_for(up.inbox.get(), timeout=2)
        await asyncio.sleep(0)
        self.assertEqual(len(up._tasks), 0)

    async def test_close_cancels_work_still_in_flight(self):
        up = HttpUpstream("http://x/mcp")

        def blocking_urlopen(request, timeout=None):
            import time
            time.sleep(5)
            return FakeResponse(b'{"id":1}')

        with mock.patch("urllib.request.urlopen", blocking_urlopen):
            await up.send({"id": 1})
            self.assertEqual(len(up._tasks), 1)
            await up.close()
            self.assertTrue(all(t.cancelled() or t.cancelling() for t in up._tasks))

    async def test_start_is_a_no_op(self):
        """Nothing is opened before the first POST."""
        up = HttpUpstream("http://x/mcp")
        self.assertIsNone(await up.start())


if __name__ == "__main__":
    unittest.main()
