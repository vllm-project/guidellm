"""Integration tests for OpenAIHTTPBackend request timing over real HTTP."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from unittest.mock import patch

import httpx
import pytest

from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.schemas import GenerationRequest, RequestInfo
from guidellm.schemas.backends import OpenAIHTTPBackendArgs


class _LocalChatServer:
    """Keep-alive chat completions server that records connections and arrivals."""

    def __init__(self, delay: float = 0.0, hold_open: bool = False):
        self.delay = delay
        self.hold_open = hold_open
        self.connections = 0
        self.received: list[float] = []
        self.paths: list[str] = []
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> _LocalChatServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.received.append(time.time())
                request_line, *header_lines = head.decode().split("\r\n")
                path = request_line.split(" ")[1]
                self.paths.append(path)
                headers = dict(
                    line.lower().split(": ", 1) for line in header_lines if ": " in line
                )
                length = int(headers.get("content-length", "0"))
                body = json.loads(await reader.readexactly(length) or b"{}")
                if path.startswith("/redirect/"):
                    writer.write(
                        b"HTTP/1.1 307 Temporary Redirect\r\n"
                        + f"Location: {path.removeprefix('/redirect')}\r\n".encode()
                        + b"Content-Length: 0\r\n\r\n"
                    )
                elif body.get("stream"):
                    await asyncio.sleep(self.delay)
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                        b"Transfer-Encoding: chunked\r\n\r\n"
                    )
                    for event in (
                        b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
                        b"data: [DONE]\n\n",
                    ):
                        writer.write(b"%x\r\n%s\r\n" % (len(event), event))
                    if self.hold_open:
                        await writer.drain()
                        await reader.read()
                        return
                    writer.write(b"0\r\n\r\n")
                else:
                    await asyncio.sleep(self.delay)
                    payload = b'{"choices": [{"message": {"content": "Hi"}}]}'
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                        + payload
                    )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        finally:
            writer.close()


class _DelayedTransport(httpx.AsyncHTTPTransport):
    """Transport that waits before handing each request to httpcore."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.delay)
        return await super().handle_async_request(request)


def _make_backend(target: str, stream: bool = True) -> OpenAIHTTPBackend:
    return OpenAIHTTPBackend(
        OpenAIHTTPBackendArgs(target=target, model="test-model", stream=stream)
    )


async def _resolve_once(
    backend: OpenAIHTTPBackend, request_info: RequestInfo | None = None
) -> None:
    request = GenerationRequest(columns={"text_column": ["Hello"]})
    async for _ in backend.resolve(request, request_info or RequestInfo()):
        pass


async def _resolve_all(
    backend: OpenAIHTTPBackend,
    request_info: RequestInfo,
    transport: httpx.AsyncBaseTransport | None = None,
) -> float:
    """Resolve one request and return the time resolve was called."""
    await backend.process_startup()
    if transport is not None:
        assert backend._async_client is not None
        await backend._async_client.aclose()
        backend._async_client = httpx.AsyncClient(transport=transport)
    try:
        called = time.time()
        await _resolve_once(backend, request_info)
    finally:
        await backend.process_shutdown()
    return called


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("stream", [True, False])
async def test_request_start_excludes_wait_before_sending(stream: bool):
    """Exclude a client-side wait before the headers are written from request time.

    ## WRITTEN BY AI ##
    """
    wait = 0.5
    async with _LocalChatServer(delay=0.02) as server:
        request_info = RequestInfo(request_id="test-id")
        called = await _resolve_all(
            _make_backend(server.url, stream), request_info, _DelayedTransport(wait)
        )

    timings = request_info.timings
    assert timings.request_start is not None
    assert timings.request_start - called >= 0.9 * wait
    assert timings.request_start <= server.received[0]
    response_time = timings.first_token_iteration if stream else timings.request_end
    assert response_time is not None
    assert response_time - timings.request_start < wait


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_request_start_kept_across_redirect():
    """Use the first request's start when the server redirects.

    ## WRITTEN BY AI ##
    """
    async with _LocalChatServer() as server:
        request_info = RequestInfo(request_id="test-id")
        await _resolve_all(_make_backend(f"{server.url}/redirect"), request_info)

    assert len(server.received) == 2
    assert request_info.timings.request_start is not None
    assert request_info.timings.request_start <= server.received[0]


@pytest.mark.sanity
@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_request_start_set_when_connection_fails():
    """Set ``request_start`` even when the request fails before it is sent.

    ## WRITTEN BY AI ##
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    request_info = RequestInfo(request_id="test-id")

    with pytest.raises(httpx.ConnectError):
        await _resolve_all(_make_backend(f"http://127.0.0.1:{port}"), request_info)

    assert request_info.timings.request_start is not None


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("concurrency", [1, 4])
async def test_streaming_requests_reuse_connections(concurrency: int):
    """Reuse one connection per concurrent stream across sequential requests.

    ## WRITTEN BY AI ##
    """
    async with _LocalChatServer() as server:
        backend = _make_backend(server.url)
        await backend.process_startup()
        try:
            for _ in range(3):
                await asyncio.gather(
                    *(_resolve_once(backend) for _ in range(concurrency))
                )
        finally:
            await backend.process_shutdown()

    assert len(server.received) == 3 * concurrency
    assert server.connections == concurrency


@pytest.mark.sanity
@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_response_held_open_after_final_event():
    """Stop reading a response held open after its final event.

    ## WRITTEN BY AI ##
    """
    async with _LocalChatServer(hold_open=True) as server:
        backend = _make_backend(server.url)
        request_info = RequestInfo(request_id="test-id")
        with patch("guidellm.backends.openai.http._READ_TO_END_TIMEOUT", 0.2):
            started = time.time()
            await _resolve_all(backend, request_info)
            elapsed = time.time() - started

    timings = request_info.timings
    assert 0.2 <= elapsed < 1.0
    assert timings.request_end is not None
    assert timings.last_token_iteration is not None
    assert timings.request_end - timings.last_token_iteration < 0.2


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_requests_use_environment_proxy(monkeypatch: pytest.MonkeyPatch):
    """Send requests through the proxy set in the environment.

    ## WRITTEN BY AI ##
    """
    async with _LocalChatServer() as proxy:
        monkeypatch.setenv("http_proxy", proxy.url)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        backend = _make_backend("http://guidellm.invalid")
        await backend.process_startup()
        try:
            await _resolve_once(backend)
        finally:
            await backend.process_shutdown()

    assert len(proxy.paths) == 1
    assert proxy.paths[0].startswith("http://guidellm.invalid/")
