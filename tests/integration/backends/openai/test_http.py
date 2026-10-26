"""Integration tests for OpenAIHTTPBackend request timing over real HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import time
from unittest.mock import patch

import httpx2
import pytest

from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)
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


class _LocalSpeechServer:
    """Serve deterministic speech responses over a real local HTTP connection."""

    def __init__(self, *, sse: bool):
        """Initialize one local speech server.

        :param sse: Whether the server should return speech SSE events.
        """
        self.sse = sse
        self.audio = b"\x00\x00" * 2400
        self.request_path: str | None = None
        self.request_body: dict[str, object] | None = None
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> _LocalSpeechServer:
        """Start the local server and return this test fixture."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the local listening socket."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        """Return this server's base URL."""
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Read one request and return the configured speech response.

        :param reader: Stream for reading the HTTP request.
        :param writer: Stream for writing the HTTP response.
        :return: None.
        """
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            request_line, *header_lines = head.decode("latin-1").split("\r\n")
            self.request_path = request_line.split(" ")[1]
            headers = dict(
                line.lower().split(": ", 1) for line in header_lines if ": " in line
            )
            body_size = int(headers.get("content-length", "0"))
            self.request_body = json.loads(await reader.readexactly(body_size))

            if self.sse:
                await self._send_sse(writer)
            else:
                await self._send_headerless_pcm(writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        finally:
            writer.close()

    async def _send_headerless_pcm(self, writer: asyncio.StreamWriter) -> None:
        """Return complete PCM without sample-rate or channel headers.

        :param writer: Stream for writing the HTTP response.
        :return: None.
        """
        headers = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: audio/pcm\r\n"
            + f"Content-Length: {len(self.audio)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
        )
        writer.write(headers + self.audio)
        await writer.drain()

    async def _send_sse(self, writer: asyncio.StreamWriter) -> None:
        """Return one layout-bearing audio delta and its terminal usage event.

        :param writer: Stream for writing the HTTP response.
        :return: None.
        """
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
        )
        await writer.drain()

        delta = {
            "type": "speech.audio.delta",
            "audio": base64.b64encode(self.audio).decode("ascii"),
            "response_format": "pcm",
            "sample_rate": 24000,
            "channels": 1,
        }
        done = {
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
                "input_token_details": {
                    "text_tokens": 3,
                    "audio_tokens": 4,
                },
            },
        }
        events = (
            "event: speech.audio.delta\ndata: " + json.dumps(delta) + "\n\n",
            "event: speech.audio.done\ndata: " + json.dumps(done) + "\n\n",
        )

        for index, event in enumerate(events):
            encoded = event.encode("utf-8")
            if index == 0:
                split_at = len(encoded) // 2
                parts = (encoded[:split_at], encoded[split_at:])
            else:
                parts = (encoded,)

            for part in parts:
                writer.write(f"{len(part):X}\r\n".encode("ascii") + part + b"\r\n")
                await writer.drain()

        writer.write(b"0\r\n\r\n")
        await writer.drain()


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("mode", ["non_streaming_headerless", "sse"])
async def test_speech_http_handles_headerless_pcm_and_sse(mode: str):
    """Resolve headerless non-streaming PCM and a complete SSE response over HTTP.

    :param mode: Headerless non-streaming PCM or SSE response mode.
    :return: None.
    ## WRITTEN BY AI ##
    """
    sse = mode == "sse"
    async with _LocalSpeechServer(sse=sse) as server:
        body = {"response_format": "pcm"}
        if sse:
            body["stream_format"] = "sse"

        backend = OpenAIHTTPBackend(
            OpenAIHTTPBackendArgs(
                target=server.url,
                model="test-speech-model",
                request_format="/v1/audio/speech",
                stream=sse,
                validate_backend=False,
                extras=GenerationRequestArguments(body=body),
            )
        )
        request = GenerationRequest(columns={"text_column": ["Speak this."]})
        request_info = RequestInfo(request_id="speech-http-test")
        responses: list[GenerationResponse] = []

        await backend.process_startup()
        try:
            async for response, _ in backend.resolve(request, request_info):
                if response is not None:
                    responses.append(response)
        finally:
            await backend.process_shutdown()

    assert server.request_path == "/v1/audio/speech"
    assert server.request_body is not None
    assert server.request_body["response_format"] == "pcm"
    assert len(responses) == 1

    response = responses[0]
    assert response.output_media[0]["data"] == server.audio
    assert response.output_media[0]["format"] == "pcm"
    assert response.output_metrics.audio_bytes == len(server.audio)

    if sse:
        assert server.request_body["stream_format"] == "sse"
        assert server.request_body["stream"] is True
        assert response.output_metrics.audio_samples == 2400
        assert response.output_metrics.audio_seconds == pytest.approx(0.1)
        assert response.input_metrics.text_tokens == 3
        assert response.input_metrics.audio_tokens == 4
        assert response.output_metrics.audio_tokens == 2
        assert request_info.timings.first_audio_iteration is not None
    else:
        assert "stream" not in server.request_body
        assert response.output_metrics.audio_samples is None
        assert response.output_metrics.audio_seconds is None


class _DelayedTransport(httpx2.AsyncHTTPTransport):
    """Transport that waits before handing each request to httpcore."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
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
    transport: httpx2.AsyncBaseTransport | None = None,
) -> float:
    """Resolve one request and return the time resolve was called."""
    await backend.process_startup()
    if transport is not None:
        assert backend._async_client is not None
        await backend._async_client.aclose()
        backend._async_client = httpx2.AsyncClient(transport=transport)
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

    with pytest.raises(httpx2.ConnectError):
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
    """Return before reading a held-open response, and stop reading at shutdown.

    ## WRITTEN BY AI ##
    """
    async with _LocalChatServer(hold_open=True) as server:
        backend = _make_backend(server.url)
        request_info = RequestInfo(request_id="test-id")
        with patch("guidellm.backends.openai.http._READ_TO_END_TIMEOUT", 0.5):
            await backend.process_startup()
            started = time.time()
            await _resolve_once(backend, request_info)
            resolved = time.time() - started
            await backend.process_shutdown()
            shut_down = time.time() - started

    timings = request_info.timings
    assert resolved < 0.3
    assert 0.5 <= shut_down < 1.5
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
