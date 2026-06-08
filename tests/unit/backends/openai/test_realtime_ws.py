"""Tests for OpenAIWebSocketBackend.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from pydantic import ValidationError
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from guidellm.backends.backend import Backend
from guidellm.backends.openai.websocket import (
    OpenAIWebSocketBackend,
    OpenAIWebSocketBackendArgs,
)
from guidellm.schemas import GenerationRequest, RequestInfo, RequestTimings


@pytest.fixture(autouse=True)
def _patch_pcm16_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid torchcodec decode in unit tests; handler chunks audio in format()."""
    monkeypatch.setattr(
        "guidellm.backends.openai.request_handlers.pcm16_append_b64_chunks",
        lambda *a, **k: ["YWFhYQ=="],
    )


def _make_ws_backend(**kwargs: Any) -> OpenAIWebSocketBackend:
    return OpenAIWebSocketBackend(OpenAIWebSocketBackendArgs(**kwargs))


async def _bounded_ws_recv(ws: object, *, timeout: float = 5.0) -> None:
    """Recv once with a cap so stub handlers never block ``serve()`` teardown."""
    with contextlib.suppress(asyncio.TimeoutError, ConnectionClosed):
        await asyncio.wait_for(ws.recv(), timeout=timeout)


@pytest.mark.asyncio
async def test_resolve_streams_deltas_and_done() -> None:
    """Fake server speaks vLLM-style realtime events; PCM path is patched."""

    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        commits: list[bool | None] = []
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit":
                commits.append(data.get("final"))
                if data.get("final"):
                    break
        assert commits == [False, True]
        await ws.send(json.dumps({"type": "transcription.delta", "delta": "hi"}))
        await ws.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": "hi",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            )
        )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="test-model",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={
                "audio_column": [
                    {"audio": b"fake", "format": "mp3", "file_name": "f.mp3"}
                ]
            },
        )
        info = RequestInfo(timings=RequestTimings())
        out: list = []
        async for item in be.resolve(req, info):
            out.append(item)
        await be.process_shutdown()

    assert len(out) == 2
    assert out[0][0] is None
    final_resp, _ = out[1]
    assert final_resp.text == "hi"
    assert final_resp.input_metrics.audio_tokens == 5
    assert final_resp.output_metrics.text_tokens == 1


@pytest.mark.asyncio
async def test_empty_delta_does_not_count_as_ignored_event() -> None:
    """Empty transcription.delta updates request iteration without token timing.

    ## WRITTEN BY AI ##
    """

    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send(json.dumps({"type": "transcription.delta", "delta": ""}))
        await ws.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": "ok",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
        )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="test-model",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"fake"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        out: list = []
        async for item in be.resolve(req, info):
            out.append(item)
        await be.process_shutdown()

    assert len(out) == 2
    _, final_info = out[1]
    assert final_info.timings.request_iterations >= 1
    assert final_info.timings.token_iterations == 1


@pytest.mark.asyncio
async def test_done_without_deltas_sets_first_token_and_prefetch_yield() -> None:
    """Only ``transcription.done`` (no deltas): TTFT and two yields match delta path."""

    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        commits: list[bool | None] = []
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit":
                commits.append(data.get("final"))
                if data.get("final"):
                    break
        assert commits == [False, True]
        await ws.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": "only-done",
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 7,
                        "total_tokens": 9,
                    },
                }
            )
        )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="test-model",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={
                "audio_column": [
                    {"audio": b"fake", "format": "mp3", "file_name": "f.mp3"}
                ]
            },
        )
        info = RequestInfo(timings=RequestTimings())
        out: list = []
        async for item in be.resolve(req, info):
            out.append(item)
        await be.process_shutdown()

    assert len(out) == 2
    prefetch, prefetch_info = out[0]
    assert prefetch is None
    assert prefetch_info.timings.first_token_iteration is not None
    assert prefetch_info.timings.last_token_iteration is not None
    assert prefetch_info.timings.token_iterations == 1
    final_resp, final_info = out[1]
    assert final_resp.text == "only-done"
    assert final_resp.input_metrics.audio_tokens == 2
    assert final_resp.output_metrics.text_tokens == 7
    assert final_info.timings.request_end is not None


@pytest.mark.asyncio
async def test_server_error_event_raises() -> None:
    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and not data.get(
                "final"
            ):
                await ws.send(
                    json.dumps({"type": "error", "error": "bad", "code": "e1"})
                )
                await _bounded_ws_recv(ws)
                return

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="bad"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_first_message_error_event_raises() -> None:
    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "error", "error": "auth failed"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="auth failed"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_first_message_not_session_created_raises() -> None:
    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "unexpected.ping"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="session.created"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_invalid_json_from_server_raises() -> None:
    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send("{not-json")

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_resolve_requires_process_startup() -> None:
    be = _make_ws_backend(
        target="http://127.0.0.1:9",
        model="m",
        validate_backend=False,
    )
    req = GenerationRequest(
        request_id="r1",
        columns={"audio_column": [{"audio": b"x"}]},
    )
    info = RequestInfo(timings=RequestTimings())
    with pytest.raises(RuntimeError, match="started"):
        async for _ in be.resolve(req, info):
            pass


@pytest.mark.asyncio
async def test_resolve_rejects_history() -> None:
    be = _make_ws_backend(
        target="http://127.0.0.1:9",
        model="m",
        validate_backend=False,
    )
    await be.process_startup()
    prev = GenerationRequest(request_id="prev", columns={})
    req = GenerationRequest(
        request_id="r1",
        columns={"audio_column": [{"audio": b"x"}]},
    )
    info = RequestInfo(timings=RequestTimings())
    with pytest.raises(NotImplementedError, match="history"):
        async for _ in be.resolve(req, info, history=[(prev, None)]):
            pass
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_resolve_rejects_wrong_audio_column_count() -> None:
    be = _make_ws_backend(
        target="http://127.0.0.1:9",
        model="m",
        validate_backend=False,
    )
    await be.process_startup()
    info = RequestInfo(timings=RequestTimings())

    async def drain(req: GenerationRequest) -> None:
        async for _ in be.resolve(req, info):
            pass

    req_empty = GenerationRequest(request_id="r1", columns={"audio_column": []})
    with pytest.raises(ValueError, match="exactly one"):
        await drain(req_empty)
    req_two = GenerationRequest(
        request_id="r2",
        columns={"audio_column": [{"audio": b"a"}, {"audio": b"b"}]},
    )
    with pytest.raises(ValueError, match="exactly one"):
        await drain(req_two)
    await be.process_shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(45)
async def test_resolve_cancelled_after_delta_yields_partial_then_reraises() -> None:
    delta_seen = asyncio.Event()

    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "session.created", "id": "s", "created": 0}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send(json.dumps({"type": "transcription.delta", "delta": "partial"}))
        delta_seen.set()
        await _bounded_ws_recv(ws)

    results: list = []
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())

        async def collect() -> None:
            async for item in be.resolve(req, info):
                results.append(item)

        task = asyncio.create_task(collect())
        await asyncio.wait_for(delta_seen.wait(), timeout=5.0)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await be.process_shutdown()

    assert len(results) == 2
    assert results[0][0] is None
    assert results[1][0] is not None
    assert results[1][0].text == "partial"


@pytest.mark.asyncio
async def test_non_object_json_after_handshake_raises() -> None:
    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "session.created", "id": "s", "created": 0}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send("[]")

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="JSON object"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_excessive_ignored_events_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guidellm.backends.openai.websocket._MAX_IGNORED_WS_EVENT_TYPES",
        2,
    )

    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "session.created", "id": "s", "created": 0}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        for _ in range(10):
            await ws.send(json.dumps({"type": "noise.event"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="Exceeded maximum"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


@pytest.mark.asyncio
async def test_available_models_parses_response(httpx_mock: object) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:9/v1/models",
        json={"data": [{"id": "a"}, {"id": "b"}]},
    )
    be = _make_ws_backend(
        target="http://127.0.0.1:9",
        validate_backend=False,
    )
    await be.process_startup()
    assert await be.available_models() == ["a", "b"]
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_resolve_raises_when_no_model_and_empty_catalog(
    httpx_mock: object,
) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:9/v1/models",
        json={"data": []},
    )
    be = _make_ws_backend(
        target="http://127.0.0.1:9",
        model="",
        validate_backend=False,
    )
    await be.process_startup()
    req = GenerationRequest(
        request_id="r1",
        columns={"audio_column": [{"audio": b"x"}]},
    )
    info = RequestInfo(timings=RequestTimings())
    with pytest.raises(RuntimeError, match="No model configured"):
        async for _ in be.resolve(req, info):
            pass
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_resolve_invalid_ws_target_url_raises() -> None:
    be = _make_ws_backend(
        target="",
        model="m",
        validate_backend=False,
    )
    await be.process_startup()
    req = GenerationRequest(
        request_id="r1",
        columns={"audio_column": [{"audio": b"x"}]},
    )
    info = RequestInfo(timings=RequestTimings())
    with pytest.raises(ValueError, match="Invalid target"):
        async for _ in be.resolve(req, info):
            pass
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_error_event_dict_formatted_message() -> None:
    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "error": {"message": "auth failed", "code": "401"},
                }
            )
        )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = _make_ws_backend(
            target=f"http://127.0.0.1:{port}",
            model="m",
            validate_backend=False,
        )
        await be.process_startup()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        info = RequestInfo(timings=RequestTimings())
        with pytest.raises(RuntimeError, match="401"):
            async for _ in be.resolve(req, info):
                pass
        await be.process_shutdown()


def test_openai_websocket_backend_args_model() -> None:
    """## WRITTEN BY AI ##"""
    a = OpenAIWebSocketBackendArgs(target="http://localhost:8000", model="x")
    assert a.request_format == "/v1/realtime"
    assert a.chunk_samples == 3200
    assert a.timeout is None


def test_openai_websocket_backend_args_rejects_non_path_request_format() -> None:
    """## WRITTEN BY AI ##"""
    with pytest.raises(ValidationError):
        OpenAIWebSocketBackendArgs(
            target="http://localhost:8000",
            request_format="realtime",
        )


def test_openai_websocket_backend_args_accepts_explicit_v1_realtime() -> None:
    """## WRITTEN BY AI ##"""
    args = OpenAIWebSocketBackendArgs(
        target="http://localhost:8000",
        request_format="/v1/realtime",
    )
    assert args.request_format == "/v1/realtime"


def test_openai_websocket_backend_resolves_websocket_path_from_request_format() -> None:
    """## WRITTEN BY AI ##"""
    backend = Backend.create(
        OpenAIWebSocketBackendArgs(
            target="http://127.0.0.1:9",
            request_format="/v1/realtime",
        )
    )
    assert backend.websocket_path == "/v1/realtime"


def test_openai_websocket_backend_args_invalid_request_format_rejected() -> None:
    """## WRITTEN BY AI ##"""
    with pytest.raises(ValidationError):
        OpenAIWebSocketBackendArgs(
            target="http://localhost:8000",
            request_format="nope",
        )
