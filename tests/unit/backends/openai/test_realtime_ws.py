"""Tests for OpenAIWebSocketBackend.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
from pydantic import ValidationError

try:
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed
except ImportError:
    pytest.skip(
        "websockets not installed; install guidellm[audio] for realtime tests",
        allow_module_level=True,
    )

from guidellm.backends.backend import Backend
from guidellm.backends.openai.websocket import (
    _DEFAULT_WS_RECV_TIMEOUT,
    OpenAIWebSocketBackend,
    OpenAIWebSocketBackendArgs,
)
from guidellm.schemas import GenerationRequest, RequestInfo, RequestTimings


async def _bounded_ws_recv(ws: object, *, timeout: float = 5.0) -> None:
    """Recv once with a cap so stub handlers never block ``serve()`` teardown."""
    with contextlib.suppress(asyncio.TimeoutError, ConnectionClosed):
        await asyncio.wait_for(ws.recv(), timeout=timeout)


@pytest.mark.asyncio
async def test_resolve_streams_deltas_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YWFhYQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_transcription_done_without_deltas_sets_first_token_and_prefetch_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YWFhYQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_transcription_done_usage_string_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """String token counts in usage should still feed AudioRequestHandler metrics."""

    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": "sess-x", "created": 0})
        )
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send(json.dumps({"type": "transcription.delta", "delta": "x"}))
        await ws.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": "x",
                    "usage": {
                        "prompt_tokens": "12",
                        "completion_tokens": "3",
                        "total_tokens": "15",
                    },
                }
            )
        )

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YWFhYQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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

    final_resp, _ = out[1]
    assert final_resp.input_metrics.audio_tokens == 12
    assert final_resp.output_metrics.text_tokens == 3


@pytest.mark.asyncio
async def test_server_error_event_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_first_message_error_event_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "error", "error": "auth failed"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_first_message_not_session_created_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "unexpected.ping"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_invalid_json_from_server_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
    be = OpenAIWebSocketBackend(
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
    be = OpenAIWebSocketBackend(
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
    be = OpenAIWebSocketBackend(
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
async def test_resolve_cancelled_after_delta_yields_partial_then_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    results: list = []
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
async def test_non_object_json_after_handshake_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(ws: object) -> None:
        await ws.send(json.dumps({"type": "session.created", "id": "s", "created": 0}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg if isinstance(msg, str) else msg.decode())
            if data.get("type") == "input_audio_buffer.commit" and data.get("final"):
                break
        await ws.send("[]")

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
    be = OpenAIWebSocketBackend(
        target="http://127.0.0.1:9",
        validate_backend=False,
    )
    await be.process_startup()
    assert await be.available_models() == ["a", "b"]
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_available_models_bad_data_shape_raises(httpx_mock: object) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:9/v1/models",
        json={"data": "not-a-list"},
    )
    be = OpenAIWebSocketBackend(
        target="http://127.0.0.1:9",
        validate_backend=False,
    )
    await be.process_startup()
    with pytest.raises(RuntimeError, match="list"):
        await be.available_models()
    await be.process_shutdown()


@pytest.mark.asyncio
async def test_resolve_raises_when_no_model_and_empty_catalog(
    httpx_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:9/v1/models",
        json={"data": []},
    )
    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )
    be = OpenAIWebSocketBackend(
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
async def test_resolve_invalid_ws_target_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )
    be = OpenAIWebSocketBackend(
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
async def test_error_event_dict_formatted_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(ws: object) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "error": {"message": "auth failed", "code": "401"},
                }
            )
        )

    monkeypatch.setattr(
        "guidellm.backends.openai.websocket.pcm16_append_b64_chunks",
        lambda *a, **k: ["YQ=="],
    )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        be = OpenAIWebSocketBackend(
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
    a = OpenAIWebSocketBackendArgs(target="http://localhost:8000", model="x")
    assert a.request_format is None
    assert a.chunk_samples == 3200
    assert a.timeout == _DEFAULT_WS_RECV_TIMEOUT


def test_openai_websocket_backend_args_normalizes_request_format_alias() -> None:
    args = OpenAIWebSocketBackendArgs(
        target="http://localhost:8000",
        request_format="realtime",
    )
    assert args.request_format == "/v1/realtime"


def test_openai_websocket_backend_resolves_websocket_path_from_request_format() -> None:
    backend = Backend.create(
        "openai_websocket",
        target="http://127.0.0.1:9",
        request_format="/custom/ws",
    )
    assert backend.websocket_path == "/custom/ws"


def test_openai_websocket_backend_args_invalid_request_format_rejected() -> None:
    with pytest.raises(ValidationError):
        OpenAIWebSocketBackendArgs(
            target="http://localhost:8000",
            request_format="nope",
        )
