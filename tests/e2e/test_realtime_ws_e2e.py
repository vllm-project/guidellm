"""End-to-end integration: realtime backend + PCM encoding + WebSocket (same loop).

## WRITTEN BY AI ##
"""

from __future__ import annotations

import json
import socket
import struct
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

try:
    from websockets.asyncio.server import serve
except ImportError:
    pytest.skip(
        "websockets not installed; install guidellm[audio] for realtime e2e",
        allow_module_level=True,
    )

from guidellm.backends.openai.websocket import OpenAIWebSocketBackend
from guidellm.schemas import GenerationRequest, RequestInfo, RequestTimings


def make_realtime_transcription_stub_handler(
    *,
    delta_text: str = "hello",
    done_text: str | None = None,
    usage: dict[str, Any] | None = None,
    session_id: str = "stub-sess",
) -> Callable[[Any], Awaitable[None]]:
    """Build an async handler that completes one transcription after two commits."""

    resolved_done = done_text if done_text is not None else delta_text
    resolved_usage = usage or {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    async def handler(ws: Any) -> None:
        await ws.send(
            json.dumps({"type": "session.created", "id": session_id, "created": 0})
        )
        commits: list[bool | None] = []
        while True:
            msg = await ws.recv()
            payload = json.loads(msg if isinstance(msg, str) else msg.decode())
            if payload.get("type") == "input_audio_buffer.commit":
                commits.append(payload.get("final"))
                if payload.get("final"):
                    break
        assert commits == [False, True]
        await ws.send(json.dumps({"type": "transcription.delta", "delta": delta_text}))
        await ws.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": resolved_done,
                    "usage": resolved_usage,
                }
            )
        )

    return handler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_minimal_wav_16k_mono(path: Path) -> None:
    n_samples = 4000
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(n_samples))
        wf.writeframes(frames)


@pytest.fixture
def requires_audio_stack():
    pytest.importorskip("torchcodec")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.e2e
async def test_realtime_ws_full_stack_in_one_event_loop(
    requires_audio_stack,
    tmp_path: Path,
) -> None:
    """
    In-process: WebSocket server, OpenAI realtime backend, and torchcodec PCM path.

    No ``guidellm benchmark`` subprocess (avoids worker/hang issues in test envs).
    For a live vLLM run, use ``scripts/e2e_realtime_external.sh``.
    """
    port = _free_port()
    wav_path = tmp_path / "clip.wav"
    _write_minimal_wav_16k_mono(wav_path)
    audio_item = {
        "audio": wav_path.read_bytes(),
        "file_name": "clip.wav",
        "format": "wav",
    }
    request = GenerationRequest(
        request_id="e2e-1",
        columns={"audio_column": [audio_item]},
    )
    info = RequestInfo(timings=RequestTimings())

    stub = make_realtime_transcription_stub_handler(session_id="e2e-stub-sess")
    async with serve(stub, "127.0.0.1", port):
        be = OpenAIWebSocketBackend(
            target=f"http://127.0.0.1:{port}",
            model="stub-model",
            validate_backend=False,
        )
        await be.process_startup()
        try:
            out: list = []
            async for item in be.resolve(request, info):
                out.append(item)
        finally:
            await be.process_shutdown()

    assert len(out) == 2
    assert out[0][0] is None
    final = out[1][0]
    assert final is not None
    assert final.text == "hello"
    assert final.input_metrics.audio_tokens == 10
    assert final.output_metrics.text_tokens == 5
