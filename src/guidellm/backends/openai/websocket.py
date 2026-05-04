"""
WebSocket backend for vLLM-compatible realtime audio transcription.

Implements the JSON event protocol used by vLLM's ``/v1/realtime`` endpoint:
``session.created`` → ``session.update`` → ``input_audio_buffer.append`` →
``input_audio_buffer.commit`` (``final: false`` starts transcription, then
``final: true`` ends the audio stream) → ``transcription.delta`` /
``transcription.done``.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import ParseResult, urlparse

import httpx
from pydantic import Field

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.common import (
    FALLBACK_TIMEOUT,
    build_headers,
    resolve_validate_kwargs,
)
from guidellm.backends.openai.request_handlers import AudioRequestHandler
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)

__all__ = [
    "OpenAIWebSocketBackend",
    "OpenAIWebSocketBackendArgs",
]

_WS_API_ROUTES = {
    "/health": "health",
    "/v1/models": "v1/models",
}

# Guard against a misbehaving server that only emits ignored event types.
_MAX_IGNORED_WS_EVENT_TYPES = 50_000

# Per-message WebSocket recv timeout default so benchmark workers do not hang forever
# on a silent peer. Pass ``timeout=None`` to wait indefinitely.
_DEFAULT_WS_RECV_TIMEOUT = 120.0

_AUDIO_EXTRA_HINT = (
    "Install optional audio extras: pip install 'guidellm[audio]' "
    "(includes websockets and torchcodec for realtime transcription)."
)


def _require_ws_connect() -> Any:
    try:
        from websockets.asyncio.client import connect as ws_connect
    except ImportError as exc:
        raise ImportError(
            "The openai_realtime_ws backend requires the 'websockets' package. "
            + _AUDIO_EXTRA_HINT
        ) from exc
    return ws_connect


def _ws_error_message(err: Any) -> str:
    """Format WebSocket ``error`` for exceptions (supports dict payloads)."""
    if isinstance(err, dict):
        msg = err.get("message") or err.get("msg")
        code = err.get("code")
        parts = [str(p) for p in (code, msg) if p]
        if parts:
            return ": ".join(parts)
        try:
            return json.dumps(err)[:500]
        except (TypeError, ValueError):
            return repr(err)
    if err is None or err == "":
        return "WebSocket error"
    return str(err)


def _model_ids_from_openai_models_payload(payload: Any) -> list[str]:
    """Parse ``GET /v1/models`` JSON body; raise RuntimeError if shape is unexpected."""
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Unexpected /v1/models response: top-level JSON must be an object, "
            f"got {type(payload).__name__}"
        )
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected /v1/models response: 'data' must be a list, "
            f"got {type(data).__name__}"
        )
    ids: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "id" not in item:
            raise RuntimeError(
                "Unexpected /v1/models response: each entry must be an object with "
                f"'id' (index {i})"
            )
        ids.append(str(item["id"]))
    return ids


def _load_ws_event(raw: str) -> dict[str, Any]:
    """Parse a JSON WebSocket text frame; raise RuntimeError on invalid JSON."""
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON from realtime WebSocket: {exc.msg} at position {exc.pos}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Expected JSON object from realtime WebSocket, got {type(parsed).__name__}"
        )
    return parsed


# Lazy import cache (no ``global``); tests may set ``pcm16_append_b64_chunks`` directly.
pcm16_append_b64_chunks: Any = None
_pcm_imported_fn: dict[str, Any] = {"fn": None}


def _ensure_pcm16_append_b64_chunks() -> Any:
    if pcm16_append_b64_chunks is not None:
        return pcm16_append_b64_chunks
    if _pcm_imported_fn["fn"] is not None:
        return _pcm_imported_fn["fn"]
    try:
        from guidellm.extras.audio import pcm16_append_b64_chunks as fn
    except ImportError as exc:
        raise ImportError(
            "The openai_realtime_ws backend requires the audio extras for PCM "
            "handling used in realtime transcription. " + _AUDIO_EXTRA_HINT
        ) from exc
    _pcm_imported_fn["fn"] = fn
    return fn


def _coerce_usage_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _normalize_transcription_usage(
    raw_usage: Any,
) -> dict[str, int | dict[str, int]] | None:
    """Coerce OpenAI-style usage dict values to ints (including numeric strings)."""
    if not isinstance(raw_usage, dict):
        return None
    result: dict[str, int | dict[str, int]] = {}
    for key, val in raw_usage.items():
        if isinstance(val, dict):
            inner: dict[str, int] = {}
            for ik, iv in val.items():
                num = _coerce_usage_int(iv)
                if num is not None:
                    inner[ik] = num
            if inner:
                result[key] = inner
        else:
            num = _coerce_usage_int(val)
            if num is not None:
                result[key] = num
    return result if result else None


class OpenAIWebSocketBackendArgs(BackendArgs):
    """Arguments for creating the realtime WebSocket backend."""

    target: str = Field(
        description=(
            "HTTP(S) base URL of the server (WebSocket URL is derived from it)."
        ),
        json_schema_extra={
            "error_message": (
                "Backend '{backend_type}' requires --target with a valid URL."
            )
        },
    )
    model: str | None = Field(
        default=None,
        description="Model identifier (required unless discoverable from /v1/models).",
    )
    websocket_path: str = Field(
        default="/v1/realtime",
        description="WebSocket path on the server (default /v1/realtime).",
    )
    chunk_samples: int = Field(
        default=3200,
        ge=1,
        description="PCM16 frames per input_audio_buffer.append chunk (16 kHz).",
    )
    api_key: str | None = Field(default=None, description="Bearer token if required.")
    verify: bool = Field(default=False, description="Verify TLS certificates.")
    timeout: float | None = Field(
        default=_DEFAULT_WS_RECV_TIMEOUT,
        description=(
            "Per-message read timeout for WebSocket receives (seconds). "
            f"Defaults to {_DEFAULT_WS_RECV_TIMEOUT}s so hung servers do not block "
            "workers; use ``None`` for no limit."
        ),
    )
    timeout_connect: float = Field(
        default=FALLBACK_TIMEOUT,
        description="Timeout for establishing the WebSocket connection.",
    )
    validate_backend: bool | str | dict[str, Any] = Field(
        default=True,
        description=(
            "HTTP health check before benchmarks (same semantics as openai_http)."
        ),
    )
    extras: dict[str, Any] | None = Field(
        default=None,
        description="Extra fields merged into session.update (backend model wins).",
    )


@Backend.register("openai_realtime_ws")
class OpenAIWebSocketBackend(Backend):
    """WebSocket client for realtime (streaming) audio transcription."""

    @classmethod
    def backend_args(cls) -> type[BackendArgs]:
        return OpenAIWebSocketBackendArgs

    def __init__(
        self,
        target: str,
        model: str = "",
        websocket_path: str = "/v1/realtime",
        chunk_samples: int = 3200,
        api_key: str | None = None,
        verify: bool = False,
        timeout: float | None = _DEFAULT_WS_RECV_TIMEOUT,
        timeout_connect: float = FALLBACK_TIMEOUT,
        validate_backend: bool | str | dict[str, Any] = True,
        extras: dict[str, Any] | None = None,
    ):
        super().__init__(type_="openai_realtime_ws")
        self.target = target.rstrip("/").removesuffix("/v1")
        self.model = model or ""
        self.websocket_path = websocket_path
        self.chunk_samples = chunk_samples
        self.api_key = api_key
        self.verify = verify
        self.timeout = timeout
        self.timeout_connect = timeout_connect
        self.api_routes = _WS_API_ROUTES
        self.validate_backend: dict[str, Any] | None = resolve_validate_kwargs(
            validate_backend
        )
        self.extras = extras or {}
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None

    @property
    def info(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "model": self.model,
            "websocket_path": self.websocket_path,
            "chunk_samples": self.chunk_samples,
            "timeout": self.timeout,
            "timeout_connect": self.timeout_connect,
            "verify": self.verify,
            "validate_backend": self.validate_backend,
        }

    def _parsed_target(self) -> ParseResult:
        raw = self.target if "://" in self.target else f"http://{self.target}"
        return urlparse(raw)

    def _ws_url(self) -> str:
        parsed = self._parsed_target()
        if not parsed.netloc:
            raise ValueError(f"Invalid target URL for WebSocket: {self.target!r}")
        ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
        path = self.websocket_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{ws_scheme}://{parsed.netloc}{path}"

    def _ssl_context(self) -> ssl.SSLContext | None:
        if self._parsed_target().scheme in ("http", "ws"):
            return None
        ctx = ssl.create_default_context()
        if not self.verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx


    def _build_headers(
        self, existing_headers: dict[str, str] | None = None
    ) -> dict[str, str] | None:
        return build_headers(self.api_key, existing_headers)

    async def process_startup(self) -> None:
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")
        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                FALLBACK_TIMEOUT,
                read=self.timeout,
                connect=self.timeout_connect,
            ),
            verify=self.verify,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
                keepalive_expiry=5.0,
            ),
        )
        self._in_process = True

    async def process_shutdown(self) -> None:
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")
        client = self._async_client
        if client is None:
            raise RuntimeError("Backend not started up for process.")
        await client.aclose()
        self._async_client = None
        self._in_process = False

    async def validate(self) -> None:
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        if not self.validate_backend:
            return
        validate_kwargs = {**self.validate_backend}
        existing_headers = validate_kwargs.get("headers")
        validate_kwargs["headers"] = build_headers(existing_headers)
        try:
            response = await self._async_client.request(**validate_kwargs)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the server "
                "or validate the backend configuration."
            ) from exc

    async def available_models(self) -> list[str]:
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        target = f"{self.target}/v1/models"
        response = await self._async_client.get(target, headers=build_headers(self.api_key))
        response.raise_for_status()
        try:
            payload: Any = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Unexpected /v1/models response: body is not valid JSON"
            ) from exc
        return _model_ids_from_openai_models_payload(payload)

    async def default_model(self) -> str:
        if self.model:
            return self.model
        if not self._in_process:
            return ""
        models = await self.available_models()
        self.model = models[0] if models else ""
        return self.model

    async def resolve(  # type: ignore[override, misc]  # noqa: C901, PLR0912, PLR0915
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse | None]]
        | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        if history:
            raise NotImplementedError(
                "openai_realtime_ws does not support multiturn/history yet."
            )

        audio_columns = request.columns.get("audio_column", [])
        if len(audio_columns) != 1:
            raise ValueError(
                "Realtime transcription expects exactly one audio_column entry; "
                f"got {len(audio_columns)}."
            )

        model_name = await self.default_model()
        if not str(model_name).strip():
            raise RuntimeError(
                "No model configured for openai_realtime_ws and /v1/models returned "
                "none. Pass --model or ensure the server lists at least one model."
            )

        arguments = GenerationRequestArguments(
            body={
                "model": model_name,
                "websocket_path": self.websocket_path,
                "chunk_samples": self.chunk_samples,
            }
        )

        pcm_fn = _ensure_pcm16_append_b64_chunks()
        chunks = pcm_fn(
            audio_columns[0],
            chunk_samples=self.chunk_samples,
        )

        session_update: dict[str, Any] = {"type": "session.update"}
        if self.extras:
            for key, val in self.extras.items():
                if key not in ("type", "model"):
                    session_update[key] = val
        session_update["model"] = model_name

        ssl_ctx = self._ssl_context()
        ws_headers = build_headers(self.api_key)
        audio_handler = AudioRequestHandler()
        full_text_parts: list[str] = []

        try:
            request_info.timings.request_start = time.time()
            connect_kw: dict[str, Any] = {
                "ssl": ssl_ctx,
                "open_timeout": self.timeout_connect,
            }
            if ws_headers:
                connect_kw["additional_headers"] = ws_headers
            ws_connect = _require_ws_connect()
            async with ws_connect(self._ws_url(), **connect_kw) as ws:
                raw_first = await self._recv_ws(ws)
                first_event = _load_ws_event(raw_first)
                if first_event.get("type") == "error":
                    raise RuntimeError(_ws_error_message(first_event.get("error")))
                if first_event.get("type") != "session.created":
                    raise RuntimeError(
                        f"Expected session.created, got {first_event.get('type')!r}"
                    )
                await ws.send(json.dumps(session_update))
                for b64_chunk in chunks:
                    await ws.send(
                        json.dumps(
                            {"type": "input_audio_buffer.append", "audio": b64_chunk}
                        )
                    )
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.commit", "final": False})
                )
                # Sentinel end-of-stream for vLLM's audio queue
                # (see RealtimeConnection).
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.commit", "final": True})
                )

                ignored_events = 0
                while True:
                    raw = await self._recv_ws(ws)
                    event = _load_ws_event(raw)
                    et = event.get("type")
                    if et == "transcription.delta":
                        iter_time = time.time()
                        if request_info.timings.first_request_iteration is None:
                            request_info.timings.first_request_iteration = iter_time
                        request_info.timings.last_request_iteration = iter_time
                        request_info.timings.request_iterations += 1
                        delta = event.get("delta") or ""
                        full_text_parts.append(delta)
                        if request_info.timings.first_token_iteration is None:
                            request_info.timings.first_token_iteration = iter_time
                            request_info.timings.token_iterations = 0
                            yield None, request_info
                        request_info.timings.last_token_iteration = iter_time
                        request_info.timings.token_iterations += 1 if delta else 0

                    elif et == "transcription.done":
                        iter_time = time.time()
                        request_info.timings.request_end = iter_time
                        full_text = event.get("text") or "".join(full_text_parts)
                        if request_info.timings.first_token_iteration is None:
                            if request_info.timings.first_request_iteration is None:
                                request_info.timings.first_request_iteration = iter_time
                            request_info.timings.last_request_iteration = iter_time
                            request_info.timings.request_iterations += 1
                            request_info.timings.first_token_iteration = iter_time
                            request_info.timings.token_iterations = 0
                            yield None, request_info
                            request_info.timings.last_token_iteration = iter_time
                            request_info.timings.token_iterations += (
                                1 if full_text else 0
                            )
                        usage_dict = _normalize_transcription_usage(event.get("usage"))
                        inp, outp = audio_handler.extract_metrics(usage_dict, full_text)
                        yield (
                            GenerationResponse(
                                request_id=request.request_id,
                                request_args=arguments.model_dump_json(),
                                text=full_text,
                                input_metrics=inp,
                                output_metrics=outp,
                            ),
                            request_info,
                        )
                        break
                    elif et == "error":
                        raise RuntimeError(_ws_error_message(event.get("error")))
                    else:
                        ignored_events += 1
                        if ignored_events > _MAX_IGNORED_WS_EVENT_TYPES:
                            raise RuntimeError(
                                "Exceeded maximum ignored realtime WebSocket events "
                                f"without transcription.done (last type={et!r})."
                            )
                        continue

        except asyncio.CancelledError as err:
            text_so_far = "".join(full_text_parts)
            inp, outp = audio_handler.extract_metrics(None, text_so_far or "")
            yield (
                GenerationResponse(
                    request_id=request.request_id,
                    request_args=arguments.model_dump_json(),
                    text=text_so_far,
                    input_metrics=inp,
                    output_metrics=outp,
                ),
                request_info,
            )
            raise err
        finally:
            if (
                request_info.timings.request_start is not None
                and request_info.timings.request_end is None
            ):
                request_info.timings.request_end = time.time()

    async def _recv_ws(self, ws: ClientConnection) -> str:
        if self.timeout is None:
            msg = await ws.recv()
        else:
            msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
        if isinstance(msg, bytes):
            return msg.decode()
        return str(msg)
