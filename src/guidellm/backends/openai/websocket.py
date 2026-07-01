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
import ssl
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import ParseResult, urlparse

import httpx
from pydantic import Field, field_validator
from websockets.asyncio.client import connect as ws_connect

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.common import (
    FALLBACK_TIMEOUT,
    build_headers,
    format_ws_error,
    resolve_validate_kwargs,
)
from guidellm.backends.openai.request_handlers import (
    WS_AUDIO_CHUNKS_BODY_KEY,
    OpenAIWSRequestHandlerFactory,
    WSEventResult,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)
from guidellm.utils.imports import json

__all__ = [
    "OpenAIWebSocketBackend",
    "OpenAIWebSocketBackendArgs",
]

_WS_API_ROUTES = {
    "/health": "health",
    "/v1/models": "v1/models",
}

_MAX_IGNORED_WS_EVENT_TYPES = 50_000


def _record_content_tokens(
    request_info: RequestInfo,
    *,
    content_tokens: int,
    record_request_iteration: bool,
) -> bool:
    """
    Update request/token timings for a WebSocket content event.

    :param request_info: Mutable timing state for the in-flight request.
    :param content_tokens: New content tokens from this event (0 for empty deltas).
    :param record_request_iteration: Whether to increment request iteration counters.
    :return: True when a prefetch ``(None, request_info)`` yield is needed.
    """
    iter_time = time.time()
    if record_request_iteration:
        if request_info.timings.first_request_iteration is None:
            request_info.timings.first_request_iteration = iter_time
        request_info.timings.last_request_iteration = iter_time
        request_info.timings.request_iterations += 1

    if content_tokens <= 0:
        return False

    request_info.timings.token_received_sum += iter_time
    request_info.timings.token_received_count += 1

    if request_info.timings.first_token_iteration is None:
        request_info.timings.first_token_iteration = iter_time
        request_info.timings.token_iterations = 0
        request_info.timings.last_token_iteration = iter_time
        request_info.timings.token_iterations += content_tokens
        return True

    request_info.timings.last_token_iteration = iter_time
    request_info.timings.token_iterations += content_tokens
    return False


def _record_request_sent(request_info: RequestInfo) -> None:
    """
    Record the timestamp of one outbound WebSocket frame for round-trip metrics.

    :param request_info: Mutable timing state for the in-flight request.
    """
    sent_time = time.time()
    request_info.timings.last_request_sent = sent_time
    request_info.timings.request_sent_sum += sent_time
    request_info.timings.request_sent_count += 1


def _load_ws_event(raw: str) -> dict[str, Any]:
    """Parse a JSON WebSocket text frame; raise RuntimeError on invalid JSON."""
    try:
        parsed: Any = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid JSON from realtime WebSocket: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Expected JSON object from realtime WebSocket, got {type(parsed).__name__}"
        )
    return parsed


def _json_text(obj: Any) -> str:
    """Serialize *obj* to a JSON string (handles orjson bytes transparently)."""
    raw = json.dumps(obj)
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


@BackendArgs.register("openai_websocket")
class OpenAIWebSocketBackendArgs(BackendArgs):
    """Typed configuration for :class:`OpenAIWebSocketBackend`."""

    kind: Literal["openai_websocket"] = Field(
        default="openai_websocket",
        description="Type identifier for the backend configuration.",
    )
    target: str = Field(
        description=(
            "HTTP(S) base URL of the server (WebSocket URL is derived from it)."
        ),
    )
    model: str = Field(
        default_factory=str,
        description="Model identifier for generation requests.",
    )
    request_format: str = Field(
        default="/v1/realtime",
        description=(
            "Realtime WebSocket path (only /v1/realtime is supported today). "
            "Use the same top-level CLI flags as ``openai_http``: "
            "--request-format / --request-type."
        ),
    )
    chunk_samples: int = Field(
        default=3200,
        ge=1,
        description="PCM16 frames per input_audio_buffer.append chunk (16 kHz).",
    )
    api_key: str | None = Field(default=None, description="Bearer token if required.")
    verify: bool = Field(default=False, description="Verify TLS certificates.")
    timeout: float | None = Field(
        default=None,
        description="Per-message read timeout for WebSocket receives (seconds).",
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

    @field_validator("target", mode="after")
    @classmethod
    def strip_target(cls, value: str) -> str:
        """Strip trailing slashes and ``/v1`` suffix from the target URL."""
        return value.rstrip("/").removesuffix("/v1")

    @field_validator("request_format")
    @classmethod
    def validate_request_format(cls, v: str) -> str:
        """Validate ``request_format`` against allowed WebSocket paths."""
        stripped = v.strip()
        if stripped != "/v1/realtime":
            raise ValueError(f"request_format must be '/v1/realtime', got {stripped!r}")
        return stripped


@Backend.register("openai_websocket")
class OpenAIWebSocketBackend(Backend):
    """
    WebSocket client for realtime (streaming) audio transcription.

    Connects to a vLLM-style ``/v1/realtime`` WebSocket, streams PCM16 audio chunks,
    and maps ``transcription.*`` events into ``GenerationResponse`` with timings.

    Example:
    ::
        args = OpenAIWebSocketBackendArgs(
            target="http://localhost:8000",
            model="my-model",
        )
        backend = OpenAIWebSocketBackend(args)

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            ...
        await backend.process_shutdown()
    """

    def __init__(self, arguments: OpenAIWebSocketBackendArgs):
        """
        Initialize the WebSocket backend from validated args.

        :param arguments: Typed configuration including target, model, and paths.
        """
        super().__init__(arguments)
        self._args = arguments
        self._resolved_model = (arguments.model or "").strip()
        self.validate_backend: dict[str, Any] | None = resolve_validate_kwargs(
            arguments.validate_backend,
            self._args.target,
            _WS_API_ROUTES,
        )
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None

    @property
    def websocket_path(self) -> str:
        """
        HTTP path segment on the host used for the WebSocket URL.

        :return: Resolved path from ``request_format``.
        """
        return self._args.request_format

    @property
    def info(self) -> dict[str, Any]:
        """
        Return a snapshot of backend configuration for logging or debugging.

        :return: Dict of target, model, WebSocket path, timeouts, and validation opts.
        """
        return {
            "target": self._args.target,
            "model": self._resolved_model or self._args.model,
            "websocket_path": self.websocket_path,
            "chunk_samples": self._args.chunk_samples,
            "timeout": self._args.timeout,
            "timeout_connect": self._args.timeout_connect,
            "verify": self._args.verify,
            "validate_backend": self.validate_backend,
        }

    def _parsed_target(self) -> ParseResult:
        """Parse ``target`` into a URL structure for scheme and host lookup."""
        raw = (
            self._args.target
            if "://" in self._args.target
            else f"http://{self._args.target}"
        )
        return urlparse(raw)

    def _ws_url(self) -> str:
        """Build ``ws://`` or ``wss://`` URL including :attr:`websocket_path`."""
        parsed = self._parsed_target()
        if not parsed.netloc:
            raise ValueError(f"Invalid target URL for WebSocket: {self._args.target!r}")
        ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
        path = self.websocket_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{ws_scheme}://{parsed.netloc}{path}"

    def _ssl_context(self) -> ssl.SSLContext | None:
        """TLS context for secure WebSockets; ``None`` when using plain ``ws``."""
        if self._parsed_target().scheme in ("http", "ws"):
            return None
        ctx = ssl.create_default_context()
        if not self._args.verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _build_headers(
        self, existing_headers: dict[str, str] | None = None
    ) -> dict[str, str] | None:
        """Merge bearer auth and optional headers for HTTP and WebSocket handshakes."""
        return build_headers(self._args.api_key, existing_headers)

    async def process_startup(self) -> None:
        """
        Create the shared :class:`httpx.AsyncClient` used for health and ``/v1/models``.

        :raises RuntimeError: If the backend was already started in this process.
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")
        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                FALLBACK_TIMEOUT,
                read=self._args.timeout,
                connect=self._args.timeout_connect,
            ),
            verify=self._args.verify,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
                keepalive_expiry=5.0,
            ),
        )
        self._in_process = True

    async def process_shutdown(self) -> None:
        """
        Close the HTTP client and reset process-local state.

        :raises RuntimeError: If the backend was not started.
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")
        client = self._async_client
        if client is None:
            raise RuntimeError("Backend not started up for process.")
        await client.aclose()
        self._async_client = None
        self._in_process = False

    async def validate(self) -> None:
        """
        Run the configured HTTP probe (same semantics as ``openai_http``).

        :raises RuntimeError: If the client is not started or the probe fails.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        if not self.validate_backend:
            return
        validate_kwargs = {**self.validate_backend}
        existing_headers = validate_kwargs.get("headers")
        validate_kwargs["headers"] = build_headers(self._args.api_key, existing_headers)
        try:
            response = await self._async_client.request(**validate_kwargs)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the server "
                "or validate the backend configuration."
            ) from exc

    async def available_models(self) -> list[str]:
        """
        List model IDs from ``GET /v1/models`` on the HTTP target.

        :return: Model identifiers from the OpenAI-style payload.
        :raises RuntimeError: If the client is not started or the response is invalid.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        target = f"{self._args.target}/v1/models"
        response = await self._async_client.get(
            target, headers=build_headers(self._args.api_key)
        )
        response.raise_for_status()
        return [item["id"] for item in response.json()["data"]]

    async def default_model(self) -> str:
        """
        Return the configured model, or the first from ``available_models`` if empty.

        :return: Non-empty model name when discoverable; otherwise ``""``.
        """
        if self._resolved_model:
            return self._resolved_model
        if not self._in_process:
            return ""
        models = await self.available_models()
        self._resolved_model = models[0] if models else ""
        return self._resolved_model

    async def resolve(  # type: ignore[override, misc]  # noqa: C901, PLR0912, PLR0915
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: (
            list[tuple[GenerationRequest, GenerationResponse | None]] | None
        ) = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Stream one realtime transcription over WebSocket for a single audio column.

        Delegates event interpretation to the registered
        :class:`~guidellm.backends.openai.request_handlers.OpenAIWSRequestHandler`
        via ``add_streaming_event`` / ``compile_streaming``, while this method handles
        only I/O and timing.

        :param request: Must contain exactly one ``audio_column`` entry.
        :param request_info: Timings updated as events arrive.
        :param history: Not supported; raises ``NotImplementedError`` if non-empty.
        :raises NotImplementedError: If ``history`` is provided.
        :raises RuntimeError: If the client is not started, model is missing, or the
            peer returns an error event.
        :yields: ``(response_or_none, request_info)`` until stream completion.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        if history:
            raise NotImplementedError(
                "openai_websocket does not support multiturn/history yet."
            )

        model_name = await self.default_model()
        if not str(model_name).strip():
            raise RuntimeError(
                "No model configured for openai_websocket and /v1/models returned "
                "none. Pass --model or ensure the server lists at least one model."
            )

        handler = OpenAIWSRequestHandlerFactory.create(self.websocket_path)
        arguments = handler.format(
            request,
            model=model_name,
            websocket_path=self.websocket_path,
            chunk_samples=self._args.chunk_samples,
        )
        body = arguments.body or {}
        chunks = body.get(WS_AUDIO_CHUNKS_BODY_KEY)
        if not isinstance(chunks, list):
            raise RuntimeError(
                "Realtime WebSocket handler format() did not provide "
                f"{WS_AUDIO_CHUNKS_BODY_KEY!r}."
            )

        session_update: dict[str, Any] = {"type": "session.update"}
        extras = self._args.extras or {}
        if extras:
            for key, val in extras.items():
                if key not in ("type", "model"):
                    session_update[key] = val
        session_update["model"] = model_name

        ssl_ctx = self._ssl_context()
        ws_headers = build_headers(self._args.api_key)

        try:
            request_info.timings.request_start = time.time()
            connect_kw: dict[str, Any] = {
                "ssl": ssl_ctx,
                "open_timeout": self._args.timeout_connect,
            }
            if ws_headers:
                connect_kw["additional_headers"] = ws_headers
            async with ws_connect(self._ws_url(), **connect_kw) as ws:
                raw_first = await self._recv_ws(ws)
                first_event = _load_ws_event(raw_first)
                if first_event.get("type") == "error":
                    raise RuntimeError(format_ws_error(first_event.get("error")))
                if first_event.get("type") != "session.created":
                    raise RuntimeError(
                        f"Expected session.created, got {first_event.get('type')!r}"
                    )
                await ws.send(_json_text(session_update))
                _record_request_sent(request_info)
                for b64_chunk in chunks:
                    await ws.send(
                        _json_text(
                            {"type": "input_audio_buffer.append", "audio": b64_chunk}
                        )
                    )
                    _record_request_sent(request_info)
                await ws.send(
                    _json_text({"type": "input_audio_buffer.commit", "final": False})
                )
                _record_request_sent(request_info)
                await ws.send(
                    _json_text({"type": "input_audio_buffer.commit", "final": True})
                )
                _record_request_sent(request_info)

                ignored_events = 0
                while True:
                    raw = await self._recv_ws(ws)
                    event = _load_ws_event(raw)
                    update = handler.add_streaming_event(event)

                    if update.kind is WSEventResult.STREAM_END:
                        iter_time = time.time()
                        request_info.timings.request_end = iter_time
                        # Done-only path: first-token timing when text exists.
                        if (
                            request_info.timings.first_token_iteration is None
                            and _record_content_tokens(
                                request_info,
                                content_tokens=1 if handler.streaming_text else 0,
                                record_request_iteration=True,
                            )
                        ):
                            yield None, request_info
                        break

                    if update.kind in (
                        WSEventResult.CONTENT,
                        WSEventResult.REQUEST_ITERATION,
                    ):
                        if _record_content_tokens(
                            request_info,
                            content_tokens=update.content_tokens,
                            record_request_iteration=True,
                        ):
                            yield None, request_info
                    elif update.kind is WSEventResult.IGNORED:
                        ignored_events += 1
                        if ignored_events > _MAX_IGNORED_WS_EVENT_TYPES:
                            raise RuntimeError(
                                "Exceeded maximum ignored realtime WebSocket events "
                                f"(last type={event.get('type')!r})."
                            )

                yield handler.compile_streaming(request, arguments), request_info

        except asyncio.CancelledError as err:
            yield handler.compile_streaming(request, arguments), request_info
            raise err
        finally:
            if (
                request_info.timings.request_start is not None
                and request_info.timings.request_end is None
            ):
                request_info.timings.request_end = time.time()

    async def _recv_ws(self, ws: ClientConnection) -> str:
        """
        Receive one text frame from the WebSocket, honoring per-message timeout.

        :param ws: Active realtime connection.
        :return: Decoded UTF-8 text from the server.
        """
        if self._args.timeout is None:
            msg = await ws.recv()
        else:
            msg = await asyncio.wait_for(ws.recv(), timeout=self._args.timeout)
        if isinstance(msg, bytes):
            return msg.decode()
        return str(msg)
