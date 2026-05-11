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
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import ParseResult, urlparse

import httpx
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.common import (
    FALLBACK_TIMEOUT,
    build_headers,
    resolve_validate_kwargs,
)
from guidellm.backends.openai.request_handlers import (
    OpenAIRequestHandlerFactory,
    RealtimeWebSocketRequestHandler,
)
from guidellm.schemas import (
    GenerationRequest,
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
            "The openai_websocket backend requires the 'websockets' package. "
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


@BackendArgs.register("openai_websocket")
class OpenAIWebSocketBackendArgs(BackendArgs):
    """
    Typed configuration for :class:`OpenAIWebSocketBackend`.

    ``request_format`` is validated against
    :class:`~guidellm.backends.openai.request_handlers.RealtimeWebSocketRequestHandler`
    so allowed paths stay aligned with the registered handler (``/v1/realtime``).
    """

    type_: Literal["openai_websocket"] = Field(
        alias="type",
        default="openai_websocket",
        description="Type identifier for the backend configuration.",
    )
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
    request_format: str | None = Field(
        default=None,
        description=(
            "Realtime WebSocket path (only /v1/realtime is supported today). "
            "Use the same top-level CLI flags as ``openai_http``: "
            "--request-format / --request-type."
        ),
        json_schema_extra={
            "error_message": (
                "Backend '{backend_type}' received an invalid --request-format / "
                "request_format; "
                + RealtimeWebSocketRequestHandler.request_format_options_description()
                + "."
            )
        },
    )

    @field_validator("target", mode="after")
    @classmethod
    def strip_target(cls, value: str) -> str:
        """Strip trailing slashes and ``/v1`` suffix from the target URL."""
        return value.rstrip("/").removesuffix("/v1")

    @field_validator("request_format")
    @classmethod
    def validate_request_format(cls, v: str | None) -> str | None:
        """Delegate path validation to the realtime WebSocket request handler."""
        return RealtimeWebSocketRequestHandler.validate_request_format_field(v)

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


@Backend.register("openai_websocket")
class OpenAIWebSocketBackend(Backend):
    """
    WebSocket client for realtime (streaming) audio transcription.

    Connects to a vLLM-style ``/v1/realtime`` WebSocket, streams PCM16 audio chunks,
    and maps ``transcription.*`` events into ``GenerationResponse`` with timings.
    Request-shape validation and allowed ``request_format`` paths are delegated to
    :class:`~guidellm.backends.openai.request_handlers.RealtimeWebSocketRequestHandler`
    (same pattern as :class:`~guidellm.backends.openai.http.OpenAIHTTPBackend` uses
    per-endpoint handlers).

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

    @staticmethod
    def append_pcm16_chunks(*args: Any, **kwargs: Any) -> Any:
        """Encode audio to PCM16 base64 chunks (lazy-imports ``guidellm[audio]``)."""
        try:
            from guidellm.extras.audio import pcm16_append_b64_chunks
        except ImportError as exc:
            raise ImportError(
                "The openai_websocket backend requires the audio extras for PCM "
                "handling used in realtime transcription. " + _AUDIO_EXTRA_HINT
            ) from exc
        return pcm16_append_b64_chunks(*args, **kwargs)

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

        :return: Resolved path (default when ``request_format`` was omitted).
        """
        return RealtimeWebSocketRequestHandler.resolved_websocket_path(
            self._args.request_format
        )

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
            raise ValueError(
                f"Invalid target URL for WebSocket: {self._args.target!r}"
            )
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
        validate_kwargs["headers"] = build_headers(
            self._args.api_key, existing_headers
        )
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
        try:
            payload: Any = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Unexpected /v1/models response: body is not valid JSON"
            ) from exc
        return _model_ids_from_openai_models_payload(payload)

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
        history: list[tuple[GenerationRequest, GenerationResponse | None]]
        | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Stream one realtime transcription over WebSocket for a single audio column.

        Uses :class:`RealtimeWebSocketRequestHandler` for request arguments and
        metrics, performs the vLLM-style handshake and chunk protocol, and yields
        ``None`` for intermediate timing updates followed by a final
        :class:`~guidellm.schemas.GenerationResponse`.

        :param request: Must contain exactly one ``audio_column`` entry.
        :param request_info: Timings updated as deltas and final text arrive.
        :param history: Not supported; raises ``NotImplementedError`` if non-empty.
        :raises NotImplementedError: If ``history`` is provided.
        :raises RuntimeError: If the client is not started, model is missing, or the
            peer returns an error event.
        :yields: ``(response_or_none, request_info)`` until ``transcription.done``.
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

        handler_raw = OpenAIRequestHandlerFactory.create(self.websocket_path)
        if not isinstance(handler_raw, RealtimeWebSocketRequestHandler):
            raise TypeError(
                "Expected RealtimeWebSocketRequestHandler for "
                f"{self.websocket_path!r}, got {type(handler_raw).__name__}"
            )
        handler = handler_raw
        arguments = handler.format(
            request,
            model=model_name,
            websocket_path=self.websocket_path,
            chunk_samples=self._args.chunk_samples,
        )
        audio_entry = handler.extract_single_audio(request)
        chunks = OpenAIWebSocketBackend.append_pcm16_chunks(
            audio_entry,
            chunk_samples=self._args.chunk_samples,
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
        full_text_parts: list[str] = []

        try:
            request_info.timings.request_start = time.time()
            connect_kw: dict[str, Any] = {
                "ssl": ssl_ctx,
                "open_timeout": self._args.timeout_connect,
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
                        inp, outp = handler.extract_metrics(usage_dict, full_text)
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
            inp, outp = handler.extract_metrics(None, text_so_far or "")
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
