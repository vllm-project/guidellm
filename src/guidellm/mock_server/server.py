"""
High-performance mock server for OpenAI and vLLM API compatibility testing.

This module provides a Sanic-based mock server that simulates OpenAI and vLLM APIs
with configurable latency, token generation patterns, and response characteristics.
The server supports both streaming and non-streaming endpoints, enabling realistic
performance testing and validation of GuideLLM benchmarking workflows without
requiring actual model deployments.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, cast

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic.log import logger
from sanic.logging.formatter import LegacyAccessFormatter, LegacyFormatter
from sanic.request import File, Request
from sanic.response import BaseHTTPResponse, HTTPResponse, ResponseStream

from guidellm.mock_server.handlers import (
    ChatCompletionsHandler,
    CompletionsHandler,
    ResponsesHandler,
    TokenizerHandler,
)
from guidellm.mock_server.multimodal import (
    WAV_BYTES_PER_SECOND,
    estimate_audio_seconds,
    mock_pcm16,
    mock_wav,
)
from guidellm.schemas.mock_server import MockServerConfig

__all__ = ["MockServer"]


def _configure_sanic_logging(*, access_log: bool) -> None:
    """
    Configure Sanic loggers for interactive or shared-TTY (test) use.

    When ``access_log`` is false, switch handlers to legacy formatters so start,
    stop, and error logs still print without ANSI cursor controls that overwrite
    shared terminal lines. Access logging itself remains controlled by Sanic's
    ``access_log`` flag on ``app.run``.
    """
    if access_log:
        return

    legacy = LegacyFormatter()
    legacy_access = LegacyAccessFormatter()
    for name in (
        "sanic.root",
        "sanic.error",
        "sanic.access",
        "sanic.server",
        "sanic.websockets",
    ):
        sanic_logger = logging.getLogger(name)
        for handler in sanic_logger.handlers:
            if name == "sanic.access":
                handler.setFormatter(legacy_access)
            else:
                handler.setFormatter(legacy)


class MockServer:
    """
    High-performance mock server implementing OpenAI and vLLM API endpoints.

    Provides a Sanic-based web server that simulates API responses with configurable
    timing characteristics for testing and benchmarking purposes. Supports chat
    completions, text completions, tokenization endpoints, and model listing with
    realistic latency patterns to enable comprehensive performance validation.

    Example:
    ::
        config = ServerConfig(model="test-model", port=8080)
        server = MockServer(config)
        server.run()
    """

    def __init__(self, config: MockServerConfig) -> None:
        """
        Initialize the mock server with configuration.

        :param config: Server configuration containing network settings and response
            timing parameters
        """
        self.config = config
        self.app = Sanic("guidellm-mock-server")
        self.chat_handler = ChatCompletionsHandler(config)
        self.completions_handler = CompletionsHandler(config)
        self.responses_handler = ResponsesHandler(config)
        self.tokenizer_handler = TokenizerHandler(config)
        # Deterministic test controls: count accepted generation requests and
        # optionally serialize them with a semaphore (lazy-created in the loop).
        self._accepted_generation_requests = 0
        self._concurrency_semaphore: asyncio.Semaphore | None = None

        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()

    def _get_concurrency_semaphore(self) -> asyncio.Semaphore | None:
        """Return the concurrency semaphore, creating it in the running loop."""
        if self.config.max_concurrent_requests is None:
            return None
        if self._concurrency_semaphore is None:
            self._concurrency_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_requests
            )
        return self._concurrency_semaphore

    async def _run_generation(
        self,
        handler: Callable[[Request], Awaitable[HTTPResponse]],
        request: Request,
    ) -> HTTPResponse:
        """
        Run a generation handler with optional fail-after and concurrency limits.

        :param handler: Async generation endpoint handler
        :param request: Incoming Sanic request
        :return: Handler response, or HTTP 500 when fail_after_requests is exceeded
        """
        fail_after = self.config.fail_after_requests
        if fail_after is not None and self._accepted_generation_requests >= fail_after:
            return response.json(
                {
                    "error": {
                        "message": (
                            f"Mock server fail_after_requests={fail_after} exceeded"
                        ),
                        "type": "server_error",
                        "code": "fail_after_requests",
                    }
                },
                status=500,
            )

        self._accepted_generation_requests += 1
        semaphore = self._get_concurrency_semaphore()
        if semaphore is None:
            return await handler(request)

        await semaphore.acquire()
        try:
            resp = await handler(request)
        except BaseException:
            semaphore.release()
            raise

        if not isinstance(resp, ResponseStream):
            semaphore.release()
            return resp

        # A streaming handler returns before any tokens are generated; Sanic runs
        # the generation afterwards through streaming_fn. Hold the slot until that
        # finishes, including when the client disconnects mid-stream.
        streaming_fn = resp.streaming_fn

        async def stream_holding_slot(stream: BaseHTTPResponse | ResponseStream):
            try:
                await streaming_fn(stream)
            finally:
                semaphore.release()

        resp.streaming_fn = stream_holding_slot
        return resp

    def _audio_usage(self, file: File, text: str) -> dict[str, int | float]:
        """
        Build usage statistics for an audio endpoint response.

        Charges prompt tokens for the uploaded audio using the configured
        per-second rate applied to the duration estimated from the payload
        size, and counts completion tokens from the generated text.

        :param file: Uploaded audio file from the multipart form
        :param text: Text returned in the response body
        :return: Usage dict with prompt, completion, and total token counts
        """
        audio_seconds = estimate_audio_seconds(len(file.body), file.type or file.name)
        prompt_tokens = math.ceil(audio_seconds * self.config.audio_tokens_per_second)
        completion_tokens = len(self.tokenizer_handler.tokenizer.tokenize(text))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "seconds": round(audio_seconds, 3),
        }

    def _validate_audio_speech_options(
        self,
        body: dict[str, Any],
    ) -> tuple[str, int] | HTTPResponse:
        """
        Validate and return audio format and sample rate (or error response).

        :param body: Parsed speech request JSON object.
        :return: The selected format and sample rate, or a validation response.
        """
        audio_format = body.get("response_format") or "wav"
        if audio_format not in ("wav", "pcm"):
            return response.json(
                {"error": "Mock speech supports wav and pcm"},
                status=400,
            )

        # WAV_BYTES_PER_SECOND assumes mono PCM16: two bytes per sample.
        sample_rate = body.get("sample_rate", WAV_BYTES_PER_SECOND // 2)
        if type(sample_rate) is not int or sample_rate <= 0:
            return response.json(
                {"error": "sample_rate must be a positive integer"},
                status=400,
            )

        return audio_format, sample_rate

    async def _handle_audio_speech(self, request: Request) -> HTTPResponse:
        """
        Build a mock WAV or PCM speech response.

        :param request: Speech request containing format and streaming options.
        :return: Audio response, stream response, or a validation error response.
        """
        body = request.json or {}
        if not isinstance(body, dict):
            return response.json({"error": "Expected a JSON object"}, status=400)

        audio_options = self._validate_audio_speech_options(
            cast("dict[str, Any]", body)
        )
        if isinstance(audio_options, HTTPResponse):
            return audio_options
        audio_format, sample_rate = audio_options

        pcm = mock_pcm16(sample_rate=sample_rate, seconds=0.4)
        wav = mock_wav(pcm, sample_rate=sample_rate)
        mime = "audio/wav" if audio_format == "wav" else "audio/pcm"
        audio_headers = {
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": "1",
        }
        stream_format = body.get("stream_format")
        streaming = stream_format in ("audio", "sse") or bool(body.get("stream"))
        if not streaming:
            return response.raw(
                wav if audio_format == "wav" else pcm,
                content_type=mime,
                headers={
                    **audio_headers,
                    "x-vllm-omni-input-text-tokens": "0",
                    "x-vllm-omni-input-audio-tokens": "0",
                    "x-vllm-omni-output-tokens": "0",
                },
            )

        half = len(pcm) // 2
        packets = [pcm[:half], pcm[half:]]
        if audio_format == "wav":
            packets.insert(0, wav[: len(wav) - len(pcm)])

        if stream_format == "audio":
            return ResponseStream(  # type: ignore[return-value]
                partial(self._send_raw, packets=packets),
                content_type=mime,
                headers=audio_headers,
            )

        return ResponseStream(  # type: ignore[return-value]
            partial(
                self._send_sse,
                packets=packets,
                audio_format=audio_format,
                sample_rate=sample_rate,
            ),
            content_type="text/event-stream",
        )

    async def _send_raw(
        self,
        stream_response: BaseHTTPResponse | ResponseStream,
        *,
        packets: list[bytes],
    ) -> None:
        """Write raw audio packets to the response stream.
        :param stream_response: Sanic response stream receiving the packets.
        :param packets: Raw audio packets to write.
        :return: None, after all packets were written.
        """
        raw_response: BaseHTTPResponse
        if isinstance(stream_response, ResponseStream):
            raw_response = cast("BaseHTTPResponse", stream_response.response)
        else:
            raw_response = stream_response

        for packet in packets:
            await raw_response.send(packet)

    async def _send_sse(
        self,
        stream_response: BaseHTTPResponse | ResponseStream,
        *,
        packets: list[bytes],
        audio_format: str,
        sample_rate: int,
    ) -> None:
        """
        Write speech audio events to the SSE response stream.
        :param stream_response: Sanic response stream receiving the events.
        :param packets: Audio packets to encode and send.
        :param audio_format: Format of the audio packets.
        :param sample_rate: Audio sample rate in Hz.
        :return: None, after the audio-done event was written.
        """
        sse_response: BaseHTTPResponse
        if isinstance(stream_response, ResponseStream):
            sse_response = cast("BaseHTTPResponse", stream_response.response)
        else:
            sse_response = stream_response

        for packet in packets:
            event = {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(packet).decode("ascii"),
                "response_format": audio_format,
                "sample_rate": sample_rate,
                "channels": 1,
            }
            await sse_response.send(
                f"event: speech.audio.delta\ndata: {json.dumps(event)}\n\n"
            )

        done = {
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_token_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                },
            },
        }
        await sse_response.send(
            f"event: speech.audio.done\ndata: {json.dumps(done)}\n\n"
        )

    def _setup_middleware(self):
        """Setup middleware for CORS, logging, etc."""

        @self.app.middleware("request")
        async def log_request_received(request: Request) -> None:
            """Log request arrival when log_request_received is enabled."""
            if self.config.log_request_received:
                logger.info(
                    "Request received: %s %s from %s",
                    request.method,
                    request.path,
                    request.ip,
                )

        @self.app.middleware("request")
        async def add_cors_headers(_request: Request) -> None:
            """Add CORS headers to all requests."""
            return None  # noqa: RET501

        @self.app.middleware("response")
        async def add_response_headers(
            _request: Any, resp: BaseHTTPResponse
        ) -> HTTPResponse:
            """Add standard response headers."""
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Server"] = "guidellm-mock-server"
            return resp  # type: ignore[return-value]

    def _setup_routes(self):  # noqa: C901
        @self.app.get("/health")
        async def health_check(_request: Request):
            return response.json({"status": "healthy", "timestamp": time.time()})

        @self.app.get("/v1/models")
        async def list_models(_request: Request):
            return response.json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.config.model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "guidellm-mock",
                        }
                    ],
                }
            )

        @self.app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
        async def chat_completions(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self._run_generation(self.chat_handler.handle, request)

        @self.app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
        async def audio_speech(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self._run_generation(self._handle_audio_speech, request)

        @self.app.route("/v1/completions", methods=["POST", "OPTIONS"])
        async def completions(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self._run_generation(self.completions_handler.handle, request)

        @self.app.route("/v1/responses", methods=["POST", "OPTIONS"])
        async def responses(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self._run_generation(self.responses_handler.handle, request)

        @self.app.route("/tokenize", methods=["POST", "OPTIONS"])
        async def tokenize(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self.tokenizer_handler.tokenize(request)

        @self.app.route("/detokenize", methods=["POST", "OPTIONS"])
        async def detokenize(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self.tokenizer_handler.detokenize(request)

        @self.app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
        async def audio_transcriptions(request: Request) -> HTTPResponse:
            """
            Mock OpenAI audio transcription endpoint:
            - receives multipart/form-data
            - file field contains audio file
            - model field is optional, default to "mock-model"
            - returns "transcribed text"
            """
            if request.method == "OPTIONS":
                return response.text("", status=204)
            if request.files is None or request.form is None:
                return response.json({"error": "No form data provided"}, status=400)
            file: File | None = request.files.get("file")
            if "file" not in request.files or "model" not in request.form:
                return response.json(
                    {"error": "Missing 'file' in form-data"}, status=400
                )

            file = cast("File", file)
            model = request.form.get("model", "mock-model")
            text = f"Mock transcription for {file.name}"

            return response.json(
                {
                    "text": text,
                    "file_size": len(file.body),
                    "model_used": model,
                    "transcription": f"Transcribed({file.name}) using {model}",
                    "usage": self._audio_usage(file, text),
                }
            )

        @self.app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
        async def audio_translations(request: Request) -> HTTPResponse:
            """
            Mock OpenAI audio translation endpoint:
            - receives multipart/form-data
            - file field contains audio file
            - model field is optional, default to "mock-model"
            - returns translated text
            """
            if request.method == "OPTIONS":
                return response.text("", status=204)
            if request.files is None or request.form is None:
                return response.json({"error": "No form data provided"}, status=400)
            file: File | None = request.files.get("file")
            if "file" not in request.files or "model" not in request.form:
                return response.json(
                    {"error": "Missing 'file' in form-data"}, status=400
                )

            file = cast("File", file)
            decoded_text = (
                "This is a mock translation result."  # mock output tranlated text
            )

            return response.json(
                {
                    "text": decoded_text,
                    "file_size": len(file.body),
                    "filename": file.name,
                    "model_used": request.form.get("model", "mock-model"),
                    "mimetype": file.type,
                    "usage": self._audio_usage(file, decoded_text),
                }
            )

    def _setup_error_handlers(self):
        """Setup error handlers."""

        @self.app.exception(Exception)
        async def generic_error_handler(_request: Request, exception: Exception):
            logger.error("Unhandled exception: %s", exception)
            return response.json(
                {
                    "error": {
                        "message": "Internal server error",
                        "type": type(exception).__name__,
                        "error": str(exception),
                    }
                },
                status=500,
            )

        @self.app.exception(NotFound)
        async def not_found_handler(_request: Request, _exception):
            return response.json(
                {
                    "error": {
                        "message": "Not Found",
                        "type": "not_found_error",
                        "code": "not_found",
                    }
                },
                status=404,
            )

    def run(self, *, access_log: bool = True) -> None:
        """
        Start the mock server with configured settings.

        Runs the Sanic application in single-process mode with access logging and
        the Sanic startup MOTD enabled by default. Pass ``access_log=False`` in
        shared-TTY test runners to disable access logs and the MOTD, and to use
        plain (non-ANSI) Sanic formatters so start/stop logs do not overwrite
        the terminal.

        :param access_log: Whether to enable Sanic per-request access logging and
            the startup MOTD banner. When false, start/stop logs still print but
            without ANSI cursor controls.
        """
        # Reconfigure after Sanic.__init__ so quiet mode applies before serve.
        _configure_sanic_logging(access_log=access_log)
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            single_process=True,
            access_log=access_log,
            motd=access_log,
            register_sys_signals=True,
        )
