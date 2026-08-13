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
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic.log import logger
from sanic.logging.formatter import LegacyAccessFormatter, LegacyFormatter
from sanic.request import File, Request
from sanic.response import BaseHTTPResponse, HTTPResponse

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.handlers import (
    ChatCompletionsHandler,
    CompletionsHandler,
    ResponsesHandler,
    TokenizerHandler,
)

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
        async with semaphore:
            return await handler(request)

    def _setup_middleware(self):
        """Setup middleware for CORS, logging, etc."""

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

            return response.json(
                {
                    "text": f"Mock transcription for {file.name}",
                    "file_size": len(file.body),
                    "model_used": model,
                    "transcription": f"Transcribed({file.name}) using {model}",
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
                    "filename": {file.name},
                    "model_used": request.form.get("model", "mock-model"),
                    "mimetype": file.type,
                }
            )

    def _setup_error_handlers(self):
        """Setup error handlers."""

        @self.app.exception(Exception)
        async def generic_error_handler(_request: Request, exception: Exception):
            logger.error("Unhandled exception: {}", exception)
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
