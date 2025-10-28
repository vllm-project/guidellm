"""
High-performance mock server for OpenAI and vLLM API compatibility testing.

This module provides a Sanic-based mock server that simulates OpenAI and vLLM APIs
with configurable latency, token generation patterns, and response characteristics.
The server supports both streaming and non-streaming endpoints, enabling realistic
performance testing and validation of GuideLLM benchmarking workflows without
requiring actual model deployments.
"""

from __future__ import annotations

import time

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic.log import logger
from sanic.request import Request
from sanic.response import HTTPResponse

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.handlers import (
    ChatCompletionsHandler,
    CompletionsHandler,
    TokenizerHandler,
)

__all__ = ["MockServer"]


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
        self.tokenizer_handler = TokenizerHandler(config)

        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()

    def _setup_middleware(self):
        """Setup middleware for CORS, logging, etc."""

        @self.app.middleware("request")
        async def add_cors_headers(_request: Request):
            """Add CORS headers to all requests."""

        @self.app.middleware("response")
        async def add_response_headers(_request: Request, resp: HTTPResponse):
            """Add standard response headers."""
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Server"] = "guidellm-mock-server"

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
            return await self.chat_handler.handle(request)

        @self.app.route("/v1/completions", methods=["POST", "OPTIONS"])
        async def completions(request: Request):
            if request.method == "OPTIONS":
                return response.text("", status=204)
            return await self.completions_handler.handle(request)

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

    def _setup_error_handlers(self):
        """Setup error handlers."""

        @self.app.exception(Exception)
        async def generic_error_handler(_request: Request, exception: Exception):
            logger.error(f"Unhandled exception: {exception}")
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

    def run(self) -> None:
        """
        Start the mock server with configured settings.

        Runs the Sanic application in single-process mode with access logging enabled
        for debugging and monitoring request patterns during testing.
        """
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            single_process=True,
            access_log=True,
            register_sys_signals=False,  # Disable signal handlers for threading
        )
