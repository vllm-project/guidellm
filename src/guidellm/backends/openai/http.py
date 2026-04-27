"""
OpenAI HTTP backend implementation for GuideLLM.

Provides HTTP-based backend for OpenAI-compatible servers including OpenAI API,
vLLM servers, and other compatible inference engines. Supports text and chat
completions with streaming, authentication, and multimodal capabilities.
Handles request formatting, response parsing, error handling, and token usage
tracking with flexible parameter customization.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.request_handlers import OpenAIRequestHandlerFactory
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)
from guidellm.utils.dict import deep_filter

__all__ = [
    "OpenAIHTTPBackend",
    "OpenAIHTTPBackendArgs",
]

# NOTE: This value is taken from httpx's default
FALLBACK_TIMEOUT = 5.0

DEFAULT_API_PATHS = {
    "/health": "health",
    "/v1/models": "v1/models",
    "/v1/completions": "v1/completions",
    "/v1/chat/completions": "v1/chat/completions",
    "/v1/embeddings": "v1/embeddings",
    "/v1/responses": "v1/responses",
    "/v1/audio/transcriptions": "v1/audio/transcriptions",
    "/v1/audio/translations": "v1/audio/translations",
    "/pooling": "pooling",
}


@BackendArgs.register("openai_http")
class OpenAIHTTPBackendArgs(BackendArgs):
    """Pydantic model for OpenAI HTTP backend creation arguments."""

    type_: Literal["openai_http"] = Field(
        alias="type",
        default="openai_http",
        description="Type identifier for the backend configuration.",
    )
    target: str = Field(
        description="Base URL of the OpenAI-compatible server",
    )
    model: str = Field(
        default_factory=str,
        description="Model identifier for generation requests",
    )
    request_format: Literal[
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/responses",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/pooling",
    ] = Field(
        default="/v1/chat/completions",
        description="Request format for OpenAI-compatible server.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for authentication (for Bearer auth)",
    )
    api_routes: dict[str, str] = Field(
        default_factory=dict,
        validate_default=True,
        description=(
            "Custom API endpoint routes mapping. Keys should be request types "
            "like '/v1/completions' and values should be the corresponding "
            "endpoint paths relative to the target URL."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds for reading response.",
    )
    timeout_connect: float | None = Field(
        default=FALLBACK_TIMEOUT,
        description="Request timeout in seconds for establishing connection.",
    )
    http2: bool = Field(
        default=True,
        description="Enable HTTP/2 protocol.",
    )
    follow_redirects: bool = Field(
        default=True,
        description="Follow HTTP redirects automatically.",
    )
    verify: bool = Field(
        default=False,
        description="Verify the server's TLS certificate.",
    )
    validate_backend: bool = Field(
        default=True,
        description="Send a health check request to validate backend configuration.",
    )
    stream: bool = Field(
        default=True,
        description="Use streaming responses for generation requests when supported.",
    )
    extras: GenerationRequestArguments | None = Field(
        default=None,
        description="Additional parameters to include in generation requests.",
    )
    max_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
        description="Maximum number of tokens to request in any response.",
    )
    server_history: bool = Field(
        default=False,
        description=(
            "Use server-side conversation history (previous_response_id) for "
            "multi-turn requests. Only supported with /v1/responses."
        ),
    )
    tool_call_missing_behavior: str = Field(
        default="error_stop",
        description=(
            "What the worker does when a tool call is expected but the model "
            "does not produce one. Options: ignore_continue (continue to next "
            "turn), ignore_stop (cancel remaining turns), error_stop (error "
            "and cancel remaining turns)."
        ),
    )

    @field_validator("tool_call_missing_behavior")
    @classmethod
    def validate_tool_call_missing_behavior(cls, v: str) -> str:
        valid = {"ignore_continue", "ignore_stop", "error_stop"}
        if v not in valid:
            raise ValueError(
                f"Invalid tool_call_missing_behavior '{v}'. "
                f"Must be one of: {', '.join(sorted(valid))}"
            )
        return v

    @field_validator("target", mode="after")
    @classmethod
    def strip_target(cls, value: str) -> str:
        """Strip trailing slashes and API paths from the target URL."""
        return value.rstrip("/").removesuffix("/v1")

    @field_validator("api_routes", mode="after")
    @classmethod
    def merge_api_routes(cls, value: dict[str, str]) -> dict[str, str]:
        """Merge user-provided API routes with default routes."""
        return DEFAULT_API_PATHS | value

    @model_validator(mode="after")
    def validate_server_history(self):
        """Validate that server_history is only True with supported endpoints."""
        if self.server_history and self.request_format != "/v1/responses":
            raise ValueError(
                "server_history=True is only supported with the /v1/responses "
                "request format. Current request_format: "
                f"'{self.request_format}'"
            )
        return self


@Backend.register("openai_http")
class OpenAIHTTPBackend(Backend):
    """
    HTTP backend for OpenAI-compatible servers.

    Supports OpenAI API, vLLM servers, and other compatible endpoints with
    text/chat completions, streaming, authentication, and multimodal inputs.
    Handles request formatting, response parsing, error handling, and token
    usage tracking with flexible parameter customization.

    Example:
    ::
        backend_args = OpenAIHTTPBackendArgs(
            target="http://localhost:8000",
            model="gpt-3.5-turbo",
            api_key="your-api-key",
        )
        backend = OpenAIHTTPBackend(backend_args)

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            process_response(response)
        await backend.process_shutdown()
    """

    def __init__(
        self,
        arguments: OpenAIHTTPBackendArgs,
    ):
        """
        Initialize OpenAI HTTP backend with server configuration.
        """
        super().__init__(arguments)
        self._args = arguments

        # Runtime state
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None

    @property
    def info(self) -> dict[str, Any]:
        """
        Get backend configuration details.

        :return: Dictionary containing backend configuration details
        """
        return self._args.model_dump()

    async def process_startup(self):
        """
        Initialize HTTP client and backend resources.

        :raises RuntimeError: If backend is already initialized
        :raises httpx.RequestError: If HTTP client cannot be created
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        self._async_client = httpx.AsyncClient(
            http2=self._args.http2,
            timeout=httpx.Timeout(
                FALLBACK_TIMEOUT,
                read=self._args.timeout,
                connect=self._args.timeout_connect,
            ),
            follow_redirects=self._args.follow_redirects,
            verify=self._args.verify,
            # Allow unlimited connections
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
                keepalive_expiry=5.0,  # default
            ),
        )
        self._in_process = True

    async def process_shutdown(self):
        """
        Clean up HTTP client and backend resources.

        :raises RuntimeError: If backend was not properly initialized
        :raises httpx.RequestError: If HTTP client cannot be closed
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        await self._async_client.aclose()  # type: ignore [union-attr]
        self._async_client = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend connectivity and configuration.

        :raises RuntimeError: If backend cannot connect or validate configuration
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if not self._args.validate_backend:
            return

        try:
            validate_kwargs: dict[str, Any] = {
                "method": "GET",
                "url": f"{self._args.target}/{self._args.api_routes['/health']}",
            }
            existing_headers = validate_kwargs.get("headers")
            built_headers = self._build_headers(existing_headers)
            validate_kwargs["headers"] = built_headers
            response = await self._async_client.request(**validate_kwargs)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the server "
                "or validate the backend configuration."
            ) from exc

    async def available_models(self) -> list[str]:
        """
        Get available models from the target server.

        :return: List of model identifiers
        :raises httpx.HTTPError: If models endpoint returns an error
        :raises RuntimeError: If backend is not initialized
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        target = f"{self._args.target}/{self._args.api_routes['/v1/models']}"
        response = await self._async_client.get(target, headers=self._build_headers())
        response.raise_for_status()

        return [item["id"] for item in response.json()["data"]]

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: Model name or None if no model is available
        """
        if self._args.model or not self._in_process:
            return self._args.model

        models = await self.available_models()
        self._args.model = models[0] if models else ""
        return self._args.model

    async def resolve(  # type: ignore[override, misc]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse | None]]
        | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Process generation request and yield progressive responses.

        Handles request formatting, timing tracking, API communication, and
        response parsing with streaming support.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialized
        :raises ValueError: If request type is unsupported
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if (
            request_path := self._args.api_routes.get(self._args.request_format)
        ) is None:
            raise ValueError(
                f"Unsupported request format '{self._args.request_format}'"
            )

        request_handler = OpenAIRequestHandlerFactory.create(
            self._args.request_format,
        )
        arguments: GenerationRequestArguments = request_handler.format(
            data=request,
            history=history,
            model=(await self.default_model()),
            stream=self._args.stream,
            extras=self._args.extras,
            max_tokens=self._args.max_tokens,
            server_history=self._args.server_history,
        )

        request_url = f"{self._args.target}/{request_path}"
        request_files = (
            {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in arguments.files.items()
            }
            if arguments.files
            else None
        )
        # Omit `None` from output JSON
        deep_filter(arguments.body or {}, lambda _, v: v is not None)
        request_json = arguments.body if not request_files else None
        request_data = arguments.body if request_files else None

        if not arguments.stream:
            request_info.timings.request_start = time.time()
            response = await self._async_client.request(
                arguments.method or "POST",
                request_url,
                params=arguments.params,
                headers=self._build_headers(arguments.headers),
                json=request_json,
                data=request_data,
                files=request_files,
            )
            request_info.timings.request_end = time.time()
            response.raise_for_status()
            data = response.json()
            yield (
                request_handler.compile_non_streaming(request, arguments, data),
                request_info,
            )
            return

        try:
            request_info.timings.request_start = time.time()

            async with self._async_client.stream(
                arguments.method or "POST",
                request_url,
                params=arguments.params,
                headers=self._build_headers(arguments.headers),
                json=request_json,
                data=request_data,
                files=request_files,
            ) as stream:
                stream.raise_for_status()
                end_reached = False

                async for chunk in self._aiter_lines(stream):
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1

                    iterations = request_handler.add_streaming_line(chunk)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        if end_reached:
                            # Break eagerly once the handler signals completion
                            # (e.g. "data: [DONE]" or "response.completed").
                            # Using continue instead would hang on servers that
                            # keep the HTTP/2 stream open after the last event.
                            break
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0
                        yield None, request_info

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations

            request_info.timings.request_end = time.time()
            yield request_handler.compile_streaming(request, arguments), request_info
        except asyncio.CancelledError as err:
            # Yield current result to store iterative results before propagating
            yield request_handler.compile_streaming(request, arguments), request_info
            raise err

    async def _aiter_lines(self, stream: httpx.Response) -> AsyncIterator[str]:
        """
        Asynchronously iterate over lines in an HTTP response stream.

        :param stream: HTTP response object with streaming content
        :yield: Lines of text from the response stream
        """
        async for line in stream.aiter_lines():
            if not line.strip():
                continue  # Skip blank lines
            yield line

    def _build_headers(
        self, existing_headers: dict[str, str] | None = None
    ) -> dict[str, str] | None:
        """
        Build headers dictionary with bearer token authentication.

        Merges the Authorization bearer token header (if api_key is set) with any
        existing headers. User-provided headers take precedence over the bearer token.

        :param existing_headers: Optional existing headers to merge with
        :return: Dictionary of headers with bearer token included if api_key is set
        """
        headers: dict[str, str] = {}

        # Add bearer token if api_key is set
        if self._args.api_key:
            token = self._args.api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"

        # Merge with existing headers (user headers take precedence)
        if existing_headers:
            headers = {**headers, **existing_headers}

        return headers or None
