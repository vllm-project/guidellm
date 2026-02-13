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
from typing import Any

import httpx

from guidellm.backends.backend import Backend
from guidellm.backends.openai.request_handlers import OpenAIRequestHandlerFactory
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)

__all__ = [
    "OpenAIHTTPBackend",
]


DEFAULT_API_PATHS = {
    "/health": "health",
    "/v1/models": "v1/models",
    "/v1/completions": "v1/completions",
    "/v1/chat/completions": "v1/chat/completions",
    "/v1/audio/transcriptions": "v1/audio/transcriptions",
    "/v1/audio/translations": "v1/audio/translations",
    "/v1/embeddings": "v1/embeddings",
    "embeddings": "v1/embeddings",  # Alias for convenience
}

DEFAULT_API = "/v1/chat/completions"

# Legacy aliases for common API paths
LEGACY_API_ALIASES = {
    "text_completions": "/v1/completions",
    "chat_completions": "/v1/chat/completions",
    "audio_transcriptions": "/v1/audio/transcriptions",
    "audio_translations": "/v1/audio/translations",
}

# NOTE: This value is taken from httpx's default
FALLBACK_TIMEOUT = 5.0


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
        backend = OpenAIHTTPBackend(
            target="http://localhost:8000",
            model="gpt-3.5-turbo",
            api_key="your-api-key"
        )

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            process_response(response)
        await backend.process_shutdown()
    """

    def __init__(
        self,
        target: str,
        model: str = "",
        request_format: str | None = None,
        api_key: str | None = None,
        api_routes: dict[str, str] | None = None,
        request_handlers: dict[str, Any] | None = None,
        timeout: float | None = None,
        timeout_connect: float | None = FALLBACK_TIMEOUT,
        http2: bool = True,
        follow_redirects: bool = True,
        verify: bool = False,
        validate_backend: bool | str | dict[str, Any] = True,
        stream: bool = True,
        extras: dict[str, Any] | GenerationRequestArguments | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ):
        """
        Initialize OpenAI HTTP backend with server configuration.

        :param target: Base URL of the OpenAI-compatible server
        :param model: Model identifier for generation requests
        :param api_key: API key for authentication (for Bearer auth)
        :param api_routes: Custom API endpoint routes mapping
        :param response_handlers: Custom response handlers for different request types
        :param timeout: Request timeout in seconds
        :param http2: Enable HTTP/2 protocol support
        :param follow_redirects: Follow HTTP redirects automatically
        :param verify: Enable SSL certificate verification
        :param validate_backend: Backend validation configuration
        """
        super().__init__(type_="openai_http")

        # Request Values
        self.target = target.rstrip("/").removesuffix("/v1")
        self.model = model
        self.api_key = api_key

        # Resolve request format
        if request_format is None:
            request_format = DEFAULT_API
        elif request_format in LEGACY_API_ALIASES:
            request_format = LEGACY_API_ALIASES[request_format]

        # Validate that the request handler exists
        valid_formats = OpenAIRequestHandlerFactory.registered_names()
        if request_format not in valid_formats:
            raise ValueError(
                f"Invalid request_format '{request_format}'. Must be one of: "
                f"{', '.join(valid_formats)}"
            )
        self.request_type = request_format

        # Store configuration
        self.api_routes = api_routes or DEFAULT_API_PATHS
        self.request_handlers = request_handlers
        self.timeout = timeout
        self.timeout_connect = timeout_connect
        self.http2 = http2
        self.follow_redirects = follow_redirects
        self.verify = verify
        self.validate_backend: dict[str, Any] | None = self._resolve_validate_kwargs(
            validate_backend
        )
        self.stream: bool = stream
        self.extras = (
            GenerationRequestArguments(**extras)
            if extras and isinstance(extras, dict)
            else extras
        )
        self.max_tokens: int | None = max_tokens or max_completion_tokens

        # Runtime state
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None

    @property
    def info(self) -> dict[str, Any]:
        """
        Get backend configuration details.

        :return: Dictionary containing backend configuration details
        """
        return {
            "target": self.target,
            "model": self.model,
            "timeout": self.timeout,
            "timeout_connect": self.timeout_connect,
            "http2": self.http2,
            "follow_redirects": self.follow_redirects,
            "verify": self.verify,
            "openai_paths": self.api_routes,
            "validate_backend": self.validate_backend,
            # Auth token excluded for security
        }

    async def process_startup(self):
        """
        Initialize HTTP client and backend resources.

        :raises RuntimeError: If backend is already initialized
        :raises httpx.RequestError: If HTTP client cannot be created
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        self._async_client = httpx.AsyncClient(
            http2=self.http2,
            timeout=httpx.Timeout(
                FALLBACK_TIMEOUT,
                read=self.timeout,
                connect=self.timeout_connect,
            ),
            follow_redirects=self.follow_redirects,
            verify=self.verify,
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

        if not self.validate_backend:
            return

        try:
            # Merge bearer token headers into validate_backend dict
            validate_kwargs = {**self.validate_backend}
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

        target = f"{self.target}/{self.api_routes['/v1/models']}"
        response = await self._async_client.get(target, headers=self._build_headers())
        response.raise_for_status()

        return [item["id"] for item in response.json()["data"]]

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: Model name or None if no model is available
        """
        if self.model or not self._in_process:
            return self.model

        models = await self.available_models()
        self.model = models[0] if models else ""
        return self.model

    async def resolve(  # type: ignore[override]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
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

        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")

        if (request_path := self.api_routes.get(self.request_type)) is None:
            raise ValueError(f"Unsupported request type '{self.request_type}'")

        request_handler = OpenAIRequestHandlerFactory.create(
            self.request_type, handler_overrides=self.request_handlers
        )
        arguments: GenerationRequestArguments = request_handler.format(
            request,
            model=(await self.default_model()),
            stream=self.stream,
            extras=self.extras,
            max_tokens=self.max_tokens,
        )

        request_url = f"{self.target}/{request_path}"
        request_files = (
            {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in arguments.files.items()
            }
            if arguments.files
            else None
        )
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

                async for chunk in stream.aiter_lines():
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1

                    iterations = request_handler.add_streaming_line(chunk)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations

            request_info.timings.request_end = time.time()
            yield request_handler.compile_streaming(request, arguments), request_info
        except asyncio.CancelledError as err:
            # Yield current result to store iterative results before propagating
            yield request_handler.compile_streaming(request, arguments), request_info
            raise err

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
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Merge with existing headers (user headers take precedence)
        if existing_headers:
            headers = {**headers, **existing_headers}

        return headers or None

    def _resolve_validate_kwargs(
        self, validate_backend: bool | str | dict[str, Any]
    ) -> dict[str, Any] | None:
        if not (validate_kwargs := validate_backend):
            return None

        if validate_kwargs is True:
            validate_kwargs = "/health"

        if isinstance(validate_kwargs, str) and validate_kwargs in self.api_routes:
            validate_kwargs = f"{self.target}/{self.api_routes[validate_kwargs]}"

        if isinstance(validate_kwargs, str):
            validate_kwargs = {
                "method": "GET",
                "url": validate_kwargs,
            }

        if not isinstance(validate_kwargs, dict) or "url" not in validate_kwargs:
            raise ValueError(
                "validate_backend must be a boolean, string, or dictionary and contain "
                f"a target URL. Got: {validate_kwargs}"
            )

        if "method" not in validate_kwargs:
            validate_kwargs["method"] = "GET"

        return validate_kwargs
