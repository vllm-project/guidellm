"""
OpenAI HTTP backend implementation for GuideLLM.

Provides HTTP-based backend for OpenAI-compatible servers including OpenAI API,
vLLM servers, and other compatible inference engines. Supports text and chat
completions with streaming, authentication, and multimodal capabilities.

Classes:
    UsageStats: Token usage statistics for generation requests.
    OpenAIHTTPBackend: HTTP backend for OpenAI-compatible API servers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from guidellm.backends.backend import Backend
from guidellm.backends.response_handlers import GenerationResponseHandlerFactory
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False

__all__ = ["OpenAIHTTPBackend"]


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
        model: str | None = None,
        api_routes: dict[str, str] | None = None,
        response_handlers: dict[str, Any] | None = None,
        timeout: float = 60.0,
        http2: bool = True,
        follow_redirects: bool = True,
        verify: bool = False,
        validate_backend: bool | str | dict[str, Any] = True,
    ):
        super().__init__(type_="openai_http")

        # Request Values
        self.target = target.rstrip("/").removesuffix("/v1")
        self.model = model

        # Store configuration
        self.api_routes = api_routes or {
            "health": "health",
            "models": "v1/models",
            "text_completions": "v1/completions",
            "chat_completions": "v1/chat/completions",
            "audio_transcriptions": "v1/audio/transcriptions",
            "audio_translations": "v1/audio/translations",
        }
        self.response_handlers = response_handlers
        self.timeout = timeout
        self.http2 = http2
        self.follow_redirects = follow_redirects
        self.verify = verify
        self.validate_backend: dict[str, Any] | None = self._resolve_validate_kwargs(
            validate_backend
        )

        # Runtime state
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: Dictionary containing backend configuration details.
        """
        return {
            "target": self.target,
            "model": self.model,
            "timeout": self.timeout,
            "http2": self.http2,
            "follow_redirects": self.follow_redirects,
            "verify": self.verify,
            "openai_paths": self.api_routes,
            "validate_backend": self.validate_backend,
        }

    async def process_startup(self):
        """
        Initialize HTTP client and backend resources.

        :raises RuntimeError: If backend is already initialized.
        :raises httpx.Exception: If HTTP client cannot be created.
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        self._async_client = httpx.AsyncClient(
            http2=self.http2,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            verify=self.verify,
        )
        self._in_process = True

    async def process_shutdown(self):
        """
        Clean up HTTP client and backend resources.

        :raises RuntimeError: If backend was not properly initialized.
        :raises httpx.Exception: If HTTP client cannot be closed.
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        await self._async_client.aclose()  # type: ignore [union-attr]
        self._async_client = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend configuration and connectivity.

        Validate backend configuration and connectivity through test requests,
        and auto-selects first available model if none is configured.

        :raises RuntimeError: If backend cannot connect or validate configuration.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if not self.validate_backend:
            return

        try:
            response = await self._async_client.request(**self.validate_backend)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the server "
                "or validate the backend configuration."
            ) from exc

    async def available_models(self) -> list[str]:
        """
        Get available models from the target server.

        :return: List of model identifiers.
        :raises HTTPError: If models endpoint returns an error.
        :raises RuntimeError: If backend is not initialized.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        target = f"{self.target}/{self.api_routes['models']}"
        response = await self._async_client.get(target)
        response.raise_for_status()

        return [item["id"] for item in response.json()["data"]]

    async def default_model(self) -> str | None:
        """
        Get the default model for this backend.

        :return: Model name or None if no model is available.
        """
        if self.model or not self._in_process:
            return self.model

        models = await self.available_models()
        return models[0] if models else None

    async def resolve(  # noqa: C901
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process a generation request and yield progressive responses.

        Handles request formatting, timing tracking, API communication, and
        response parsing with streaming support.

        :param request: Generation request with content and parameters.
        :param request_info: Request tracking info updated with timing metadata.
        :param history: Conversation history. Currently not supported.
        :raises NotImplementedError: If history is provided.
        :yields: Tuples of (response, updated_request_info) as generation progresses.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if history is not None:
            raise NotImplementedError(
                "Multi-turn requests with conversation history are not yet supported"
            )

        response_handler = (
            self.response_handlers.get(request.request_type)
            if self.response_handlers
            else None
        )
        if response_handler is None:
            response_handler_class = (
                GenerationResponseHandlerFactory.get_registered_object(
                    request.request_type
                )
            )
            if response_handler_class is None:
                raise ValueError(
                    "No response handler registered for request type "
                    f"'{request.request_type}'"
                )
            response_handler = response_handler_class()

        if (request_path := self.api_routes.get(request.request_type)) is None:
            raise ValueError(f"Unsupported request type '{request.request_type}'")
        request_url = f"{self.target}/{request_path}"
        request_info.timings.request_start = time.time()

        if not request.arguments.stream:
            response = await self._async_client.request(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=request.arguments.headers,
                json=request.arguments.body if not request.arguments.files else None,
                data=request.arguments.body if request.arguments.files else None,
                files=(
                    {
                        key: tuple(value) if isinstance(value, list) else value
                        for key, value in request.arguments.files.items()
                    }
                    if request.arguments.files
                    else None
                ),
            )
            request_info.timings.request_end = time.time()
            response.raise_for_status()
            data = response.json()
            yield response_handler.compile_non_streaming(request, data), request_info
            return

        try:
            async with self._async_client.stream(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=request.arguments.headers,
                json=request.arguments.body if not request.arguments.files else None,
                data=request.arguments.body if request.arguments.files else None,
                files=(
                    {
                        key: tuple(value) if isinstance(value, list) else value
                        for key, value in request.arguments.files.items()
                    }
                    if request.arguments.files
                    else None
                ),
            ) as stream:
                stream.raise_for_status()
                end_reached = False

                async for chunk in stream.aiter_lines():
                    if end_reached:
                        continue

                    if (
                        iterations := response_handler.add_streaming_line(chunk)
                    ) is None or iterations < 0:
                        end_reached = end_reached or iterations is None
                        continue

                    if request_info.timings.first_iteration is None:
                        request_info.timings.first_iteration = time.time()
                    request_info.timings.last_iteration = time.time()

                    if request_info.timings.iterations is None:
                        request_info.timings.iterations = 0
                    request_info.timings.iterations += iterations

                request_info.timings.request_end = time.time()

            yield response_handler.compile_streaming(request), request_info
        except asyncio.CancelledError as err:
            yield response_handler.compile_streaming(request), request_info
            raise err

    def _resolve_validate_kwargs(
        self, validate_backend: bool | str | dict[str, Any]
    ) -> dict[str, Any] | None:
        if not (validate_kwargs := validate_backend):
            return None

        if validate_kwargs is True:
            validate_kwargs = "health"

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
