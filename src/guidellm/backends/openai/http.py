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
import json
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx
from loguru import logger

from guidellm.backends.backend import Backend
from guidellm.backends.openai.common import FALLBACK_TIMEOUT
from guidellm.backends.openai.request_handlers import (
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)
from guidellm.schemas.backends import OpenAIHTTPBackendArgs
from guidellm.utils.dict import deep_filter

__all__ = [
    "OpenAIHTTPBackend",
]


_SERVER_CONFIG_TIMEOUT = 5.0
_SERVER_CONFIG_MAX_BYTES = 1024 * 1024


def _redact_server_config(value: Any) -> Any:
    """Redact common credential fields without hiding token-count settings."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            sensitive = (
                normalized in {"token", "authorization", "credentials"}
                or any(
                    part in normalized
                    for part in (
                        "api_key",
                        "apikey",
                        "password",
                        "secret",
                        "access_key",
                    )
                )
                or normalized.endswith(("_token", "_credentials"))
            )
            result[key] = "[REDACTED]" if sensitive else _redact_server_config(item)
        return result
    if isinstance(value, list):
        return [_redact_server_config(item) for item in value]
    return value


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

    _args: OpenAIHTTPBackendArgs

    def __init__(
        self,
        arguments: OpenAIHTTPBackendArgs,
    ):
        """
        Initialize OpenAI HTTP backend with server configuration.
        """
        super().__init__(arguments)

        # Runtime state
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None
        self._server_config_attempted = False
        self._server_info: dict[str, Any] | None = None

    @property
    def info(self) -> dict[str, Any]:
        """Return backend arguments and the optional server configuration snapshot.

        :return: JSON-serializable backend metadata for benchmark results
        """
        info = super().info
        if self._server_info is not None:
            info["server_info"] = deepcopy(self._server_info)
        return info

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

        if self._args.validate_backend:
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
                    "Backend validation request failed. Could not connect to the "
                    "server or validate the backend configuration."
                ) from exc

        if self._args.capture_server_config and not self._server_config_attempted:
            # resolve_backend validates in the parent before scheduling. Keep this
            # flag when pickled so workers do not repeat the optional request.
            self._server_config_attempted = True
            try:
                self._server_info = await asyncio.wait_for(
                    self._fetch_server_config(), timeout=_SERVER_CONFIG_TIMEOUT
                )
            except (
                httpx.HTTPError,
                ValueError,
                RecursionError,
                asyncio.TimeoutError,
            ) as exc:
                # URLs, response bodies and exception messages may contain secrets.
                logger.warning(
                    "Server configuration was not recorded ({}). "
                    "Benchmarking will continue without it.",
                    type(exc).__name__,
                )

    async def _fetch_server_config(self) -> dict[str, Any]:
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        route = self._args.api_routes["/server_info"].lstrip("/")
        headers = self._args.extras.headers if self._args.extras else None
        # Bound both elapsed time and decoded body size, even for streaming or
        # compressed responses. Never follow a metadata redirect to another host.
        async with self._async_client.stream(
            "GET",
            f"{self._args.target}/{route}",
            params={"config_format": "json"},
            headers=self._build_headers(headers),
            timeout=_SERVER_CONFIG_TIMEOUT,
            follow_redirects=False,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > _SERVER_CONFIG_MAX_BYTES:
                    raise ValueError("Server configuration exceeds the size limit")
                body.extend(chunk)

        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("vllm_config"), dict
        ):
            raise ValueError("Expected structured vllm_config")

        # Environment dumps are not needed for this feature and can contain
        # private deployment details. Do not persist legacy repr strings either.
        return {"vllm_config": _redact_server_config(payload["vllm_config"])}

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

        (
            request_handler,
            arguments,
            request_kwargs,
        ) = await self._prepare_resolve_request(request, history)

        if not arguments.stream:
            async for item in self._resolve_non_streaming(
                request, request_info, request_handler, arguments, request_kwargs
            ):
                yield item
            return

        async for item in self._resolve_streaming(
            request, request_info, request_handler, arguments, request_kwargs
        ):
            yield item

    async def _prepare_resolve_request(
        self,
        request: GenerationRequest,
        history: list[tuple[GenerationRequest, GenerationResponse | None]]
        | None = None,
    ) -> tuple[
        OpenAIRequestHandler,
        GenerationRequestArguments,
        dict[str, Any],
    ]:
        """
        Build the request handler, format arguments, and prepare HTTP kwargs.

        :param request: Generation request with content and parameters
        :param history: Optional conversation history for multi-turn requests
        :return: Tuple of (request_handler, formatted_arguments, http_kwargs)
        :raises ValueError: If request format is unsupported
        """
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
            multiturn_reasoning=self._args.multiturn_reasoning,
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

        request_kwargs: dict[str, Any] = {
            "url": request_url,
            "method": arguments.method or "POST",
            "params": arguments.params,
            "headers": self._build_headers(arguments.headers),
            "json": request_json,
            "data": request_data,
            "files": request_files,
        }

        return request_handler, arguments, request_kwargs

    async def _resolve_non_streaming(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        request_handler: OpenAIRequestHandler,
        arguments: GenerationRequestArguments,
        request_kwargs: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Handle a non-streaming generation request.

        :param request: The original generation request
        :param request_info: Request tracking info updated with timing metadata
        :param request_handler: Handler for compiling the response
        :param arguments: Formatted request arguments
        :param request_kwargs: Prepared HTTP request keyword arguments
        :yields: Single (response, request_info) tuple
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        request_info.timings.request_start = time.time()
        response = await self._async_client.request(**request_kwargs)
        request_info.timings.request_end = time.time()
        response.raise_for_status()
        data = response.json()
        gen_response = request_handler.compile_non_streaming(request, arguments, data)
        request_handler.post_validation(gen_response)
        try:
            self._check_tool_call_expectations(request, gen_response)
        except asyncio.CancelledError:
            yield gen_response, request_info
            raise
        yield gen_response, request_info

    async def _resolve_streaming(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        request_handler: OpenAIRequestHandler,
        arguments: GenerationRequestArguments,
        request_kwargs: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Handle a streaming generation request with progressive timing updates.

        :param request: The original generation request
        :param request_info: Request tracking info updated with timing metadata
        :param request_handler: Handler for processing stream lines and compiling
        :param arguments: Formatted request arguments
        :param request_kwargs: Prepared HTTP request keyword arguments
        :yields: Tuples of (response, request_info) as generation progresses
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        try:
            request_info.timings.request_start = time.time()

            async with self._async_client.stream(**request_kwargs) as stream:
                stream.raise_for_status()
                end_reached = False

                async for chunk in self._aiter_lines(stream):
                    stream.raise_for_status()
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

                    # TTFOT: record the first content (non-reasoning) token.
                    # For non-reasoning models this fires on the same iteration
                    # as first_token_iteration, making TTFOT == TTFT.
                    if (
                        request_info.timings.first_output_token_iteration is None
                        and request_handler.last_iteration_had_content
                    ):
                        request_info.timings.first_output_token_iteration = iter_time

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations

            request_info.timings.request_end = time.time()
            gen_response = request_handler.compile_streaming(request, arguments)
            request_handler.post_validation(gen_response)
            self._check_tool_call_expectations(request, gen_response)
            yield gen_response, request_info
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

    def _check_tool_call_expectations(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> None:
        """Validate that a tool-call turn actually produced tool calls.

        Called before the final yield in ``resolve`` so that any raised
        exception prevents the normal yield and is instead handled by the
        ``except`` block (which yields the response once before propagating).
        When the request expected a tool call but the model didn't produce one,
        raises an exception according to ``tool_call_missing_behavior``:

        * ``ignore_continue`` -- no-op; the conversation proceeds normally.
        * ``ignore_stop`` -- raises :class:`asyncio.CancelledError` so the
          worker cancels remaining turns.
        * ``error_stop`` -- raises :class:`ValueError` so the worker marks
          the current turn as errored and cancels remaining turns.

        :param request: The generation request that was resolved.
        :param response: The compiled response from the model.
        """
        if request.turn_type != "client_tool_call" or response.tool_calls:
            return

        behavior = self._args.tool_call_missing_behavior
        if behavior == "ignore_continue":
            pass
        elif behavior == "ignore_stop":
            raise asyncio.CancelledError("Expected tool call but model produced none")
        elif behavior == "error_stop":
            raise ValueError("Expected tool call but model produced none")
