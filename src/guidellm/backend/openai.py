"""
OpenAI HTTP backend implementation for GuideLLM.

Provides HTTP-based backend for OpenAI-compatible servers including OpenAI API,
vLLM servers, and other compatible inference engines. Supports text and chat
completions with streaming, authentication, and multimodal capabilities.

Classes:
    UsageStats: Token usage statistics for generation requests.
    OpenAIHTTPBackend: HTTP backend for OpenAI-compatible API servers.
"""

import base64
import contextlib
import copy
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import httpx
from PIL import Image
from pydantic import dataclasses

from guidellm.backend.backend import Backend
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import ScheduledRequestInfo

__all__ = ["OpenAIHTTPBackend", "UsageStats"]


@dataclasses.dataclass
class UsageStats:
    """Token usage statistics for generation requests."""

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


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

    HEALTH_PATH: ClassVar[str] = "/health"
    MODELS_PATH: ClassVar[str] = "/v1/models"
    TEXT_COMPLETIONS_PATH: ClassVar[str] = "/v1/completions"
    CHAT_COMPLETIONS_PATH: ClassVar[str] = "/v1/chat/completions"

    MODELS_KEY: ClassVar[str] = "models"
    TEXT_COMPLETIONS_KEY: ClassVar[str] = "text_completions"
    CHAT_COMPLETIONS_KEY: ClassVar[str] = "chat_completions"

    def __init__(
        self,
        target: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        http2: bool = True,
        follow_redirects: bool = True,
        max_output_tokens: Optional[int] = None,
        stream_response: bool = True,
        extra_query: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        remove_from_body: Optional[list[str]] = None,
        headers: Optional[dict] = None,
        verify: bool = False,
    ):
        """
        Initialize OpenAI HTTP backend.

        :param target: Target URL for the OpenAI server (e.g., "http://localhost:8000").
        :param model: Model to use for requests. If None, uses first available model.
        :param api_key: API key for authentication. Adds Authorization header
            if provided.
        :param organization: Organization ID. Adds OpenAI-Organization header
            if provided.
        :param project: Project ID. Adds OpenAI-Project header if provided.
        :param timeout: Request timeout in seconds. Defaults to 60 seconds.
        :param http2: Whether to use HTTP/2. Defaults to True.
        :param follow_redirects: Whether to follow redirects. Default True.
        :param max_output_tokens: Maximum tokens for completions. If None, none is set.
        :param stream_response: Whether to stream responses by default. Can be
            overridden per request. Defaults to True.
        :param extra_query: Additional query parameters. Both general and
            endpoint-specific with type keys supported.
        :param extra_body: Additional body parameters. Both general and
            endpoint-specific with type keys supported.
        :param remove_from_body: Parameter names to remove from request bodies.
        :param headers: Additional HTTP headers.
        :param verify: Whether to verify SSL certificates. Default False.
        """
        super().__init__(type_="openai_http")

        # Request Values
        self.target = target.rstrip("/").removesuffix("/v1")
        self.model = model
        self.headers = self._build_headers(api_key, organization, project, headers)

        # Store configuration
        self.timeout = timeout
        self.http2 = http2
        self.follow_redirects = follow_redirects
        self.verify = verify
        self.max_output_tokens = max_output_tokens
        self.stream_response = stream_response
        self.extra_query = extra_query or {}
        self.extra_body = extra_body or {}
        self.remove_from_body = remove_from_body or []

        # Runtime state
        self._in_process = False
        self._async_client: Optional[httpx.AsyncClient] = None

    async def info(self) -> dict[str, Any]:
        """
        :return: Dictionary containing backend configuration details.
        """
        return {
            "target": self.target,
            "model": self.model,
            "headers": self.headers,
            "timeout": self.timeout,
            "http2": self.http2,
            "follow_redirects": self.follow_redirects,
            "verify": self.verify,
            "max_output_tokens": self.max_output_tokens,
            "stream_response": self.stream_response,
            "extra_query": self.extra_query,
            "extra_body": self.extra_body,
            "remove_from_body": self.remove_from_body,
            "health_path": self.HEALTH_PATH,
            "models_path": self.MODELS_PATH,
            "text_completions_path": self.TEXT_COMPLETIONS_PATH,
            "chat_completions_path": self.CHAT_COMPLETIONS_PATH,
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

        await self._async_client.aclose()
        self._async_client = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend configuration and connectivity.

        Validate backend configuration and connectivity through test requests,
        and auto-selects first available model if none is configured.

        :raises RuntimeError: If backend cannot connect or validate configuration.
        """
        self._check_in_process()

        if self.model:
            with contextlib.suppress(httpx.TimeoutException, httpx.HTTPStatusError):
                # Model is set, use /health endpoint as first check
                target = f"{self.target}{self.HEALTH_PATH}"
                headers = self._get_headers()
                response = await self._async_client.get(target, headers=headers)
                response.raise_for_status()

                return

        with contextlib.suppress(httpx.TimeoutException, httpx.HTTPStatusError):
            # Check if models endpoint is available next
            models = await self.available_models()
            if models and not self.model:
                self.model = models[0]
            elif not self.model:
                raise RuntimeError(
                    "No model available and could not set a default model "
                    "from the server's available models."
                )

            return

        with contextlib.suppress(httpx.TimeoutException, httpx.HTTPStatusError):
            # Last check, fall back on dummy request to text completions
            async for _, __ in self.text_completions(
                prompt="Validate backend",
                request_id="validate",
                output_token_count=1,
                stream_response=False,
            ):
                pass

            return

        raise RuntimeError(
            "Backend validation failed. Could not connect to the server or "
            "validate the backend configuration."
        )

    async def available_models(self) -> list[str]:
        """
        Get available models from the target server.

        :return: List of model identifiers.
        :raises HTTPError: If models endpoint returns an error.
        :raises RuntimeError: If backend is not initialized.
        """
        self._check_in_process()

        target = f"{self.target}{self.MODELS_PATH}"
        headers = self._get_headers()
        params = self._get_params(self.MODELS_KEY)
        response = await self._async_client.get(target, headers=headers, params=params)
        response.raise_for_status()

        return [item["id"] for item in response.json()["data"]]

    async def default_model(self) -> Optional[str]:
        """
        Get the default model for this backend.

        :return: Model name or None if no model is available.
        """
        if self.model or not self._in_process:
            return self.model

        models = await self.available_models()
        return models[0] if models else None

    async def resolve(
        self,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[
        tuple[GenerationResponse, ScheduledRequestInfo[GenerationRequestTimings]]
    ]:
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
        self._check_in_process()
        if history is not None:
            raise NotImplementedError(
                "Multi-turn requests with conversation history are not yet supported"
            )

        response = GenerationResponse(
            request_id=request.request_id,
            request_args={
                "request_type": request.request_type,
                "output_token_count": request.constraints.get("max_output_tokens"),
                **request.params,
            },
            value="",
            request_prompt_tokens=request.stats.get("prompt_tokens"),
            request_output_tokens=request.constraints.get("output_tokens"),
        )
        request_info.request_timings = GenerationRequestTimings()
        request_info.request_timings.request_start = time.time()

        completion_method = (
            self.text_completions
            if request.request_type == "text_completions"
            else self.chat_completions
        )
        completion_kwargs = (
            {
                "prompt": request.content,
                "request_id": request.request_id,
                "output_token_count": request.constraints.get("max_output_tokens"),
                "stream_response": request.params.get("stream", self.stream_response),
                **request.params,
            }
            if request.request_type == "text_completions"
            else {
                "content": request.content,
                "request_id": request.request_id,
                "output_token_count": request.constraints.get("max_output_tokens"),
                "stream_response": request.params.get("stream", self.stream_response),
                **request.params,
            }
        )

        async for delta, usage_stats in completion_method(**completion_kwargs):
            if request_info.request_timings.request_start is None:
                request_info.request_timings.request_start = time.time()

            if delta is not None:
                if request_info.request_timings.first_iteration is None:
                    request_info.request_timings.first_iteration = time.time()
                response.value += delta
                response.delta = delta
                request_info.request_timings.last_iteration = time.time()
                response.iterations += 1

            if usage_stats is not None:
                request_info.request_timings.request_end = time.time()
                response.request_output_tokens = usage_stats.output_tokens
                response.request_prompt_tokens = usage_stats.prompt_tokens

            yield response, request_info

        if request_info.request_timings.request_end is None:
            request_info.request_timings.request_end = time.time()
        response.delta = None
        yield response, request_info

    async def text_completions(
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str],  # noqa: ARG002
        output_token_count: Optional[int] = None,
        stream_response: bool = True,
        **kwargs,
    ) -> AsyncIterator[tuple[Optional[str], Optional[UsageStats]]]:
        """
        Generate text completions using the /v1/completions endpoint.

        :param prompt: Text prompt(s) for completion. Single string or list.
        :param request_id: Request identifier for tracking.
        :param output_token_count: Maximum tokens to generate. Overrides default
            if specified.
        :param stream_response: Whether to stream response progressively.
        :param kwargs: Additional request parameters (temperature, top_p, etc.).
        :yields: Tuples of (generated_text, usage_stats). First yield is (None, None).
        :raises RuntimeError: If backend is not initialized.
        :raises HTTPError: If API request fails.
        """
        self._check_in_process()
        target = f"{self.target}{self.TEXT_COMPLETIONS_PATH}"
        headers = self._get_headers()
        params = self._get_params(self.TEXT_COMPLETIONS_KEY)
        body = self._get_body(
            endpoint_type=self.TEXT_COMPLETIONS_KEY,
            request_kwargs=kwargs,
            max_output_tokens=output_token_count,
            prompt=prompt,
        )
        yield None, None  # Initial yield for async iterator to signal start

        if not stream_response:
            response = await self._async_client.post(
                target,
                headers=headers,
                params=params,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            yield (
                self._get_completions_text_content(data),
                self._get_completions_usage_stats(data),
            )
            return

        body.update({"stream": True, "stream_options": {"include_usage": True}})
        async with self._async_client.stream(
            "POST",
            target,
            headers=headers,
            params=params,
            json=body,
        ) as stream:
            stream.raise_for_status()
            async for line in stream.aiter_lines():
                if not line or not line.strip().startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                data = json.loads(line.strip()[len("data: ") :])
                yield (
                    self._get_completions_text_content(data),
                    self._get_completions_usage_stats(data),
                )

    async def chat_completions(
        self,
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        request_id: Optional[str] = None,  # noqa: ARG002
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        stream_response: bool = True,
        **kwargs,
    ) -> AsyncIterator[tuple[Optional[str], Optional[UsageStats]]]:
        """
        Generate chat completions using the /v1/chat/completions endpoint.

        Supports multimodal inputs including text and images with message formatting.

        :param content: Chat content - string, list of mixed content, or raw content
            when raw_content=True.
        :param request_id: Request identifier (currently unused).
        :param output_token_count: Maximum tokens to generate. Overrides default
            if specified.
        :param raw_content: If True, passes content directly without formatting.
        :param stream_response: Whether to stream response progressively.
        :param kwargs: Additional request parameters (temperature, top_p, tools, etc.).
        :yields: Tuples of (generated_text, usage_stats). First yield is (None, None).
        :raises RuntimeError: If backend is not initialized.
        :raises HTTPError: If API request fails.
        """
        self._check_in_process()
        target = f"{self.target}{self.CHAT_COMPLETIONS_PATH}"
        headers = self._get_headers()
        params = self._get_params(self.CHAT_COMPLETIONS_KEY)
        body = self._get_body(
            endpoint_type=self.CHAT_COMPLETIONS_KEY,
            request_kwargs=kwargs,
            max_output_tokens=output_token_count,
            messages=self._get_chat_messages(content) if not raw_content else content,
            **kwargs,
        )
        yield None, None  # Initial yield for async iterator to signal start

        if not stream_response:
            response = await self._async_client.post(
                target, headers=headers, params=params, json=body
            )
            response.raise_for_status()
            data = response.json()
            yield (
                self._get_completions_text_content(data),
                self._get_completions_usage_stats(data),
            )
            return

        body.update({"stream": True, "stream_options": {"include_usage": True}})
        async with self._async_client.stream(
            "POST", target, headers=headers, params=params, json=body
        ) as stream:
            stream.raise_for_status()
            async for line in stream.aiter_lines():
                if not line or not line.strip().startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                data = json.loads(line.strip()[len("data: ") :])
                yield (
                    self._get_completions_text_content(data),
                    self._get_completions_usage_stats(data),
                )

    def _build_headers(
        self,
        api_key: Optional[str],
        organization: Optional[str],
        project: Optional[str],
        user_headers: Optional[dict],
    ) -> dict[str, str]:
        headers = {}

        if api_key:
            headers["Authorization"] = (
                f"Bearer {api_key}" if not api_key.startswith("Bearer") else api_key
            )
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        if user_headers:
            headers.update(user_headers)

        return {key: val for key, val in headers.items() if val is not None}

    def _check_in_process(self):
        if not self._in_process or self._async_client is None:
            raise RuntimeError(
                "Backend not started up for process, cannot process requests."
            )

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **self.headers,
        }

    def _get_params(self, endpoint_type: str) -> dict[str, str]:
        if endpoint_type in self.extra_query:
            return copy.deepcopy(self.extra_query[endpoint_type])
        return copy.deepcopy(self.extra_query)

    def _get_chat_messages(
        self,
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]

        if not isinstance(content, list):
            raise ValueError(f"Unsupported content type: {type(content)}")

        resolved_content = []
        for item in content:
            if isinstance(item, dict):
                resolved_content.append(item)
            elif isinstance(item, str):
                resolved_content.append({"type": "text", "text": item})
            elif isinstance(item, (Image.Image, Path)):
                resolved_content.append(self._get_chat_message_media_item(item))
            else:
                raise ValueError(f"Unsupported content item type: {type(item)}")

        return [{"role": "user", "content": resolved_content}]

    def _get_chat_message_media_item(
        self, item: Union[Path, Image.Image]
    ) -> dict[str, Any]:
        if isinstance(item, Image.Image):
            encoded = base64.b64encode(item.tobytes()).decode("utf-8")
            return {
                "type": "image",
                "image": {"url": f"data:image/jpeg;base64,{encoded}"},
            }

        # Handle file paths
        suffix = item.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            image = Image.open(item)
            encoded = base64.b64encode(image.tobytes()).decode("utf-8")
            return {
                "type": "image",
                "image": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        elif suffix == ".wav":
            encoded = base64.b64encode(item.read_bytes()).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            }
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _get_body(
        self,
        endpoint_type: str,
        request_kwargs: Optional[dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Start with endpoint-specific extra body parameters
        extra_body = self.extra_body.get(endpoint_type, self.extra_body)

        body = copy.deepcopy(extra_body)
        body.update(request_kwargs or {})
        body.update(kwargs)
        body["model"] = self.model

        # Handle token limits
        max_tokens = max_output_tokens or self.max_output_tokens
        if max_tokens is not None:
            body.update(
                {
                    "max_tokens": max_tokens,
                    "max_completion_tokens": max_tokens,
                }
            )
            # Set stop conditions only for request-level limits
            if max_output_tokens:
                body.update({"stop": None, "ignore_eos": True})

        return {key: val for key, val in body.items() if val is not None}

    def _get_completions_text_content(self, data: dict) -> Optional[str]:
        if not data.get("choices"):
            return None

        choice = data["choices"][0]
        return choice.get("text") or choice.get("delta", {}).get("content")

    def _get_completions_usage_stats(self, data: dict) -> Optional[UsageStats]:
        if not data.get("usage"):
            return None

        return UsageStats(
            prompt_tokens=data["usage"].get("prompt_tokens"),
            output_tokens=data["usage"].get("completion_tokens"),
        )
