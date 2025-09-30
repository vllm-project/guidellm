"""
OpenAI HTTP backend implementation for GuideLLM.

Provides HTTP-based backend for OpenAI-compatible servers including OpenAI API,
vLLM servers, and other compatible inference engines. Supports text and chat
completions with streaming, authentication, and multimodal capabilities.

Classes:
    UsageStats: Token usage statistics for generation requests.
    OpenAIHTTPBackend: HTTP backend for OpenAI-compatible API servers.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx
from pydantic import dataclasses

from guidellm.backend.backend import Backend
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
    GenerationTokenStats,
)
from guidellm.scheduler import ScheduledRequestInfo

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False

__all__ = ["OpenAIHTTPBackend", "UsageStats"]


@dataclasses.dataclass
class UsageStats:
    """Token usage statistics for generation requests."""

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


open_ai_paths: dict[str, str] = {
    "health": "health",
    "models": "v1/models",
    "text_completions": "v1/completions",
    "chat_completions": "v1/chat/completions",
    "audio_transcriptions": "v1/audio/transcriptions",
    "audio_translations": "v1/audio/translations",
}


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
        model: Optional[str] = None,
        timeout: float = 60.0,
        http2: bool = True,
        follow_redirects: bool = True,
        verify: bool = False,
    ):
        super().__init__(type_="openai_http")

        # Request Values
        self.target = target.rstrip("/").removesuffix("/v1")
        self.model = model

        # Store configuration
        self.timeout = timeout
        self.http2 = http2
        self.follow_redirects = follow_redirects
        self.verify = verify

        # Runtime state
        self._in_process = False
        self._async_client: Optional[httpx.AsyncClient] = None

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
            "openai_paths": open_ai_paths,
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
        self._check_in_process()

        if self.model:
            with contextlib.suppress(httpx.TimeoutException, httpx.HTTPStatusError):
                # Model is set, use /health endpoint as first check
                target = f"{self.target}{open_ai_paths['health']}"
                response = await self._async_client.get(target)
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

        target = f"{self.target}{open_ai_paths['models']}"
        response = await self._async_client.get(target)
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

    async def resolve(  # noqa: C901
        self,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo,
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[tuple[GenerationResponse, ScheduledRequestInfo]]:
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

        request_info.request_timings = GenerationRequestTimings()
        request.arguments.url = (
            request.arguments.url or f"{self.target}/{request.arguments.path}"
            if request.arguments.path is not None
            else f"{self.target}/{open_ai_paths[request.request_type]}"
        )
        request_info.request_timings.request_start = time.time()

        if not request.arguments.stream:
            response = await self._async_client.request(
                request.arguments.method or "POST",
                request.arguments.url,
                content=request.arguments.content,
                files=request.arguments.files,
                json=request.arguments.json,
                params=request.arguments.params,
                headers=request.arguments.headers,
            )
            response.raise_for_status()
            data = response.json()
            prompt_stats, output_stats = self._extract_response_stats(data, request)
            request_info.request_timings.request_end = time.time()

            yield (
                GenerationResponse(
                    request_id=request.request_id,
                    request_args=request.arguments,
                    text=self._extract_response_text(data),
                    iterations=0,
                    prompt_stats=prompt_stats,
                    output_stats=output_stats,
                ),
                request_info,
            )
            return

        deltas = []
        prompt_stats = None
        output_stats = None
        end_reached = False

        try:
            async with self._async_client.stream(
                request.arguments.method or "POST",
                request.arguments.url,
                content=request.arguments.content,
                files=request.arguments.files,
                json=request.arguments.json,
                params=request.arguments.params,
                headers=request.arguments.headers,
            ) as stream:
                stream.raise_for_status()
                buffer = bytearray()

                async for chunk in stream.aiter_bytes():
                    if not chunk or end_reached:
                        continue
                    buffer.extend(chunk)

                    while (start := buffer.find(b"data:")) != -1 and (
                        end := buffer.find(b"\n", start)
                    ) != -1:
                        line = buffer[start + len(b"data:") : end].strip()
                        buffer = buffer[end + 1 :]

                        if not line:
                            continue

                        if line == b"[DONE]":
                            if request_info.request_timings.request_end is None:
                                request_info.request_timings.request_end = time.time()
                            end_reached = True
                            break

                        data = (
                            json.loads(line) if not HAS_ORJSON else orjson.loads(line)
                        )

                        if "usage" in data:
                            request_info.request_timings.request_end = time.time()
                            prompt_stats, output_stats = self._extract_response_stats(
                                data, request
                            )
                        else:
                            if request_info.request_timings.first_iteration is None:
                                request_info.request_timings.first_iteration = (
                                    time.time()
                                )
                            request_info.request_timings.last_iteration = time.time()
                            deltas.append(self._extract_response_text(data))

            yield (
                GenerationResponse(
                    request_id=request.request_id,
                    request_args=request.arguments,
                    text="".join(deltas) if deltas else None,
                    iterations=len(deltas),
                    prompt_stats=prompt_stats or GenerationTokenStats(),
                    output_stats=output_stats or GenerationTokenStats(),
                ),
                request_info,
            )
        except asyncio.CancelledError as err:
            yield (  # Ensure we yield what we have so far before stopping
                GenerationResponse(
                    request_id=request.request_id,
                    request_args=request.arguments,
                    text="".join(deltas) if deltas else None,
                    iterations=len(deltas),
                    prompt_stats=prompt_stats or GenerationTokenStats(),
                    output_stats=output_stats or GenerationTokenStats(),
                ),
                request_info,
            )
            raise err

    def _extract_response_text(self, data: dict) -> str:
        if not data:
            return None

        object_type = data.get("object") or data.get("type")

        if object_type == "text_completion":
            return data.get("choices", [{}])[0].get("text", "")

        if object_type == "chat.completion":
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if object_type == "chat.completion.chunk":
            return data.get("choices", [{}])[0].get("delta", {}).get("content", "")

        if "text" in data:
            return data.get("text", "")

        if "delta" in data:
            return data.get("delta", "")

        raise ValueError(f"Unsupported response format: {data}")

    def _extract_response_stats(
        self, data: dict, request: GenerationRequest
    ) -> tuple[GenerationTokenStats, GenerationTokenStats]:
        prompt_stats = GenerationTokenStats()
        output_stats = GenerationTokenStats()

        if not data or not (usage := data.get("usage")):
            return prompt_stats, output_stats

        prompt_stats.request = request.stats.get("prompt_tokens")
        prompt_stats.response = usage.get("prompt_tokens", usage.get("input_tokens"))
        prompt_token_details = usage.get(
            "prompt_tokens_details", usage.get("input_tokens_details")
        )
        if prompt_token_details:
            for key, val in prompt_token_details.items():
                setattr(prompt_stats, key, val)

        output_stats.request = request.constraints.get("output_tokens")
        output_stats.response = usage.get(
            "completion_tokens", usage.get("output_tokens")
        )
        output_token_details = usage.get(
            "completion_tokens_details", usage.get("output_tokens_details")
        )
        if output_token_details:
            for key, val in output_token_details.items():
                setattr(output_stats, key, val)

        return prompt_stats, output_stats

    def _check_in_process(self):
        if not self._in_process or self._async_client is None:
            raise RuntimeError(
                "Backend not started up for process, cannot process requests."
            )
