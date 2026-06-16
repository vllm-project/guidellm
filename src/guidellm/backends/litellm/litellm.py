"""
LiteLLM backend implementation for GuideLLM.

Provides SDK-based backend for LiteLLM, enabling benchmarking across 100+
providers (Anthropic, Gemini, Bedrock, Groq, Cohere, Mistral, and more) via
a unified interface. Supports streaming, token usage tracking, and timing
measurements compatible with GuideLLM's benchmark pipeline.

Install the optional dependency to use this backend:
    pip install "guidellm[litellm]"
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import Field, SecretStr

import guidellm.extras.litellm as _litellm
from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.request_handlers import (
    ChatCompletionsRequestHandler,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)
from guidellm.schemas.request import UsageMetrics

__all__ = [
    "LiteLLMBackend",
    "LiteLLMBackendArgs",
]


@BackendArgs.register("litellm")
class LiteLLMBackendArgs(BackendArgs):
    """Pydantic model for LiteLLM backend creation arguments."""

    kind: Literal["litellm"] = Field(
        default="litellm",
        description="Type identifier for the LiteLLM backend configuration.",
    )
    model: str = Field(
        description=(
            "Model identifier in LiteLLM format, e.g. "
            "'anthropic/claude-haiku-4-5', 'gemini/gemini-1.5-flash', "
            "'bedrock/anthropic.claude-3-sonnet-20240229-v1:0', 'gpt-4o'."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "API key for the provider (overrides provider env var)."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description="Custom base URL, e.g. a LiteLLM proxy URL.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate per request.",
    )
    extras: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments forwarded to "
            "litellm.acompletion()."
        ),
    )


@Backend.register("litellm")
class LiteLLMBackend(Backend):
    """
    LiteLLM SDK backend for GuideLLM.

    Routes generation requests through the LiteLLM SDK, enabling load-testing
    and benchmarking across any provider supported by LiteLLM.

    Example:
    ::
        args = LiteLLMBackendArgs(
            model="anthropic/claude-haiku-4-5",
            api_key="sk-ant-...",
        )
        backend = LiteLLMBackend(args)
        await backend.process_startup()
        async for response, info in backend.resolve(request, info):
            process(response)
        await backend.process_shutdown()
    """

    def __init__(self, args: LiteLLMBackendArgs):
        super().__init__(args)
        self._args = args
        self._in_process = False

    async def process_startup(self):
        """Validate that litellm is importable and model is set."""
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")
        _ = _litellm.acompletion  # noqa: F841
        self._in_process = True

    async def process_shutdown(self):
        """Clean up backend resources."""
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")
        self._in_process = False

    async def validate(self):
        """Validate that model string is non-empty."""
        if not self._args.model:
            raise RuntimeError(
                "LiteLLM backend requires a non-empty model string."
            )

    async def available_models(self) -> list[str]:
        """Return the configured model as the only available model."""
        return [self._args.model]

    async def default_model(self) -> str:
        """Return the configured model identifier."""
        return self._args.model

    async def resolve(  # type: ignore[override]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: (
            list[tuple[GenerationRequest, GenerationResponse | None]]
            | None
        ) = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """
        Process generation request via litellm.acompletion().

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info with timing metadata
        :param history: Optional conversation history for multi-turn
        :yields: Tuples of (response, updated_request_info)
        """
        model = self._args.model
        messages, request_args_json = self._format_request(
            request, history, model,
        )
        call_kwargs = self._build_call_kwargs()

        texts: list[str] = []
        response_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            request_info.timings.request_start = time.time()

            response_stream = await _litellm.acompletion(
                model=model,
                messages=messages,
                stream=True,
                **call_kwargs,
            )

            async for chunk in response_stream:
                token_info = self._process_chunk(
                    chunk, request_info, texts,
                )
                if token_info.response_id and response_id is None:
                    response_id = token_info.response_id
                if token_info.input_tokens is not None:
                    input_tokens = token_info.input_tokens
                if token_info.output_tokens is not None:
                    output_tokens = token_info.output_tokens
                if token_info.first_token:
                    yield None, request_info

            request_info.timings.request_end = time.time()

            yield self._build_response(
                request, response_id, request_args_json,
                texts, input_tokens, output_tokens,
            ), request_info

        except asyncio.CancelledError as err:
            yield self._build_response(
                request, response_id, request_args_json,
                texts, input_tokens, output_tokens,
            ), request_info
            raise err

    def _format_request(
        self,
        request: GenerationRequest,
        history: (
            list[tuple[GenerationRequest, GenerationResponse | None]]
            | None
        ),
        model: str,
    ) -> tuple[list[dict[str, Any]], str]:
        handler = ChatCompletionsRequestHandler()
        arguments = handler.format(
            data=request,
            history=history,
            model=model,
            stream=True,
            max_tokens=self._args.max_tokens,
        )
        messages: list[dict[str, Any]] = (
            arguments.body.get("messages", [])
            if arguments.body
            else []
        )
        return messages, arguments.model_dump_json()

    def _build_call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "drop_params": True,
            "stream_options": {"include_usage": True},
        }
        if self._args.api_key:
            kwargs["api_key"] = self._args.api_key.get_secret_value()
        if self._args.api_base:
            kwargs["api_base"] = self._args.api_base
        if self._args.max_tokens is not None:
            kwargs["max_tokens"] = self._args.max_tokens
        if self._args.extras:
            kwargs.update(self._args.extras)
        return kwargs

    def _process_chunk(
        self,
        chunk: Any,
        request_info: RequestInfo,
        texts: list[str],
    ) -> _ChunkResult:
        iter_time = time.time()

        if request_info.timings.first_request_iteration is None:
            request_info.timings.first_request_iteration = iter_time
        request_info.timings.last_request_iteration = iter_time
        request_info.timings.request_iterations += 1

        response_id = chunk.id if chunk.id else None

        usage = getattr(chunk, "usage", None)
        input_tokens = (
            getattr(usage, "prompt_tokens", None) if usage else None
        )
        output_tokens = (
            getattr(usage, "completion_tokens", None) if usage else None
        )

        first_token = False
        if chunk.choices:
            delta = chunk.choices[0].delta
            content = delta.content if delta else None

            if content:
                texts.append(content)

                if request_info.timings.first_token_iteration is None:
                    request_info.timings.first_token_iteration = iter_time
                    request_info.timings.first_output_token_iteration = (
                        iter_time
                    )
                    request_info.timings.token_iterations = 0
                    first_token = True

                request_info.timings.last_token_iteration = iter_time
                request_info.timings.token_iterations += 1

        return _ChunkResult(
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            first_token=first_token,
        )

    @staticmethod
    def _build_response(
        request: GenerationRequest,
        response_id: str | None,
        request_args_json: str,
        texts: list[str],
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> GenerationResponse:
        return GenerationResponse(
            request_id=request.request_id,
            response_id=response_id,
            request_args=request_args_json,
            text="".join(texts) if texts else None,
            input_metrics=UsageMetrics(text_tokens=input_tokens),
            output_metrics=UsageMetrics(text_tokens=output_tokens),
        )


class _ChunkResult:
    __slots__ = (
        "first_token",
        "input_tokens",
        "output_tokens",
        "response_id",
    )

    def __init__(
        self,
        *,
        response_id: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        first_token: bool,
    ):
        self.response_id = response_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.first_token = first_token
