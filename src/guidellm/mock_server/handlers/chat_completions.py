"""
OpenAI Chat Completions API endpoint handler for the mock server.

Provides a complete implementation of the /v1/chat/completions endpoint that simulates
realistic LLM behavior with configurable timing characteristics. Supports both streaming
and non-streaming responses with proper token counting, latency simulation including
TTFT (Time To First Token) and ITL (Inter-Token Latency), and OpenAI-compatible error
handling for comprehensive benchmarking scenarios.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

from pydantic import ValidationError
from sanic import response
from sanic.request import Request
from sanic.response import HTTPResponse, ResponseStream
from transformers import PreTrainedTokenizer

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.models import (
    ChatCompletionChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
    Usage,
)
from guidellm.mock_server.utils import (
    MockTokenizer,
    create_fake_text,
    create_fake_tokens_str,
    sample_number,
    times_generator,
)

__all__ = ["ChatCompletionsHandler"]


class ChatCompletionsHandler:
    """
    Handles OpenAI Chat Completions API requests with realistic LLM simulation.

    Implements the /v1/chat/completions endpoint behavior including request validation,
    response generation, and timing simulation. Supports both streaming and
    non-streaming modes with configurable latency characteristics for comprehensive
    benchmarking. Uses either a mock tokenizer or a real tokenizer for accurate token
    counting and realistic text generation.

    Example:
    ::
        config = MockServerConfig(ttft_ms=100, itl_ms=50)
        handler = ChatCompletionsHandler(config)
        response = await handler.handle(request)
    """

    def __init__(self, config: MockServerConfig) -> None:
        """
        Initialize the Chat Completions handler with server configuration.

        :param config: Mock server configuration containing timing and behavior settings
        """
        self.config = config
        self.tokenizer = (
            MockTokenizer()
            if config.processor is None
            else PreTrainedTokenizer.from_pretrained(config.processor)
        )

    async def handle(self, request: Request) -> HTTPResponse:
        """
        Process incoming chat completion requests with validation and routing.

        Validates the request payload, handles errors gracefully, and routes to
        appropriate streaming or non-streaming response handlers based on the
        request configuration.

        :param request: Sanic HTTP request containing chat completion parameters
        :return: HTTP response with completion data or error information
        :raises ValidationError: When request payload fails validation
        :raises JSONDecodeError: When request contains invalid JSON
        """
        try:
            # Parse and validate request
            req_data = ChatCompletionsRequest(**request.json)
        except ValidationError as exc:
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid request: {str(exc)}",
                        type="invalid_request_error",
                        code="invalid_request",
                    )
                ).model_dump(),
                status=400,
            )
        except (json.JSONDecodeError, TypeError):
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message="Invalid JSON in request body",
                        type="invalid_request_error",
                        code="invalid_json",
                    )
                ).model_dump(),
                status=400,
            )

        # Handle streaming vs non-streaming
        if req_data.stream:
            return await self._handle_stream(req_data)
        else:
            return await self._handle_non_stream(req_data)

    async def _handle_non_stream(self, req: ChatCompletionsRequest) -> HTTPResponse:
        """
        Generate complete non-streaming chat completion response.

        Simulates realistic LLM behavior with TTFT and ITL delays, generates
        appropriate token counts, and returns a complete response with usage
        statistics and generated content.

        :param req: Validated chat completion request parameters
        :return: Complete HTTP response with generated completion data
        """
        # TTFT delay
        await asyncio.sleep(
            sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
        )

        # Token counts
        prompt_text = self.tokenizer.apply_chat_template(req.messages)
        prompt_tokens = len(self.tokenizer(prompt_text))  # type: ignore[arg-type]
        max_tokens = req.max_completion_tokens or req.max_tokens or math.inf
        completion_tokens_count = min(
            sample_number(self.config.output_tokens, self.config.output_tokens_std),
            max_tokens,
        )

        # ITL delay
        itl_delay = 0.0
        delays_iter = iter(times_generator(self.config.itl_ms, self.config.itl_ms_std))
        for _ in range(int(completion_tokens_count) - 1):
            itl_delay += next(delays_iter)
        await asyncio.sleep(itl_delay / 1000.0)

        # Response
        chat_response = ChatCompletionsResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=create_fake_text(
                            int(completion_tokens_count), self.tokenizer
                        ),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=int(completion_tokens_count),
            ),
            system_fingerprint=f"fp_{uuid.uuid4().hex[:10]}",
        )

        return response.json(chat_response.model_dump())

    async def _handle_stream(self, req: ChatCompletionsRequest) -> HTTPResponse:
        """
        Generate streaming chat completion response with real-time token delivery.

        Creates a streaming response that delivers tokens incrementally with
        realistic timing delays. Supports optional usage statistics in the final
        stream chunk when requested via stream_options.

        :param req: Validated chat completion request with streaming enabled
        :return: Streaming HTTP response delivering tokens with proper timing
        """

        async def generate_stream(stream_response):
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

            # TTFT delay
            await asyncio.sleep(
                sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
            )

            # Token counts
            prompt_text = self.tokenizer.apply_chat_template(req.messages)
            prompt_tokens = len(self.tokenizer(prompt_text))  # type: ignore[arg-type]
            max_tokens = req.max_completion_tokens or req.max_tokens or math.inf
            completion_tokens_count = int(
                min(
                    sample_number(
                        self.config.output_tokens, self.config.output_tokens_std
                    ),
                    max_tokens,
                )
            )

            # Send tokens
            tokens = create_fake_tokens_str(completion_tokens_count, self.tokenizer)
            delays_iter = iter(
                times_generator(self.config.itl_ms, self.config.itl_ms_std)
            )

            for index, token in enumerate(tokens):
                if index > 0:
                    itl_delay = next(delays_iter)
                    await asyncio.sleep(itl_delay / 1000.0)

                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                await stream_response.write(f"data: {json.dumps(chunk_data)}\n\n")

            # Send final chunk with finish reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            await stream_response.write(f"data: {json.dumps(final_chunk)}\n\n")

            # Send usage if requested
            if req.stream_options and req.stream_options.include_usage:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens_count,
                        "total_tokens": prompt_tokens + completion_tokens_count,
                    },
                }
                await stream_response.write(f"data: {json.dumps(usage_chunk)}\n\n")

            # End stream
            await stream_response.write("data: [DONE]\n\n")

        return ResponseStream(  # type: ignore[return-value]
            generate_stream,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
