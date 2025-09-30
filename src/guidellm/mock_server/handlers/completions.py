"""
Legacy OpenAI Completions API handler for the mock server.

This module provides the CompletionsHandler class that implements the /v1/completions
endpoint for the guidellm mock server. It supports both streaming and non-streaming
completions with configurable timing parameters (TTFT, ITL) and token generation to
simulate realistic LLM behavior for benchmarking and testing purposes.
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
    CompletionChoice,
    CompletionsRequest,
    CompletionsResponse,
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

__all__ = ["CompletionsHandler"]


class CompletionsHandler:
    """
    Handler for the OpenAI /v1/completions endpoint in the mock server.

    This handler simulates the legacy OpenAI completions API by processing incoming
    requests and generating responses with configurable timing and token generation
    patterns. It supports both streaming and non-streaming modes, applying realistic
    timing delays (TTFT and ITL) to mimic actual LLM behavior for benchmarking.

    Example:
    ::
        config = MockServerConfig(ttft_ms=100, itl_ms=50)
        handler = CompletionsHandler(config)
        response = await handler.handle(sanic_request)
    """

    def __init__(self, config: MockServerConfig) -> None:
        """
        Initialize the completions handler with configuration settings.

        :param config: Mock server configuration containing timing parameters
            and tokenizer settings
        """
        self.config = config
        self.tokenizer = (
            MockTokenizer()
            if config.processor is None
            else PreTrainedTokenizer.from_pretrained(config.processor)
        )

    async def handle(self, request: Request) -> HTTPResponse:
        """
        Process a completions request and return the appropriate response.

        Validates the incoming request, determines whether to use streaming or
        non-streaming mode, and delegates to the appropriate handler method.

        :param request: Sanic request object containing the completions request data
        :return: HTTP response with completion data or error information
        :raises ValidationError: When request validation fails
        :raises json.JSONDecodeError: When request JSON is malformed
        """
        try:
            # Parse and validate request
            req_data = CompletionsRequest(**request.json)
        except ValidationError as e:
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid request: {str(e)}",
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

    async def _handle_non_stream(self, req: CompletionsRequest) -> HTTPResponse:
        """
        Generate a non-streaming completion response.

        Simulates TTFT and ITL delays, generates appropriate token counts, and returns
        a complete response with the generated text and usage statistics.

        :param req: Validated completions request containing prompt and parameters
        :return: JSON HTTP response with completion text and usage data
        :raises NotImplementedError: When batch processing is requested
        """
        if isinstance(req.prompt, list):
            raise NotImplementedError("Batch processing is not supported.")

        # TTFT delay
        await asyncio.sleep(
            sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
        )

        # Token counts
        prompt_tokens = len(self.tokenizer(req.prompt))
        max_tokens = req.max_tokens or math.inf
        completion_tokens_count = int(
            min(
                sample_number(self.config.output_tokens, self.config.output_tokens_std),
                max_tokens,
            )
            if req.stop
            else max_tokens
        )

        # ITL delay
        itl_delay = 0.0
        delays_iter = iter(times_generator(self.config.itl_ms, self.config.itl_ms_std))
        for _ in range(int(completion_tokens_count) - 1):
            itl_delay += next(delays_iter)
        await asyncio.sleep(itl_delay / 1000.0)

        # Response
        completion_response = CompletionsResponse(
            id=f"cmpl-{uuid.uuid4().hex[:29]}",
            model=req.model,
            choices=[
                CompletionChoice(
                    text=create_fake_text(completion_tokens_count, self.tokenizer),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens_count,
            ),
            system_fingerprint=f"fp_{uuid.uuid4().hex[:10]}",
        )

        return response.json(completion_response.model_dump())

    async def _handle_stream(self, req: CompletionsRequest) -> HTTPResponse:
        """
        Generate a streaming completion response.

        Creates a server-sent events stream that delivers tokens incrementally with
        realistic timing delays between each token. Includes usage statistics if
        requested and properly terminates the stream.

        :param req: Validated completions request containing prompt and streaming
            options
        :return: ResponseStream object that generates server-sent events
        """

        async def generate_stream(stream_response):
            completion_id = f"cmpl-{uuid.uuid4().hex[:29]}"

            # TTFT delay
            await asyncio.sleep(
                sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
            )

            # Token counts
            prompt_tokens = len(self.tokenizer(req.prompt))
            max_tokens = req.max_tokens or math.inf
            completion_tokens_count = int(
                min(
                    sample_number(
                        self.config.output_tokens, self.config.output_tokens_std
                    ),
                    max_tokens,
                )
                if req.stop
                else max_tokens
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
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "text": token,
                            "index": index,
                            "finish_reason": None,
                        }
                    ],
                }
                await stream_response.write(f"data: {json.dumps(chunk_data)}\n\n")

            # Send final chunk with finish reason
            final_chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "text": "",
                        "index": index,
                        "finish_reason": "stop",
                    }
                ],
            }
            await stream_response.write(f"data: {json.dumps(final_chunk)}\n\n")

            # Send usage if requested
            if req.stream_options and req.stream_options.include_usage:
                usage_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
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
