"""
OpenAI Responses API endpoint handler for the mock server.

Provides an implementation of the /v1/responses endpoint that simulates
realistic LLM behavior with configurable timing characteristics. Supports both
streaming and non-streaming responses with proper token counting, latency
simulation including TTFT and ITL, and Responses API-compatible event formats.
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
    ErrorDetail,
    ErrorResponse,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from guidellm.mock_server.utils import (
    MockTokenizer,
    create_fake_text,
    create_fake_tokens_str,
    sample_number,
    times_generator,
)

__all__ = ["ResponsesHandler"]


class ResponsesHandler:
    """
    Handles OpenAI Responses API requests with realistic LLM simulation.

    Implements the /v1/responses endpoint behavior including request validation,
    response generation, and timing simulation. Supports both streaming and
    non-streaming modes with configurable latency characteristics.
    """

    def __init__(self, config: MockServerConfig) -> None:
        self.config = config
        self.tokenizer = (
            MockTokenizer()
            if config.processor is None
            else PreTrainedTokenizer.from_pretrained(config.processor)
        )

    def _extract_input_text(self, req: ResponsesRequest) -> str:
        """Extract plain text from the input field for tokenization."""
        parts: list[str] = []
        if req.instructions:
            parts.append(req.instructions)

        if isinstance(req.input, str):
            parts.append(req.input)
        else:
            for item in req.input:
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in (
                            "input_text",
                            "text",
                        ):
                            parts.append(part.get("text", ""))
        return " ".join(parts)

    def _build_response_object(
        self,
        *,
        response_id: str,
        model: str,
        status: str,
        output: list,
        usage: ResponsesUsage | None = None,
    ) -> dict:
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": model,
            "output": output,
            "usage": usage.model_dump() if usage else None,
        }

    async def handle(self, request: Request) -> HTTPResponse:
        try:
            req_data = ResponsesRequest(**request.json)
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

        if req_data.stream:
            return await self._handle_stream(req_data)
        else:
            return await self._handle_non_stream(req_data)

    async def _handle_non_stream(self, req: ResponsesRequest) -> HTTPResponse:
        await asyncio.sleep(
            sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
        )

        input_text = self._extract_input_text(req)
        input_tokens = len(self.tokenizer(input_text))  # type: ignore[arg-type]
        max_tokens = req.max_output_tokens or math.inf
        output_tokens_count = min(
            sample_number(self.config.output_tokens, self.config.output_tokens_std),
            max_tokens,
        )

        itl_delay = 0.0
        delays_iter = iter(times_generator(self.config.itl_ms, self.config.itl_ms_std))
        for _ in range(int(output_tokens_count) - 1):
            itl_delay += next(delays_iter)
        await asyncio.sleep(itl_delay / 1000.0)

        response_id = f"resp_{uuid.uuid4().hex[:29]}"
        msg_id = f"msg_{uuid.uuid4().hex[:29]}"
        generated_text = create_fake_text(int(output_tokens_count), self.tokenizer)

        resp = ResponsesResponse(
            id=response_id,
            status="completed",
            model=req.model,
            output=[
                {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": generated_text},
                    ],
                }
            ],
            usage=ResponsesUsage(
                input_tokens=input_tokens,
                output_tokens=int(output_tokens_count),
            ),
        )

        return response.json(resp.model_dump())

    async def _handle_stream(self, req: ResponsesRequest) -> HTTPResponse:
        async def generate_stream(stream_response):
            response_id = f"resp_{uuid.uuid4().hex[:29]}"
            msg_id = f"msg_{uuid.uuid4().hex[:29]}"

            await asyncio.sleep(
                sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
            )

            input_text = self._extract_input_text(req)
            input_tokens = len(self.tokenizer(input_text))  # type: ignore[arg-type]
            max_tokens = req.max_output_tokens or math.inf
            output_tokens_count = int(
                min(
                    sample_number(
                        self.config.output_tokens, self.config.output_tokens_std
                    ),
                    max_tokens,
                )
            )

            seq = 0

            def _write_event(event_type: str, data: dict) -> str:
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            empty_response = self._build_response_object(
                response_id=response_id,
                model=req.model,
                status="in_progress",
                output=[],
            )

            await stream_response.write(
                _write_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": empty_response,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            await stream_response.write(
                _write_event(
                    "response.in_progress",
                    {
                        "type": "response.in_progress",
                        "response": empty_response,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            message_item = {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
            await stream_response.write(
                _write_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": message_item,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            content_part = {"type": "output_text", "text": ""}
            await stream_response.write(
                _write_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": content_part,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            tokens = create_fake_tokens_str(output_tokens_count, self.tokenizer)
            delays_iter = iter(
                times_generator(self.config.itl_ms, self.config.itl_ms_std)
            )

            for index, token in enumerate(tokens):
                if index > 0:
                    itl_delay = next(delays_iter)
                    await asyncio.sleep(itl_delay / 1000.0)

                await stream_response.write(
                    _write_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": token,
                            "sequence_number": seq,
                        },
                    )
                )
                seq += 1

            full_text = "".join(tokens)
            await stream_response.write(
                _write_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            done_message = {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text}],
            }
            await stream_response.write(
                _write_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": done_message,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

            usage = ResponsesUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens_count,
            )
            completed_response = self._build_response_object(
                response_id=response_id,
                model=req.model,
                status="completed",
                output=[done_message],
                usage=usage,
            )
            await stream_response.write(
                _write_event(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": completed_response,
                        "sequence_number": seq,
                    },
                )
            )

            # Real vLLM does NOT send "data: [DONE]" for the Responses API
            # (unlike chat completions). We include it here so the mock server
            # also works with clients that rely on it as a stream terminator.
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
