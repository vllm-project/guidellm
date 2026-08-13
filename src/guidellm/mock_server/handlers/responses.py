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
from typing import Any

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

    def _should_return_tool_calls(self, req: ResponsesRequest) -> bool:
        """Check if this request expects a function_call response.

        Returns True when tool_choice is "required", which is what GuideLLM
        sets on client_tool_call turns.
        """
        return req.tool_choice == "required" and bool(req.tools)

    def _build_function_call_item(
        self,
        req: ResponsesRequest,
    ) -> dict[str, Any]:
        """Build a mock function_call output item from the request's tools.

        Uses the first tool's name and returns empty arguments.
        """
        func_name = "mock_function"
        if req.tools:
            first_tool = req.tools[0]
            func_name = first_tool.get("name", func_name)

        call_id = f"call_{uuid.uuid4().hex[:24]}"
        return {
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": call_id,
            "name": func_name,
            "arguments": "{}",
            "status": "completed",
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

        if self._should_return_tool_calls(req):
            output = [self._build_function_call_item(req)]
        else:
            msg_id = f"msg_{uuid.uuid4().hex[:29]}"
            generated_text = create_fake_text(int(output_tokens_count), self.tokenizer)
            output = [
                {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": generated_text},
                    ],
                }
            ]

        resp = ResponsesResponse(
            id=response_id,
            status="completed",
            model=req.model,
            output=output,
            usage=ResponsesUsage(
                input_tokens=input_tokens,
                output_tokens=int(output_tokens_count),
            ),
        )

        return response.json(resp.model_dump())

    async def _handle_stream(self, req: ResponsesRequest) -> HTTPResponse:
        is_tool_call = self._should_return_tool_calls(req)

        async def generate_stream(stream_response):
            response_id = f"resp_{uuid.uuid4().hex[:29]}"

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

            if is_tool_call:
                done_item, seq = await self._stream_function_call(
                    stream_response,
                    req,
                    _write_event,
                    seq,
                )
            else:
                done_item, seq = await self._stream_text_content(
                    stream_response,
                    _write_event,
                    seq,
                    output_tokens_count,
                )

            usage = ResponsesUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens_count,
            )
            completed_response = self._build_response_object(
                response_id=response_id,
                model=req.model,
                status="completed",
                output=[done_item],
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

        return ResponseStream(  # type: ignore[return-value]
            generate_stream,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_text_content(
        self,
        stream_response,
        _write_event,
        seq,
        output_tokens_count,
    ) -> tuple[dict, int]:
        """Stream text content tokens and return the completed message item."""
        msg_id = f"msg_{uuid.uuid4().hex[:29]}"

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
        delays_iter = iter(times_generator(self.config.itl_ms, self.config.itl_ms_std))

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

        return done_message, seq

    async def _stream_function_call(
        self,
        stream_response,
        req,
        _write_event,
        seq,
    ) -> tuple[dict, int]:
        """Stream a function_call item with argument deltas and ITL delays."""
        fc_item = self._build_function_call_item(req)
        arguments = fc_item["arguments"]

        # Emit output_item.added with the function_call item (empty arguments)
        added_item = {
            "type": "function_call",
            "id": fc_item["id"],
            "call_id": fc_item["call_id"],
            "name": fc_item["name"],
            "arguments": "",
            "status": "in_progress",
        }
        await stream_response.write(
            _write_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": added_item,
                    "sequence_number": seq,
                },
            )
        )
        seq += 1

        # Stream argument chunks with ITL delays
        arg_chunks = list(arguments) if len(arguments) > 1 else [arguments]
        delays_iter = iter(times_generator(self.config.itl_ms, self.config.itl_ms_std))

        for chunk in arg_chunks:
            itl_delay = next(delays_iter)
            await asyncio.sleep(itl_delay / 1000.0)

            await stream_response.write(
                _write_event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "delta": chunk,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1

        # Emit arguments done
        await stream_response.write(
            _write_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 0,
                    "arguments": arguments,
                    "sequence_number": seq,
                },
            )
        )
        seq += 1

        # Emit output_item.done with the completed function_call item
        done_item = {
            "type": "function_call",
            "id": fc_item["id"],
            "call_id": fc_item["call_id"],
            "name": fc_item["name"],
            "arguments": arguments,
            "status": "completed",
        }
        await stream_response.write(
            _write_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": done_item,
                    "sequence_number": seq,
                },
            )
        )
        seq += 1

        return done_item, seq
