"""
Tool call data models for streaming and non-streaming responses.

Provides Pydantic models for representing tool calls returned by OpenAI-compatible
APIs. Used by both the response and request statistics schemas to carry tool call
payloads through the benchmarking pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "StreamingToolCall",
    "StreamingToolCallFunction",
]


class StreamingToolCallFunction(BaseModel):
    """Accumulated function name and arguments for a single streamed tool call."""

    name: str = ""
    arguments: str = ""


class StreamingToolCall(BaseModel):
    """A single tool call reassembled from streaming deltas."""

    id: str = ""
    type: str = "function"
    function: StreamingToolCallFunction = Field(
        default_factory=StreamingToolCallFunction
    )
