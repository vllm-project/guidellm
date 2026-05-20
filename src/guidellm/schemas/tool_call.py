"""
Tool call data models for OpenAI-compatible API responses.

Provides Pydantic models for representing tool calls returned by OpenAI-compatible
APIs. Used by both the response and request statistics schemas to carry tool call
payloads through the benchmarking pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ToolCall",
    "ToolCallFunction",
]


class ToolCallFunction(BaseModel):
    """Function name and arguments for a single tool call."""

    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    """A single tool call from an OpenAI-compatible API response."""

    id: str = ""
    type: str = "function"
    function: ToolCallFunction = Field(default_factory=ToolCallFunction)
