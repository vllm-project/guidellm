from __future__ import annotations

from pathlib import Path

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["TraceDataArgs"]


class TraceDataArgs(DataArgs):
    """Abstract class meant to be inherited by a trace format.
    For testing, use `trace_minimal.MinimalTraceFormatArgs` instead."""

    kind: str = Field(
        description="Type identifier for the trace dataset deserializer.",
    )
    path: Path = Field(description="Path to the trace file.")
    timestamp_column: str = Field(
        default="timestamp",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="input_length",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="output_length",
        description="Column name for output token counts in the trace file.",
    )
    conversation_id_column: str | None = Field(
        default=None,
        description=(
            "Column name for conversation IDs. Required for formats "
            "with conversation-scoped trace data such as hash IDs."
        ),
    )
