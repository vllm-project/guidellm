from __future__ import annotations

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["TraceDataArgs"]


class TraceDataArgs(DataArgs):
    """Abstract class meant to be inherited by a trace format.
    For testing, use `trace_minimal.MinimalTraceFormatArgs` instead."""

    kind: str = Field(
        description="Type identifier for the trace dataset deserializer.",
    )
    source: DataArgs = Field(
        description="Source dataset to read trace data from.",
    )
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
    max_wait: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum wait in original trace seconds between consecutive requests "
            "within one session. Applied independently per conversation. "
            "Larger gaps are shortened to this value; later requests in that session "
            "shift earlier by the trimmed amount."
        ),
    )
    max_session_wait: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum wait in original trace seconds between consecutive sessions. "
            "If the next session starts more than this many seconds after "
            "the previous session's last request, that session is shifted earlier."
        ),
    )
    min_concurrent_sessions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Pack sessions so at least this many overlap during steady state. "
            "The first N sessions start together; each later session starts as soon "
            "as session i-N ends, but never later than its original start."
        ),
    )
    time_scale: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Scale factor applied to relative timestamps after wait and pack caps. "
            "1.0 preserves original timing; values above 1.0 stretch intervals; "
            "values below 1.0 compress them."
        ),
    )
