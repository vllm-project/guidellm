from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.deserializers.trace_common import TraceDataArgs
from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["OTELTraceFormatArgs"]


@DataArgs.register(["otel", "opentelemetry", "otel_trace"])
class OTELTraceFormatArgs(TraceDataArgs):
    kind: Literal["otel", "opentelemetry", "otel_trace"] = Field(
        default="otel",
        description="Type identifier for the OpenTelemetry trace format.",
    )
    spans_column: str = Field(
        default="spans",
        description="Column name for nested span lists in session-per-line files.",
    )
    trace_id_column: str = Field(
        default="trace_id",
        description="Column name used to group span-per-line files into conversations.",
    )
    span_timestamp_field: str = Field(
        default="start_time",
        description="Span field holding the request start time.",
    )
    input_tokens_attributes: list[str] = Field(
        default_factory=lambda: [
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
        ],
        description="Attribute keys tried in order for prompt token counts.",
    )
    output_tokens_attributes: list[str] = Field(
        default_factory=lambda: [
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
        ],
        description="Attribute keys tried in order for output token counts.",
    )
