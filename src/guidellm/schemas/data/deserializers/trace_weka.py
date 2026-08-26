from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.deserializers.trace_common import TraceDataArgs
from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["WEKATraceFormatArgs"]


@DataArgs.register("weka")
class WEKATraceFormatArgs(TraceDataArgs):
    kind: Literal["weka"] = Field(
        default="weka",
        description="Type identifier for the WEKA trace format.",
    )
    timestamp_column: str = Field(
        default="t",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="in",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="out",
        description="Column name for output token counts in the trace file.",
    )
    conversation_id_column: str = Field(
        default="id",
        description="Column name for conversation UUIDs in the trace file.",
    )
    hash_ids_column: str = Field(
        default="hash_ids",
        description="Column name for lists of hash IDs in the trace file.",
    )
    hash_id_block_size: int = Field(
        gt=0,
        # Recommended in original github repository callanjfox/agentic-coding-analysis
        default=64,
        description="Amount of tokens represented by one hash ID.",
    )
