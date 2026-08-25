from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.deserializers.trace_common import TraceDataArgs
from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["MinimalTraceFormatArgs"]


@DataArgs.register("trace_synthetic")
class MinimalTraceFormatArgs(TraceDataArgs):
    kind: Literal["trace_synthetic"] = Field(
        default="trace_synthetic",
        description="Type identifier for the minimal trace format.",
    )
