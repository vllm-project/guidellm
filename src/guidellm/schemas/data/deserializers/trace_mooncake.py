from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.deserializers.trace_common import TraceDataArgs
from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["MooncakeTraceFormatArgs"]


@DataArgs.register("mooncake")
class MooncakeTraceFormatArgs(TraceDataArgs):
    kind: Literal["mooncake"] = Field(
        default="mooncake",
        description="Type identifier for the Mooncake trace format.",
    )
    hash_ids_column: str = Field(
        default="hash_ids",
        description="Column name for lists of hash IDs in the trace file.",
    )
    hash_id_block_size: int = Field(
        gt=0,
        # Default used in Mooncake's paper https://arxiv.org/pdf/2407.00079
        default=512,
        description="Amount of tokens represented by one hash ID.",
    )
