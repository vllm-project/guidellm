"""
A minimal trace file format primarily used for testing. Designed to do the bare minimum
needed to complete a fully functioning trace deserializer with synthetic prompt
generation.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from typing import Literal

from datasets import Features
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.trace_common import (
    TraceDataArgs,
    TraceFormatBase,
    TraceFormatRegistry,
    decode_prompt,
    generate_token_ids,
)
from guidellm.data.schemas import DataArgs

__all__ = ["MinimalTraceFormatArgs"]


@DataArgs.register("trace_synthetic")
class MinimalTraceFormatArgs(TraceDataArgs):
    kind: Literal["trace_synthetic"] = Field(
        default="trace_synthetic",
        description="Type identifier for the minimal trace format.",
    )


@TraceFormatRegistry.register("trace_synthetic")
class MinimalTraceFormat(TraceFormatBase):
    def __init__(self) -> None:
        pass

    def required_columns(
        self,
        config: MinimalTraceFormatArgs,  # noqa: ARG002
    ) -> Features:
        return []

    def validate_row(
        self,
        config: MinimalTraceFormatArgs,  # noqa: ARG002
        row: dict,  # noqa: ARG002
    ) -> None:
        return

    def create_prompt(
        self,
        config: MinimalTraceFormatArgs,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        token_ids = generate_token_ids(
            row[config.prompt_tokens_column], processor, faker
        )
        return decode_prompt(processor, token_ids)
