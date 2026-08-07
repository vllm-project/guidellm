"""
A minimal trace file format primarily used for testing. Designed to do the bare minimum
needed to complete a fully functioning trace deserializer with synthetic prompt
generation.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from collections.abc import Iterable

from datasets import Dataset, Features
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.trace_common import (
    TraceFormatBase,
    TraceFormatRegistry,
    decode_prompt,
    generate_token_ids,
    get_missing_columns,
)
from guidellm.schemas.data.deserializers import MinimalTraceFormatArgs

__all__ = ["MinimalTraceFormatArgs"]


@TraceFormatRegistry.register("trace_synthetic")
class MinimalTraceFormat(TraceFormatBase):
    def __init__(self, config: MinimalTraceFormatArgs, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dataset]:
        yield self.dataset.sort(self.config.timestamp_column)

    def required_columns(self) -> Features:
        return {}

    def find_required_columns(self, columns: list[str]) -> list[str]:
        return get_missing_columns(columns, self.dataset.column_names)

    def validate_row(
        self,
        row: dict,  # noqa: ARG002
    ) -> None:
        return

    def create_prompt(
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        token_ids = generate_token_ids(
            row[self.config.prompt_tokens_column], processor, faker
        )
        return decode_prompt(processor, list(token_ids))
