"""
The Mooncake trace format and data arguments.

Reads a trace file (timestamp, input_length, output_length, hash_ids) and yields one
row per line with a synthetic prompt matching the requested input_length for replay
benchmarks. Checks for distinctness between hash IDs that share the
same previous hash ID.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

from datasets import Dataset, Features, List, Value
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_common import (
    TraceDataArgs,
    TraceDatasetDeserializer,
    TraceFormatBase,
    TraceFormatRegistry,
    create_distinct_token_block,
    create_prompt_from_hash_ids,
    get_missing_columns,
)
from guidellm.data.schemas import DataArgs

__all__ = ["MooncakeTraceFormatArgs"]


def _calculate_required_prompt_tokens(
    config: MooncakeTraceFormatArgs, row: dict, hash_id: int
) -> int:
    """Returns the number of prompt tokens needed to satisfy the row input length.
    This will be less than the block_size if the input length is not divisible by it
    and `hash_id` is the final ID for the row."""
    remainder = row[config.prompt_tokens_column] % config.hash_id_block_size
    if row[config.hash_ids_column][-1] == hash_id and remainder != 0:
        return remainder
    return config.hash_id_block_size


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "mooncake")


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


@TraceFormatRegistry.register("mooncake")
class MooncakeTraceFormat(TraceFormatBase):
    """Mooncake trace format requires a column for timestamps, prompt token counts,
    ouput token counts and lists of hash IDs.

    Hash IDs are globally unique identifiers based on the current and previous token
    blocks in a prompt. The relationships of IDs forms a tree, where every first ID
    in a prompt has a parent node of `None`. Parent nodes can have an unbounded
    number of children. Two hash IDs can represent identical blocks of tokens so long
    as they do not share the same parent (previous ID).

    For more details, see section 4 of https://arxiv.org/pdf/2407.00079.

    Generated prompts match the prompt token count of the row."""

    def __init__(self, config: MooncakeTraceFormatArgs, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset
        self.conversation_locations = [[0]]

        self.hash_id_table: dict[int, tuple[int, ...]] = {}
        self.sibling_token_blocks: dict[Any, set[tuple[int, ...]]] = {}
    
    def __iter__(self) -> Iterable[Dataset]:
        yield self.dataset.sort(self.config.timestamp_column)

    def required_columns(self) -> Features:
        return Features({self.config.hash_ids_column: List(Value("int32"))})

    def find_required_columns(self, columns: list[str]) -> list[str]:
        return get_missing_columns(columns, self.dataset.column_names)

    def get_conversation_id_trace(
        self,
        conversation_location: list[int],  # noqa: ARG002
    ) -> list[str] | None:
        return None

    def validate_row(self, row: dict) -> None:
        n_in = row[self.config.prompt_tokens_column]
        n_blocks = len(row[self.config.hash_ids_column])
        block_size = self.config.hash_id_block_size
        for hash_id in row[self.config.hash_ids_column]:
            if hash_id < 0:
                raise DataNotSupportedError(
                    f"Hash ID must be non-negative, got {hash_id}"
                )
        if math.ceil(n_in / block_size) != n_blocks:
            raise DataNotSupportedError(
                f"Input token count of {n_in} split into blocks of size "
                f"{block_size} does not match given {n_blocks} blocks"
            )

    def create_prompt(
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        """Before generating the prompt, this first generates a block of tokens for
        each hash ID that has not already been seen."""
        ids = row[self.config.hash_ids_column]
        for idx, hash_id in enumerate(ids):
            if hash_id not in self.hash_id_table:
                prev_id = None if idx == 0 else ids[idx - 1]
                num_tokens = _calculate_required_prompt_tokens(
                    self.config, row, hash_id
                )
                self.sibling_token_blocks.setdefault(prev_id, set())
                self.hash_id_table[hash_id] = create_distinct_token_block(
                    num_tokens,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].add(self.hash_id_table[hash_id])
        return create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
