"""
The Mooncake trace format and data arguments.

Reads a trace file (timestamp, input_length, output_length, hash_ids) and yields one
row per line with a synthetic prompt matching the requested input_length for replay
benchmarks. Checks for distinctness between hash IDs that share the
same previous hash ID.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from datasets import List, Value
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
    decode_prompt,
    generate_token_ids,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import TraceColumn

__all__ = ["MooncakeTraceFormatArgs"]


def _is_in_table(hash_id_table: list[Any], hash_id: int) -> bool:
    return (
        hash_id < len(hash_id_table)
        and hash_id >= 0
        and hash_id_table[hash_id] is not None
    )


def _resize_to_hold_id(hash_id_table: list[Any], hash_id: int) -> None:
    num_new_entries = hash_id - (len(hash_id_table) - 1)
    hash_id_table.extend(None for _ in range(num_new_entries))


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


def _create_distinct_token_block(
    block_size: int,
    sibling_token_blocks: list[list[int]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    max_attempts: int = 20,
) -> list[int]:
    """Constructs a new token block of `block_size` that does not appear in
    `sibling_token_blocks`."""
    attempt = 0
    while attempt < max_attempts:
        token_ids = generate_token_ids(block_size, processor, faker)
        if token_ids not in sibling_token_blocks:
            return token_ids
        attempt += 1
    raise ValueError(
        f"Failed to generate distinct synthetic token block after {attempt} attempts"
    )


def _create_prompt_from_hash_ids(
    hash_ids: list[int],
    hash_id_table: list[list[int]],
    processor: PreTrainedTokenizerBase,
) -> str:
    """Returns a synthetic prompt from `hash_ids` using pre-generated token blocks.

    Precondition: All ids in `hash_ids` appear in `hash_id_table`."""
    prompt_token_ids = [
        token for hash_id in hash_ids for token in hash_id_table[hash_id]
    ]
    return decode_prompt(processor, prompt_token_ids)


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "mooncake")


@DataArgs.register("mooncake")
class MooncakeTraceFormatArgs(TraceDataArgs):
    kind: Literal["mooncake"] = Field(
        default="mooncake",
        description="Type identifier for the trace Mooncake dataset deserializer.",
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
    as they do not share the same parent (previous ID). For more details, see section 4
    of https://arxiv.org/pdf/2407.00079.

    Generated prompts match the prompt token count of the row."""

    def __init__(self) -> None:
        self.hash_id_table: list[Any] = []
        self.sibling_token_blocks: dict[Any, list[list[int]]] = {}

    def required_columns(self, config: MooncakeTraceFormatArgs) -> list[TraceColumn]:
        return [TraceColumn(config.hash_ids_column, List(Value("int32")))]

    def validate_row(self, config: MooncakeTraceFormatArgs, row: dict) -> None:
        n_in = row[config.prompt_tokens_column]
        n_blocks = len(row[config.hash_ids_column])
        for hash_id in row[config.hash_ids_column]:
            if hash_id < 0:
                raise DataNotSupportedError(
                    f"Hash ID must be non-negative, got {hash_id}"
                )
        if math.ceil(n_in / config.hash_id_block_size) != n_blocks:
            raise DataNotSupportedError(
                f"Input token count of {n_in} split into blocks of size "
                f"{config.hash_id_block_size} does not match given {n_blocks} blocks"
            )

    def create_prompt(
        self,
        config: MooncakeTraceFormatArgs,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        ids = row[config.hash_ids_column]
        for idx, hash_id in enumerate(ids):
            if not _is_in_table(self.hash_id_table, hash_id):
                _resize_to_hold_id(self.hash_id_table, hash_id)
                prev_id = None if idx == 0 else ids[idx - 1]
                num_tokens = _calculate_required_prompt_tokens(config, row, hash_id)
                self.sibling_token_blocks.setdefault(prev_id, [])
                self.hash_id_table[hash_id] = _create_distinct_token_block(
                    num_tokens,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].append(self.hash_id_table[hash_id])
        return _create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
