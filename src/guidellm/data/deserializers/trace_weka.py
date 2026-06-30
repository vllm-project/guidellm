"""
The WEKA trace format and data arguments.

TODO
"""

from __future__ import annotations

import math
from typing import Any, Literal

from datasets import Features, List, Value
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

__all__ = ["WEKATraceFormatArgs"]


def _is_in_table(hash_id_table: list[Any], hash_id: int) -> bool:
    return (
        hash_id <= len(hash_id_table)
        and hash_id > 0
        and hash_id_table[hash_id] is not None
    )


def _resize_to_hold_id(hash_id_table: list[Any], hash_id: int) -> None:
    num_new_entries = hash_id - len(hash_id_table)
    hash_id_table.extend(None for _ in range(num_new_entries))


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


def _generate_remaining_prompt(
    num_tokens: int, processor: PreTrainedTokenizerBase, faker: Faker
) -> str:
    token_ids = generate_token_ids(num_tokens, processor, faker)
    return decode_prompt(processor, token_ids)


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "weka")


@DataArgs.register("weka")
class WEKATraceFormatArgs(TraceDataArgs):
    kind: Literal["weka"] = Field(
        default="weka",
        description="Type identifier for the WEKA trace format.",
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


@TraceFormatRegistry.register("weka")
class WEKATraceFormat(TraceFormatBase):
    """TODO"""

    def __init__(self) -> None:
        self.hash_id_table: list[Any] = []
        self.sibling_token_blocks: dict[Any, list[list[int]]] = {}

    def required_columns(self,config: WEKATraceFormatArgs) -> Features:
        return Features({config.hash_ids_column: List(Value("int32"))})

    def validate_row(self, config: WEKATraceFormatArgs, row: dict) -> None:
        n_in = row[config.prompt_tokens_column]
        n_blocks = len(row[config.hash_ids_column])
        for hash_id in row[config.hash_ids_column]:
            if hash_id < 1:
                raise DataNotSupportedError(
                    f"Hash ID must be greater than 0, got {hash_id}"
                )
        # WEKA format drops what would be the partially filled hash ID
        if math.ceil(n_in / config.hash_id_block_size) - 1 != n_blocks:
            raise DataNotSupportedError(
                f"Input token count of {n_in} split into blocks of size "
                f"{config.hash_id_block_size} full blocks does not match given"
                f"{n_blocks} blocks"
            )

    def create_prompt(
        self,
        config: WEKATraceFormatArgs,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        """Before generating the prompt, this first generates a block of tokens for
        each hash ID that has not already been seen."""
        ids = row[config.hash_ids_column]
        for idx, hash_id in enumerate(ids):
            if not _is_in_table(self.hash_id_table, hash_id):
                _resize_to_hold_id(self.hash_id_table, hash_id)
                prev_id = None if idx == 0 else ids[idx - 1]
                self.sibling_token_blocks.setdefault(prev_id, [])
                self.hash_id_table[hash_id] = _create_distinct_token_block(
                    config.hash_id_block_size,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].append(self.hash_id_table[hash_id])
        prompt = _create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
        remainder = _generate_remaining_prompt(
            config.prompt_tokens_column % config.hash_id_block_size,
            processor,
            faker,
        )
        return prompt + remainder
