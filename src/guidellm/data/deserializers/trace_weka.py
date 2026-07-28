"""
The WEKA trace format and data arguments.

Reads a trace file and yields one row per line with a
synthetic prompt matching the requested input_length for replay
benchmarks. Checks for distinctness between hash IDs that share the
same previous hash ID.

Generates prompts starting from the first conversation.
When the conversation ends, the next conversation will be used.

Some features such as subagent conversations,
tool call events and non-linear histories are still missing.
The results from datasets including these features will be unreliable.
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


def _create_distinct_token_block(
    block_size: int,
    sibling_token_blocks: set[tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    max_attempts: int = 20,
) -> tuple[int]:
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
    hash_id_table: dict[int, tuple[int]],
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
    if num_tokens == 0:
        return ""
    token_ids = generate_token_ids(num_tokens, processor, faker)
    return decode_prompt(processor, list(token_ids))


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "weka")


@DataArgs.register("weka")
class WEKATraceFormatArgs(TraceDataArgs):
    kind: Literal["weka"] = Field(
        default="weka",
        description="Type identifier for the WEKA trace format.",
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


@TraceFormatRegistry.register("weka")
class WEKATraceFormat(TraceFormatBase):
    """WEKA trace format requires a column for timestamps, prompt token counts,
    ouput token counts and lists of hash IDs.

    Hash IDs are unique identifiers based on the current and previous token
    blocks in a prompt. The relationships of IDs forms a tree, where every first ID
    in a prompt has a parent node of `None`. Parent nodes can have an unbounded
    number of children. Two hash IDs can represent identical blocks of tokens so long
    as they do not share the same parent (previous ID).

    For more details, see [the WEKA trace format specification][trace-spec].

    [trace-spec]: https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md

    Generated prompts match the prompt token count of the row."""

    def __init__(self) -> None:
        self.hash_id_table: dict[int, tuple[int]] = {}
        self.sibling_token_blocks: dict[Any, set[tuple[int, ...]]] = {}

    def reset(self) -> None:
        self.hash_id_table = {}
        self.sibling_token_blocks = {}

    def required_columns(self, config: WEKATraceFormatArgs) -> Features:
        return Features(
            {
                config.conversation_id_column: Value("string"),
                config.hash_ids_column: List(Value("int32")),
            }
        )

    def validate_row(self, config: WEKATraceFormatArgs, row: dict) -> None:
        """WEKA format drops what would be the partially filled hash ID at the end of
        the chain. Some popular datasets
        (e.g. `semianalysisai/cc-traces-weka-no-subagents-051226`) still contain the
        trailing hash ID.
        In this case, `validate_row` tolerates the addition, and handles it in
        `create_prompt`."""
        n_in = row[config.prompt_tokens_column]
        n_blocks = len(row[config.hash_ids_column])
        for hash_id in row[config.hash_ids_column]:
            if hash_id < 0:
                raise DataNotSupportedError(
                    f"Hash ID must be non-negative, got {hash_id}"
                )
        expected = n_in / config.hash_id_block_size
        if math.floor(expected) != n_blocks and math.ceil(expected) != n_blocks:
            raise DataNotSupportedError(
                f"Input token count of {n_in} split into blocks of size "
                f"{config.hash_id_block_size} full blocks and "
                f"{config.hash_id_block_size} full blocks + partially filled "
                f"trailing block does not match given {n_blocks} blocks"
            )

    def create_prompt(
        self,
        config: WEKATraceFormatArgs,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        """Before generating the prompt, this first generates a block of tokens for
        each hash ID that has not already been seen.

        Hash IDs that are partially filled are discarded to match the specification.
        Remainder of the prompt is created after the creation via hash IDs token
        blocks."""
        ids = row[config.hash_ids_column]
        expected = row[config.prompt_tokens_column] / config.hash_id_block_size
        if math.floor(expected) != len(ids) and math.ceil(expected) == len(ids):
            ids.pop()
        for idx, hash_id in enumerate(ids):
            if hash_id not in self.hash_id_table:
                prev_id = None if idx == 0 else ids[idx - 1]
                self.sibling_token_blocks.setdefault(prev_id, set())
                self.hash_id_table[hash_id] = _create_distinct_token_block(
                    config.hash_id_block_size,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].add(self.hash_id_table[hash_id])
        prompt = _create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
        remainder = _generate_remaining_prompt(
            row[config.prompt_tokens_column] % config.hash_id_block_size,
            processor,
            faker,
        )
        return prompt + " " + remainder
