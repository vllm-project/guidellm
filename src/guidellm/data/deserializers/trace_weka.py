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
    decode_prompt,
    generate_token_ids,
    get_missing_columns,
)
from guidellm.data.schemas import DataArgs

__all__ = ["WEKATraceFormatArgs"]


def _find_requests_column(dataset: Dataset) -> str | None:
    for name, val in dataset.features.items():
        if isinstance(val, List):
            if len(dataset[name][0]) > 0 and isinstance(dataset[name][0][0], dict):
                return name
    return None


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

    def __init__(self, dataset: Dataset) -> None:
        self.conversation_locations = [[loc] for loc in list(range(len(dataset)))]

        self.hash_id_table: dict[int, tuple[int, ...]] = {}
        self.sibling_token_blocks: dict[Any, set[tuple[int, ...]]] = {}
        self.requests_col = _find_requests_column(dataset)
        if self.requests_col is None:
            raise DataNotSupportedError(
                "WEKA format: Failed to find requests column or requests was empty"
            )

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

    def find_required_columns(
        self, config: WEKATraceFormatArgs, columns: list[str], dataset: Dataset
    ) -> list[str]:
        """TODO: Handle edge cases"""
        conv_col = config.conversation_id_column
        if conv_col not in dataset.column_names:
            return [config.conversation_id_column]
        columns.remove(conv_col)
        return get_missing_columns(columns, dataset[self.requests_col][0][0].keys())

    def get_conversation_id_trace(
        self,
        config: WEKATraceFormatArgs,
        conversation_location: list[int],
        dataset: Dataset,
    ) -> list[str] | None:
        return [dataset[conversation_location[0]][config.conversation_id_column]]

    def get_conversation_iter(
        self, config: WEKATraceFormatArgs, dataset: Dataset
    ) -> Iterable[Dataset]:
        curr_conv = 0
        while True:
            try:
                trace_rows = dataset[self.requests_col][
                    self.conversation_locations[curr_conv][0]
                ]
                trace_rows = Dataset.from_list(trace_rows)
                trace_rows.sort(config.timestamp_column)
            except IndexError:
                break
            curr_conv += 1
            yield trace_rows

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
                self.hash_id_table[hash_id] = create_distinct_token_block(
                    config.hash_id_block_size,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].add(self.hash_id_table[hash_id])
        prompt = create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
        remainder = _generate_remaining_prompt(
            row[config.prompt_tokens_column] % config.hash_id_block_size,
            processor,
            faker,
        )
        if not prompt:
            return remainder
        if not remainder:
            return prompt
        return f"{prompt} {remainder}"
