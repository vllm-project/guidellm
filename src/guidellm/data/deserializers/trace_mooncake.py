"""
OUTDATED:
Trace deserializer for Mooncake formatted files that generates synthetic prompts per
row.

Reads a trace file (timestamp, input_length, output_length, hash_ids) and yields one
row per line with a synthetic prompt matching the requested input_length for replay
benchmarks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from datasets import Features, List, Value
from datasets.exceptions import DatasetGenerationError
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import DataNotSupportedError
from guidellm.data.deserializers.trace_common import TraceDataArgs
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import TraceColumn, load_trace_rows

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
    row: dict, config: MooncakeTraceFormatArgs, hash_id: int
) -> int:
    """Returns the number of prompt tokens needed to satisfy the row input length.
    This will be less than the block_size if the input length is not divisible by it
    and `hash_id` is the final ID for the row."""
    remainder = row[config.prompt_tokens_column] % config.hash_id_block_size
    if row[config.hash_ids_column][-1] == hash_id and remainder != 0:
        return remainder
    return config.hash_id_block_size


def _generate_token_ids(
    token_count: int,
    processor: PreTrainedTokenizerBase,
    faker: Faker,
) -> list[int]:
    """Generate `token_count` synthetic token ids for trace prompt construction."""
    # Ideally, `margin_of_safety` should be set to slighty more than
    # the average number of characters used by tokenizers to form one token.
    margin_of_safety = 8
    attempt = 0
    while True:
        attempt += 1
        # The Faker.text() can only generate text of at least 5 characters.
        num_chars = max(token_count * margin_of_safety * attempt, 5)
        text = faker.text(max_nb_chars=num_chars)
        token_ids = processor.encode(text)
        if len(token_ids) >= token_count:
            return token_ids[:token_count]


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
        token_ids = _generate_token_ids(block_size, processor, faker)
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
    prompt = processor.decode(prompt_token_ids, skip_special_tokens=True)
    if isinstance(prompt, list):
        return prompt[0] if prompt else ""
    return prompt


def _validate_row(row: dict, config: MooncakeTraceFormatArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    n_blocks = len(row[config.hash_ids_column])
    if n_in < 0 or n_out < 0:
        raise DataNotSupportedError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )
    for hash_id in row[config.hash_ids_column]:
        if hash_id < 0:
            raise DataNotSupportedError(f"Hash ID must be non-negative, got {hash_id}")
    if math.ceil(n_in / config.hash_id_block_size) != n_blocks:
        raise DataNotSupportedError(
            f"Input token count of {n_in} split into blocks of size "
            f"{config.hash_id_block_size} does not match given {n_blocks} blocks"
        )


class _MooncakeFormatExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: MooncakeTraceFormatArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        try:
            self.trace_rows = load_trace_rows(
                config.path,
                TraceColumn(config.timestamp_column, Value("float")),
                required_columns=[
                    TraceColumn(config.prompt_tokens_column, Value("int32")),
                    TraceColumn(config.output_tokens_column, Value("int32")),
                    TraceColumn(config.hash_ids_column, List(Value("int32"))),
                ],
            )
        except (DatasetGenerationError, KeyError, ValueError) as e:
            raise DataNotSupportedError(str(e)) from e
        for row in self.trace_rows:
            _validate_row(row, self.config)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        row_idx = 0
        hash_id_table: list[Any] = []
        sibling_token_blocks: dict[Any, list[list[int]]] = {}
        timestamps = self.trace_rows[self.config.timestamp_column]
        while True:
            try:
                row = self.trace_rows[row_idx]
            except IndexError:
                break

            ids = row[self.config.hash_ids_column]
            for idx, hash_id in enumerate(ids):
                if not _is_in_table(hash_id_table, hash_id):
                    _resize_to_hold_id(hash_id_table, hash_id)
                    prev_id = None if idx == 0 else ids[idx - 1]
                    num_tokens = _calculate_required_prompt_tokens(
                        row, self.config, hash_id
                    )
                    sibling_token_blocks.setdefault(prev_id, [])
                    hash_id_table[hash_id] = _create_distinct_token_block(
                        num_tokens,
                        sibling_token_blocks[prev_id],
                        self.processor,
                        self.faker,
                    )
                    sibling_token_blocks[prev_id].append(hash_id_table[hash_id])
            prompt = _create_prompt_from_hash_ids(ids, hash_id_table, self.processor)
            relative_timestamp = timestamps[row_idx] - timestamps[0]
            yield (
                row_idx,
                {
                    "prompt_tokens_count": row[self.config.prompt_tokens_column],
                    "output_tokens_count": row[self.config.output_tokens_column],
                    "prompt": prompt,
                    "relative_timestamp": relative_timestamp,
                },
            )
            row_idx += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features(
            {
                "prompt": Value("string"),
                "prompt_tokens_count": Value("int32"),
                "output_tokens_count": Value("int32"),
                "relative_timestamp": Value("float"),
            }
        )

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _MooncakeFormatExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _MooncakeFormatExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self):
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


@DataArgs.register("mooncake")
class MooncakeTraceFormatArgs(TraceDataArgs):
    format: Literal["mooncake"] = Field(
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
    ex_iterable = _MooncakeFormatExamplesIterable
