"""
Trace deserializer for Mooncake formatted files that generates synthetic prompts per
row.

Reads a trace file (timestamp, input_length, output_length, hash_ids) and yields one
row per line with a synthetic prompt matching the requested input_length for replay
benchmarks.
"""

import dataclasses
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, DatasetInfo, Features, IterableDataset, List, Value
from datasets.exceptions import DatasetGenerationError
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_synthetic import TraceSyntheticDataArgs
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import load_trace_rows

__all__ = ["TraceMooncakeDataArgs", "TraceMooncakeDatasetDeserializer"]


@DataArgs.register("mooncake")
class TraceMooncakeDataArgs(TraceSyntheticDataArgs):
    kind: Literal["trace_mooncake"] = Field(  # type: ignore[assignment]
        default="trace_mooncake",
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
    row: dict, config: TraceMooncakeDataArgs, hash_id: int
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
    margin_of_safety = 2
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


@dataclasses.dataclass
class Column:
    name: str
    feature_type: Value


def _load_formatted_trace_rows(
    path: Path,
    timestamp_column: Column,
    required_columns: list[Column],
) -> Dataset:
    """Load a trace file and format the columns."""
    try:
        rows = load_trace_rows(
            path,
            [col.name for col in required_columns],
            timestamp_column.name,
        )
    except (DatasetGenerationError, KeyError, ValueError) as e:
        raise DataNotSupportedError(str(e)) from e
    if not rows:
        raise DataNotSupportedError("Trace file is empty")
    for col in [timestamp_column] + required_columns:
        if rows.data[col.name].null_count != 0:
            raise DataNotSupportedError(f"NoneType found in {col}")
        try:
            rows.cast_column(col.name, col.feature_type)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e
    return rows


def _validate_row(row: dict, config: TraceMooncakeDataArgs) -> None:
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


class _TraceMooncakeExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: TraceMooncakeDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        self.trace_rows = _load_formatted_trace_rows(
            config.path,
            Column(config.timestamp_column, Value("float")),
            required_columns=[
                Column(config.prompt_tokens_column, Value("int32")),
                Column(config.output_tokens_column, Value("int32")),
                Column(config.hash_ids_column, List(Value("int32"))),
            ],
        )
        for row in self.trace_rows:
            _validate_row(row, self.config)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        hash_id_table: list[Any] = []
        sibling_token_blocks: dict[Any, list[list[int]]] = {}
        while True:
            try:
                row = self.trace_rows[self.iteration_count]
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
            yield (
                self.iteration_count,
                {
                    "prompt_tokens_count": row[self.config.prompt_tokens_column],
                    "output_tokens_count": row[self.config.output_tokens_column],
                    "prompt": prompt,
                },
            )
            self.iteration_count += 1

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
            }
        )

    @property
    def num_shards(self) -> int:
        return 1

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self):
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class _TraceMooncakeDataset(IterableDataset):
    def __init__(
        self,
        config: TraceMooncakeDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        ex_iterable = _TraceMooncakeExamplesIterable(config, processor, random_seed)
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Mooncake trace dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if isinstance(self._ex_iterable, _TraceMooncakeExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("mooncake")
class TraceMooncakeDatasetDeserializer(DatasetDeserializer):
    """Mooncake trace format requires a column for timestamps, prompt token counts,
    ouput token counts and lists of hash IDs.

    Hash IDs are globally unique identifiers based on the current and previous token
    blocks in a prompt. The relationships of IDs forms a tree, where every first ID
    in a prompt has a parent node of `None`. Parent nodes can have an unbounded
    number of children. Two hash IDs can represent identical blocks of tokens so long
    as they do not share the same parent (previous ID). For more details, see section 4
    of https://arxiv.org/pdf/2407.00079.

    Generated prompts match the prompt token count of the row."""

    def __call__(
        self,
        config: TraceMooncakeDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        if not config.path.is_file():
            raise DataNotSupportedError(
                f"{type(self).__name__} expects a path to a trace file, "
                f"got {config.path}"
            )
        return _TraceMooncakeDataset(config, processor_factory(), random_seed)
