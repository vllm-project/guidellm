"""Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (consisting of at least the columns timestamp, input_length,
output_length) and yields one row per line with a synthetic prompt matching the
requested input_length for replay benchmarks."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from datasets import (
    Dataset,
    DatasetInfo,
    Features,
    IterableDataset,
    Value,
)
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.utils.hf_datasets import load_dataset_from_file
from guidellm.utils.json_unwrap import try_json_load
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "MissingColumnsLocation",
    "TraceDataArgs",
    "TraceDatasetDeserializer",
    "TraceFormatBase",
    "TraceFormatRegistry",
    "create_distinct_token_block",
    "create_prompt_from_hash_ids",
    "decode_prompt",
    "generate_token_ids",
    "get_missing_columns",
]


def decode_prompt(
    processor: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    """Decode token ids into a prompt string."""
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


def generate_token_ids(
    token_count: int,
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    margin_of_safety: int = 8,
) -> tuple[int, ...]:
    """Generate `token_count` synthetic token ids for trace prompt construction.

    Ideally, `margin_of_safety` should be set to slighty more than
    the average number of characters used by tokenizers to form one token."""
    attempt = 0
    while True:
        attempt += 1
        # The Faker.text() can only generate text of at least 5 characters.
        num_chars = max(token_count * margin_of_safety * attempt, 5)
        text = faker.text(num_chars)
        token_ids = processor.encode(text)
        if len(token_ids) >= token_count:
            return tuple(token_ids[:token_count])


def get_missing_columns(
    required_columns: list[str], actual_columns: list[str]
) -> list[str]:
    return [c for c in required_columns if c not in actual_columns]


def create_prompt_from_hash_ids(
    hash_ids: list[int],
    hash_id_table: dict[int, tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
) -> str:
    """Returns a synthetic prompt from `hash_ids` using pre-generated token blocks.

    Precondition: All ids in `hash_ids` appear in `hash_id_table`."""
    prompt_token_ids = [
        token for hash_id in hash_ids for token in hash_id_table[hash_id]
    ]
    return decode_prompt(processor, prompt_token_ids)


def create_distinct_token_block(
    block_size: int,
    sibling_token_blocks: set[tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    max_attempts: int = 20,
) -> tuple[int, ...]:
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


class TraceFormatBase(Protocol):
    conversation_locations: list[list[int]]

    def __init__(self, config, dataset: Dataset) -> None: ...

    def __iter__(self) -> Iterable[Dataset]:
        """TODO"""

    def reset(self) -> None:
        pass

    def required_columns(self) -> Features: ...

    def find_required_columns(self, columns: list[str]) -> list[str]:
        """TODO"""

    def get_conversation_id_trace(
        self, conversation_location: list[int]
    ) -> list[str] | None:
        """TODO"""

    def validate_row(self, row: dict) -> None:
        """OUTDATED: Called within `trace_common.TraceExamplesIterable` on
        initialization, immediately after doing its own checks on the row."""

    def create_prompt(
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        """Called within `trace_common.TraceExamplesIterable` on each iteration.
        Returns a generated synthetic prompt."""


class TraceFormatRegistry(RegistryMixin[type[TraceFormatBase]]):
    @classmethod
    def dispatch(cls, config: TraceDataArgs, dataset: Dataset) -> TraceFormatBase:
        format_from_type = cls.get_registered_object(config.kind)
        if format_from_type is None:
            raise DataNotSupportedError(
                f"Format type '{config.kind}' is not registered."
            )
        return format_from_type(config, dataset)


class TraceDataArgs(DataArgs):
    """Abstract class meant to be inherited by a trace format.
    For testing, use `trace_minimal.MinimalTraceFormatArgs` instead."""

    kind: str = Field(
        description="Type identifier for the trace dataset deserializer.",
    )
    path: Path = Field(description="Path to the trace file.")
    timestamp_column: str = Field(
        default="timestamp",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="input_length",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="output_length",
        description="Column name for output token counts in the trace file.",
    )
    conversation_id_column: str | None = Field(
        default=None,
        description=(
            "Column name for conversation IDs. Required for formats "
            "with conversation-scoped trace data such as hash IDs."
        ),
    )


class TraceExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: TraceDataArgs,
        trace_format: TraceFormatBase,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.format = trace_format
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        samples_count = 0
        for conv in self.format:  # type: ignore[attr-defined]
            start_ts = conv[0][self.config.timestamp_column]
            row = self._create_conversation_row(conv, start_ts)
            samples_count += len(conv)
            yield samples_count, row
            self.format.reset()

    def _create_conversation_row(
        self, conversation: Dataset, start_ts: float
    ) -> dict[str, Any]:
        """Build a ``conversation_turns`` payload for linear or branched graphs."""
        turns = []
        for turn_idx, turn in enumerate(conversation):
            parents = []
            if turn_idx > 0:
                parents.append(
                    ConversationParentRef(parent_node_id=f"main_{turn_idx - 1}")
                )

            # TODO: Branches & subagents

            prompt = self.format.create_prompt(turn, self.processor, self.faker)
            relative_timestamp = turn[self.config.timestamp_column] - start_ts
            columns = {
                "text_column": [prompt],
                "prompt_tokens_count_column": [turn[self.config.prompt_tokens_column]],
                "output_tokens_count_column": [turn[self.config.output_tokens_column]],
                "relative_timestamp_column": [relative_timestamp],
            }
            turns.append(
                ConversationTurnData(
                    node_id=f"main_{turn_idx}",
                    agent_id="default",
                    parents=parents,
                    columns=columns,
                )
            )
        graph_data = ConversationGraphData(turns=turns)
        payload = json.dumps(graph_data.model_dump(mode="json"))
        return {
            "conversation_turns": (
                payload.decode() if isinstance(payload, bytes) else payload
            )
        }

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features({"conversation_turns": Value("large_string")})

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> TraceExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> TraceExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self):
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class TraceDataset(IterableDataset):
    def __init__(
        self,
        config: TraceDataArgs,
        trace_format: TraceFormatBase,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        ex_iterable = TraceExamplesIterable(
            config, trace_format, processor, random_seed
        )
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic trace dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if hasattr(self._ex_iterable, "iteration_count"):
            self._ex_iterable.iteration_count = epoch


@dataclasses.dataclass
class MissingColumnsLocation:
    conversation_location: list[int]
    columns: list[str]


def _validate_path(path: Path) -> None:
    if not path.exists():
        raise DataNotSupportedError(f"Trace file not found: {path}")
    if not path.is_file():
        raise DataNotSupportedError(f"Trace path is not a file: {path}")
    if path.stat().st_size == 0:
        raise DataNotSupportedError(f"Trace file is empty: {path}")


def _validate_row(row: dict, config: TraceDataArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    if n_in < 0 or n_out < 0:
        raise DataNotSupportedError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )


def _raise_if_nonetype_found(dataset: Dataset) -> None:
    for col in dataset.column_names:
        if dataset.data[col].null_count != 0:
            raise DataNotSupportedError(f"Missing column values in {col}")


def _raise_if_incorrect_types(dataset: Dataset, features: Features) -> None:
    try:
        dataset.cast(features)
    except ValueError as e:
        raise DataNotSupportedError(str(e)) from e


def _validate_dataset(config: TraceDataArgs, trace_format: TraceFormatBase) -> None:
    features = Features(
        {
            config.timestamp_column: Value("float"),
            config.prompt_tokens_column: Value("int32"),
            config.output_tokens_column: Value("int32"),
            **dict(trace_format.required_columns()),
        }
    )
    for conv in trace_format:  # type: ignore[attr-defined]
        if config.conversation_id_column in features:
            features.pop(config.conversation_id_column)
        _raise_if_nonetype_found(conv)
        _raise_if_incorrect_types(conv, features)
        for row in conv:
            _validate_row(row, config)
            trace_format.validate_row(row)


def _handle_column_search(config: TraceDataArgs, trace_format: TraceFormatBase) -> None:
    features = Features(
        {
            config.timestamp_column: Value("float"),
            config.prompt_tokens_column: Value("int32"),
            config.output_tokens_column: Value("int32"),
            **dict(trace_format.required_columns()),
        }
    )
    missing = trace_format.find_required_columns(list(features.keys()))
    if missing:
        raise DataNotSupportedError(f"Trace missing required columns: {missing}")


def _load_all_json_strings(data: list | dict) -> list | dict:
    iterable = enumerate(data) if isinstance(data, list) else data.items()
    for k, v in iterable:
        if isinstance(v, str):
            res = try_json_load(v)
            if isinstance(res, (list, dict)):
                data[k] = res
        if isinstance(v, (list, dict)):
            data[k] = _load_all_json_strings(data[k])
    return data


def _deserialize_nested_data(batch: dict[str, list]) -> dict[str, list]:
    """Intended to be used with `datasets.Dataset.map()`."""
    sample = {k: v[0] for k, v in batch.items()}
    for col, val in sample.items():
        if isinstance(val, str) and isinstance(try_json_load(val), (list, dict)):
            batch[col] = list(map(try_json_load, batch[col]))
        if isinstance(val, (list, dict)):
            batch[col] = list(map(_load_all_json_strings, batch[col]))
    return batch


@DatasetDeserializerFactory.register(["trace_synthetic"])
class TraceDatasetDeserializer(DatasetDeserializer):
    """Dataset deserializer for all trace formats."""

    def __call__(
        self,
        config: TraceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
    ) -> IterableDataset:
        _validate_path(config.path)
        try:
            dataset = load_dataset_from_file(config.path, **config.load_kwargs)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e
        if not dataset:
            raise DataNotSupportedError(f"Trace file has no valid rows: {config.path}")
        dataset.map(_deserialize_nested_data, batched=True)
        trace_format = TraceFormatRegistry.dispatch(config, dataset)
        _handle_column_search(config, trace_format)
        _validate_dataset(config, trace_format)
        return TraceDataset(config, trace_format, processor_factory(), random_seed)
