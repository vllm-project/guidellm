"""Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (consisting of at least the columns timestamp, input_length,
output_length) and yields one row per line with a synthetic prompt matching the
requested input_length for replay benchmarks."""

from __future__ import annotations

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
from guidellm.data.schemas import DataArgs
from guidellm.utils.hf_datasets import load_dataset_from_file
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "TraceDataArgs",
    "TraceDatasetDeserializer",
    "TraceFormatBase",
    "TraceFormatRegistry",
    "decode_prompt",
    "generate_token_ids",
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
) -> list[int]:
    """Generate `token_count` synthetic token ids for trace prompt construction.

    Ideally, `margin_of_safety` should be set to slighty more than
    the average number of characters used by tokenizers to form one token."""
    attempt = 0
    while True:
        attempt += 1
        # The Faker.text() can only generate text of at least 5 characters.
        num_chars = max(token_count * margin_of_safety * attempt, 5)
        text = faker.text(max_nb_chars=num_chars)
        token_ids = processor.encode(text)
        if len(token_ids) >= token_count:
            return token_ids[:token_count]


def validate_trace_path(path: Path | str) -> Path:
    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Trace file is empty: {path}")
    return path


def check_and_raise_missing_columns(
    required_columns: list[str], actual_columns: list[str]
) -> None:
    missing = [c for c in required_columns if c not in actual_columns]
    if missing:
        raise KeyError(f"Trace row missing required columns: {missing}")


def load_trace_rows(
    path: Path | str,
    timestamp_column_name: str,
    required_columns: Features,
    **data_kwargs: Any,
) -> Dataset:
    """
    Load trace file rows as a HuggingFace Dataset.

    Every column in required_columns must exist in the dataset;
    otherwise KeyError is raised with a descriptive message.
    Rows are sorted by column timestamp_column_name.

    :param path: Path to the trace file.
    :param timestamp_column_name: Name of the timestamp column used to sort trace rows.
    :param required_columns: List of column/fields that each row must have. Must contain
    the timestamp column.
    :param data_kwargs: Additional keyword arguments forwarded to load_dataset.
    :return: HuggingFace Dataset (iterable as dicts, column-accessible).
    :raises DataNotSupportedError: For any of the following reasons:
    - The dataset is empty or has no valid rows
    - A required column contains a NoneType
    - A required column failed during cast to feature type

    :raises KeyError: If a required column is missing in the dataset.
    :raises ValueError: If the file format is not .jsonl, .json, .csv or .parquet.
    """
    path = validate_trace_path(path)
    trace_dataset = load_dataset_from_file(path, **data_kwargs)
    if required_columns:
        check_and_raise_missing_columns(
            required_columns.keys(), trace_dataset.column_names
        )

    if not trace_dataset:
        raise DataNotSupportedError(f"Trace file has no valid rows: {path}")
    for name, val in required_columns.items():
        if trace_dataset.data[name].null_count != 0:
            raise DataNotSupportedError(f"Missing column values in {name}")
        try:
            trace_dataset.cast_column(name, val)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e

    return trace_dataset.sort(timestamp_column_name)


class TraceFormatBase(Protocol):
    def __init__(self) -> None: ...

    def required_columns(self, config) -> Features: ...

    def validate_row(self, config, row: dict) -> None:
        """Called within `trace_common.TraceExamplesIterable` on initialization,
        immediately after doing its own checks on the row."""

    def create_prompt(
        self,
        config,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        """Called within `trace_common.TraceExamplesIterable` on each iteration.
        Returns a generated synthetic prompt."""


class TraceFormatRegistry(RegistryMixin[type[TraceFormatBase]]):
    @classmethod
    def dispatch(cls, config: TraceDataArgs) -> TraceFormatBase:
        format_from_type = cls.get_registered_object(config.kind)
        if format_from_type is None:
            raise DataNotSupportedError(
                f"Format type '{config.kind}' is not registered."
            )
        return format_from_type()


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


def validate_row(row: dict, config: TraceDataArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    if n_in < 0 or n_out < 0:
        raise DataNotSupportedError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )


class TraceExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: TraceDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.format = TraceFormatRegistry.dispatch(self.config)
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        try:
            self.trace_rows = load_trace_rows(
                config.path,
                config.timestamp_column,
                required_columns=Features(
                    {
                        config.timestamp_column: Value("float"),
                        config.prompt_tokens_column: Value("int32"),
                        config.output_tokens_column: Value("int32"),
                        **dict(self.format.required_columns(self.config)),
                    }
                ),
                **config.load_kwargs,
            )
        except (DatasetGenerationError, KeyError, ValueError) as e:
            raise DataNotSupportedError(str(e)) from e

        for row in self.trace_rows:
            validate_row(row, self.config)
            self.format.validate_row(self.config, row)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        row_idx = 0
        timestamps = self.trace_rows[self.config.timestamp_column]
        while True:
            try:
                row = self.trace_rows[row_idx]
            except IndexError:
                break

            prompt = self.format.create_prompt(
                self.config, row, self.processor, self.faker
            )
            relative_timestamp = timestamps[row_idx] - timestamps[0]
            yield (
                row_idx,
                {
                    "prompt": prompt,
                    "prompt_tokens_count": row[self.config.prompt_tokens_column],
                    "output_tokens_count": row[self.config.output_tokens_column],
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
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        ex_iterable = TraceExamplesIterable(config, processor, random_seed)
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


@DatasetDeserializerFactory.register(["trace_synthetic"])
class TraceDatasetDeserializer(DatasetDeserializer):
    """Dataset deserializer for all trace formats."""

    def __call__(
        self,
        config: TraceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
    ) -> IterableDataset:
        if not config.path.exists():
            raise DataNotSupportedError(f"Trace file not found: {config.path}")
        if not config.path.is_file():
            raise DataNotSupportedError(f"Trace path is not a file: {config.path}")
        return TraceDataset(config, processor_factory(), random_seed)
