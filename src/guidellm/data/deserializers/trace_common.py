"""Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (consisting of at least the columns timestamp, input_length,
output_length) and yields one row per line with a synthetic prompt matching the
requested input_length for replay benchmarks."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

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
from guidellm.utils.json_unwrap import (
    VirtualColumnLocation,
    construct_virtual_column_locations,
    get_json_column_names,
    unzip_virtual_column_locations,
)
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "TraceDataArgs",
    "TraceDatasetDeserializer",
    "TraceFormatBase",
    "TraceFormatRegistry",
    "create_prompt_from_hash_ids",
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
) -> tuple[int]:
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


def create_prompt_from_hash_ids(
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


class TraceFormatBase(Protocol):
    def __init__(self) -> None: ...

    def reset(self) -> None:
        pass

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
        trace_rows: Dataset,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.format = TraceFormatRegistry.dispatch(self.config)
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        self.trace_rows = trace_rows
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        timestamps = self.trace_rows[self.config.timestamp_column]
        conv_col = self.config.conversation_id_column
        current_conv = None
        conv_start_ts = timestamps[0]
        for row_idx, row in enumerate(self.trace_rows):
            if conv_col:
                conv_id = row[conv_col]
                if conv_id != current_conv:
                    current_conv = conv_id
                    conv_start_ts = row[self.config.timestamp_column]
                    self.format.reset()

            prompt = self.format.create_prompt(
                self.config, row, self.processor, self.faker
            )
            relative_timestamp = timestamps[row_idx] - conv_start_ts
            yield (
                row_idx,
                {
                    "prompt": prompt,
                    "prompt_tokens_count": row[self.config.prompt_tokens_column],
                    "output_tokens_count": row[self.config.output_tokens_column],
                    "relative_timestamp": relative_timestamp,
                },
            )

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
        trace_rows: Dataset,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        ex_iterable = TraceExamplesIterable(config, trace_rows, processor, random_seed)
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


def _get_missing_columns(
    required_columns: list[str], actual_columns: list[str]
) -> list[str]:
    return [c for c in required_columns if c not in actual_columns]


class Status(enum.Enum):
    FAILURE = enum.auto()
    SUCCESS = enum.auto()


@dataclasses.dataclass
class ColumnSearchResult:
    status: Status
    checked_json_columns: bool
    apropos_column_names: Sequence[str | VirtualColumnLocation]


def _is_supported_json_data_type(data: Any) -> bool:
    """Currently, only JSON data in the form of a list of JSON objects is supported."""
    return isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict))


def _find_virtual_columns(
    sample_row: dict[str, Any],
    json_column_names: list[str],
    target_columns: list[str],
) -> ColumnSearchResult:
    """Required columns must all be stored within the same column."""
    completely_missing = set(target_columns)
    for col in json_column_names:
        parsed = sample_row[col]
        if _is_supported_json_data_type(parsed) and len(parsed) > 0:
            virtual_columns = [] if parsed is None else list(parsed[0].keys())
            missing = _get_missing_columns(target_columns, virtual_columns)
            if not missing:
                locations = construct_virtual_column_locations(col, target_columns)
                return ColumnSearchResult(Status.SUCCESS, True, locations)
            completely_missing = completely_missing.difference(missing)
    if json_column_names:
        return ColumnSearchResult(Status.FAILURE, True, list(completely_missing))
    return ColumnSearchResult(Status.FAILURE, False, [])


def _find_required_columns(
    columns: list[str], dataset: Dataset, conversation_id_col: str | None = None
) -> ColumnSearchResult:
    """Returns a list of all missing columns on failure. Otherwise returns a list of
    the locations of any required columns embedded inside a JSON dict."""
    missing = _get_missing_columns(columns, dataset.column_names)
    if missing:
        # The conversation IDs should always be top-level.
        if conversation_id_col:
            if conversation_id_col in missing:
                return ColumnSearchResult(Status.FAILURE, False, conversation_id_col)
            columns.remove(conversation_id_col)
        json_column_names = get_json_column_names(dataset)
        sample = dataset[0]
        result = _find_virtual_columns(sample, json_column_names, columns)
        if result.status is Status.FAILURE and not result.checked_json_columns:
            result.apropos_column_names = missing
        return result
    return ColumnSearchResult(Status.SUCCESS, False, [])


def _make_columns_from_virtual(
    batch: dict[str, list],
    *args,
    wrapper_col: str,
    virtual_cols: list[str],
    conversation_id_col: str | None = None,
) -> dict[str, list]:
    """Intended to be used with `datasets.Dataset.map()`."""
    indices = args[0] if args else []
    json_dicts = []
    conv_ids = []
    for batch_idx, json_dicts_list in enumerate(batch[wrapper_col]):
        json_dicts.extend(json_dicts_list)
        if conversation_id_col:
            conv_ids.extend([indices[batch_idx]] * len(json_dicts_list))
    result = {c: [row[c] for row in json_dicts] for c in virtual_cols}
    if conversation_id_col:
        result[conversation_id_col] = conv_ids
    return result


def _make_dataset_from_virtual(
    dataset: Dataset,
    columns: list[VirtualColumnLocation],
    conversation_id_col: str | None = None,
) -> Dataset:
    """Assumes all virtual columns are stored inside the same column.
    (Currently ensured by `_is_supported_json_data_type`)."""
    wrapper_cols, virt_cols = unzip_virtual_column_locations(columns)
    return dataset.map(
        _make_columns_from_virtual,
        batched=True,
        with_indices=conversation_id_col is not None,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "wrapper_col": wrapper_cols[0],
            "virtual_cols": virt_cols,
            "conversation_id_col": conversation_id_col,
        },
    )


def _handle_column_search_result(
    result: ColumnSearchResult,
    dataset: Dataset,
    conversation_id_col: str | None = None,
) -> Dataset:
    """Returns an updated dataset where any required columns found wrapped inside
    JSON dicts are unwrapped and added as columns to the dataset.

    :raises KeyError: If a required column is missing in the dataset."""
    if result.status is Status.FAILURE:
        additional_info = ""
        if result.checked_json_columns:
            additional_info = (
                "Note: GuideLLM searched columns with lists of JSON objects after "
                "failing to find them at the top level. "
                "Ensure that all required columns are wrapped in the same column if "
                "this is where they are intended to be found."
            )
        raise KeyError(
            f"Trace row missing required columns: {result.apropos_column_names} "
            f"{additional_info}"
        )
    if not result.checked_json_columns:
        return dataset
    return _make_dataset_from_virtual(
        dataset,
        cast("list[VirtualColumnLocation]", result.apropos_column_names),
        conversation_id_col,
    )


def _load_trace_rows(
    dataset: Dataset,
    timestamp_column_name: str,
    required_columns: Features,
    conversation_id_column_name: str | None = None,
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
    """
    result = _find_required_columns(
        list(required_columns.keys()), dataset, conversation_id_column_name
    )
    dataset = _handle_column_search_result(result, dataset, conversation_id_column_name)

    for name, val in required_columns.items():
        if dataset.data[name].null_count != 0:
            raise DataNotSupportedError(f"Missing column values in {name}")
        try:
            dataset.cast_column(name, val)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e

    if conversation_id_column_name:
        return dataset.sort([conversation_id_column_name, timestamp_column_name])
    return dataset.sort(timestamp_column_name)


def validate_path(path: Path) -> None:
    if not path.exists():
        raise DataNotSupportedError(f"Trace file not found: {path}")
    if not path.is_file():
        raise DataNotSupportedError(f"Trace path is not a file: {path}")
    if path.stat().st_size == 0:
        raise DataNotSupportedError(f"Trace file is empty: {path}")


def try_load_trace(config: TraceDataArgs, dataset: Dataset) -> Dataset:
    trace_format = TraceFormatRegistry.dispatch(config)
    try:
        return _load_trace_rows(
            dataset,
            config.timestamp_column,
            required_columns=Features(
                {
                    config.timestamp_column: Value("float"),
                    config.prompt_tokens_column: Value("int32"),
                    config.output_tokens_column: Value("int32"),
                    **dict(trace_format.required_columns(config)),
                }
            ),
            conversation_id_column_name=config.conversation_id_column,
        )
    except (DatasetGenerationError, KeyError, ValueError) as e:
        raise DataNotSupportedError(str(e)) from e


def _validate_row(row: dict, config: TraceDataArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    if n_in < 0 or n_out < 0:
        raise DataNotSupportedError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )


def validate_rows(config: TraceDataArgs, trace_rows: Dataset) -> None:
    trace_format = TraceFormatRegistry.dispatch(config)
    for row in trace_rows:
        _validate_row(row, config)
        trace_format.validate_row(config, row)


@DatasetDeserializerFactory.register(["trace_synthetic"])
class TraceDatasetDeserializer(DatasetDeserializer):
    """Dataset deserializer for all trace formats."""

    def __call__(
        self,
        config: TraceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
    ) -> IterableDataset:
        validate_path(config.path)
        try:
            dataset = load_dataset_from_file(config.path, **config.load_kwargs)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e
        if not dataset:
            raise DataNotSupportedError(f"Trace file has no valid rows: {config.path}")
        trace_rows = try_load_trace(config, dataset)
        validate_rows(config, trace_rows)
        return TraceDataset(config, trace_rows, processor_factory(), random_seed)
