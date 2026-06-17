from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.exceptions import DatasetGenerationError
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field, field_serializer, field_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config
from guidellm.utils.trace_io import TraceColumn, load_trace_rows

__all__ = [
    "TraceDataArgs",
    "TraceDataset",
    "TraceDatasetDeserializer",
    "TraceFormatArgs",
]


class TraceFormatArgs(PydanticClassRegistryMixin["TraceFormatArgs"], ABC):
    model_config = standard_model_config()
    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[TraceFormatArgs]:
        """Return base type for polymorphic validation hierarchy.

        :return: Base Profile class for schema validation"""
        if cls.__name__ == "TraceFormatArgs":
            return cls
        return TraceFormatArgs

    kind: str = Field(
        description="Type identifier for the format arguments.",
    )

    required_columns: ClassVar[list[TraceColumn]] = []

    @classmethod
    def validate_row(
        cls,
        row: dict,  # noqa: ARG002
        config: TraceDataArgs,  # noqa: ARG002
    ) -> None:
        """Called within `trace_common.TraceExamplesIterable` on initialization,
        immediately after doing its own checks on the row."""

    @classmethod
    @abstractmethod
    def create_prompt(
        cls,
        row: dict,  # noqa: ARG002
        config: TraceDataArgs,  # noqa: ARG002
        processor: PreTrainedTokenizerBase,  # noqa: ARG002
        faker: Faker,  # noqa: ARG002
    ) -> str:
        """Called within `trace_common.TraceExamplesIterable` on each iteration.
        Returns a generated synthetic prompt."""


@DataArgs.register("trace")
class TraceDataArgs(DataArgs):
    kind: Literal["trace"] = Field(
        default="trace",
        description="Type identifier for the trace dataset deserializer.",
    )
    format: TraceFormatArgs = Field(
        description="Format the trace file adhers to, "
        "including its specific configuration arguments."
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

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, format: Any) -> TraceFormatArgs:
        """Validates format against `TraceFormatArgs`. Returns
        the subclass format which `TraceFormatArgs` dispatched to on success."""
        return TraceFormatArgs.model_validate(format)

    @field_serializer("path")
    @classmethod
    def serialize_path(cls, path: Path) -> str:
        """Serialize path as a string because Path is not JSON serializable."""
        return str(path)


def _validate_row(row: dict, config: TraceDataArgs) -> None:
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
                    *self.config.format.required_columns,
                ],
            )
        except (DatasetGenerationError, KeyError, ValueError) as e:
            raise DataNotSupportedError(str(e)) from e

        for row in self.trace_rows:
            _validate_row(row, self.config)
            self.config.format.validate_row(row, self.config)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        row_idx = 0
        while True:
            try:
                row = self.trace_rows[row_idx]
            except IndexError:
                break

            prompt = self.config.format.create_prompt(
                row, self.config, self.processor, self.faker
            )
            timestamps = self.trace_rows[self.config.timestamp_column]
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


@DatasetDeserializerFactory.register("trace_file")
class TraceDatasetDeserializer(DatasetDeserializer):
    """TODO"""

    def __call__(
        self,
        config: TraceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
    ) -> IterableDataset:
        if not config.path.is_file():
            raise DataNotSupportedError(
                f"{type(self).__name__} expects a path to a trace file, "
                f"got {config.path}"
            )
        return TraceDataset(config, processor_factory(), random_seed)
