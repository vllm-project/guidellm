from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from datasets import DatasetInfo, IterableDataset
from datasets.iterable_dataset import _BaseExamplesIterable
from pydantic import Field, field_serializer
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs

__all__ = ["TraceDataArgs", "TraceDataset", "TraceDatasetDeserializer"]


@DataArgs.register("trace_file")
class TraceDataArgs(DataArgs):
    kind: Literal["trace_file"] = Field(
        default="trace_file",
        description="Type identifier for the trace dataset deserializer.",
    )
    format: str = Field(description="Format the trace file adhers to.")
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

    @field_serializer("path")
    @classmethod
    def serialize_path(cls, path: Path) -> str:
        """Serialize path as a string because Path is not JSON serializable."""
        return str(path)

    ex_iterable: ClassVar[type[_BaseExamplesIterable]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is TraceDataArgs:
            return
        if not hasattr(cls, "ex_iterable"):
            raise NotImplementedError(f"{cls.__name__} must define ex_iterable")
        if not issubclass(cls.ex_iterable, _BaseExamplesIterable):
            raise TypeError(
                f"{cls.__name__}.ex_iterable must inherit _BaseExamplesIterable"
            )


class TraceDataset(IterableDataset):
    def __init__(self, ex_iterable: _BaseExamplesIterable):
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

        ex_iterable = config.ex_iterable()(config, processor_factory(), random_seed)
        return TraceDataset(ex_iterable)
