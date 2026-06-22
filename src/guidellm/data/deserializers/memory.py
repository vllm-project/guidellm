from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from datasets import Dataset
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs

__all__ = [
    "InMemoryDictDataArgs",
    "InMemoryDictDatasetDeserializer",
    "InMemoryDictListDataArgs",
    "InMemoryDictListDatasetDeserializer",
    "InMemoryItemListDataArgs",
    "InMemoryItemListDatasetDeserializer",
]


@DataArgs.register("in_memory_dict")
class InMemoryDictDataArgs(DataArgs):
    """Model for in-memory data deserializer arguments."""

    kind: Literal["in_memory_dict"] = Field(  # type: ignore[assignment]
        default="in_memory_dict",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: dict[str, list] = Field(
        description="In-memory data input for the dataset deserializer.",
        examples=[{"column1": [1, 2, 3], "column2": [4, 5, 6]}],
    )


@DatasetDeserializerFactory.register("in_memory_dict")
class InMemoryDictDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: InMemoryDictDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if not (data := config.data):
            raise DataNotSupportedError(
                f"Unsupported data for InMemoryDictDatasetDeserializer, "
                f"expected dict[str, list], got {data}"
            )

        rows = len(list(data.values())[0])
        if not all(len(val) == rows for val in data.values()):
            raise DataNotSupportedError(
                "All lists in the data dictionary must have the same length, "
                f"expected {rows} for all keys {list(data.keys())}"
            )

        return Dataset.from_dict(data, **config.load_kwargs)


@DataArgs.register("in_memory_dict_list")
class InMemoryDictListDataArgs(DataArgs):
    kind: Literal["in_memory_dict_list"] = Field(  # type: ignore[assignment]
        default="in_memory_dict_list",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: list[dict[str, Any]] = Field(
        description="In-memory list of dicts input for the dataset deserializer.",
        examples=[{"column1": 1, "column2": 2}, {"column1": 3, "column2": 4}],
    )


@DatasetDeserializerFactory.register("in_memory_dict_list")
class InMemoryDictListDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: InMemoryDictListDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if not (typed_data := config.data):
            raise DataNotSupportedError(
                f"Unsupported data for InMemoryDictListDatasetDeserializer, "
                f"expected list of dicts, got {typed_data}"
            )

        first_keys = set(typed_data[0].keys())
        for index, item in enumerate(typed_data):
            if set(item.keys()) != first_keys:
                raise DataNotSupportedError(
                    f"All dictionaries must have the same keys. "
                    f"Expected keys: {first_keys}, "
                    f"got keys at index {index}: {set(item.keys())}"
                )

        result_dict: dict[str, list] = {key: [] for key in first_keys}
        for item in typed_data:
            for key, value in item.items():
                result_dict[key].append(value)

        return Dataset.from_dict(result_dict, **config.load_kwargs)


@DataArgs.register("in_memory_item_list")
class InMemoryItemListDataArgs(DataArgs):
    kind: Literal["in_memory_item_list"] = Field(  # type: ignore[assignment]
        default="in_memory_item_list",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: list[str | int | float | bool | None] = Field(
        description="In-memory list of primitive items for the dataset deserializer.",
        examples=[1, 2, 3, 4, 5],
    )
    column_name: str = Field(
        default="data",
        description="Column name to use when creating the dataset.",
    )


@DatasetDeserializerFactory.register("in_memory_item_list")
class InMemoryItemListDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: InMemoryItemListDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if not (data := config.data):
            raise DataNotSupportedError(
                f"Unsupported data for InMemoryItemListDatasetDeserializer, "
                f"expected list of primitive items, got {data}"
            )

        return Dataset.from_dict({config.column_name: data}, **config.load_kwargs)
