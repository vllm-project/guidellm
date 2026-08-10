from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.schemas.data.deserializers import (
    InMemoryDictDataArgs,
    InMemoryDictListDataArgs,
    InMemoryItemListDataArgs,
)

__all__ = [
    "InMemoryDictDatasetDeserializer",
    "InMemoryDictListDatasetDeserializer",
    "InMemoryItemListDatasetDeserializer",
]


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
