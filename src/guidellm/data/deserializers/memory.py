from __future__ import annotations

import contextlib
import csv
import json
from collections.abc import Callable
from io import StringIO
from typing import Any, cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)

__all__ = [
    "InMemoryCsvDatasetDeserializer",
    "InMemoryDictDatasetDeserializer",
    "InMemoryDictListDatasetDeserializer",
    "InMemoryItemListDatasetDeserializer",
    "InMemoryJsonStrDatasetDeserializer",
]


@DatasetDeserializerFactory.register("in_memory_dict")
class InMemoryDictDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if (
            not data
            or not isinstance(data, dict)
            or not all(
                isinstance(key, str) and isinstance(val, list)
                for key, val in data.items()
            )
        ):
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

        return Dataset.from_dict(data, **data_kwargs)


@DatasetDeserializerFactory.register("in_memory_dict_list")
class InMemoryDictListDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if (
            not data
            or not isinstance(data, list)
            or not all(isinstance(item, dict) for item in data)
            or not all(isinstance(key, str) for item in data for key in item)
        ):
            raise DataNotSupportedError(
                f"Unsupported data for InMemoryDictListDatasetDeserializer, "
                f"expected list of dicts, got {data}"
            )

        typed_data: list[dict[str, Any]] = cast("list[dict[str, Any]]", data)
        first_keys = set(typed_data[0].keys())
        for index, item in enumerate(typed_data):
            if set(item.keys()) != first_keys:
                raise DataNotSupportedError(
                    f"All dictionaries must have the same keys. "
                    f"Expected keys: {first_keys}, "
                    f"got keys at index {index}: {set(item.keys())}"
                )

        # Convert list of dicts to dict of lists
        result_dict: dict = {key: [] for key in first_keys}
        for item in typed_data:
            for key, value in item.items():
                result_dict[key].append(value)

        return Dataset.from_dict(result_dict, **data_kwargs)


@DatasetDeserializerFactory.register("in_memory_item_list")
class InMemoryItemListDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        primitive_types = (str, int, float, bool, type(None))
        if (
            not data
            or not isinstance(data, list)
            or not all(isinstance(item, primitive_types) for item in data)
        ):
            raise DataNotSupportedError(
                f"Unsupported data for InMemoryItemListDatasetDeserializer, "
                f"expected list of primitive items, got {data}"
            )

        column_name = data_kwargs.pop("column_name", "data")

        return Dataset.from_dict({column_name: data}, **data_kwargs)


@DatasetDeserializerFactory.register("in_memory_json_str")
class InMemoryJsonStrDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        if (
            isinstance(data, str)
            and (json_str := data.strip())
            and (
                (json_str.startswith("{") and json_str.endswith("}"))
                or (json_str.startswith("[") and json_str.endswith("]"))
            )
        ):
            with contextlib.suppress(Exception):
                parsed_data = json.loads(data)

            deserializers = [
                InMemoryDictDatasetDeserializer(),
                InMemoryDictListDatasetDeserializer(),
                InMemoryItemListDatasetDeserializer(),
            ]

            for deserializer in deserializers:
                with contextlib.suppress(DataNotSupportedError):
                    return deserializer(
                        parsed_data, processor_factory, random_seed, **data_kwargs
                    )

        raise DataNotSupportedError(
            f"Unsupported data for InMemoryJsonStrDatasetDeserializer, "
            f"expected JSON string with a list or dict of items, got {data}"
        )


@DatasetDeserializerFactory.register("in_memory_csv_str")
class InMemoryCsvDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        if (
            isinstance(data, str)
            and (csv_str := data.strip())
            and len(csv_str.split("\n")) > 0
        ):
            with contextlib.suppress(Exception):
                csv_buffer = StringIO(data)
                reader = csv.DictReader(csv_buffer)
                rows = list(reader)

                return InMemoryDictListDatasetDeserializer()(
                    rows, processor_factory, random_seed, **data_kwargs
                )

        raise DataNotSupportedError(
            f"Unsupported data for InMemoryCsvDatasetDeserializer, "
            f"expected CSV string, got {type(data)}"
        )
