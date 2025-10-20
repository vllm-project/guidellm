from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Protocol, Union, runtime_checkable

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.utils import resolve_dataset_split
from guidellm.utils import RegistryMixin

__all__ = [
    "DataNotSupportedError",
    "DatasetDeserializer",
    "DatasetDeserializerFactory",
]


class DataNotSupportedError(Exception):
    """Exception raised when data format is not supported by deserializer."""


@runtime_checkable
class DatasetDeserializer(Protocol):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> dict[str, list]: ...


class DatasetDeserializerFactory(
    RegistryMixin[Union["type[DatasetDeserializer]", DatasetDeserializer]],
):
    @classmethod
    def deserialize(
        cls,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        type_: str | None = None,
        resolve_split: bool = True,
        select_columns: list[str] | None = None,
        remove_columns: list[str] | None = None,
        **data_kwargs: dict[str, Any],
    ) -> Dataset | IterableDataset:
        dataset = None

        if type_ is None:
            for priority, objects in cls.get_priority_grouped_objects() :
                for deserializer in objects:
                    deserializer_fn: DatasetDeserializer = (
                        deserializer() if isinstance(deserializer, type) else deserializer
                    )

                    with contextlib.suppress(DataNotSupportedError):
                        dataset = deserializer_fn(
                            data=data,
                            processor_factory=processor_factory,
                            random_seed=random_seed,
                            **data_kwargs,
                        )
                if dataset is not None:
                    break
        elif deserializer := cls.get_registered_object(type_) is not None:
            deserializer_fn: DatasetDeserializer = (
                deserializer() if isinstance(deserializer, type) else deserializer
            )

            dataset = deserializer_fn(
                data=data,
                processor_factory=processor_factory,
                random_seed=random_seed,
                **data_kwargs,
            )

        if dataset is None:
            raise DataNotSupportedError(
                f"No suitable deserializer found for data {data} "
                f"with kwargs {data_kwargs} and deserializer type {type_}."
            )

        if resolve_split:
            dataset = resolve_dataset_split(dataset)

        if select_columns is not None or remove_columns is not None:
            column_names = dataset.column_names or list(next(iter(dataset)).keys())
            if select_columns is not None:
                remove_columns = [
                    col for col in column_names if col not in select_columns
                ]

            dataset = dataset.remove_columns(remove_columns)

        return dataset
