from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Protocol, Union, runtime_checkable

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
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
    ) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict: ...


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
        dataset: Dataset | None = None

        if type_ is None:
            dataset = cls._deserialize_with_registered_deserializers(
                data, processor_factory, random_seed, **data_kwargs
            )

        elif (deserializer_from_type := cls.get_registered_object(type_)) is not None:
            if isinstance(deserializer_from_type, type):
                deserializer_fn = deserializer_from_type()
            else:
                deserializer_fn = deserializer_from_type

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

    @classmethod
    def _deserialize_with_registered_deserializers(
        cls,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        if cls.registry is None:
            raise RuntimeError("registry is None; cannot deserialize dataset")
        dataset: Dataset | None = None

        errors = []
        # Note: There is no priority order for the deserializers, so all deserializers
        #  must be mutually exclusive to ensure deterministic behavior.
        for _name, deserializer in cls.registry.items():
            deserializer_fn: DatasetDeserializer = (
                deserializer() if isinstance(deserializer, type) else deserializer
            )

            try:
                with contextlib.suppress(DataNotSupportedError):
                    dataset = deserializer_fn(
                        data=data,
                        processor_factory=processor_factory,
                        random_seed=random_seed,
                        **data_kwargs,
                    )
            except Exception as e:  # noqa: BLE001 # The exceptions are saved.
                errors.append(e)

            if dataset is not None:
                break  # Found one that works. Continuing could overwrite it.

        if dataset is None and len(errors) > 0:
            raise DataNotSupportedError(
                f"data deserialization failed; {len(errors)} errors occurred while "
                f"attempting to deserialize data {data}: {errors}"
            )
        return dataset
