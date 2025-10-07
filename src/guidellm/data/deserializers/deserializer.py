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
        **data_kwargs: dict[str, Any],
    ) -> Dataset | IterableDataset:
        dataset = None

        if type_ is None:
            for deserializer in cls.registered_objects():
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
                f"with kwargs {data_kwargs} and type_ {type_}."
            )

        return resolve_dataset_split(dataset) if resolve_split else dataset
