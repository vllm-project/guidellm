from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Protocol, Union, runtime_checkable

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizerBase

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
        data_kwargs: dict[str, Any],
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> dict[str, list]: ...


class DatasetDeserializerFactory(
    RegistryMixin[Union["type[DatasetDeserializer]", DatasetDeserializer]],
):
    @classmethod
    def deserialize(
        cls,
        data: Any,
        data_kwargs: dict[str, Any],
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        type_: str | None = None,
    ) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
        if type_ is not None:
            deserializer = cls.get_registered_object(type_)

            if deserializer is None:
                raise DataNotSupportedError(
                    f"Deserializer type '{type_}' is not registered. "
                    f"Available types: {cls.registry}"
                )
            elif isinstance(deserializer, type):
                deserializer_fn = deserializer()
            else:
                deserializer_fn = deserializer

            return deserializer_fn(
                data=data,
                data_kwargs=data_kwargs,
                processor_factory=processor_factory,
                random_seed=random_seed,
            )

        for deserializer in cls.registered_objects():
            deserializer_fn: DatasetDeserializer = (
                deserializer() if isinstance(deserializer, type) else deserializer
            )

            with contextlib.suppress(DataNotSupportedError):
                return deserializer_fn(
                    data=data,
                    data_kwargs=data_kwargs,
                    processor_factory=processor_factory,
                    random_seed=random_seed,
                )

        raise DataNotSupportedError(
            f"No suitable deserializer found for data {data} with kwargs {data_kwargs}."
        )
