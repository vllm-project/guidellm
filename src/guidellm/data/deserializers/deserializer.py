from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from disdantic import RegistryMixin
from transformers import PreTrainedTokenizerBase

from guidellm.data.schemas import DataArgs, DataNotSupportedError, DatasetType

__all__ = [
    "DatasetDeserializer",
    "DatasetDeserializerFactory",
]


@runtime_checkable
class DatasetDeserializer(Protocol):
    def __call__(
        self,
        config,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> DatasetType: ...


class DatasetDeserializerFactory(RegistryMixin["type[DatasetDeserializer]"]):
    @classmethod
    def deserialize(
        cls,
        config: DataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> DatasetType:
        deserializer_from_type = cls.get_registered_object(config.kind)
        if deserializer_from_type is None:
            raise DataNotSupportedError(
                f"Deserializer type '{config.kind}' is not registered."
            )

        deserializer_fn = deserializer_from_type()
        dataset: DatasetType = deserializer_fn(
            config=config,
            processor_factory=processor_factory,
            random_seed=random_seed,
        )

        return dataset
