from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from guidellm.data.schemas import DataFinalizerArgs
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "DatasetFinalizer",
    "FinalizerRegistry",
]

DataT_co = TypeVar("DataT_co", covariant=True)


@runtime_checkable
class DatasetFinalizer(Protocol[DataT_co]):
    """
    Protocol for finalizing dataset rows into a desired data type.
    """

    def __init__(self, config: DataFinalizerArgs) -> None: ...

    def __call__(self, items: list[dict[str, Any]]) -> DataT_co: ...


class FinalizerRegistry(RegistryMixin[type[DatasetFinalizer]]):
    @classmethod
    def create(cls, config: DataFinalizerArgs) -> DatasetFinalizer:
        """
        Factory method to create a DatasetFinalizer instance based on configuration.

        :param config: A DataFinalizerArgs object containing the configuration.
        """
        kind = config.kind
        finalizer_cls = cls.get_registered_object(kind)

        if finalizer_cls is None:
            raise ValueError(
                f"DatasetFinalizer type '{kind}' is not registered."
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return finalizer_cls(config)
