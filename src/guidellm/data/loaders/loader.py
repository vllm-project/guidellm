from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from guidellm.data.schemas import DataEntrypointArgs
from guidellm.utils.registry import RegistryMixin

__all__ = ["DataLoader", "DataLoaderRegistry"]

DataT_co = TypeVar("DataT_co", covariant=True)


class DataLoader(Protocol[DataT_co]):
    def __init__(self, config: DataEntrypointArgs) -> None: ...
    def __iter__(self) -> Iterator[DataT_co]: ...
    @property
    def info(self) -> dict[str, Any]: ...


class DataLoaderRegistry(RegistryMixin[type[DataLoader]]):
    @classmethod
    def create(cls, config: DataEntrypointArgs, **kwargs) -> DataLoader:
        """
        Factory method to create a DataLoader instance based on provided configuration.

        :param config: A DataEntrypointArgs object containing the configuration.
        """
        kind = config.loader.kind
        data_loader_cls = cls.get_registered_object(kind)

        if data_loader_cls is None:
            raise ValueError(
                f"DataLoader type '{kind}' is not registered."
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return data_loader_cls(config, **kwargs)
