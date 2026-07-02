from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic, Protocol, TypeVar

from disdantic import RegistryMixin

from guidellm.data.finalizers import DatasetFinalizer
from guidellm.data.preprocessors import DatasetPreprocessor
from guidellm.data.schemas import DataLoaderArgs, DatasetType

__all__ = ["DataLoader", "DataLoaderRegistry"]

DataT_co = TypeVar("DataT_co", covariant=True)


class DataLoader(Protocol[DataT_co]):
    def __init__(
        self,
        config: DataLoaderArgs,
        datasets: list[DatasetType],
        preprocessors: list[DatasetPreprocessor],
        finalizer: DatasetFinalizer[DataT_co],
        random_seed: int,
        **kwargs: Any,
    ) -> None: ...
    def __iter__(self) -> Iterator[DataT_co]: ...
    @property
    def info(self) -> dict[str, Any]: ...


class DataLoaderRegistry(Generic[DataT_co], RegistryMixin[type[DataLoader]]):
    @classmethod
    def create(
        cls,
        config: DataLoaderArgs,
        datasets: list[DatasetType],
        preprocessors: list[DatasetPreprocessor],
        finalizer: DatasetFinalizer[DataT_co],
        random_seed: int,
        **kwargs: Any,
    ) -> DataLoader[DataT_co]:
        """
        Factory method to create a DataLoader instance based on provided configuration.

        :param config: A DataEntrypointArgs object containing the configuration.
        """
        kind = config.kind
        data_loader_cls = cls.get_registered_object(kind)

        if data_loader_cls is None:
            raise ValueError(
                f"DataLoader type '{kind}' is not registered."
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return data_loader_cls(
            config=config,
            datasets=datasets,
            preprocessors=preprocessors,
            finalizer=finalizer,
            random_seed=random_seed,
            **kwargs,
        )
