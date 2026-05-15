from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from datasets import Dataset, IterableDataset

from guidellm.data.schemas import DataPreprocessorArgs
from guidellm.utils.registry import RegistryMixin

__all__ = ["DataDependentPreprocessor", "DatasetPreprocessor", "PreprocessorRegistry"]


@runtime_checkable
class DatasetPreprocessor(Protocol):
    def __init__(self, config: DataPreprocessorArgs) -> None: ...

    def __call__(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


@runtime_checkable
class DataDependentPreprocessor(DatasetPreprocessor, Protocol):
    def setup_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[dict[str, Any]],
    ): ...


class PreprocessorRegistry(
    RegistryMixin[type[DatasetPreprocessor] | type[DataDependentPreprocessor]]
):
    @classmethod
    def create(cls, config: DataPreprocessorArgs) -> DatasetPreprocessor:
        """
        Factory method to create a DatasetPreprocessor instance based on configuration.

        :param config: A DataPreprocessorArgs object containing the configuration.
        """
        kind = config.kind
        preprocessor_cls = cls.get_registered_object(kind)

        if preprocessor_cls is None:
            raise ValueError(
                f"DatasetPreprocessor type '{kind}' is not registered."
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return preprocessor_cls(config)
