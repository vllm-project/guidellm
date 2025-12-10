from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from datasets import Dataset, IterableDataset

from guidellm.schemas import GenerationRequest
from guidellm.utils import RegistryMixin

__all__ = ["DataDependentPreprocessor", "DatasetPreprocessor", "PreprocessorRegistry"]


@runtime_checkable
class DatasetPreprocessor(Protocol):
    def __call__(self, item: dict[str, Any]) -> GenerationRequest | dict[str, Any]: ...


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
    pass
