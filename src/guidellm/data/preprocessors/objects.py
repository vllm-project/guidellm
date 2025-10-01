from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from datasets import Dataset, IterableDataset

from guidellm.data.objects import GenerativeDatasetArgs

__all__ = ["DatasetPreprocessor"]


@runtime_checkable
class DatasetPreprocessor(Protocol):
    def init_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[GenerativeDatasetArgs],
    ): ...

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]: ...
