from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from torch.utils.data.dataloader import DataLoader as PyTorchDataLoader
from torch.utils.data.dataset import IterableDataset as TorchIterableDataset

from guidellm.data.finalizers import DatasetFinalizer
from guidellm.data.loaders.loader import DataLoader, DataLoaderRegistry
from guidellm.data.preprocessors import (
    DataDependentPreprocessor,
    DatasetPreprocessor,
)
from guidellm.data.schemas import DatasetType
from guidellm.logger import logger
from guidellm.schemas.data.loaders import TorchDataLoaderArgs
from guidellm.utils.mixins import InfoMixin

__all__ = ["DatasetsIterator", "TorchDataLoader"]


def _collate_first(batch: list) -> Any:
    return batch[0]


@runtime_checkable
class _InfiniteDataset(Protocol):
    """Dataset that can report whether its iteration ends."""

    def is_infinite(self) -> bool:
        """
        :return: ``True`` when iteration does not stop on its own
        """


def dataset_is_infinite(dataset: DatasetType) -> bool:
    """
    Whether one deserialized dataset iterates without end.

    Sources that do not implement ``is_infinite`` are finite.

    :param dataset: Dataset produced by a deserializer
    :return: ``True`` when the dataset reports that it does not end
    """
    return isinstance(dataset, _InfiniteDataset) and dataset.is_infinite()


def datasets_are_infinite(datasets: list[object], samples: int) -> bool:
    """
    Whether a loader over these datasets would iterate without end.

    A positive ``samples`` cap makes the loader finite. Otherwise it is
    infinite only when every dataset reports that it does not end.

    :param datasets: Datasets produced by deserializers
    :param samples: Loader sample cap; ``0`` or negative means no cap
    :return: ``True`` when iteration does not stop on its own
    """
    if samples > 0:
        return False
    return bool(datasets) and all(dataset_is_infinite(item) for item in datasets)


DataT = TypeVar("DataT")


class DatasetsIterator(TorchIterableDataset[DataT]):
    def __init__(
        self,
        datasets: list[DatasetType],
        data_samples: int,
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        finalizer: DatasetFinalizer[DataT],
    ):
        self.datasets = datasets
        self.preprocessors = preprocessors
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, DataDependentPreprocessor):
                preprocessor.setup_data(
                    datasets=self.datasets,
                )
        self.finalizer = finalizer
        self.precache: list[Any] | None = (
            list(self.generator(data_samples)) if data_samples else None
        )
        self.epoch = 0

    def __iter__(self) -> Iterator[DataT]:
        worker_info = torch.utils.data.get_worker_info()
        worker_modulus = worker_info.num_workers if worker_info is not None else 1
        worker_index = worker_info.id if worker_info is not None else 0

        if self.precache:
            for index, item in enumerate(self.precache):
                if (index + worker_index) % worker_modulus == 0:
                    yield item
        else:
            yield from self.generator(
                modulus=worker_modulus, offset=worker_index, epoch=self.epoch
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generator(  # noqa: C901
        self,
        max_items: int | None = None,
        modulus: int | None = None,
        offset: int | None = None,
        epoch: int = 0,
    ) -> Iterator[DataT]:
        gen_count = 0
        yield_count = 0
        error_count = 0

        with contextlib.suppress(StopIteration):
            dataset_iters = []
            for dataset in self.datasets:
                if hasattr(dataset, "set_epoch"):
                    with contextlib.suppress(Exception):
                        dataset.set_epoch(epoch)
                dataset_iters.append(iter(dataset))

            while max_items is None or gen_count < max_items:
                try:
                    row: list[dict[str, Any]] = [
                        {"dataset": next(dataset_iter)}
                        for dataset_iter in dataset_iters
                    ]
                    gen_count += 1

                    if (
                        modulus is not None
                        and offset is not None
                        and (gen_count % modulus) != offset
                    ):
                        continue

                    # Apply preprocessors in sequence
                    for preprocessor in self.preprocessors:
                        row = preprocessor(row)

                    result = self.finalizer(row)
                    # Filter empty results (e.g. column mapper matched
                    # no columns, so finalizer returned an empty list)
                    if not result:
                        continue
                    yield result
                    yield_count += 1
                except StopIteration:
                    raise  # Stop iteration when any dataset is exhausted
                except Exception as err:  # noqa: BLE001 # Exception logged
                    error_count += 1
                    logger.error(
                        "Skipping data row due to error: {}. "
                        "Check data format and preprocessor configuration.",
                        err,
                    )
                    gen_count -= 1

        if gen_count > 0 and yield_count == 0:
            raise ValueError(
                f"Dataset iterator processed {gen_count} rows but yielded "
                f"zero results ({error_count} errors; {gen_count - error_count} "
                f"empty). Check your data and data arguments."
            )

        if max_items is not None and gen_count < max_items:
            raise ValueError(
                f"Requested {max_items} samples, but only {gen_count} "
                "available from the provided datasets."
            )


@DataLoaderRegistry.register("pytorch")
class TorchDataLoader(PyTorchDataLoader[DataT], InfoMixin, DataLoader[DataT]):
    def __init__(
        self,
        config: TorchDataLoaderArgs,
        datasets: list[DatasetType],
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        finalizer: DatasetFinalizer[DataT],
        random_seed: int = 42,
        **kwargs: Any,
    ):
        self._samples = config.samples
        iterator: DatasetsIterator[DataT] = DatasetsIterator(
            datasets=datasets,
            data_samples=config.samples,
            preprocessors=preprocessors,
            finalizer=finalizer,
        )
        self._info: dict[str, Any] = config.model_dump(mode="json")
        self.epoch = 0

        gen = torch.Generator()
        gen.manual_seed(random_seed)
        super().__init__(
            dataset=iterator,
            batch_size=1,
            shuffle=config.shuffle,
            collate_fn=_collate_first,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            generator=gen,
            **kwargs,
        )

    def __iter__(self):
        if isinstance(self.dataset, DatasetsIterator):
            self.dataset.set_epoch(self.epoch)
        self.epoch += 1

        return super().__iter__()

    def is_infinite(self) -> bool:
        """
        Whether prefetching every request would run without end.

        A positive sample cap ends generation. Otherwise the loader is infinite
        only when every dataset reports that it does not end.

        :return: ``True`` when iteration does not stop on its own
        """
        if self._samples > 0:
            return False
        datasets = (
            self.dataset.datasets if isinstance(self.dataset, DatasetsIterator) else []
        )
        return datasets_are_infinite(datasets, samples=self._samples)

    @property
    def info(self) -> dict[str, Any]:
        return self._info
