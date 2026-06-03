from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal, TypeVar

import torch
from pydantic import Field
from torch.utils.data.dataloader import DataLoader as PyTorchDataLoader
from torch.utils.data.dataset import IterableDataset as TorchIterableDataset

from guidellm.data.finalizers import DatasetFinalizer
from guidellm.data.loaders.loader import DataLoader, DataLoaderRegistry
from guidellm.data.preprocessors import (
    DataDependentPreprocessor,
    DatasetPreprocessor,
)
from guidellm.data.schemas import (
    DataLoaderArgs,
    DatasetType,
)
from guidellm.logger import logger
from guidellm.utils.mixins import InfoMixin

__all__ = ["DatasetsIterator", "TorchDataLoader", "TorchDataLoaderArgs"]


@DataLoaderArgs.register("pytorch")
class TorchDataLoaderArgs(DataLoaderArgs):
    kind: Literal["pytorch"] = Field(  # type: ignore[assignment]
        default="pytorch",
        description="Type identifier for the generative data loader.",
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle data rows at every epoch.",
    )
    num_workers: int = Field(
        default=1,
        description=(
            "Number of worker processes for data loading. If 0, data loading "
            "will be performed in the main process."
        ),
    )


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
        iterator: DatasetsIterator[DataT] = DatasetsIterator(
            datasets=datasets,
            data_samples=config.samples,
            preprocessors=preprocessors,
            finalizer=finalizer,
        )
        self._info: dict[str, Any] = config.model_dump()
        self.epoch = 0

        gen = torch.Generator()
        gen.manual_seed(random_seed)
        super().__init__(
            dataset=iterator,
            batch_size=1,
            shuffle=config.shuffle,
            collate_fn=lambda batch: batch[0],
            num_workers=config.num_workers,
            generator=gen,
            **kwargs,
        )

    def __iter__(self):
        if isinstance(self.dataset, DatasetsIterator):
            self.dataset.set_epoch(self.epoch)
        self.epoch += 1

        return super().__iter__()

    @property
    def info(self) -> dict[str, Any]:
        return self._info
