from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader as PyTorchDataLoader
from torch.utils.data.dataset import IterableDataset as TorchIterableDataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.preprocessors import DataDependentPreprocessor, DatasetPreprocessor
from guidellm.logger import logger

__all__ = ["DataLoader", "DatasetsIterator"]



class DatasetsIterator(TorchIterableDataset):
    def __init__(
        self,
        data: list[Any],
        data_args: list[dict[str, Any]] | None,
        data_samples: int,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        random_seed: int,
    ):
        if not data or not isinstance(data, list):
            raise ValueError(f"Data must be a non-empty list, got {data}.")

        if not data_args:
            data_args = [{} for _ in data]

        if len(data) != len(data_args):
            raise ValueError(
                f"Length of data ({len(data)}) must match length of data_args "
                f"({len(data_args)})."
            )

        self.datasets = []
        for datum, data_kwargs in zip(data, data_args, strict=False):
            self.datasets.append(
                DatasetDeserializerFactory.deserialize(
                    data=datum,
                    processor_factory=processor_factory,
                    random_seed=random_seed,
                    **data_kwargs,
                )
            )
        self.preprocessors = preprocessors
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, DataDependentPreprocessor):
                preprocessor.setup_data(
                    datasets=self.datasets,
                    data_args=data_args,
                )
        self.precache: list[Any] | None = (
            list(self.generator(data_samples)) if data_samples else None
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_modulus = worker_info.num_workers if worker_info is not None else 1
        worker_index = worker_info.id if worker_info is not None else 0

        if self.precache:
            for index, item in enumerate(self.precache):
                if (index + worker_index) % worker_modulus == 0:
                    yield item
        else:
            yield from self.generator(modulus=worker_modulus, offset=worker_index)

    def generator(
        self,
        max_items: int | None = None,
        modulus: int | None = None,
        offset: int | None = None,
    ) -> Iterator[Any]:
        gen_count = 0

        with contextlib.suppress(StopIteration):
            dataset_iters = [iter(dataset) for dataset in self.datasets]

            while max_items is None or gen_count < max_items:
                try:
                    row: dict[str, Any] = {
                        "items": [next(dataset_iter) for dataset_iter in dataset_iters]
                    }
                    gen_count += 1

                    if (
                        modulus is not None
                        and offset is not None
                        and (gen_count % modulus) != offset
                    ):
                        continue

                    for preprocessor in self.preprocessors:
                        # This can assign a GenerationRequest, which would then be
                        # passed into the preprocessor, which is a type violation.
                        # This should be fixed at some point.
                        row = preprocessor(row)  # type: ignore[assignment]
                    yield row
                except Exception as err:  # noqa: BLE001 # Exception logged
                    logger.error(f"Skipping data row due to error: {err}")
                    gen_count -= 1

        if max_items is not None and gen_count < max_items:
            raise ValueError(
                f"Requested {max_items} samples, but only {gen_count} "
                "available from the provided datasets."
            )


class DataLoader(PyTorchDataLoader):
    def __init__(
        self,
        data: list[Any],
        data_args: list[dict[str, Any]] | None,
        data_samples: int,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        collator: Callable,
        sampler: Sampler[int] | Literal["shuffle"] | None = None,
        num_workers: int | None = 1,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        iterator = DatasetsIterator(
            data=data,
            data_args=data_args,
            data_samples=data_samples,
            processor_factory=processor_factory,
            preprocessors=preprocessors,
            random_seed=random_seed,
        )

        super().__init__(
            dataset=iterator,
            batch_size=1,
            shuffle=sampler == "shuffle",
            sampler=sampler if sampler != "shuffle" else None,
            collate_fn=collator,
            num_workers=num_workers or 0,
            **kwargs,
        )
