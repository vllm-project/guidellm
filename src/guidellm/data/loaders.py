from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import Any, Literal

from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader as PyTorchDataLoader
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.objects import GenerationRequest
from guidellm.data.preprocessors import DataDependentPreprocessor, DatasetPreprocessor

__all__ = ["DataIterator", "DataLoader"]


class DataIterator:
    def __init__(
        self,
        datasets: list[Dataset | IterableDataset],
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        precache_size: int | None = None,
    ):
        self.datasets = datasets
        self.preprocessors = preprocessors
        self.precache = (
            None if not precache_size else list(self.generator(precache_size))
        )

    def __iter__(self):
        if self.precache is not None:
            yield from self.precache
        else:
            yield from self.generator()

    def generator(self, max_items: int | None = None) -> Iterator[Any]:
        gen_count = 0

        with contextlib.suppress(StopIteration):
            dataset_iters = [iter(dataset) for dataset in self.datasets]

            while max_items is None or gen_count < max_items:
                row = {"items": [next(dataset_iter) for dataset_iter in dataset_iters]}
                for preprocessor in self.preprocessors:
                    row = preprocessor(row)
                yield row
                gen_count += 1

        if max_items is not None and gen_count < max_items:
            raise ValueError(
                f"Requested {max_items} samples, but only {gen_count} "
                "available from the provided datasets."
            )


class DataLoader(PyTorchDataLoader[GenerationRequest]):
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
        if not data or not isinstance(data, list):
            raise ValueError(f"Data must be a non-empty list, got {data}.")

        if data_args is None:
            data_args = [{} for _ in data]

        if len(data) != len(data_args):
            raise ValueError(
                f"Length of data ({len(data)}) must match length of data_args "
                f"({len(data_args)})."
            )

        datasets = []
        for datum, data_kwargs in zip(data, data_args):
            datasets.append(
                DatasetDeserializerFactory.deserialize(
                    data=datum,
                    processor_factory=processor_factory,
                    random_seed=random_seed,
                    **data_kwargs,
                )
            )
        for preprocessor in preprocessors:
            if isinstance(preprocessor, DataDependentPreprocessor):
                preprocessor.setup_data(
                    datasets=datasets,
                    data_args=data_args,
                )

        data_iterator = DataIterator(
            datasets=datasets,
            preprocessors=preprocessors,
            precache_size=data_samples
            if data_samples != math.inf and data_samples > 0
            else None,
        )
        dataset = IterableDataset.from_generator(data_iterator.__iter__)

        super().__init__(
            dataset=dataset,
            batch_size=1,
            shuffle=sampler == "shuffle",
            sampler=sampler if sampler != "shuffle" else None,
            collate_fn=collator,
            num_workers=num_workers,
            **kwargs,
        )
