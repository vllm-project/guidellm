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

__all__ = ["DataLoader", "datasets_item_iterator"]


def datasets_item_iterator(
    datasets: list[Dataset | IterableDataset],
    data_samples: int,
    preprocessors: tuple[DatasetPreprocessor | DataDependentPreprocessor],
) -> Iterator[Any]:
    gen_count = 0
    dataset_iters = [iter(dataset) for dataset in datasets]

    with contextlib.suppress(StopIteration):
        while gen_count < data_samples or data_samples == math.inf:
            row = {"items": [next(dataset_iter) for dataset_iter in dataset_iters]}
            for preprocessor in preprocessors:
                row = preprocessor(row)
            yield row
            gen_count += 1

    if data_samples != math.inf and gen_count < data_samples:
        raise ValueError(
            f"Requested {data_samples} samples, but only {gen_count} "
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
            type_ = data_kwargs.pop("type_") if "type_" in data_kwargs else None
            datasets.append(
                DatasetDeserializerFactory.deserialize(
                    data=datum,
                    data_kwargs=data_args,
                    processor_factory=processor_factory,
                    random_seed=random_seed,
                    type_=type_,
                    **data_kwargs,
                )
            )
        for preprocessor in preprocessors:
            if isinstance(preprocessor, DataDependentPreprocessor):
                preprocessor.setup_data(
                    datasets=datasets,
                    data_args=data_args,
                )
        if data_samples != math.inf and data_samples > 0:
            cached_samples = list(
                datasets_item_iterator(datasets, data_samples, tuple(preprocessors))
            )
            dataset = IterableDataset.from_generator(lambda: cached_samples)
        else:
            dataset = IterableDataset.from_generator(
                datasets_item_iterator,
                gen_kwargs={
                    "datasets": datasets,
                    "data_samples": math.inf,
                    "preprocessors": tuple(preprocessors),
                },
            )

        super().__init__(
            dataset=dataset,
            batch_size=1,
            shuffle=sampler == "shuffle",
            sampler=sampler if sampler != "shuffle" else None,
            collate_fn=collator,
            num_workers=num_workers,
            **kwargs,
        )
