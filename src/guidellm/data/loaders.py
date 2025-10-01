from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizerBase

from guidellm.data.datasets import GenerativeRequestsDataset
from guidellm.data.formatters import GenerativeRequestFormatter
from guidellm.data.objects import GenerationRequest, GenerativeDatasetArgs
from guidellm.data.preprocessors import (
    DatasetPreprocessor,
    GenerativeColumnMapper,
)

__all__ = ["GenerativeDataLoader", "GenerativeRequestCollator"]


class GenerativeRequestCollator:
    def __call__(
        self, batch: list[dict[Literal["request"], dict[str, Any]]]
    ) -> GenerationRequest:
        if len(batch) != 1:
            raise NotImplementedError(
                f"Batch size greater than 1 is not currently supported. "
                f"Got batch size: {len(batch)}"
            )

        return GenerationRequest.model_validate(batch[0]["request"])


class GenerativeDataLoader(DataLoader[GenerationRequest]):
    def __init__(
        self,
        data: list[Any],
        data_args: list[GenerativeDatasetArgs] | None,
        data_samples: int,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        column_mapper: GenerativeColumnMapper,
        preprocessors: list[DatasetPreprocessor],
        request_formatter: GenerativeRequestFormatter,
        sampler: Sampler[int] | Literal["shuffle"] | None = None,
        collate_fn: GenerativeRequestCollator | None = None,
        num_workers: int | None = None,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        dataset = GenerativeRequestsDataset.build(
            data=data,
            data_args=data_args,
            data_samples=data_samples,
            processor_factory=processor_factory,
            column_mapper=column_mapper,
            request_formatter=request_formatter,
            preprocessors=preprocessors,
            random_seed=random_seed,
        )

        if collate_fn is None:
            collate_fn = GenerativeRequestCollator()

        # Handle sampler/shuffle logic based on dataset type
        if sampler == "shuffle":
            shuffle = True
            sampler = None
        elif isinstance(sampler, str) and sampler != "shuffle":
            raise ValueError(
                f"Invalid string sampler: {sampler}. "
                f"Only 'shuffle' is supported as a string value."
            )
        else:
            shuffle = False

        if isinstance(dataset, IterableDataset) and sampler is not None:
            raise ValueError(
                "Samplers are not supported with IterableDataset. "
                "Use shuffle=True or apply shuffling to the dataset directly."
            )
        elif isinstance(dataset, Dataset) and shuffle:
            dataset = dataset.shuffle(seed=random_seed)
            shuffle = False

        super().__init__(
            dataset=dataset,
            batch_size=1,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers or 0,
            **kwargs,
        )
