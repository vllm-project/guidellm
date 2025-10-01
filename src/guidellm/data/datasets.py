from __future__ import annotations

from collections.abc import Callable
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.formatters import GenerativeRequestFormatter
from guidellm.data.objects import GenerativeDatasetArgs
from guidellm.data.preprocessors import (
    DatasetPreprocessor,
    GenerativeColumnMapper,
)
from guidellm.data.utils import datasets_item_iterator, resolve_dataset_split

__all__ = ["GenerativeRequestsDataset"]


class GenerativeRequestsDataset:
    @classmethod
    def build(
        cls,
        data: list[Any],
        data_args: list[GenerativeDatasetArgs] | None,
        data_samples: int,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        column_mapper: GenerativeColumnMapper,
        preprocessors: list[DatasetPreprocessor],
        request_formatter: GenerativeRequestFormatter,
        random_seed: int = 42,
    ) -> Dataset | IterableDataset:
        if not data or not isinstance(data, list):
            raise ValueError(f"Data must be a non-empty list, got {data}.")

        if data_args is None:
            data_args = [GenerativeDatasetArgs() for _ in data]

        if len(data) != len(data_args):
            raise ValueError(
                f"Length of data ({len(data)}) must match length of data_args "
                f"({len(data_args)})."
            )

        datasets = []
        for datum, args in zip(data, data_args):
            datasets.append(
                resolve_dataset_split(
                    dataset=DatasetDeserializerFactory.deserialize(
                        data=datum,
                        data_kwargs=args.to_kwargs(),
                        processor_factory=processor_factory,
                        random_seed=random_seed,
                        type_=args.type_,
                    ),
                    split=args.split,
                )
            )

        column_mapper.init_data(datasets=datasets, data_args=data_args)
        request_formatter.init_data(datasets=datasets, data_args=data_args)
        for preprocessor in preprocessors:
            preprocessor.init_data(datasets=datasets, data_args=data_args)

        if data_samples > 0:
            dataset = Dataset.from_list(
                list(
                    datasets_item_iterator(
                        datasets=datasets,
                        data_samples=data_samples,
                    )
                )
            )
        else:
            dataset = IterableDataset.from_generator(
                datasets_item_iterator,
                gen_kwargs={
                    "datasets": datasets,
                    "data_samples": data_samples,
                },
            )

        dataset = dataset.map(column_mapper)
        for preprocessor in preprocessors:
            dataset = dataset.map(preprocessor)

        return dataset.map(request_formatter)
