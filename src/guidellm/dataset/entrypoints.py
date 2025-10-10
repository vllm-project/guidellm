from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

from guidellm.dataset.creator import ColumnInputTypes
from guidellm.dataset.file import FileDatasetCreator
from guidellm.dataset.hf_datasets import HFDatasetsCreator
from guidellm.dataset.in_memory import InMemoryDatasetCreator
from guidellm.dataset.synthetic import SyntheticDatasetCreator

__all__ = ["load_dataset"]


def load_dataset(
    data: Any,
    data_args: dict[str, Any] | None,
    processor: str | Path | PreTrainedTokenizerBase | None,
    processor_args: dict[str, Any] | None,
    random_seed: int = 42,
    split_pref_order: list[str] | None = None,
) -> tuple[Dataset | IterableDataset, dict[ColumnInputTypes, str]]:
    creators = [
        InMemoryDatasetCreator,
        SyntheticDatasetCreator,
        FileDatasetCreator,
        HFDatasetsCreator,
    ]

    for creator in creators:
        if creator.is_supported(data, data_args):
            return creator.create(
                data,
                data_args,
                processor,
                processor_args,
                random_seed,
                split_pref_order,
            )

    raise ValueError(f"Unsupported data type: {type(data)} given for {data}. ")
