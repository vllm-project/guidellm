from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd  # type: ignore[import]
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

from guidellm.dataset.creator import DatasetCreator

__all__ = ["FileDatasetCreator"]


class FileDatasetCreator(DatasetCreator):
    SUPPORTED_TYPES = {
        ".txt",
        ".text",
        ".csv",
        ".json",
        ".jsonl",
        ".parquet",
        ".arrow",
        ".hdf5",
        ".tar",
    }

    @classmethod
    def is_supported(cls, data: Any, data_args: Optional[dict[str, Any]]) -> bool:  # noqa: ARG003
        if isinstance(data, str | Path) and (path := Path(data)).exists():
            # local folder or py file, assume supported
            return path.suffix.lower() in cls.SUPPORTED_TYPES

        return False

    @classmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],  # noqa: ARG003
        processor_args: Optional[dict[str, Any]],  # noqa: ARG003
        random_seed: int,  # noqa: ARG003
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        if not isinstance(data, str | Path):
            raise ValueError(f"Unsupported data type: {type(data)} given for {data}. ")

        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Unsupported data type: {path} given for {path}. ")

        if path.suffix.lower() not in cls.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {path.suffix} given for {path}. ")

        return cls.load_dataset(path, data_args)

    @classmethod
    def load_dataset(
        cls, path: Path, data_args: Optional[dict[str, Any]]
    ) -> Union[Dataset, IterableDataset]:
        if path.suffix.lower() in {".txt", ".text"}:
            with path.open("r") as file:
                items = file.readlines()

            dataset = Dataset.from_dict({"text": items}, **(data_args or {}))
        elif path.suffix.lower() == ".csv":
            dataset = load_dataset("csv", data_files=str(path), **(data_args or {}))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files=str(path), **(data_args or {}))
        elif path.suffix.lower() == ".parquet":
            dataset = load_dataset("parquet", data_files=str(path), **(data_args or {}))
        elif path.suffix.lower() == ".arrow":
            dataset = load_dataset("arrow", data_files=str(path), **(data_args or {}))
        elif path.suffix.lower() == ".hdf5":
            dataset = Dataset.from_pandas(pd.read_hdf(str(path)), **(data_args or {}))
        elif path.suffix.lower() == ".db":
            dataset = Dataset.from_sql(con=str(path), **(data_args or {}))
        elif path.suffix.lower() == ".tar":
            dataset = load_dataset(
                "webdataset", data_files=str(path), **(data_args or {})
            )
        else:
            raise ValueError(f"Unsupported file type: {path.suffix} given for {path}. ")

        return dataset
