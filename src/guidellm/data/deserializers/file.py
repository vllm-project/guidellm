from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, IterableDataset, load_dataset
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.data.utils import resolve_dataset_split

__all__ = [
    "ArrowFileDatasetDeserializer",
    "CSVFileDatasetDeserializer",
    "DBFileDatasetDeserializer",
    "FileDataArgs",
    "HDF5FileDatasetDeserializer",
    "JSONFileDatasetDeserializer",
    "ParquetFileDatasetDeserializer",
    "TarFileDatasetDeserializer",
    "TextFileDatasetDeserializer",
]


def _load_file_dataset(
    loader: str, path: Path, **load_kwargs: Any
) -> Dataset | IterableDataset:
    """
    Load a local data file and return a single dataset split.

    ``load_dataset`` returns a ``DatasetDict`` keyed by split for file-based
    loaders, which downstream preprocessing does not expect. This resolves the
    result to a single ``Dataset`` (or ``IterableDataset``) split.

    :param loader: The ``datasets`` loader name (e.g. ``"json"``, ``"csv"``).
    :param path: Path to the local data file.
    :param load_kwargs: Additional keyword arguments forwarded to ``load_dataset``.
    :return: The resolved dataset split.
    """
    dataset = load_dataset(loader, data_files=str(path), **load_kwargs)
    return resolve_dataset_split(dataset)


@DataArgs.register(
    [
        "text_file",
        "csv_file",
        "json_file",
        "parquet_file",
        "arrow_file",
        "hdf5_file",
        "db_file",
        "tar_file",
    ]
)
class FileDataArgs(DataArgs):
    kind: Literal[  # type: ignore[assignment]
        "text_file",
        "csv_file",
        "json_file",
        "parquet_file",
        "arrow_file",
        "hdf5_file",
        "db_file",
        "tar_file",
    ] = Field(
        default="text_file",
        description="Type identifier for the data arguments configuration.",
    )
    path: Path = Field(
        description="Path to the data file.",
        examples=["data.txt"],
    )


@DatasetDeserializerFactory.register("text_file")
class TextFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".txt", ".text"}
        ):
            raise DataNotSupportedError(
                "Unsupported data for TextFileDatasetDeserializer, "
                f"expected str or Path to a local .txt or .text file, got {path}"
            )

        with path.open() as file:
            lines = file.readlines()

        return Dataset.from_dict({"text": lines}, **config.load_kwargs)


@DatasetDeserializerFactory.register("csv_file")
class CSVFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset:
        _ = (processor_factory, random_seed)

        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() != ".csv"
        ):
            raise DataNotSupportedError(
                "Unsupported data for CSVFileDatasetDeserializer, "
                f"expected str or Path to a valid local .csv file, got {path}"
            )

        return _load_file_dataset("csv", path, **config.load_kwargs)


@DatasetDeserializerFactory.register("json_file")
class JSONFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".json", ".jsonl"}
        ):
            raise DataNotSupportedError(
                f"Unsupported data for JSONFileDatasetDeserializer, "
                f"expected str or Path to a local .json or .jsonl file, got {path}"
            )

        return _load_file_dataset("json", path, **config.load_kwargs)


@DatasetDeserializerFactory.register("parquet_file")
class ParquetFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() != ".parquet"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for ParquetFileDatasetDeserializer, "
                f"expected str or Path to a local .parquet file, got {path}"
            )

        return _load_file_dataset("parquet", path, **config.load_kwargs)


@DatasetDeserializerFactory.register("arrow_file")
class ArrowFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() != ".arrow"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for ArrowFileDatasetDeserializer, "
                f"expected str or Path to a local .arrow file, got {path}"
            )

        return _load_file_dataset("arrow", path, **config.load_kwargs)


@DatasetDeserializerFactory.register("hdf5_file")
class HDF5FileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".hdf5", ".h5"}
        ):
            raise DataNotSupportedError(
                f"Unsupported data for HDF5FileDatasetDeserializer, "
                f"expected str or Path to a local .hdf5 or .h5 file, got {path}"
            )

        return Dataset.from_pandas(pd.read_hdf(str(path)), **config.load_kwargs)


@DatasetDeserializerFactory.register("db_file")
class DBFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> dict[str, list]:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() != ".db"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for DBFileDatasetDeserializer, "
                f"expected str or Path to a local .db file, got {path}"
            )

        return Dataset.from_sql(con=str(path), **config.load_kwargs)


@DatasetDeserializerFactory.register("tar_file")
class TarFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: FileDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset:
        _ = (processor_factory, random_seed)
        if (
            not (path := config.path).exists()
            or not path.is_file()
            or path.suffix.lower() != ".tar"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for TarFileDatasetDeserializer, "
                f"expected str or Path to a local .tar file, got {path}"
            )

        return _load_file_dataset("webdataset", path, **config.load_kwargs)
