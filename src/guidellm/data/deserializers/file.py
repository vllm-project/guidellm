from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)

__all__ = [
    "ArrowFileDatasetDeserializer",
    "CSVFileDatasetDeserializer",
    "DBFileDatasetDeserializer",
    "HDF5FileDatasetDeserializer",
    "JSONFileDatasetDeserializer",
    "ParquetFileDatasetDeserializer",
    "TarFileDatasetDeserializer",
    "TextFileDatasetDeserializer",
]


@DatasetDeserializerFactory.register("text_file")
class TextFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)  # Ignore unused args format errors

        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".txt", ".text"}
        ):
            raise DataNotSupportedError(
                "Unsupported data for TextFileDatasetDeserializer, "
                f"expected str or Path to a local .txt or .text file, got {data}"
            )

        with path.open() as file:
            lines = file.readlines()

        return Dataset.from_dict({"text": lines}, **data_kwargs)


@DatasetDeserializerFactory.register("csv_file")
class CSVFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)

        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() != ".csv"
        ):
            raise DataNotSupportedError(
                "Unsupported data for CSVFileDatasetDeserializer, "
                f"expected str or Path to a valid local .csv file, got {data}"
            )

        return load_dataset("csv", data_files=str(path), **data_kwargs)


@DatasetDeserializerFactory.register("json_file")
class JSONFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".json", ".jsonl"}
        ):
            raise DataNotSupportedError(
                f"Unsupported data for JSONFileDatasetDeserializer, "
                f"expected str or Path to a local .json or .jsonl file, got {data}"
            )

        return load_dataset("json", data_files=str(path), **data_kwargs)


@DatasetDeserializerFactory.register("parquet_file")
class ParquetFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() != ".parquet"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for ParquetFileDatasetDeserializer, "
                f"expected str or Path to a local .parquet file, got {data}"
            )

        return load_dataset("parquet", data_files=str(path), **data_kwargs)


@DatasetDeserializerFactory.register("arrow_file")
class ArrowFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() != ".arrow"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for ArrowFileDatasetDeserializer, "
                f"expected str or Path to a local .arrow file, got {data}"
            )

        return load_dataset("arrow", data_files=str(path), **data_kwargs)


@DatasetDeserializerFactory.register("hdf5_file")
class HDF5FileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() not in {".hdf5", ".h5"}
        ):
            raise DataNotSupportedError(
                f"Unsupported data for HDF5FileDatasetDeserializer, "
                f"expected str or Path to a local .hdf5 or .h5 file, got {data}"
            )

        return Dataset.from_pandas(pd.read_hdf(str(path)), **data_kwargs)


@DatasetDeserializerFactory.register("db_file")
class DBFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> dict[str, list]:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() != ".db"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for DBFileDatasetDeserializer, "
                f"expected str or Path to a local .db file, got {data}"
            )

        return Dataset.from_sql(con=str(path), **data_kwargs)


@DatasetDeserializerFactory.register("tar_file")
class TarFileDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> dict[str, list]:
        _ = (processor_factory, random_seed)
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
            or path.suffix.lower() != ".tar"
        ):
            raise DataNotSupportedError(
                f"Unsupported data for TarFileDatasetDeserializer, "
                f"expected str or Path to a local .tar file, got {data}"
            )

        return load_dataset("webdataset", data_files=str(path), **data_kwargs)
