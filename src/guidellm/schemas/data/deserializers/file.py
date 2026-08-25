from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["FileDataArgs"]


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
