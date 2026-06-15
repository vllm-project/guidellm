"""
Shared trace file I/O for replay benchmarks.

Reads trace files (.jsonl only for now) and exposes rows or relative timestamps.
Used by replay profiles and the trace_synthetic deserializer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from datasets import Dataset, Value, load_dataset

from guidellm.data.deserializers.deserializer import DataNotSupportedError

__all__ = ["load_relative_timestamps", "load_trace_rows"]


@dataclasses.dataclass
class TraceColumn:
    """Holds metadata for trace file columns in a HuggingFace Dataset."""

    name: str
    feature_type: Value


def get_column_names(columns: list[TraceColumn]) -> list[str]:
    return [c.name for c in columns]


def validate_trace_path(path: Path | str) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix != ".jsonl":
        raise ValueError(f"Unsupported trace file format: {suffix}")
    if path.stat().st_size == 0:
        raise ValueError(f"Trace file is empty or has no valid rows: {path}")
    return path


def check_and_raise_missing_columns(
    required_columns: list[str], actual_columns: list[str]
) -> None:
    missing = [c for c in required_columns if c not in actual_columns]
    if missing:
        raise KeyError(f"Trace row missing required columns: {missing}")


def load_trace_rows(
    path: Path | str,
    timestamp_column: TraceColumn,
    required_columns: list[TraceColumn] | None = None,
    **data_kwargs: Any,
) -> Dataset:
    """
    Load trace file rows as a HuggingFace Dataset.

    Supports .jsonl only (one JSON object per line).
    If required_columns is set, every column must exist in the dataset;
    otherwise KeyError is raised with a descriptive message.
    Rows are sorted by timestamp_column.

    :param path: Path to the trace file.
    :param timestamp_column: Timestamp column used to sort trace rows.
    :param required_columns: Optional list of column/fields that each row must have.
    :param data_kwargs: Additional keyword arguments forwarded to load_dataset.
    :return: HuggingFace Dataset (iterable as dicts, column-accessible).
    :raises DataNotSupportedError: For any of the following reasons:
    - The dataset is empty or has no valid rows
    - A required column or timestamp_column contains a NoneType
    - A required column or timestamp_column failed during cast to feature type

    :raises KeyError: If a required column is missing in the dataset.
    :raises ValueError: If the file format is not .jsonl.
    """
    if required_columns is None:
        required_columns = []
    if timestamp_column not in required_columns:
        required_columns = [*required_columns, timestamp_column]
    path = validate_trace_path(path)
    trace_dataset = load_dataset(
        "json", data_files=str(path), split="train", **data_kwargs
    )
    if required_columns:
        check_and_raise_missing_columns(
            get_column_names(required_columns), trace_dataset.column_names
        )

    if not trace_dataset:
        raise DataNotSupportedError(f"Trace file is empty or has no valid rows: {path}")
    for col in required_columns:
        if trace_dataset.data[col.name].null_count != 0:
            raise DataNotSupportedError(f"NoneType found in {col}")
        try:
            trace_dataset.cast_column(col.name, col.feature_type)
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e

    return trace_dataset.sort(timestamp_column.name)


def load_relative_timestamps(
    path: Path | str,
    timestamp_column: str = "timestamp",
) -> list[float]:
    """
    Load timestamps from a trace file and return times relative to the first event.

    Trace file must be JSONL (one JSON object per line). The first timestamp
    becomes 0.0, and all others are relative to it (always >= 0).

    :param path: Path to the trace file.
    :param timestamp_column: Name of the column/field containing the timestamp.
    :return: List of relative timestamps in seconds (first is 0.0).
    """
    trace_dataset = load_trace_rows(
        path,
        TraceColumn(name=timestamp_column, feature_type=Value("float")),
    )
    timestamps = [float(t) for t in trace_dataset[timestamp_column]]
    t0 = timestamps[0]
    return [t - t0 for t in timestamps]
