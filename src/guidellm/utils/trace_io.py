"""
Shared trace file I/O for replay benchmarks.

Reads trace files (.jsonl only for now) and exposes raw rows or relative timestamps.
Used by the scheduler (load_relative_timestamps) and the trace_synthetic deserializer
(load_trace_rows with token columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

__all__ = ["load_relative_timestamps", "load_trace_rows"]


def load_trace_rows(
    path: Path | str,
    required_columns: list[str] | None = None,
    timestamp_column: str | None = None,
    **data_kwargs: Any,
) -> Dataset:
    """
    Load trace file rows as a HuggingFace Dataset.

    Supports .jsonl only (one JSON object per line).
    If required_columns is set, every column must exist in the dataset;
    otherwise KeyError is raised with a descriptive message.

    :param path: Path to the trace file.
    :param required_columns: Optional list of column/field names that each row
        must have.
    :param timestamp_column: Optional timestamp column used to order trace rows.
    :param data_kwargs: Additional keyword arguments forwarded to load_dataset.
    :return: HuggingFace Dataset (iterable as dicts, column-accessible).
    :raises KeyError: If a required column is missing in the dataset.
    :raises ValueError: If the file format is not .jsonl.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix != ".jsonl":
        raise ValueError(f"Unsupported trace file format: {suffix}")
    if path.stat().st_size == 0:
        raise ValueError(f"Trace file is empty or has no valid rows: {path}")

    trace_dataset = load_dataset(
        "json", data_files=str(path), split="train", **data_kwargs
    )

    required_columns = required_columns or []
    if timestamp_column and timestamp_column not in required_columns:
        required_columns = [*required_columns, timestamp_column]

    if required_columns:
        missing = [c for c in required_columns if c not in trace_dataset.column_names]
        if missing:
            raise KeyError(f"Trace row missing required columns: {missing}")

    if timestamp_column:
        trace_dataset = trace_dataset.sort(timestamp_column)

    return trace_dataset


def load_relative_timestamps(
    path: Path | str,
    timestamp_column: str = "timestamp",
) -> list[float]:
    """
    Load timestamps from a trace file and return times relative to the first event.

    Trace file must be JSONL (one JSON object per line). Timestamps are sorted
    chronologically before calculating relative times. The earliest timestamp
    becomes 0.0, and all others are relative to it (always >= 0).

    :param path: Path to the trace file.
    :param timestamp_column: Name of the column/field containing the timestamp.
    :return: List of relative timestamps in seconds (first is 0.0, always sorted).
    :raises ValueError: If the trace file is empty or has no valid rows.
    """
    trace_dataset = load_trace_rows(path, timestamp_column=timestamp_column)
    if len(trace_dataset) == 0:
        raise ValueError(f"Trace file is empty or has no valid rows: {path}")
    timestamps = [float(t) for t in trace_dataset[timestamp_column]]
    t0 = timestamps[0]
    return [t - t0 for t in timestamps]
