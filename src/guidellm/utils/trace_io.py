"""
Shared trace file I/O for replay benchmarks.

Reads trace files (.jsonl only for now) and exposes raw rows or relative timestamps.
Used by the scheduler (load_relative_timestamps) and the trace_synthetic deserializer
(load_trace_rows with token columns).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_relative_timestamps", "load_trace_rows"]


def load_trace_rows(
    path: Path | str,
    required_columns: list[str] | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load trace file rows as a list of dicts.

    Supports .jsonl only (one JSON object per line).
    If required_columns is set, every row must contain these keys; otherwise
    KeyError is raised with a descriptive message.
    If max_rows is set, only the first max_rows rows are loaded (for replay
    with a request limit).

    :param path: Path to the trace file.
    :param required_columns: Optional list of column/field names that each row
        must have.
    :param max_rows: Optional maximum number of rows to load; None means load all.
        If set to a value less than 1, returns an empty list.
    :return: List of row dicts (keys and values as in the file).
    :raises KeyError: If a required column is missing in the file or in a row.
    :raises ValueError: If the file format is not .jsonl.
    """
    path = Path(path)
    if max_rows is not None and max_rows < 1:
        return []
    suffix = path.suffix.lower()
    if suffix != ".jsonl":
        raise ValueError(f"Unsupported trace file format: {suffix}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if required_columns:
                missing = [c for c in required_columns if c not in row]
                if missing:
                    raise KeyError(f"Trace row missing required columns: {missing}")
            rows.append(row)

    return rows


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
    raw = load_trace_rows(
        path,
        required_columns=[timestamp_column],
    )
    timestamps = sorted([float(row[timestamp_column]) for row in raw])
    if not timestamps:
        raise ValueError(f"Trace file has no valid rows: {path}")
    t0 = timestamps[0]
    return [t - t0 for t in timestamps]
