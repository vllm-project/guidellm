from __future__ import annotations

from pathlib import Path

from guidellm.schemas.data import FileDataArgs


def trace_file_source(data: str | Path) -> FileDataArgs:
    path = Path(data)
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        kind = "json_file"
    elif suffix == ".csv":
        kind = "csv_file"
    elif suffix == ".parquet":
        kind = "parquet_file"
    else:
        kind = "json_file"
    return FileDataArgs(kind=kind, path=path)
