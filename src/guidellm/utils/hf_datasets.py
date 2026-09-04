from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from guidellm.utils.imports import json

SUPPORTED_TYPES = {
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
}

_JSON_SUFFIXES = {".json", ".jsonl"}


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON array or JSON Lines file into Python dicts.

    :param path: Path to a ``.json`` or ``.jsonl`` file.
    :return: One dict per record.
    :raises ValueError: If the file is not valid JSON or JSON Lines.
    """
    with path.open(encoding="utf-8") as handle:
        prefix = handle.read(1)
        while prefix and prefix.isspace():
            prefix = handle.read(1)
        if not prefix:
            return []
        handle.seek(0)
        if prefix == "[":
            try:
                data = json.loads(handle.read())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc
            if not isinstance(data, list):
                raise ValueError(
                    f"JSON file '{path}' must contain a list of records or JSON Lines."
                )
            return data
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of '{path}': {exc}"
                ) from exc
        return records


def load_json_dataset_from_file(path: str | Path) -> Dataset:
    """Load JSON or JSONL without HuggingFace chunked schema inference.

    The datasets json builder infers Arrow types from the first chunk.
    Later rows that add optional nested fields (for example WEKA
    ``ttft`` on only some ``requests``) then fail to cast. Parsing the
    file in Python and building the Dataset from the full record list
    lets Arrow union those keys.

    :param path: Path to a ``.json`` or ``.jsonl`` file.
    :return: Dataset with one row per JSON record.
    :raises ValueError: If the file cannot be parsed or converted.
    """
    path = Path(path)
    records = _read_json_records(path)
    try:
        return Dataset.from_list(records)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to load JSON dataset from '{path}': {exc}") from exc


def load_dataset_from_file(
    path: str | Path, split: str = "train", **data_kwargs: Any
) -> Dataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        # Bypass the HuggingFace json builder: its chunked Arrow schema
        # cannot grow when later records add optional nested fields.
        return load_json_dataset_from_file(path)
    if suffix in SUPPORTED_TYPES:
        return load_dataset(
            suffix.replace(".", ""), data_files=str(path), split=split, **data_kwargs
        )
    raise ValueError(
        f"Unsupported file suffix '{suffix}' in path '{path}'."
        f" Only {SUPPORTED_TYPES} are supported."
    )


def save_dataset_to_file(dataset: Dataset, output_path: str | Path) -> None:
    """
    Saves a HuggingFace Dataset to file in a supported format.

    :param dataset: Dataset to save.
    :param output_path: Output file path (.json, .jsonl, .csv, .parquet).
    :raises ValueError: If the file extension is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        dataset.to_csv(output_path)
    elif suffix in {".json", ".jsonl"}:
        dataset.to_json(output_path)
    elif suffix == ".parquet":
        dataset.to_parquet(output_path)
    else:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path '{output_path}'."
            f" Only {SUPPORTED_TYPES} are supported."
        )
