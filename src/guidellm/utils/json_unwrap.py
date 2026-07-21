import dataclasses
import itertools
import json
from typing import Any

from datasets import Dataset, IterableDataset


@dataclasses.dataclass
class VirtualColumnLocation:
    wrapper_column: str
    virtual_column: str


def try_json_load(json_string: str) -> Any:
    try:
        return json.loads(json_string)
    except (TypeError, json.JSONDecodeError):
        return None


def get_json_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    """Assumes dataset has at least one column and at least one row.

    Returns a list of all columns in the dataset containing valid JSON. This includes
    columns containing lists of valid JSON.
    """
    sample = next(iter(dataset))
    sample = {k: (v[0] if isinstance(v, list) and v else v) for k, v in sample.items()}
    column_names = dataset.column_names or list(next(iter(dataset)).keys())
    return [
        col
        for col in column_names
        if isinstance(sample[col], dict) or try_json_load(sample[col]) is not None
    ]


def _extract_list_of_json(raw: list) -> Any:
    data = list(
        itertools.takewhile(
            lambda res: res is not None,
            ((try_json_load(val) if isinstance(val, str) else val) for val in raw),
        )
    )
    if len(data) < len(raw):
        return None
    return data


def extract_json(row_data: dict[str, Any], wrapper_column: str) -> Any:
    """Parse a JSON `wrapper_column` from a row and return its inner JSON
    object or list of JSON objects.
    """
    raw = row_data.get(wrapper_column)
    if raw is None:
        return None

    if isinstance(raw, list):
        data = _extract_list_of_json(raw)
    elif isinstance(raw, str):
        data = try_json_load(raw)
    else:
        data = raw

    if isinstance(data, list | dict):
        return data
    return None
