from typing import Any

from datasets import Dataset, IterableDataset

from guidellm.utils.imports import json


def try_json_load(json_string: str) -> Any:
    try:
        return json.loads(json_string)
    except (TypeError, json.JSONDecodeError):
        return None


def is_json_serializable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def get_json_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    """Assumes dataset has at least one column and at least one row.

    Returns a list of all columns in the dataset containing valid JSON. This includes
    columns containing lists of valid JSON, as well as Python dictionaries that can be
    serialized to JSON with no issue.
    """
    sample = next(iter(dataset))
    sample = {k: (v[0] if isinstance(v, list) and v else v) for k, v in sample.items()}
    column_names = dataset.column_names or list(next(iter(dataset)).keys())
    return [
        col
        for col in column_names
        if (isinstance(sample[col], dict) and is_json_serializable(sample[col]))
        or try_json_load(sample[col]) is not None
    ]
