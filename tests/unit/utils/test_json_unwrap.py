import pytest
from datasets import Dataset, IterableDataset

from guidellm.utils.json_unwrap import (
    extract_json,
    get_json_column_names,
    try_json_load,
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("arg", "expected_out"),
    [
        ("", None),
        (r"{}", {}),
        ("'string'", None),
        ("'1'", None),
        (
            r'{"field_1": 1, "field_2": "two", "field_3": [3, 4]}',
            {"field_1": 1, "field_2": "two", "field_3": [3, 4]},
        ),
        (
            r'[{"field_1": 1, "field_2": "two"}, {"field_3": [3, 4]}]',
            [{"field_1": 1, "field_2": "two"}, {"field_3": [3, 4]}],
        ),
    ],
)
def test_try_json_load(arg, expected_out):
    assert try_json_load(arg) == expected_out


def one_row_generator(row: dict):
    yield {key: value[0] for key, value in row.items()}


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("data", "expected_out"),
    [
        ({"field": [1]}, []),
        ({"wrapper": [r'{"inner_field": 2}']}, ["wrapper"]),
        ({"field": [1], "wrapper": [r'{"inner_field": 2}']}, ["wrapper"]),
        (
            {"field": [1], "wrapper": [r'{"inner_field": 2}'], "field_2": ["two"]},
            ["wrapper"],
        ),
        ({"wrapper": [r"{}"]}, ["wrapper"]),
        (
            {
                "wrapper": [r'{"inner_field": 2}'],
                "field": [1],
                "wrapper_2": [[r'{"inner_field": 3}']],
                "field_2": [4],
                "wrapper_3": [r'{"inner_field": 5}'],
            },
            ["wrapper", "wrapper_2", "wrapper_3"],
        ),
    ],
)
def test_get_json_column_names(data, expected_out):
    dataset = Dataset.from_dict(data)
    iterable_dataset = IterableDataset.from_generator(
        one_row_generator, gen_kwargs={"row": data}
    )
    for ds in (dataset, iterable_dataset):
        assert get_json_column_names(ds) == expected_out
