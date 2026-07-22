import datetime

import pytest
from datasets import Dataset, IterableDataset

from guidellm.utils.json_unwrap import (
    VirtualColumnLocation,
    construct_virtual_column_locations,
    get_json_column_names,
    try_json_load,
    unzip_virtual_column_locations,
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("wrapper_col", "virtual_cols", "expected"),
    [
        ("", [], []),
        (
            "wrapper",
            ["field_1", "field_2"],
            [
                VirtualColumnLocation("wrapper", "field_1"),
                VirtualColumnLocation("wrapper", "field_2"),
            ],
        ),
        ("wrapper", [], []),
    ],
)
def test_construct_virtual_column_locations(wrapper_col, virtual_cols, expected):
    actual = construct_virtual_column_locations(wrapper_col, virtual_cols)
    if not actual:
        assert actual == expected
    for location in actual:
        assert isinstance(location, VirtualColumnLocation)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("arg", "expected"),
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
def test_try_json_load(arg, expected):
    assert try_json_load(arg) == expected


def one_row_generator(row: dict):
    yield {key: value[0] for key, value in row.items()}


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("data", "expected"),
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
        (
            {
                "valid_dict": [{"field_1": 1, "field_2": "two", "field_3": [3, 4]}],
                "valid_dict_2": [[{"field_1": 1, "field_2": "two", "field_3": [3, 4]}]],
                "invalid_dict": [{"time": datetime.datetime.now()}],
                "valid_dict_3": [{}],
            },
            ["valid_dict", "valid_dict_2", "valid_dict_3"],
        ),
    ],
)
def test_get_json_column_names(data, expected):
    dataset = Dataset.from_dict(data)
    iterable_dataset = IterableDataset.from_generator(
        one_row_generator, gen_kwargs={"row": data}
    )
    for ds in (dataset, iterable_dataset):
        assert get_json_column_names(ds) == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("locations", "expected"),
    [
        ([], ((), ())),
        (
            [
                VirtualColumnLocation("wrapper", "field_1"),
                VirtualColumnLocation("wrapper_2", "field_1"),
                VirtualColumnLocation("wrapper", "field_2"),
            ],
            (("wrapper", "wrapper_2", "wrapper"), ("field_1", "field_1", "field_2")),
        ),
        ([VirtualColumnLocation("wrapper", "field")], (("wrapper",), ("field",))),
    ],
)
def test_unzip_virtual_column_locations(locations, expected):
    actual = unzip_virtual_column_locations(locations)
    print(actual)
    assert isinstance(actual, tuple)
    for idx in range(len(actual)):
        assert isinstance(actual[idx], tuple)
        assert actual[idx] == expected[idx]
