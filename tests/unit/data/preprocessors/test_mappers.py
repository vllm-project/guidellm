"""Unit tests for column mapper image-cell normalization."""

import pytest
from datasets import Dataset

from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.schemas.data import GenerativeColumnMapperArgs


@pytest.mark.regression
@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("image", "first", ["first"]),
        ("image", ["first", "second"], ["first", "second"]),
        ("image", [["first"]], [["first"]]),
        ("prompt", ["first", "second"], [["first", "second"]]),
    ],
)
def test_image_lists_expand_one_level_only(column, value, expected):
    """Expand only Python image lists within their assigned turn. ## WRITTEN BY AI ##"""
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([Dataset.from_dict({column: [value]})])
    column_type = "image_column" if column == "image" else "text_column"
    assert mapper([{"dataset": {column: value}}]) == [{column_type: expected}]
