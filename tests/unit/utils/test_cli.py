"""Tests for CLI utility helpers."""

import pytest

from guidellm.utils.cli import format_validation_error


@pytest.mark.sanity
def test_format_validation_error_includes_indices_and_field_path():
    """
    ``format_validation_error`` should render integer loc components as ``[i]``
    and string components as ``.name`` so nested errors are unambiguous.

    ## WRITTEN BY AI ##
    """
    formatted = format_validation_error(
        {
            "loc": ("data", 0, "synthetic_text", "output_tokens"),
            "msg": "Field required",
        }
    )

    assert formatted == "Field required (at 'data[0].synthetic_text.output_tokens')"


@pytest.mark.sanity
def test_format_validation_error_handles_top_level_only_path():
    """
    A single-element loc should still produce a readable message without
    appending a trailing separator.

    ## WRITTEN BY AI ##
    """
    formatted = format_validation_error(
        {"loc": ("rate",), "msg": "Input should be a valid number"}
    )

    assert formatted == "Input should be a valid number (at 'rate')"
