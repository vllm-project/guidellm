"""Tests for ``guidellm benchmark run`` CLI error translation."""

import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.regression
def test_run_reports_nested_field_path_on_missing_subfield():
    """
    A missing nested data subfield should surface its full location path
    instead of only the top-level CLI option name.

    Regression test for the misleading ``Invalid value for --data: Field required``
    message produced when a synthetic data sub-argument was omitted.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target",
            "http://localhost:9",
            "--data",
            "kind=synthetic_text",
            "--max-requests",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "--data" in result.output
    assert "data[0].synthetic_text.prompt_tokens" in result.output


@pytest.mark.regression
def test_run_allows_synthetic_text_without_output_tokens():
    """
    ``output_tokens`` is optional so synthetic_text data can be used against
    endpoints that don't produce output tokens (e.g. embeddings) without forcing
    users to supply a meaningless value.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target",
            "http://localhost:9",
            "--data",
            "kind=synthetic_text,prompt_tokens=128",
            "--max-requests",
            "1",
        ],
    )

    assert "Invalid value for --data" not in result.output
    assert "output_tokens" not in result.output
