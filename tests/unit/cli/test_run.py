"""Tests for ``guidellm run`` CLI error translation."""

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
            "run",
            "--data",
            "kind=synthetic_text",
            "--constraint",
            "kind=max_requests,max_num=1",
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
            "run",
            "--data",
            "kind=synthetic_text,prompt_tokens=128",
            "--constraint",
            "kind=max_requests,max_num=1",
        ],
    )

    assert "Invalid value for --data" not in result.output
    assert "output_tokens" not in result.output


@pytest.mark.regression
def test_run_constraint_bare_word_shows_kind_usage():
    """
    A bare non-dict ``--constraint`` value documents expected format and kinds
    instead of the opaque pydantic dictionary message.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--constraint", "wrong"])

    assert result.exit_code != 0
    assert "--constraint" in result.output
    assert "Expected format" in result.output
    assert "max_requests" in result.output
    assert "valid dictionary" not in result.output.lower()


@pytest.mark.regression
def test_run_constraint_missing_argument_shows_kind_usage():
    """
    ``--constraint`` with no value documents expected format and kinds instead
    of Click's stock requires-an-argument message.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--constraint"])

    assert result.exit_code != 0
    assert "requires a value" in result.output
    assert "Expected format" in result.output
    assert "max_requests" in result.output
    assert "Option '--constraint' requires an argument." not in result.output


@pytest.mark.regression
def test_run_constraint_missing_kind_shows_kind_usage():
    """
    Dict-like ``--constraint`` input without ``kind`` lists valid kinds.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--constraint", "count=1"])

    assert result.exit_code != 0
    assert "missing required key 'kind'" in result.output
    assert "Expected format" in result.output
    assert "max_requests" in result.output


@pytest.mark.regression
def test_run_constraint_invalid_kind_lists_valid_kinds():
    """
    An unknown ``kind`` fails with a single usage block listing valid kinds,
    without duplicating pydantic's expected-tags list.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--constraint", "kind=bogus"])

    assert result.exit_code != 0
    assert "invalid kind 'bogus'" in result.output
    assert "Expected format" in result.output
    assert "max_requests" in result.output
    assert "expected tags" not in result.output


@pytest.mark.regression
def test_run_constraint_missing_field_stays_specific():
    """
    A valid kind with a missing required field reports the field error without
    drowning it in generic format help.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--data",
            "kind=synthetic_text,prompt_tokens=128",
            "--constraint",
            "kind=max_requests",
        ],
    )

    assert result.exit_code != 0
    assert "count" in result.output or "Field required" in result.output
    assert "Expected format" not in result.output
