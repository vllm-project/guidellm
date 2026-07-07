"""Tests for ``guidellm run`` CLI behavior."""

import click
import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli
from guidellm.cli.run import _build_metadata, run


def test_build_metadata_attaches_metadata_and_labels():
    """Metadata and labels should be written into the benchmark scenario.

    ## WRITTEN BY AI ##
    """
    metadata = _build_metadata(
        labels=[("env", "staging")],
        metadata=[("owner", "platform"), ("ticket", "LLM-123")],
    )

    assert metadata == {
        "labels": {"env": "staging"},
        "owner": "platform",
        "ticket": "LLM-123",
    }


def test_run_registers_metadata_aliases():
    """The run command should expose --metadata and legacy --output-extras.

    ## WRITTEN BY AI ##
    """
    metadata_option = next(param for param in run.params if param.name == "metadata")

    assert isinstance(metadata_option, click.Option)
    assert "--metadata" in metadata_option.opts
    assert "--output-extras" in metadata_option.opts


def test_run_rejects_metadata_labels_key():
    """The labels metadata field is reserved for --label values.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--disable-console",
            "--data",
            "kind=synthetic_text,prompt_tokens=128",
            "--constraint",
            "kind=max_requests,max_num=1",
            "--metadata",
            "labels=staging",
        ],
    )

    assert result.exit_code != 0
    assert "`labels` is reserved for --label values" in result.output


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
