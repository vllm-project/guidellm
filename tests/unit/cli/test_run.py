"""Tests for ``guidellm run`` CLI error translation."""

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli
from guidellm.backends.openai.http import OpenAIHTTPBackendArgs


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
def test_run_parses_append_payloads_into_openai_backend():
    """The top-level option is stored in the registered backend configuration.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    benchmark = AsyncMock()

    with patch("guidellm.cli.run.benchmark_generative_text", benchmark):
        result = runner.invoke(
            cli,
            [
                "run",
                "--backend",
                "kind=openai_http,target=http://localhost:8000",
                "--append-payloads",
                '{"metadata":{"category":"support"},"priority":1}',
                "--data",
                "kind=synthetic_text,prompt_tokens=128",
                "--disable-console",
            ],
        )

    assert result.exit_code == 0, result.output
    args = benchmark.await_args.kwargs["args"]
    assert isinstance(args.spec.backend, OpenAIHTTPBackendArgs)
    assert args.spec.backend.append_payloads == {
        "metadata": {"category": "support"},
        "priority": 1,
    }


@pytest.mark.regression
def test_run_rejects_non_object_append_payloads():
    """The CLI reports a clear error when append payloads is not an object.

    ## WRITTEN BY AI ##
    """
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--append-payloads",
            '["not", "an", "object"]',
            "--data",
            "kind=synthetic_text,prompt_tokens=128",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--append-payloads'" in result.output
    assert "must be a JSON, YAML, or key=value object" in result.output
