import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.smoke
def test_benchmark_run_with_backend_args():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--backend-args",
            '{"headers": {"Authorization": "Bearer my-token"}, "verify": false}',
            "--target",
            "http://localhost:9",
            "--data",
            "prompt_tokens=1,output_tokens=1",
            "--profile",
            "constant",
            "--rate",
            "1",
            "--max-requests",
            "1",
        ],
    )
    # This will fail because it can't connect to the server,
    # but it will pass the header parsing, which is what we want to test.
    assert result.exit_code != 0
    assert "Invalid header format" not in result.output


@pytest.mark.xfail(reason="old and broken", run=False)
@patch("guidellm.__main__.benchmark_generative_text")
def test_cli_backend_args_header_removal(mock_benchmark_func, tmp_path: Path):
    """
    Tests that --backend-args from the CLI correctly overrides scenario
    values and that `null` correctly removes a header.
    """
    scenario_path = tmp_path / "scenario.json"

    # Create a scenario file with a header that should be overridden and removed
    scenario_content = {
        "backend_type": "openai_http",
        "backend_kwargs": {"headers": {"Authorization": "should-be-removed"}},
        "data": "prompt_tokens=10,output_tokens=10",
        "max_requests": 1,
        "target": "http://dummy-target",
        "profile": "synchronous",
        "processor": "gpt2",
    }
    with scenario_path.open("w") as f:
        json.dump(scenario_content, f)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--scenario",
            str(scenario_path),
            "--backend-kwargs",
            '{"headers": {"Authorization": null, "Custom-Header": "Custom-Value"}}',
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output

    # Assert that benchmark_with_scenario was called with the correct scenario
    mock_benchmark_func.assert_called_once()
    call_args = mock_benchmark_func.call_args[1]
    scenario = call_args["args"]

    # Verify the backend_args were merged correctly
    backend_args = scenario.backend_kwargs
    expected_headers = {"Authorization": None, "Custom-Header": "Custom-Value"}
    assert backend_args["headers"] == expected_headers

@patch("guidellm.__main__.benchmark_generative_text", new_callable=AsyncMock)
def test_cli_passes_per_constraints(mock_benchmark_func):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target",
            "http://localhost:9",
            "--data",
            "prompt_tokens=1,output_tokens=1",
            "--profile",
            "sweep",
            "--rate",
            "5",
            "--max-requests",
            "1",
            "--per-constraints",
            '{"max_seconds":[5,10,15,15,20], "max_requests":[100,200,200,400,400]}',
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    mock_benchmark_func.assert_called_once()
    args = mock_benchmark_func.call_args.kwargs["args"]
    assert args.per_constraints == {
        "max_seconds": [5,10,15,15,20],
        "max_requests": [100,200,200,400,400],
    }
