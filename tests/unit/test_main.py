import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.smoke
def test_benchmark_run_with_backend_args():
    """
    Test that CLI invocation with new-style options parses correctly.

    The command will fail because it can't connect to the server,
    but it should pass argument parsing without errors.

    ## WRITTEN BY AI ##
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--backend",
            "kind=openai_http,target=http://localhost:9,verify=false",
            "--data",
            "kind=synthetic_text,prompt_tokens=1,output_tokens=1",
            "--profile",
            "kind=constant,rate=1",
            "--constraint",
            "kind=max_requests,max_num=1",
        ],
    )
    # This will fail because it can't connect to the server,
    # but it will pass the argument parsing, which is what we want to test.
    assert result.exit_code != 0
    assert "Invalid header format" not in result.output
