"""E2E coverage for parameterized strategy labels in console summary tables."""

from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)


@pytest.mark.regression
@pytest.mark.timeout(90)
def test_concurrent_output_keeps_parameterized_strategy_labels(
    server: E2EServer, tmp_path: Path
):
    """Render distinct concurrent stream counts in the final summary tables.

    ## WRITTEN BY AI ##
    """
    client = GuidellmClient(target=server.get_url(), output_dir=tmp_path)
    client.start_benchmark(
        profile="concurrent",
        rate=None,
        max_seconds=2,
        data="kind=synthetic_text,prompt_tokens=64,output_tokens=16",
        additional_args="--override profile.streams 1,2",
        disable_console=False,
    )
    client.wait_for_completion(timeout=60)

    assert client.process is not None
    assert client.process.returncode == 0, (client.stdout, client.stderr)
    assert_no_python_exceptions(client.stderr)
    output = client.stdout or ""
    assert "Strategy" in output
    for streams in (1, 2):
        label = f"concurrent@{streams}"
        assert label in output, f"final tables are missing {label!r}:\n{output}"

    benchmarks = load_benchmark_report(tmp_path / "benchmarks.json")["benchmarks"]
    assert len(benchmarks) == 2
    assert all(benchmark["requests"]["successful"] for benchmark in benchmarks)
