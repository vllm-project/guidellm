# E2E test for max error rate constraint functionality

from pathlib import Path

import pytest

from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    load_benchmark_report,
)
from tests.e2e.vllm_sim_server import VllmSimServer


@pytest.fixture
def server():
    """
    Pytest fixture to start and stop the server for each test function.

    Uses function scope (not module) because this test intentionally kills
    the server mid-run; function scope ensures reruns get a fresh server.
    """
    server = VllmSimServer(
        port=8000,
        model="databricks/dolly-v2-12b",
        mode="random",
        time_to_first_token=1,  # 1ms TTFT
        inter_token_latency=1,  # 1ms ITL
    )
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(120)
def test_max_error_benchmark(server: VllmSimServer, tmp_path: Path):
    """
    Test that the max error rate constraint is properly triggered when server goes down.
    """
    report_name = "max_error_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 10
    max_error_rate = 0.1

    # Create and configure the guidellm client
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark
    client.start_benchmark(
        rate=rate,
        max_seconds=50,
        max_error_rate=max_error_rate,
    )

    # Wait for the benchmark to complete (server will be stopped after 30 seconds).
    # CI workers need ~15-20s to initialize (fork, download/load tokenizer on first
    # run), so 30s gives enough headroom for actual benchmarking before the server
    # is killed and the error-rate constraint triggers.
    client.wait_for_completion(timeout=60, stop_server_after=30, server=server)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the max error rate constraint was triggered
    assert_constraint_triggered(
        benchmark,
        "max_error_rate",
        {
            "exceeded_error_rate": True,
            "current_error_rate": lambda rate: rate >= max_error_rate,
        },
    )
