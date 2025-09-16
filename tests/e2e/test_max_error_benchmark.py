# E2E test for max error rate constraint functionality

from pathlib import Path

import pytest

from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    cleanup_report_file,
    load_benchmark_report,
)
from tests.e2e.vllm_sim_server import VllmSimServer


@pytest.fixture(scope="module")
def server():
    """
    Pytest fixture to start and stop the server for the entire module
    using the TestServer class.
    """
    server = VllmSimServer(port=8000, model="databricks/dolly-v2-12b", mode="echo")
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(30)
def test_max_error_benchmark(server: VllmSimServer):
    """
    Test that the max error rate constraint is properly triggered when server goes down.
    """
    report_path = Path("tests/e2e/max_error_benchmarks.json")
    rate = 10
    max_error_rate = 0.1

    # Create and configure the guidellm client
    client = GuidellmClient(target=server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_seconds=25,
            max_error_rate=max_error_rate,
        )

        # Wait for the benchmark to complete (server will be stopped after 10 seconds)
        client.wait_for_completion(timeout=30, stop_server_after=10, server=server)

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

    finally:
        cleanup_report_file(report_path)
