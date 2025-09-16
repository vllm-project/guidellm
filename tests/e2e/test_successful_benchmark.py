# E2E tests for successful benchmark scenarios with timing validation

from pathlib import Path

import pytest

from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    assert_successful_requests_fields,
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
    server = VllmSimServer(
        port=8000,
        model="databricks/dolly-v2-12b",
        mode="echo",
        time_to_first_token=1,  # 1ms TTFT
        inter_token_latency=1,  # 1ms ITL
    )
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(30)
def test_max_seconds_benchmark(server: VllmSimServer):
    """
    Test that the max seconds constraint is properly triggered.
    """
    report_path = Path("tests/e2e/max_duration_benchmarks.json")
    rate = 10

    # Create and configure the guidellm client
    client = GuidellmClient(target=server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_seconds=1,
        )

        # Wait for the benchmark to complete
        client.wait_for_completion(timeout=30)

        # Assert no Python exceptions occurred
        assert_no_python_exceptions(client.stderr)

        # Load and validate the report
        report = load_benchmark_report(report_path)
        benchmark = report["benchmarks"][0]

        # Check that the max duration constraint was triggered
        assert_constraint_triggered(
            benchmark, "max_seconds", {"duration_exceeded": True}
        )

        # Validate successful requests have all expected fields
        successful_requests = benchmark["requests"]["successful"]
        assert_successful_requests_fields(successful_requests)

    finally:
        cleanup_report_file(report_path)


@pytest.mark.timeout(30)
def test_max_requests_benchmark(server: VllmSimServer):
    """
    Test that the max requests constraint is properly triggered.
    """
    report_path = Path("tests/e2e/max_number_benchmarks.json")
    rate = 10

    # Create and configure the guidellm client
    client = GuidellmClient(target=server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_requests=rate,
        )

        # Wait for the benchmark to complete
        client.wait_for_completion(timeout=30)

        # Assert no Python exceptions occurred
        assert_no_python_exceptions(client.stderr)

        # Load and validate the report
        report = load_benchmark_report(report_path)
        benchmark = report["benchmarks"][0]

        # Check that the max requests constraint was triggered
        assert_constraint_triggered(
            benchmark, "max_requests", {"processed_exceeded": True}
        )

        # Validate successful requests have all expected fields
        successful_requests = benchmark["requests"]["successful"]
        assert len(successful_requests) == rate, (
            f"Expected {rate} successful requests, got {len(successful_requests)}"
        )
        assert_successful_requests_fields(successful_requests)

    finally:
        cleanup_report_file(report_path)
