# E2E tests for successful benchmark scenarios with timing validation

from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer
from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    assert_successful_requests_fields,
    load_benchmark_report,
)


@pytest.mark.timeout(60)
@pytest.mark.sanity
def test_max_seconds_benchmark(server: E2EServer, tmp_path: Path):
    """
    Test that the max seconds constraint is properly triggered.
    """
    report_name = "max_duration_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 4
    max_seconds = 2
    # Create and configure the guidellm client
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark
    client.start_benchmark(
        rate=rate,
        max_seconds=max_seconds,
        data="kind=synthetic_text,prompt_tokens=64,output_tokens=16",
    )
    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=30)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the max duration constraint was triggered
    assert_constraint_triggered(benchmark, "max_duration", {"duration_exceeded": True})

    # Validate successful requests have all expected fields
    successful_requests = benchmark["requests"]["successful"]
    assert_successful_requests_fields(successful_requests)


@pytest.mark.timeout(60)
@pytest.mark.sanity
def test_max_requests_benchmark(server: E2EServer, tmp_path: Path):
    """
    Test that the max requests constraint is properly triggered.
    """
    report_name = "max_number_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 4
    max_requests = 8

    # Create and configure the guidellm client
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark
    client.start_benchmark(
        rate=rate,
        max_requests=max_requests,
        data="kind=synthetic_text,prompt_tokens=64,output_tokens=16",
    )
    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=30)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the max requests constraint was triggered
    assert_constraint_triggered(benchmark, "max_requests", {"processed_exceeded": True})

    # Validate successful requests have all expected fields
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} successful requests, got {len(successful_requests)}"
    )
    assert_successful_requests_fields(successful_requests)
