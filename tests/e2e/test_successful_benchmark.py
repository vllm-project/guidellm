# E2E tests for successful benchmark scenarios with timing validation

import json
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
@pytest.mark.regression
@pytest.mark.parametrize(("e2el_ms", "expected_attainment"), [(1e6, 1.0), (1e-6, 0.0)])
def test_slo_goodput_report(
    server: E2EServer, tmp_path: Path, e2el_ms: float, expected_attainment: float
):
    """
    Report independent objectives and conforming token rates through the CLI.

    ## WRITTEN BY AI ##
    """
    report_name = "slo_benchmarks.json"
    max_requests = 4
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )
    metrics_config = json.dumps(
        {"kind": "generative", "slo": {"ttft_ms": 1e6, "e2el_ms": e2el_ms}}
    )
    client.start_benchmark(
        rate=4,
        max_requests=max_requests,
        data="kind=synthetic_text,prompt_tokens=64,output_tokens=16",
        additional_args=f"--metrics '{metrics_config}'",
    )
    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    benchmark = load_benchmark_report(tmp_path / report_name)["benchmarks"][0]
    assert len(benchmark["requests"]["successful"]) == max_requests
    metrics = benchmark["metrics"]
    assert metrics["slo_attainment"] == expected_attainment
    assert metrics["slo_determined_requests"] == max_requests
    assert metrics["slo_attainment_by_metric"] == {
        "ttft_ms": {
            "conforming_requests": max_requests,
            "determined_requests": max_requests,
            "attainment": 1.0,
        },
        "e2el_ms": {
            "conforming_requests": int(max_requests * expected_attainment),
            "determined_requests": max_requests,
            "attainment": expected_attainment,
        },
    }
    throughput = metrics["output_tokens_per_second"]["successful"]["mean"]
    assert throughput > 0
    assert metrics["output_token_goodput"]["successful"]["mean"] == pytest.approx(
        throughput * expected_attainment
    )


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
