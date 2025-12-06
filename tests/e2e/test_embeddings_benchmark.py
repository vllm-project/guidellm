# E2E tests for embeddings endpoint benchmarking

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
def embeddings_server():
    """
    Pytest fixture to start and stop the embeddings server for the entire module.
    """
    server = VllmSimServer(
        port=8001,
        model="text-embedding-ada-002",
        mode="random",
        time_to_first_token=10,  # 10ms processing time
        inter_token_latency=1,  # Not really used for embeddings
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.mark.timeout(30)
def test_embeddings_max_requests_benchmark(embeddings_server: VllmSimServer):
    """
    Test that the max requests constraint works properly for embeddings endpoint.

    ## WRITTEN BY AI ##
    """
    report_path = Path("tests/e2e/embeddings_max_requests.json")
    rate = 5
    max_requests = 20

    # Create and configure the guidellm client
    client = GuidellmClient(target=embeddings_server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_requests=max_requests,
            request_type="embeddings",
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

        # Validate successful requests count
        successful_requests = benchmark["requests"]["successful"]
        assert len(successful_requests) == max_requests, (
            f"Expected {max_requests} successful requests, "
            f"got {len(successful_requests)}"
        )

    finally:
        cleanup_report_file(report_path)


@pytest.mark.timeout(30)
def test_embeddings_max_seconds_benchmark(embeddings_server: VllmSimServer):
    """
    Test that the max seconds constraint works properly for embeddings endpoint.

    ## WRITTEN BY AI ##
    """
    report_path = Path("tests/e2e/embeddings_max_seconds.json")
    rate = 4
    duration = 5
    max_seconds = duration

    # Create and configure the guidellm client
    client = GuidellmClient(target=embeddings_server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_seconds=max_seconds,
            request_type="embeddings",
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

        # Validate that we have successful requests
        successful_requests = benchmark["requests"]["successful"]
        assert len(successful_requests) > 0, "Expected at least one successful request"

    finally:
        cleanup_report_file(report_path)


@pytest.mark.timeout(30)
def test_embeddings_rate_benchmark(embeddings_server: VllmSimServer):
    """
    Test basic rate-based benchmarking for embeddings endpoint.

    ## WRITTEN BY AI ##
    """
    report_path = Path("tests/e2e/embeddings_rate.json")
    rate = 10
    max_requests = 30

    # Create and configure the guidellm client
    client = GuidellmClient(target=embeddings_server.get_url(), output_path=report_path)

    try:
        # Start the benchmark
        client.start_benchmark(
            rate=rate,
            max_requests=max_requests,
            request_type="embeddings",
        )

        # Wait for the benchmark to complete
        client.wait_for_completion(timeout=30)

        # Assert no Python exceptions occurred
        assert_no_python_exceptions(client.stderr)

        # Load and validate the report
        report = load_benchmark_report(report_path)
        benchmark = report["benchmarks"][0]

        # Validate successful requests
        successful_requests = benchmark["requests"]["successful"]
        assert len(successful_requests) == max_requests, (
            f"Expected {max_requests} successful requests, "
            f"got {len(successful_requests)}"
        )

        # Validate that all requests have the expected fields
        for request in successful_requests:
            assert "start_time" in request, "Request missing start_time"
            assert "end_time" in request, "Request missing end_time"
            # For embeddings, we don't expect output_tokens, only input_tokens
            assert "prompt" in request or "input_tokens" in request, (
                "Request missing prompt or input_tokens"
            )

    finally:
        cleanup_report_file(report_path)
