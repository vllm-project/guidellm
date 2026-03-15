# E2E tests for successful benchmark scenarios with timing validation

from pathlib import Path

import pytest

from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    assert_successful_requests_fields,
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
        mode="random",
        time_to_first_token=1,  # 1ms TTFT
        inter_token_latency=1,  # 1ms ITL
    )
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_max_seconds_benchmark(server: VllmSimServer, tmp_path: Path):
    """
    Test that the max seconds constraint is properly triggered.
    """
    report_name = "max_duration_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 4
    duration = 5
    max_seconds = duration
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
    )

    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=30)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the max duration constraint was triggered
    assert_constraint_triggered(benchmark, "max_seconds", {"duration_exceeded": True})

    # Validate successful requests have all expected fields
    successful_requests = benchmark["requests"]["successful"]
    assert_successful_requests_fields(successful_requests)


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_max_requests_benchmark(server: VllmSimServer, tmp_path: Path):
    """
    Test that the max requests constraint is properly triggered.
    """
    report_name = "max_number_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 4
    duration = 5
    max_requests = rate * duration

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


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_replay_profile_benchmark(server: VllmSimServer, tmp_path: Path):
    """
    Test trace replay profile with a simple trace file.
    Validates that requests are replayed with correct timing from trace.
    Also tests time_scale (rate) functionality.
    """
    report_name = "replay_benchmarks.json"
    report_path = tmp_path / report_name

    # Create trace file with 5 requests at 0.05s intervals
    trace_file = _create_trace_file(tmp_path, num_requests=5, interval=0.05)

    # Create and configure the guidellm client with replay profile
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark with replay profile
    # rate=2.0 means time_scale=2.0 (timestamps multiplied by 2)
    client.start_benchmark(
        profile="replay",
        rate=2.0,
        max_requests=5,
        data=str(trace_file),
        processor="gpt2",
    )

    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=30)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    assert len(report["benchmarks"]) == 1

    benchmark = report["benchmarks"][0]

    # Validate successful requests have all expected fields
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == 5, (
        f"Expected 5 successful requests, got {len(successful_requests)}"
    )
    assert_successful_requests_fields(successful_requests)

    # Verify scheduler state shows correct request count
    assert "scheduler_state" in benchmark
    scheduler_state = benchmark["scheduler_state"]
    assert scheduler_state["processed_requests"] == 5


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_replay_profile_max_requests_stronger_than_max_seconds(
    server: VllmSimServer, tmp_path: Path
):
    """
    Test replay profile where max_requests is the limiting constraint.
    Trace has 20 requests over 2 seconds, but max_requests=5 limits to 5.
    max_seconds=10 is not reached because max_requests triggers first.
    """
    report_name = "replay_max_requests_stronger.json"
    report_path = tmp_path / report_name

    # Create trace with 20 requests at 0.1s intervals (total 1.9s)
    trace_file = _create_trace_file(tmp_path, num_requests=20, interval=0.1)

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # max_requests=5 should be the limiting constraint
    # max_seconds=10 should NOT be reached
    client.start_benchmark(
        profile="replay",
        rate=1.0,
        max_requests=5,
        max_seconds=10,  # Very high, won't be reached
        data=str(trace_file),
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Should only have 5 requests (max_requests won)
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == 5, (
        f"Expected 5 requests (max_requests limit), got {len(successful_requests)}"
    )

    # Verify max_requests constraint was triggered
    assert_constraint_triggered(benchmark, "max_requests", {"processed_exceeded": True})


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_replay_profile_max_seconds_stronger_than_max_requests(
    server: VllmSimServer, tmp_path: Path
):
    """
    Test replay profile where max_seconds is the limiting constraint.
    Trace has 20 requests over 2 seconds, but max_seconds=0.3 limits to ~3 requests.
    max_requests=10 is not reached because max_seconds triggers first.
    """
    report_name = "replay_max_seconds_stronger.json"
    report_path = tmp_path / report_name

    # Create trace with 20 requests at 0.1s intervals
    # With time_scale=1.0, timestamps are: 0.0, 0.1, 0.2, 0.3, 0.4, ...
    # max_seconds=0.25 should include: 0.0, 0.1, 0.2 (3 requests, 0.3 > 0.25)
    trace_file = _create_trace_file(tmp_path, num_requests=20, interval=0.1)

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # max_seconds=0.25 should be the limiting constraint
    # Only timestamps <= 0.25 should be kept: 0.0, 0.1, 0.2
    client.start_benchmark(
        profile="replay",
        rate=1.0,
        max_requests=10,  # High, won't be reached
        max_seconds=0.25,
        data=str(trace_file),
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Should have 3 requests (0.0, 0.1, 0.2 where 0.2 <= 0.25)
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == 3, (
        f"Expected 3 requests (max_seconds=0.25 filter), got {len(successful_requests)}"
    )

    # Verify max_requests constraint was triggered
    # (max_seconds is converted to max_requests internally)
    assert_constraint_triggered(benchmark, "max_requests", {"processed_exceeded": True})


# Helper functions for trace file creation


def _create_trace_file(
    tmp_path: Path, num_requests: int = 5, interval: float = 0.1
) -> Path:
    """Create a trace file with evenly spaced timestamps for testing."""
    trace_file = tmp_path / "trace.jsonl"
    lines = [
        f'{{"timestamp": {i * interval}, '
        f'"input_length": {10 * (i + 1)}, '
        f'"output_length": {5 * (i + 1)}}}'
        for i in range(num_requests)
    ]
    trace_file.write_text("\n".join(lines))
    return trace_file


def _create_burst_trace_file(tmp_path: Path, num_requests: int = 10) -> Path:
    """Create a trace file with all requests at the same timestamp."""
    trace_file = tmp_path / "trace_burst.jsonl"
    lines = [
        f'{{"timestamp": 0.0, '
        f'"input_length": {20 * (i + 1)}, '
        f'"output_length": {10 * (i + 1)}}}'
        for i in range(num_requests)
    ]
    trace_file.write_text("\n".join(lines))
    return trace_file
