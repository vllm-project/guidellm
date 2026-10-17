"""E2E tests for stopping a benchmark at a target margin of error."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    load_benchmark_report,
)


@pytest.fixture(scope="module")
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """
    MockServer with a spread in TTFT, so the margin of error shrinks with the
    sample count instead of being met at the first check.
    """
    if e2e_server_kind != "mock":
        pytest.skip("target_moe E2E tests rely on MockServer TTFT spread")

    handle = start_mock_server(
        ttft_ms=20.0,
        ttft_ms_std=5.0,
        itl_ms=1.0,
        output_tokens=4,
        request_latency=0.05,
    )
    try:
        yield handle
    finally:
        handle.stop()


@pytest.mark.timeout(90)
def test_target_moe_stops_benchmark(server: E2EServer, tmp_path: Path):
    """
    Test that a reachable target margin of error stops the benchmark before the
    duration limit, with the reported margin at or below the target.

    ## WRITTEN BY AI ##
    """
    report_name = "target_moe_benchmarks.json"
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )

    client.start_benchmark(
        rate=50,
        max_seconds=60,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=4",
        additional_args=(
            '--constraint "kind=target_moe,metric=time_to_first_token_ms,'
            'statistic=mean,moe=0.05,min_samples=30,check_interval=10"'
        ),
    )
    client.wait_for_completion(timeout=80)

    assert_no_python_exceptions(client.stderr)
    benchmark = load_benchmark_report(tmp_path / report_name)["benchmarks"][0]
    assert_constraint_triggered(benchmark, "target_moe", {"target_moe_reached": True})
    metadata = benchmark["scheduler_state"]["end_processing_constraints"]["target_moe"][
        "metadata"
    ]
    assert metadata["relative_moe"] <= 0.05
    assert metadata["samples"] >= 30
    assert (
        "max_duration" not in benchmark["scheduler_state"]["end_processing_constraints"]
    )


@pytest.mark.timeout(60)
def test_unreachable_target_moe_leaves_other_limits(server: E2EServer, tmp_path: Path):
    """
    Test that an unreachable target does not stop the run, which then ends at
    its duration limit.

    ## WRITTEN BY AI ##
    """
    report_name = "target_moe_unreachable_benchmarks.json"
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )

    client.start_benchmark(
        rate=20,
        max_seconds=5,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=4",
        additional_args='--constraint "kind=target_moe,statistic=p95,moe=0.0001"',
    )
    client.wait_for_completion(timeout=50)

    assert_no_python_exceptions(client.stderr)
    benchmark = load_benchmark_report(tmp_path / report_name)["benchmarks"][0]
    constraints = benchmark["scheduler_state"]["end_processing_constraints"]
    assert "max_duration" in constraints
    assert "target_moe" not in constraints
