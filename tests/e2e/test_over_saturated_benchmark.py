from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, free_port, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_constraint_triggered,
    assert_no_python_exceptions,
    load_benchmark_report,
)
from tests.e2e.vllm_sim_server import VllmSimServer


@pytest.fixture(scope="module")
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """
    Server configured to create rising concurrency / TTFT for over-saturation.

    Default MockServer uses ``max_concurrent_requests=1`` and elevated TTFT.
    ``--e2e-server=llm-d`` keeps the historical slow-sim configuration.
    """
    if e2e_server_kind == "llm-d":
        sim = VllmSimServer(
            port=free_port(),
            model="databricks/dolly-v2-12b",
            mode="random",
            time_to_first_token=60000,
            inter_token_latency=100,
            max_num_seqs=1,
        )
        try:
            sim.start()
            yield sim  # Yield the URL for tests to use
        finally:
            sim.stop()  # Teardown: Stop the server after tests are done
        return

    handle = start_mock_server(
        # Very slow TTFT + single slot: in-flight concurrency rises for the full
        # run (few completions), matching the historical llm-d max_num_seqs=1
        # + 60s TTFT pattern that over-saturation detects via concurrent slope.
        ttft_ms=30000.0,
        itl_ms=10.0,
        output_tokens=4,
        max_concurrent_requests=1,
        request_latency=30.0,
    )
    try:
        yield handle  # Yield the URL for tests to use
    finally:
        handle.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(60)
def test_over_saturated_benchmark(server: E2EServer, tmp_path: Path):
    """
    Test over-saturation detection with enforce mode.

    ## WRITTEN BY AI ##
    """
    report_name = "over_saturated_benchmarks.json"
    report_path = tmp_path / report_name
    rate = 10

    # Create and configure the guidellm client
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark
    client.start_benchmark(
        rate=rate,
        max_seconds=10,
        over_saturation={
            "mode": "enforce",
            "min_seconds": 0,
            "minimum_ttft": 0.1,
            "minimum_window_size": 5,
            "moe_threshold": 2.0,
        },
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=4",
    )

    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=40)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the over-saturation constraint was triggered
    assert_constraint_triggered(
        benchmark, "over_saturation", {"is_over_saturated": True}
    )


@pytest.mark.timeout(60)
def test_over_saturated_benchmark_with_dict_config(server: E2EServer, tmp_path: Path):
    """
    Test over-saturation detection with explicit dictionary configuration.

    ## WRITTEN BY AI ##
    """
    report_name = "over_saturated_benchmarks_dict.json"
    report_path = tmp_path / report_name
    rate = 10

    # Create and configure the guidellm client
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Start the benchmark with dictionary configuration for over-saturation
    client.start_benchmark(
        rate=rate,
        max_seconds=10,
        over_saturation={
            "mode": "enforce",
            "min_seconds": 0,
            "max_window_seconds": 120.0,
            "moe_threshold": 2.0,
            "minimum_window_size": 5,
            "minimum_ttft": 0.1,
        },
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=4",
    )

    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=40)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the over-saturation constraint was triggered
    assert_constraint_triggered(
        benchmark, "over_saturation", {"is_over_saturated": True}
    )
