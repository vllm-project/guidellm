# E2E test for max error rate constraint functionality

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


@pytest.fixture
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """
    Function-scoped server that fails after a fixed number of successful requests.

    Uses function scope (not module) so each run gets a fresh server: the mock
    ``fail_after_requests`` counter is process-local, and the llm-d path
    intentionally kills the server mid-run (reruns need a new process).

    Uses MockServer ``fail_after_requests`` by default for deterministic error-rate
    triggering. With ``--e2e-server=llm-d``, falls back to killing the sim mid-run.
    """
    if e2e_server_kind == "llm-d":
        sim = VllmSimServer(
            port=free_port(),
            model="databricks/dolly-v2-12b",
            mode="random",
            time_to_first_token=1,  # 1ms TTFT
            inter_token_latency=1,  # 1ms ITL
        )
        try:
            sim.start()
            yield sim  # Yield the URL for tests to use
        finally:
            sim.stop()  # Teardown: Stop the server after tests are done
        return

    # Allow a few successes, then return 500s so max_error_rate trips quickly.
    handle = start_mock_server(
        fail_after_requests=5,
        ttft_ms=5.0,
        itl_ms=1.0,
        output_tokens=16,
    )
    try:
        yield handle  # Yield the URL for tests to use
    finally:
        handle.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(90)
def test_max_error_benchmark(server: E2EServer, e2e_server_kind: str, tmp_path: Path):
    """
    Test that the max error rate constraint is properly triggered.

    ## WRITTEN BY AI ##
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
        max_seconds=30,
        max_error_rate=max_error_rate,
        data="kind=synthetic_text,prompt_tokens=64,output_tokens=16",
    )

    if e2e_server_kind == "llm-d":
        # Kill the simulator after requests have started. Worker spawn + tokenizer
        # load often needs ~15-20s, so 30s gives headroom before the kill so
        # error-rate can actually accumulate. MockServer uses fail_after_requests
        # instead and does not need a mid-run kill.
        client.wait_for_completion(timeout=60, stop_server_after=30, server=server)
    else:
        # Wait for the benchmark to complete
        client.wait_for_completion(timeout=60)

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
            "current_error_rate": lambda current: current >= max_error_rate,
        },
    )
