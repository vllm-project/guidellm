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
    server = VllmSimServer(
        port=8000,
        model="databricks/dolly-v2-12b",
        mode="random",
        time_to_first_token=10000,
        inter_token_latency=100,
        max_num_seqs=1,
    )
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.skip(reason="Skipping future feature test")
@pytest.mark.timeout(60)
def test_over_saturated_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/over_saturated_benchmarks.json")
    rate = 100

    # Create and configure the guidellm client
    client = GuidellmClient(target=server.get_url(), output_path=report_path)

    cleanup_report_file(report_path)
    # Start the benchmark
    client.start_benchmark(
        rate=rate,
        max_seconds=20,
        stop_over_saturated=True,
        extra_env={
            "GUIDELLM__CONSTRAINT_OVER_SATURATION_MIN_SECONDS": "0",
            "GOMAXPROCS": "1",
        },
    )

    # Wait for the benchmark to complete
    client.wait_for_completion(timeout=55)

    # Assert no Python exceptions occurred
    assert_no_python_exceptions(client.stderr)

    # Load and validate the report
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check that the max duration constraint was triggered
    assert_constraint_triggered(
        benchmark, "stop_over_saturated", {"is_over_saturated": True}
    )

    cleanup_report_file(report_path)
