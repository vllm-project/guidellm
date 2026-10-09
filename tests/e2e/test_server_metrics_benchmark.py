# E2E tests for scraping server metrics while benchmarks run

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)

STREAMS = 4


@pytest.fixture(scope="module")
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """Mock server whose /metrics reflects the requests it is serving.

    ## WRITTEN BY AI ##
    """
    if e2e_server_kind == "llm-d":
        pytest.skip("Assertions rely on the mock server's /metrics contents")

    handle = start_mock_server(ttft_ms=50.0, itl_ms=10.0, output_tokens=16)
    try:
        yield handle
    finally:
        handle.stop()


@pytest.mark.sanity
@pytest.mark.timeout(120)
def test_server_metrics_match_client_measurements(server: E2EServer, tmp_path: Path):
    """
    Server-side metrics scraped during a run agree with the client's results.

    ## WRITTEN BY AI ##
    """
    report_name = "server_metrics.json"
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )
    metrics_url = f"{server.get_url()}/metrics"
    client.start_benchmark(
        profile="constant",
        rate=20,
        max_concurrency=STREAMS,
        max_seconds=6,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=16",
        additional_args=(
            f'--server-metrics "kind=prometheus,url={metrics_url},interval=0.5"'
        ),
    )
    client.wait_for_completion(timeout=90)
    assert_no_python_exceptions(client.stderr)

    benchmark = load_benchmark_report(tmp_path / report_name)["benchmarks"][0]
    [summary] = benchmark["server_metrics"]
    successful = benchmark["metrics"]["request_totals"]["successful"]

    assert summary["source"] == metrics_url
    assert summary["scrape_errors"] == 0
    assert summary["scrapes"] >= 4

    [running] = summary["gauges"]["vllm:num_requests_running"]
    assert running["samples"]
    assert running["summary"]["max"] == STREAMS

    [completed] = summary["counters"]["vllm:request_success"]
    # The window edges are interpolated between scrapes half a second apart.
    assert completed["increase"] == pytest.approx(successful, rel=0.2)

    [latency] = summary["histograms"]["vllm:e2e_request_latency_seconds"]
    client_latency = benchmark["metrics"]["request_latency"]["successful"]["mean"]
    assert latency["mean"] == pytest.approx(client_latency, rel=0.2)


@pytest.mark.sanity
@pytest.mark.timeout(120)
def test_unreachable_server_metrics_do_not_fail_benchmark(
    server: E2EServer, tmp_path: Path
):
    """
    A metrics endpoint that cannot be reached is reported, not fatal.

    ## WRITTEN BY AI ##
    """
    report_name = "server_metrics_unreachable.json"
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )
    client.start_benchmark(
        profile="constant",
        rate=2,
        max_seconds=3,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=16",
        additional_args=(
            '--server-metrics "kind=prometheus,url=http://127.0.0.1:9/metrics,'
            'interval=0.5,timeout=0.5"'
        ),
    )
    client.wait_for_completion(timeout=90)
    assert_no_python_exceptions(client.stderr)

    benchmark = load_benchmark_report(tmp_path / report_name)["benchmarks"][0]
    [summary] = benchmark["server_metrics"]

    assert benchmark["metrics"]["request_totals"]["successful"] > 0
    assert summary["scrapes"] == 0
    assert summary["scrape_errors"] >= 2
    assert summary["gauges"] == {}
