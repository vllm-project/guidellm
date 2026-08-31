"""
End-to-end checks that scheduler dispatch delay is reported.

``request_latency`` is measured from the moment a request is dispatched, so any
time it spent waiting for the scheduler is invisible to it. When the configured
rate exceeds what the concurrency cap allows, that wait is where most of the
real latency lives. These tests pin the reported dispatch delay in both
directions: near zero while the scheduler keeps up, and dominant once it cannot.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)

# Service time is roughly ttft + itl * (output_tokens - 1), about 65ms here.
SERVICE_TIME_SEC = 0.065
CAPPED_CONCURRENCY = 2
# Well above what CAPPED_CONCURRENCY can sustain, so the scheduler falls behind.
OVERSUBSCRIBED_RATE = 100
# Comfortably inside what the server handles without any concurrency cap.
SUSTAINABLE_RATE = 10


@pytest.fixture(scope="module")
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """Healthy server with a stable, known service time.

    ## WRITTEN BY AI ##
    """
    if e2e_server_kind == "llm-d":
        pytest.skip("Dispatch delay assertions require a fixed-latency mock server")

    handle = start_mock_server(
        ttft_ms=50.0,
        itl_ms=2.0,
        output_tokens=8,
        request_latency=SERVICE_TIME_SEC,
    )
    try:
        yield handle
    finally:
        handle.stop()


def _run(
    server: E2EServer,
    tmp_path: Path,
    report_name: str,
    rate: int,
    max_concurrency: int | None,
) -> dict:
    """Run a benchmark and return the first benchmark entry of its report.

    ## WRITTEN BY AI ##
    """
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs=report_name
    )
    client.start_benchmark(
        rate=rate,
        max_concurrency=max_concurrency,
        max_seconds=8,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=8",
    )
    client.wait_for_completion(timeout=60)
    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(tmp_path / report_name)
    return report["benchmarks"][0]


@pytest.mark.sanity
@pytest.mark.timeout(120)
def test_dispatch_delay_reported_when_scheduler_falls_behind(
    server: E2EServer, tmp_path: Path
):
    """
    Requested rate far above the concurrency cap surfaces a large delay.

    ## WRITTEN BY AI ##
    """
    benchmark = _run(
        server, tmp_path, "co_capped.json", OVERSUBSCRIBED_RATE, CAPPED_CONCURRENCY
    )
    metrics = benchmark["metrics"]

    latency = metrics["request_latency"]["successful"]["median"]
    delay = metrics["request_dispatch_delay"]["successful"]["median"]
    scheduled = metrics["request_scheduled_latency"]["successful"]["median"]

    # The server itself stays healthy, so dispatch is the only thing lagging.
    assert latency == pytest.approx(SERVICE_TIME_SEC, abs=0.2)
    assert delay > 2.0
    # Almost all of the real latency is the wait, not the request itself.
    assert scheduled > 3 * latency


@pytest.mark.sanity
@pytest.mark.timeout(120)
def test_dispatch_delay_near_zero_when_scheduler_keeps_up(
    server: E2EServer, tmp_path: Path
):
    """
    A sustainable rate leaves scheduled latency equal to request latency.

    ## WRITTEN BY AI ##
    """
    benchmark = _run(server, tmp_path, "co_sustainable.json", SUSTAINABLE_RATE, None)
    metrics = benchmark["metrics"]

    latency = metrics["request_latency"]["successful"]["median"]
    delay = metrics["request_dispatch_delay"]["successful"]["median"]
    scheduled = metrics["request_scheduled_latency"]["successful"]["median"]

    assert delay < 0.5
    assert scheduled == pytest.approx(latency, abs=0.5)
