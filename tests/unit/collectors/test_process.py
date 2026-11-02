"""Tests for running collectors in a separate process."""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator

import pytest

from guidellm.collectors import CollectorProcess
from guidellm.schemas.benchmark import CollectorArgs, PrometheusCollectorArgs


class _UnknownArgs(CollectorArgs):
    kind: str = "missing"


EXPOSITION = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 2
"""


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = EXPOSITION.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def metrics_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/metrics"
    server.shutdown()


@pytest.mark.sanity
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_process_returns_one_result_per_collector(metrics_url: str):
    """Run collectors in a child process and report each one in order.

    A collector whose endpoint is down still reports a result, so the results
    line up with the configured collectors.

    ## WRITTEN BY AI ##
    """
    process = CollectorProcess(
        [
            PrometheusCollectorArgs(url=metrics_url, interval=0.05),
            PrometheusCollectorArgs(url="http://127.0.0.1:9/metrics", timeout=0.5),
        ]
    )
    await process.start()
    try:
        results = []
        for _ in range(2):
            start = time.time()
            await process.benchmark_started()
            await process.benchmark_stopped()
            results.append(await process.results(start, time.time()))
    finally:
        await process.shutdown()

    for up, down in results:
        assert up["kind"] == down["kind"] == "prometheus"
        assert up["source"] == metrics_url
        assert up["scrapes"] >= 2
        assert up["gauges"]["vllm:num_requests_running"][0]["summary"]["mean"] == 2
        assert down["scrapes"] == 0
        assert down["scrape_errors"] == 2


@pytest.mark.sanity
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_process_reports_collectors_it_cannot_create():
    """Fail at startup when a collector cannot be created.

    ## WRITTEN BY AI ##
    """
    process = CollectorProcess([_UnknownArgs()])

    with pytest.raises(RuntimeError, match="missing"):
        await process.start()
    await process.shutdown()
