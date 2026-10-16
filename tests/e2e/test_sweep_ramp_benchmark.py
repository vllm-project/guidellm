# E2E tests for the sweep profile's ramp peak detection

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)

MAX_CONCURRENT_REQUESTS = 4


@pytest.fixture(scope="module")
def server() -> Iterator[E2EServer]:
    """
    Mock server that can only run a few requests at once, so it has a fixed
    capacity the ramp can saturate within a short step.
    """
    handle = start_mock_server(
        ttft_ms=100.0,
        itl_ms=10.0,
        output_tokens=8,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
    )
    try:
        yield handle
    finally:
        handle.stop()


def _run_ramp_sweep(server: E2EServer, tmp_path: Path, max_seconds: int) -> dict:
    report_name = "sweep_ramp_benchmarks.json"
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )
    client.start_benchmark(
        profile="sweep,sweep_size=3,peak_detection=ramp",
        rate=None,
        max_seconds=max_seconds,
        data="kind=synthetic_text,prompt_tokens=32,output_tokens=8",
    )
    client.wait_for_completion(timeout=90)
    assert_no_python_exceptions(client.stderr)

    return load_benchmark_report(tmp_path / report_name)


def _ramp_conclusion(report: dict) -> dict:
    conclusions = [
        entry
        for entry in report.get("conclusions", [])
        if entry.get("kind") == "sweep_ramp"
    ]
    assert len(conclusions) == 1, report.get("conclusions")
    return conclusions[0]


@pytest.mark.timeout(120)
def test_sweep_ramp_finds_server_capacity(server: E2EServer, tmp_path: Path):
    """
    The ramp saturates a server with a fixed capacity and the sweep uses the
    peak it measured as the top of its range.

    ## WRITTEN BY AI ##
    """
    report = _run_ramp_sweep(server, tmp_path, max_seconds=10)

    benchmarks = report["benchmarks"]
    assert [b["config"]["strategy"]["type_"] for b in benchmarks] == [
        "synchronous",
        "ramp",
        "constant",
    ]

    conclusion = _ramp_conclusion(report)
    assert conclusion["peak_source"] == "saturated"

    sync_latency = benchmarks[0]["metrics"]["request_latency"]["successful"]["mean"]
    capacity = MAX_CONCURRENT_REQUESTS / sync_latency
    assert 0.7 * capacity <= conclusion["peak_rate"] <= 1.3 * capacity

    # With sweep_size=3 the only constant step runs at the peak.
    constant_rate = benchmarks[2]["config"]["strategy"]["rate"]
    assert constant_rate == pytest.approx(conclusion["peak_rate"])


@pytest.mark.timeout(120)
def test_sweep_ramp_reports_when_not_saturated(server: E2EServer, tmp_path: Path):
    """
    A per-step limit that ends the ramp before the server saturates is reported
    in the conclusion.

    ## WRITTEN BY AI ##
    """
    report = _run_ramp_sweep(server, tmp_path, max_seconds=2)

    conclusion = _ramp_conclusion(report)
    assert conclusion["peak_source"] != "saturated"
