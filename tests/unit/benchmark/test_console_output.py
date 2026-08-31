"""Tests for the console benchmark output tables."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from guidellm.benchmark.outputs.console import GenerativeBenchmarkerConsole
from guidellm.schemas import StatusDistributionSummary

# Metrics read by GenerativeBenchmarkerConsole.print_request_latency_table.
LATENCY_TABLE_METRICS = (
    "request_latency",
    "time_to_first_token_ms",
    "time_to_first_output_token_ms",
    "inter_token_latency_ms",
    "time_per_output_token_ms",
)


def _make_benchmark() -> SimpleNamespace:
    """Build a benchmark stub exposing every metric the latency table reads.

    ## WRITTEN BY AI ##
    """
    distribution = StatusDistributionSummary.from_values([1.0, 2.0, 3.0], [], [])
    metrics = SimpleNamespace(
        **dict.fromkeys(LATENCY_TABLE_METRICS, distribution),
        request_dispatch_delay=distribution,
        request_scheduled_latency=distribution,
    )

    return SimpleNamespace(
        config=SimpleNamespace(strategy=SimpleNamespace(type_="constant")),
        metrics=metrics,
    )


def _render_latency_table_headers() -> str:
    """Render the latency table and return its headers flattened to text.

    ## WRITTEN BY AI ##
    """
    captured: dict[str, object] = {}
    output = GenerativeBenchmarkerConsole()
    output.console.print = lambda *args, **kwargs: None
    output.console.print_table = lambda headers, values, title=None: captured.update(
        headers=headers
    )
    output.print_request_latency_table(SimpleNamespace(benchmarks=[_make_benchmark()]))

    return " ".join(str(header) for header in captured["headers"])  # type: ignore[union-attr]


class TestRequestLatencyTable:
    """
    Verify which metrics the final console latency table exposes.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_omits_schedule_relative_metrics(self):
        """
        Dispatch delay and scheduled latency are absent from the console table.

        They are derived from targeted_start, which is an ASAP-style value under
        the synchronous, concurrent, and throughput profiles, so they are not
        meaningful in a table shared across every profile.

        ## WRITTEN BY AI ##
        """
        headers = _render_latency_table_headers()

        assert "Dispatch Delay" not in headers
        assert "Scheduled Latency" not in headers

    @pytest.mark.regression
    def test_retains_existing_latency_columns(self):
        """
        The pre-existing latency columns are still rendered unchanged.

        ## WRITTEN BY AI ##
        """
        headers = _render_latency_table_headers()

        for group in ("Request Latency", "TTFT", "TTFOT", "ITL", "TPOT"):
            assert group in headers
