"""Tests for the console benchmark output tables."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from guidellm.benchmark.outputs.console import GenerativeBenchmarkerConsole
from guidellm.schemas import StatusDistributionSummary

# Metrics read by GenerativeBenchmarkerConsole.print_server_throughput_table.
THROUGHPUT_TABLE_METRICS = (
    "request_concurrency",
    "requests_per_second",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "tokens_per_second",
)

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


def _make_throughput_benchmark(
    attainment: float | None, goodput: StatusDistributionSummary | None
) -> SimpleNamespace:
    """Build a benchmark stub exposing every metric the throughput table reads.

    ## WRITTEN BY AI ##
    """
    distribution = StatusDistributionSummary.from_values([1.0, 2.0, 3.0], [], [])
    metrics = SimpleNamespace(
        **dict.fromkeys(THROUGHPUT_TABLE_METRICS, distribution),
        slo_attainment=attainment,
        request_goodput=goodput,
    )

    return SimpleNamespace(
        config=SimpleNamespace(
            strategy=SimpleNamespace(type_="concurrent"),
            slo=None if attainment is None and goodput is None else object(),
        ),
        metrics=metrics,
    )


def _render_throughput_table(benchmarks) -> tuple[str, list]:
    """Render the throughput table, returning flattened headers and value columns.

    ## WRITTEN BY AI ##
    """
    captured: dict[str, object] = {}
    output = GenerativeBenchmarkerConsole()
    output.console.print = lambda *args, **kwargs: None
    output.console.print_table = lambda headers, values, title=None: captured.update(
        headers=headers, values=values
    )
    output.print_server_throughput_table(SimpleNamespace(benchmarks=benchmarks))

    return (
        " ".join(str(header) for header in captured["headers"]),  # type: ignore[union-attr]
        captured["values"],  # type: ignore[return-value]
    )


class TestServerThroughputTableGoodput:
    """
    Verify goodput columns appear only when latency objectives were configured.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_omits_goodput_columns_without_objectives(self):
        """
        A run with no configured objectives renders the table unchanged.

        ## WRITTEN BY AI ##
        """
        headers, _ = _render_throughput_table(
            [_make_throughput_benchmark(attainment=None, goodput=None)]
        )

        assert "Goodput" not in headers
        assert "Attainment" not in headers

    @pytest.mark.regression
    def test_renders_goodput_columns_with_objectives(self):
        """
        A run with objectives adds attainment as a percentage and goodput rate.

        ## WRITTEN BY AI ##
        """
        distribution = StatusDistributionSummary.from_values([1.0, 2.0, 3.0], [], [])
        headers, values = _render_throughput_table(
            [_make_throughput_benchmark(attainment=0.954, goodput=distribution)]
        )

        assert "Goodput" in headers
        assert "Attainment" in headers
        assert any("95.4" in str(column) for column in values)

    @pytest.mark.regression
    def test_shows_columns_when_objectives_cannot_be_evaluated(self):
        """
        Render the goodput columns with empty cells when objectives were
        configured but no request could be evaluated against them.

        Gating on the compiled metric instead would render the table exactly
        as if no objectives had been set, hiding an objective the workload
        cannot measure, such as time to first token without streaming.

        ## WRITTEN BY AI ##
        """
        benchmark = _make_throughput_benchmark(attainment=None, goodput=None)
        benchmark.config.slo = object()
        headers, values = _render_throughput_table([benchmark])

        assert "Goodput" in headers
        assert any("--" in column for column in values)

    @pytest.mark.regression
    def test_keeps_columns_aligned_across_mixed_benchmarks(self):
        """
        Every column holds one value per benchmark even when only some report
        goodput, so rows cannot shift relative to their headers.

        ## WRITTEN BY AI ##
        """
        distribution = StatusDistributionSummary.from_values([1.0, 2.0, 3.0], [], [])
        headers, values = _render_throughput_table(
            [
                _make_throughput_benchmark(attainment=0.99, goodput=distribution),
                _make_throughput_benchmark(attainment=None, goodput=None),
            ]
        )

        # The columns must be present when any benchmark reports goodput, not
        # only when all of them do.
        assert "Goodput" in headers
        assert all(len(column) == 2 for column in values)
        attainment_column = next(column for column in values if "99.0" in column)
        assert attainment_column[1] == "--"
