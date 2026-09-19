"""Tests for the console benchmark output tables."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from guidellm.benchmark.outputs.console import (
    UNSUPPORTED_PERCENTILE_FOOTNOTE,
    UNSUPPORTED_PERCENTILE_MARKER,
    ConsoleTableColumnsCollection,
    GenerativeBenchmarkerConsole,
)
from guidellm.schemas import (
    ConfidenceInterval,
    DistributionSummary,
    Percentiles,
    SampleUncertainty,
    StatusBreakdown,
    StatusDistributionSummary,
)

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


def _token_status_totals(total_sum: float) -> SimpleNamespace:
    """Build a minimal status breakdown stub with only total_sum populated.

    ## WRITTEN BY AI ##
    """
    return SimpleNamespace(
        successful=SimpleNamespace(total_sum=total_sum),
        incomplete=SimpleNamespace(total_sum=0.0),
        errored=SimpleNamespace(total_sum=0.0),
    )


def _make_run_summary_benchmark(
    *,
    successful: int = 8,
    incomplete: int = 2,
    errored: int = 1,
) -> SimpleNamespace:
    """Build a benchmark stub exposing every metric the run summary table reads.

    ## WRITTEN BY AI ##
    """
    return SimpleNamespace(
        config=SimpleNamespace(strategy=SimpleNamespace(type_="constant")),
        start_time=1_700_000_000.0,
        end_time=1_700_000_060.0,
        duration=60.0,
        warmup_duration=5.0,
        cooldown_duration=2.0,
        metrics=SimpleNamespace(
            request_totals=StatusBreakdown(
                successful=successful,
                incomplete=incomplete,
                errored=errored,
                total=successful + incomplete + errored,
            ),
            prompt_token_count=_token_status_totals(100.0),
            output_token_count=_token_status_totals(200.0),
        ),
    )


def _render_run_summary_table(benchmarks) -> tuple[str, list]:
    """Render the run summary table, returning flattened headers and value columns.

    ## WRITTEN BY AI ##
    """
    captured: dict[str, object] = {}
    output = GenerativeBenchmarkerConsole()
    output.console.print = lambda *args, **kwargs: None
    output.console.print_table = lambda headers, values, title=None: captured.update(
        headers=headers, values=values
    )
    output.print_run_summary_table(SimpleNamespace(benchmarks=benchmarks))

    return (
        " ".join(str(header) for header in captured["headers"]),  # type: ignore[union-attr]
        captured["values"],  # type: ignore[return-value]
    )


class TestRunSummaryTable:
    """
    Verify request totals appear in the console run summary table.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_renders_request_totals_by_status(self):
        """
        Successful, incomplete, and errored request counts are shown in the
        first console table alongside existing timing and token columns.

        ## WRITTEN BY AI ##
        """
        headers, values = _render_run_summary_table([_make_run_summary_benchmark()])

        assert "Requests" in headers
        for name in ("Comp", "Inc", "Err"):
            assert name in headers
        assert "Timings" in headers
        assert "Input Tokens" in headers
        assert "Output Tokens" in headers
        assert any("8" in column for column in values)
        assert any("2" in column for column in values)
        assert any("1" in column for column in values)


def _render_latency_table_values(
    with_intervals: bool = True,
) -> tuple[list[list[str]], list[list[str]]]:
    """Render the latency table and return its headers and cell values.

    ## WRITTEN BY AI ##
    """
    uncertainty = SampleUncertainty(confidence=0.95) if with_intervals else None
    samples = [float(index) for index in range(200)]
    per_request = StatusDistributionSummary.from_values(
        samples, [], [], uncertainty=uncertainty
    )
    # Weighted values stand in for ITL and TPOT, which carry an exposure weight
    # and so are reported without an interval.
    weighted = StatusDistributionSummary.from_values(
        [(value, 1.0 + index % 5) for index, value in enumerate(samples)],
        [],
        [],
        uncertainty=uncertainty,
    )
    metrics = SimpleNamespace(
        request_latency=per_request,
        time_to_first_token_ms=per_request,
        time_to_first_output_token_ms=per_request,
        inter_token_latency_ms=weighted,
        time_per_output_token_ms=weighted,
    )
    benchmark = SimpleNamespace(
        config=SimpleNamespace(strategy=SimpleNamespace(type_="constant")),
        metrics=metrics,
    )

    captured: dict[str, object] = {}
    output = GenerativeBenchmarkerConsole()
    output.console.print = lambda *args, **kwargs: None
    output.console.print_table = lambda headers, values, title=None: captured.update(
        headers=headers, values=values
    )
    output.print_request_latency_table(SimpleNamespace(benchmarks=[benchmark]))

    return captured["headers"], captured["values"]  # type: ignore[return-value]


class TestRequestLatencyTableIntervals:
    """
    Verify how the latency table renders a mean alongside its margin.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_mean_column_carries_a_margin(self):
        """
        A per-request metric renders as a mean followed by its margin.

        ## WRITTEN BY AI ##
        """
        headers, values = _render_latency_table_values()

        mean_columns = [
            index for index, header in enumerate(headers) if header[-1] == "Mean"
        ]
        assert mean_columns
        rendered = [values[index][0] for index in mean_columns]
        assert any("±" in cell for cell in rendered)

    @pytest.mark.sanity
    def test_weighted_metrics_render_without_a_margin(self):
        """
        ITL and TPOT show the mean alone, since they carry no interval.

        ## WRITTEN BY AI ##
        """
        headers, values = _render_latency_table_values()

        for index, header in enumerate(headers):
            if header[0] in ("ITL", "TPOT") and header[-1] == "Mean":
                assert "±" not in values[index][0]

    @pytest.mark.sanity
    def test_margin_is_omitted_when_no_interval_was_estimated(self):
        """
        Without an estimator the column shows the bare mean.

        ## WRITTEN BY AI ##
        """
        headers, values = _render_latency_table_values(with_intervals=False)

        for index, header in enumerate(headers):
            if header[-1] == "Mean":
                assert "±" not in values[index][0]

    @pytest.mark.regression
    def test_margin_keeps_enough_precision_to_be_visible(self):
        """
        A margin smaller than the column precision is not rendered as zero.

        Rounding it away would present an imprecise measurement as an exact one.

        ## WRITTEN BY AI ##
        """
        distribution = DistributionSummary(
            mean=1.0,
            median=1.0,
            mode=1.0,
            variance=0.0,
            std_dev=0.0,
            min=1.0,
            max=1.0,
            count=10,
            total_sum=10.0,
            percentiles=Percentiles(
                **dict.fromkeys(
                    [
                        "p001",
                        "p01",
                        "p05",
                        "p10",
                        "p25",
                        "p50",
                        "p75",
                        "p90",
                        "p95",
                        "p99",
                        "p999",
                    ],
                    1.0,
                )
            ),
            mean_ci=ConfidenceInterval(lower=0.98, upper=1.02),
        )

        rendered = ConsoleTableColumnsCollection._format_mean_with_margin(
            distribution, precision=1
        )

        assert rendered == "1.0 ±0.02"


def _render_latency_table_with_footnote(
    sample_size: int,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Render the latency table for a sample of the given size.

    ## WRITTEN BY AI ##
    """
    samples = [float(index) for index in range(sample_size)]
    distribution = StatusDistributionSummary.from_values(
        samples, [], [], uncertainty=SampleUncertainty(confidence=0.95)
    )
    metrics = SimpleNamespace(
        **dict.fromkeys(LATENCY_TABLE_METRICS, distribution),
    )
    benchmark = SimpleNamespace(
        config=SimpleNamespace(strategy=SimpleNamespace(type_="constant")),
        metrics=metrics,
    )

    captured: dict[str, object] = {}
    printed: list[str] = []
    output = GenerativeBenchmarkerConsole()
    output.console.print = lambda *args, **kwargs: printed.extend(
        str(arg) for arg in args
    )
    output.console.print_table = lambda headers, values, title=None: captured.update(
        headers=headers, values=values
    )
    output.print_request_latency_table(SimpleNamespace(benchmarks=[benchmark]))

    return captured["headers"], captured["values"], printed  # type: ignore[return-value]


class TestUnsupportedPercentileMarker:
    """
    Verify the console marks a percentile the sample cannot bound.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _p95_cells(headers, values) -> list[str]:
        """Collect the rendered p95 cells.

        ## WRITTEN BY AI ##
        """
        return [
            values[index][0]
            for index, header in enumerate(headers)
            if header[-1] == "p95"
        ]

    @pytest.mark.smoke
    def test_marks_p95_below_the_supporting_sample_size(self):
        """
        A sample too small to bound p95 renders the value with a marker.

        Seventy-one observations cannot place an observation above the 95th
        percentile often enough to form a two-sided interval, so the point
        estimate stands alone and says so.

        ## WRITTEN BY AI ##
        """
        headers, values, printed = _render_latency_table_with_footnote(71)

        cells = self._p95_cells(headers, values)
        assert cells
        assert all(cell.endswith(UNSUPPORTED_PERCENTILE_MARKER) for cell in cells)
        assert UNSUPPORTED_PERCENTILE_FOOTNOTE in printed

    @pytest.mark.sanity
    def test_leaves_p95_unmarked_once_the_sample_supports_it(self):
        """
        One more observation removes the marker and the footnote.

        ## WRITTEN BY AI ##
        """
        headers, values, printed = _render_latency_table_with_footnote(72)

        cells = self._p95_cells(headers, values)
        assert cells
        assert not any(cell.endswith(UNSUPPORTED_PERCENTILE_MARKER) for cell in cells)
        assert UNSUPPORTED_PERCENTILE_FOOTNOTE not in printed

    @pytest.mark.regression
    def test_metrics_without_intervals_are_not_marked(self):
        """
        A metric reported without intervals is not marked as sample-limited.

        ITL and TPOT carry no interval by design rather than for want of
        observations, so marking them would misattribute the reason.

        ## WRITTEN BY AI ##
        """
        headers, values = _render_latency_table_values(with_intervals=True)

        for index, header in enumerate(headers):
            if header[0] in ("ITL", "TPOT") and header[-1] == "p95":
                assert not values[index][0].endswith(UNSUPPORTED_PERCENTILE_MARKER)
