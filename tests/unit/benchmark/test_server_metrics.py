"""Unit tests for server metrics scraping and summarization."""

from __future__ import annotations

import math

import httpx
import pytest

from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.benchmark.server_metrics import (
    PrometheusServerMetricsCollector,
    ServerMetricsCollector,
    histogram_quantile,
    parse_prometheus_text,
    summarize_snapshots,
)
from guidellm.scheduler import ConcurrentStrategy
from guidellm.schemas.benchmark import PrometheusServerMetricsArgs
from tests.unit.benchmark.html_report_fixtures import make_benchmark

FAMILIES = [
    "vllm:num_requests_running",
    "vllm:prompt_tokens",
    "vllm:time_to_first_token_seconds",
]
LABELS = 'engine="0",model_name="m"'


def _exposition(running: float, tokens: float, buckets: tuple[float, ...]) -> str:
    """Build vLLM-style exposition text with one series per family.

    ## WRITTEN BY AI ##
    """
    bounds = ("0.1", "0.5", "+Inf")
    bucket_lines = "\n".join(
        f'vllm:time_to_first_token_seconds_bucket{{{LABELS},le="{bound}"}} {count}'
        for bound, count in zip(bounds, buckets, strict=True)
    )
    return f"""# HELP vllm:num_requests_running Requests in the running batch.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{{LABELS}}} {running}
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{{LABELS}}} {tokens}
# TYPE vllm:prompt_tokens_created gauge
vllm:prompt_tokens_created{{{LABELS}}} 1700000000.0
# TYPE vllm:time_to_first_token_seconds histogram
{bucket_lines}
vllm:time_to_first_token_seconds_count{{{LABELS}}} {buckets[-1]}
vllm:time_to_first_token_seconds_sum{{{LABELS}}} {buckets[-1] * 0.2}
"""


class TestParsePrometheusText:
    @pytest.mark.smoke
    def test_parses_each_metric_type(self):
        """Read gauges, counters declared with _total, and histogram parts.

        ## WRITTEN BY AI ##
        """
        snapshot = parse_prometheus_text(
            _exposition(running=3, tokens=100, buckets=(2, 5, 6)), FAMILIES, 10.0
        )
        key_labels = (("engine", "0"), ("model_name", "m"))

        assert snapshot.timestamp == 10.0
        assert snapshot.gauges == {("vllm:num_requests_running", key_labels): 3.0}
        assert snapshot.counters == {("vllm:prompt_tokens", key_labels): 100.0}
        histogram = snapshot.histograms[
            ("vllm:time_to_first_token_seconds", key_labels)
        ]
        assert histogram.buckets == {0.1: 2.0, 0.5: 5.0, math.inf: 6.0}
        assert histogram.count == 6.0
        assert histogram.sum == pytest.approx(1.2)

    @pytest.mark.sanity
    def test_skips_unconfigured_untyped_and_nan_samples(self):
        """Ignore similarly named families, samples without a TYPE line, and NaN.

        ## WRITTEN BY AI ##
        """
        text = """# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total 9
vllm:num_requests_running 4
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total NaN
"""
        snapshot = parse_prometheus_text(text, FAMILIES, 1.0)

        assert snapshot.gauges == {}
        assert snapshot.counters == {}

    @pytest.mark.sanity
    def test_parses_escaped_label_values_and_timestamps(self):
        """Unescape quoted label values and ignore a trailing sample timestamp.

        ## WRITTEN BY AI ##
        """
        text = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="a \\"quoted\\", b"} 2 1700000000000
"""
        snapshot = parse_prometheus_text(text, FAMILIES, 1.0)

        assert snapshot.gauges == {
            ("vllm:num_requests_running", (("model_name", 'a "quoted", b'),)): 2.0
        }

    @pytest.mark.regression
    def test_skips_malformed_lines_exemplars_and_infinite_values(self):
        """Keep parsing past bad lines, and drop exemplars and infinite gauges.

        ## WRITTEN BY AI ##
        """
        text = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running
vllm:num_requests_running{model_name="m"} not-a-number
vllm:num_requests_running{model_name="inf"} +Inf
vllm:num_requests_running{model_name="m"} 2
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 3 # {trace_id="a"} 0.5
"""
        snapshot = parse_prometheus_text(text, FAMILIES, 1.0)

        assert snapshot.gauges == {
            ("vllm:num_requests_running", (("model_name", "m"),)): 2.0
        }
        histogram = snapshot.histograms[("vllm:time_to_first_token_seconds", ())]
        assert histogram.buckets == {math.inf: 3.0}


class TestHistogramQuantile:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.25, 0.075), (0.5, 0.1 + 0.4 / 3), (0.99, 0.5)],
    )
    def test_interpolates_within_bucket(self, quantile: float, expected: float):
        """Interpolate linearly and fall back to the top finite bound for +Inf.

        ## WRITTEN BY AI ##
        """
        buckets = {0.1: 2.0, 0.5: 5.0, math.inf: 6.0}

        assert histogram_quantile(quantile, buckets) == pytest.approx(expected)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "buckets",
        [{}, {0.1: 0.0, math.inf: 0.0}, {0.1: 3.0}],
    )
    def test_returns_none_without_observations(self, buckets: dict[float, float]):
        """Return None for empty histograms or ones missing the +Inf bucket.

        ## WRITTEN BY AI ##
        """
        assert histogram_quantile(0.5, buckets) is None


class TestSummarizeSnapshots:
    @pytest.mark.smoke
    def test_counter_increase_is_interpolated_at_window_edges(self):
        """Interpolate counters between the scrapes around each window edge.

        ## WRITTEN BY AI ##
        """
        snapshots = [
            parse_prometheus_text(_exposition(0, tokens, (0, 0, 0)), FAMILIES, t)
            for t, tokens in [(0.0, 0.0), (10.0, 100.0), (20.0, 300.0)]
        ]

        summary = summarize_snapshots("src", snapshots, 0, 5.0, 15.0)
        counter = summary.counters["vllm:prompt_tokens"][0]

        # 50 tokens at t=5 and 200 at t=15.
        assert counter.increase == pytest.approx(150.0)
        assert counter.rate == pytest.approx(15.0)
        assert counter.labels == {"engine": "0", "model_name": "m"}

    @pytest.mark.sanity
    def test_counter_reset_is_corrected(self):
        """Treat a drop in a counter as a server restart, not a negative increase.

        ## WRITTEN BY AI ##
        """
        snapshots = [
            parse_prometheus_text(_exposition(0, tokens, (0, 0, 0)), FAMILIES, t)
            for t, tokens in [(0.0, 100.0), (10.0, 150.0), (20.0, 30.0)]
        ]

        summary = summarize_snapshots("src", snapshots, 0, 0.0, 20.0)

        assert summary.counters["vllm:prompt_tokens"][0].increase == 80.0

    @pytest.mark.regression
    def test_series_first_seen_mid_run_counts_from_zero(self):
        """Count a series that appears mid-run from zero, not from its first value.

        ## WRITTEN BY AI ##
        """
        empty = parse_prometheus_text("", FAMILIES, 0.0)
        snapshots = [empty] + [
            parse_prometheus_text(_exposition(0, tokens, buckets), FAMILIES, t)
            for t, tokens, buckets in [(10.0, 5.0, (1, 2, 2)), (20.0, 15.0, (2, 4, 4))]
        ]

        summary = summarize_snapshots("src", snapshots, 0, 0.0, 20.0)

        assert summary.counters["vllm:prompt_tokens"][0].increase == 15.0
        assert summary.histograms["vllm:time_to_first_token_seconds"][0].count == 4.0

    @pytest.mark.smoke
    def test_gauges_keep_samples_within_window(self):
        """Summarize only the gauge values scraped inside the window.

        ## WRITTEN BY AI ##
        """
        snapshots = [
            parse_prometheus_text(_exposition(running, 0, (0, 0, 0)), FAMILIES, t)
            for t, running in [(0.0, 100.0), (5.0, 2.0), (10.0, 4.0), (15.0, 100.0)]
        ]

        summary = summarize_snapshots("src", snapshots, 0, 4.0, 11.0)
        gauge = summary.gauges["vllm:num_requests_running"][0]

        assert gauge.samples == [(5.0, 2.0), (10.0, 4.0)]
        assert gauge.summary.mean == pytest.approx(3.0)
        assert gauge.summary.max == 4.0

    @pytest.mark.smoke
    def test_histogram_reports_window_delta(self):
        """Report count, sum, mean and quantiles from bucket increases.

        ## WRITTEN BY AI ##
        """
        snapshots = [
            parse_prometheus_text(_exposition(0, 0, buckets), FAMILIES, t)
            for t, buckets in [(0.0, (1, 1, 1)), (10.0, (3, 6, 7))]
        ]

        summary = summarize_snapshots("src", snapshots, 2, 0.0, 10.0)
        histogram = summary.histograms["vllm:time_to_first_token_seconds"][0]

        assert summary.scrapes == 2
        assert summary.scrape_errors == 2
        assert histogram.count == 6.0
        assert histogram.sum == pytest.approx(1.2)
        assert histogram.mean == pytest.approx(0.2)
        # Window buckets are 2, 5 and 6, so the median falls in (0.1, 0.5].
        assert histogram.quantiles["p50"] == pytest.approx(0.1 + 0.4 / 3)
        assert histogram.quantiles["p99"] == 0.5


class TestPrometheusServerMetricsCollector:
    @pytest.mark.smoke
    def test_resolves_from_args(self):
        """Build a Prometheus collector from its registered arguments.

        ## WRITTEN BY AI ##
        """
        args = PrometheusServerMetricsArgs(url="http://server/metrics", interval=2.0)

        collector = ServerMetricsCollector.resolve(args)

        assert isinstance(collector, PrometheusServerMetricsCollector)
        assert collector.url == "http://server/metrics"
        assert collector.interval == 2.0
        assert collector.metrics == args.metrics

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_scrapes_at_start_and_stop(self):
        """Scrape when a benchmark starts and again when it stops.

        ## WRITTEN BY AI ##
        """
        tokens = iter([10.0, 40.0])
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, text=_exposition(1, next(tokens), (0, 0, 0)))

        collector = PrometheusServerMetricsCollector(
            url="http://server/metrics",
            interval=3600.0,
            timeout=1.0,
            metrics=FAMILIES,
            transport=httpx.MockTransport(respond),
        )

        await collector.start()
        await collector.stop()
        summary = collector.summarize(0.0, 2**40)

        assert len(requests) == 2
        assert requests[0].headers["accept"].startswith("text/plain")
        assert summary.scrapes == 2
        assert summary.counters["vllm:prompt_tokens"][0].increase == 30.0

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_failed_scrapes_are_counted_not_raised(self):
        """Count HTTP errors and connection failures without failing the run.

        ## WRITTEN BY AI ##
        """
        outcomes = iter(["error", "down"])

        def respond(request: httpx.Request) -> httpx.Response:
            if next(outcomes) == "error":
                return httpx.Response(503)
            raise httpx.ConnectError("refused", request=request)

        collector = PrometheusServerMetricsCollector(
            url="http://server/metrics",
            interval=3600.0,
            timeout=1.0,
            metrics=FAMILIES,
            transport=httpx.MockTransport(respond),
        )

        await collector.start()
        await collector.stop()
        summary = collector.summarize(0.0, 1.0)

        assert summary.scrapes == 0
        assert summary.scrape_errors == 2
        assert summary.counters == {}

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_start_discards_previous_benchmark(self):
        """Start each benchmark with no scrapes or errors from the previous one.

        ## WRITTEN BY AI ##
        """
        statuses = iter([503, 503, 200, 200])

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(next(statuses), text=_exposition(1, 5, (0, 0, 0)))

        collector = PrometheusServerMetricsCollector(
            url="http://server/metrics",
            interval=3600.0,
            timeout=1.0,
            metrics=FAMILIES,
            transport=httpx.MockTransport(respond),
        )

        for _ in range(2):
            await collector.start()
            await collector.stop()
        summary = collector.summarize(0.0, 2**40)

        assert summary.scrapes == 2
        assert summary.scrape_errors == 0


@pytest.mark.regression
def test_server_metrics_survive_report_round_trip():
    """Serialize and reload a benchmark carrying server metrics unchanged.

    ## WRITTEN BY AI ##
    """
    snapshots = [
        parse_prometheus_text(_exposition(running, tokens, buckets), FAMILIES, t)
        for t, running, tokens, buckets in [
            (0.0, 1.0, 0.0, (0, 0, 0)),
            (10.0, 3.0, 50.0, (2, 5, 6)),
        ]
    ]
    summary = summarize_snapshots("http://server/metrics", snapshots, 1, 0.0, 10.0)
    benchmark = make_benchmark(
        strategy=ConcurrentStrategy(streams=1), rps=1.0, tps=1.0
    ).model_copy(update={"server_metrics": [summary]})

    reloaded = GenerativeBenchmark.model_validate_json(benchmark.model_dump_json())

    assert reloaded.server_metrics == [summary]
