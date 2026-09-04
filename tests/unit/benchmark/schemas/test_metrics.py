"""
Tests for metrics compilation, specifically verifying that tool call metrics
tables appear when all expected tool call requests errored.
"""

from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.accumulator import GenerativeBenchmarkAccumulator
from guidellm.benchmark.schemas.base import BenchmarkConfig
from guidellm.benchmark.schemas.metrics import (
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeToolCallMetricsSummary,
)
from guidellm.scheduler import (
    AsyncConstantStrategy,
    SchedulingStrategy,
    ThroughputStrategy,
)
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    StatusDistributionSummary,
    UsageMetrics,
)
from guidellm.schemas.benchmark import GoodputSLO


def _make_errored_tool_call_stats(request_id: str) -> GenerativeRequestStats:
    """Build a GenerativeRequestStats representing an errored tool-call request.

    The output_metrics.tool_call_count is 0 (expected but failed) and all
    other output fields are None, matching the behaviour of
    ``GenerationResponse.compile_stats`` for errored requests with
    ``turn_type='client_tool_call'``.
    """
    timings = RequestTimings(
        resolve_start=1.0,
        resolve_end=2.0,
        request_start=1.0,
        request_end=2.0,
    )
    return GenerativeRequestStats(
        request_id=request_id,
        info=RequestInfo(request_id=request_id, status="errored", timings=timings),
        input_metrics=UsageMetrics(),
        output_metrics=UsageMetrics(tool_call_count=0),
    )


class TestToolCallMetricsAllErrored:
    """
    Verify that tool call metrics are non-None when all requests
    expected tool calls but errored.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_compile_timed_metrics_zero_values_produces_distribution(self):
        """
        compile_timed_metrics returns a non-None summary when all values
        are zero (not None), representing errored tool-call requests.

        ## WRITTEN BY AI ##
        """
        errored_metrics = [
            (1.0, 2.0, None, 0),
            (2.0, 3.0, None, 0),
            (3.0, 4.0, None, 0),
        ]

        result = GenerativeMetricsSummary.compile_timed_metrics(
            successful=[],
            incomplete=[],
            errored=errored_metrics,
        )

        assert result is not None
        # Input is always None for tool_call_count (not applicable)
        assert result.input is None
        # Output should have a valid distribution with errored data
        assert result.output is not None
        assert result.output.errored is not None
        assert result.output.errored.count == 3
        assert result.output.errored.mean == 0.0

    @pytest.mark.smoke
    def test_compile_timed_metrics_empty_lists_all_fields_none(self):
        """
        compile_timed_metrics with empty lists produces a summary where
        all distribution fields are None, indicating no data.

        ## WRITTEN BY AI ##
        """
        result = GenerativeMetricsSummary.compile_timed_metrics(
            successful=[],
            incomplete=[],
            errored=[],
        )
        assert result is not None
        assert result.input is None
        assert result.output is None
        assert result.total is None

    @pytest.mark.sanity
    def test_tool_call_summary_compile_all_errored(self):
        """
        GenerativeToolCallMetricsSummary.compile produces a non-None
        count metric when all tool-call requests errored with
        tool_call_count=0.

        ## WRITTEN BY AI ##
        """
        errored = [
            _make_errored_tool_call_stats("err-1"),
            _make_errored_tool_call_stats("err-2"),
        ]

        summary = GenerativeToolCallMetricsSummary.compile(
            successful=[],
            incomplete=[],
            errored=errored,
        )

        assert summary.count is not None
        assert summary.count.output is not None
        assert summary.count.output.errored is not None
        assert summary.count.output.errored.count == 2
        assert summary.count.output.errored.mean == 0.0

    @pytest.mark.sanity
    def test_tool_call_summary_compile_no_tool_calls(self):
        """
        GenerativeToolCallMetricsSummary.compile produces metrics with
        all-None distribution fields when no requests had tool call data
        (tool calls not applicable).

        ## WRITTEN BY AI ##
        """
        timings = RequestTimings(
            resolve_start=1.0,
            resolve_end=2.0,
            request_start=1.0,
            request_end=2.0,
        )
        plain_stats = GenerativeRequestStats(
            request_id="plain-1",
            info=RequestInfo(request_id="plain-1", status="completed", timings=timings),
            input_metrics=UsageMetrics(text_tokens=10),
            output_metrics=UsageMetrics(text_tokens=20),
        )

        summary = GenerativeToolCallMetricsSummary.compile(
            successful=[plain_stats],
            incomplete=[],
            errored=[],
        )

        # All sub-metrics exist but have no distribution data
        assert summary.count is not None
        assert summary.count.input is None
        assert summary.count.output is None

        assert summary.tokens is not None
        assert summary.tokens.input is None
        assert summary.tokens.output is None

        assert summary.mixed_tokens is not None
        assert summary.mixed_tokens.input is None
        assert summary.mixed_tokens.output is None


@pytest.mark.sanity
def test_round_trip_metrics_compile():
    """
    WebSocket round-trip metrics expose GenerativeMetrics fields and compile
    into StatusDistributionSummary distributions from request timings.

    ## WRITTEN BY AI ##
    """
    timings = RequestTimings(
        resolve_start=0.0,
        resolve_end=2.0,
        request_start=0.0,
        request_end=2.0,
        last_request_sent=0.2,
        last_token_iteration=0.5,
        request_sent_sum=0.3,  # mean 0.1 over 3 sends
        request_sent_count=3,
        token_received_sum=1.2,  # mean 0.4 over 3 receives
        token_received_count=3,
    )
    stats = GenerativeRequestStats(
        request_id="rtt",
        info=RequestInfo(request_id="rtt", status="completed", timings=timings),
        input_metrics=UsageMetrics(),
        output_metrics=UsageMetrics(),
    )

    # Schema exposes the new metric fields.
    assert "time_to_last_round_trip_ms" in GenerativeMetrics.model_fields
    assert "avg_round_trip_time_ms" in GenerativeMetrics.model_fields

    # The same compile expressions used in GenerativeMetrics.compile().
    last_round_trip = StatusDistributionSummary.from_values_function(
        function=lambda req: req.time_to_last_round_trip_ms or 0.0,
        successful=[stats],
        incomplete=[],
        errored=[],
    )
    avg_round_trip = StatusDistributionSummary.from_values_function(
        function=lambda req: req.avg_round_trip_time_ms or 0.0,
        successful=[stats],
        incomplete=[],
        errored=[],
    )

    assert last_round_trip.successful.mean == pytest.approx(300.0, abs=0.1)
    assert avg_round_trip.successful.mean == pytest.approx(300.0, abs=0.1)


def _make_scheduled_stats(
    request_id: str, targeted_start: float, request_start: float, request_end: float
) -> GenerativeRequestStats:
    """Build a completed request with an explicit targeted start time.

    ## WRITTEN BY AI ##
    """
    timings = RequestTimings(
        targeted_start=targeted_start,
        resolve_start=request_start,
        resolve_end=request_end,
        request_start=request_start,
        request_end=request_end,
    )
    return GenerativeRequestStats(
        request_id=request_id,
        info=RequestInfo(request_id=request_id, status="completed", timings=timings),
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=8),
    )


# Non-zero epoch base; a measurement window starting at 0.0 reads as unset.
SCHEDULE_BASE_TIME = 1000.0


def _make_accumulator(
    successful: list[GenerativeRequestStats],
    start_time: float,
    end_time: float,
    strategy: SchedulingStrategy | None = None,
) -> GenerativeBenchmarkAccumulator:
    """Build an accumulator holding completed requests over a measurement window.

    Defaults to a constant-rate strategy, which defines an arrival schedule.

    ## WRITTEN BY AI ##
    """
    accumulator = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="schedule-metrics",
            run_index=0,
            strategy=strategy or AsyncConstantStrategy(rate=10.0),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
        )
    )
    accumulator.timings.measure_start = start_time
    accumulator.timings.measure_end = end_time
    accumulator.completed.requests_stats = list(successful)

    return accumulator


class TestScheduleRelativeMetrics:
    """
    Verify the schedule-relative distributions added alongside request_latency.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_schedule_fields_are_optional_for_existing_reports(self):
        """
        The schedule-relative fields carry defaults so reports written before
        they existed still validate.

        ## WRITTEN BY AI ##
        """
        successful = [
            _make_scheduled_stats(
                "req",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + 1.0,
                SCHEDULE_BASE_TIME + 1.5,
            )
        ]
        metrics = GenerativeMetrics.compile(
            _make_accumulator(successful, SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 10.0)
        )

        # Strip the new keys to mimic a report written before they existed.
        payload = metrics.model_dump()
        for name in ("request_dispatch_delay", "request_scheduled_latency"):
            del payload[name]

        restored = GenerativeMetrics.model_validate(payload)

        assert restored.request_dispatch_delay is None
        assert restored.request_scheduled_latency is None
        assert restored.request_latency.successful.mean == pytest.approx(0.5)

    @pytest.mark.sanity
    def test_compile_distributions_track_scheduler_backlog(self):
        """
        Compiled dispatch delay reflects how far behind its targeted start each
        request was issued, while request latency stays flat.

        ## WRITTEN BY AI ##
        """
        # Constant 0.5s service time, dispatched 0s / 1s / 2s behind schedule.
        successful = [
            _make_scheduled_stats(
                f"req-{index}",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + index,
                SCHEDULE_BASE_TIME + index + 0.5,
            )
            for index in range(3)
        ]
        metrics = GenerativeMetrics.compile(
            _make_accumulator(successful, SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 10.0)
        )

        assert metrics.request_latency.successful.mean == pytest.approx(0.5)
        assert metrics.request_dispatch_delay.successful.mean == pytest.approx(1.0)
        assert metrics.request_scheduled_latency.successful.mean == pytest.approx(1.5)
        assert metrics.request_scheduled_latency.successful.max == pytest.approx(2.5)

    @pytest.mark.sanity
    def test_compile_reports_percentiles_for_schedule_metrics(self):
        """
        Compiled schedule-relative metrics carry percentiles, so a tail hidden
        from request_latency is visible in the report.

        ## WRITTEN BY AI ##
        """
        # Steady 0.1s service time with dispatch falling a second further behind.
        successful = [
            _make_scheduled_stats(
                f"req-{index}",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + index,
                SCHEDULE_BASE_TIME + index + 0.1,
            )
            for index in range(100)
        ]
        metrics = GenerativeMetrics.compile(
            _make_accumulator(
                successful, SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 200.0
            )
        )

        assert metrics.request_scheduled_latency is not None
        latency = metrics.request_latency.successful
        scheduled = metrics.request_scheduled_latency.successful

        assert latency.percentiles.p99 == pytest.approx(0.1)
        assert scheduled.percentiles.p99 > 90.0

    @pytest.mark.regression
    def test_compile_scheduled_latency_decomposes_across_requests(self):
        """
        Every compiled request satisfies scheduled latency equal to dispatch
        delay plus request latency.

        ## WRITTEN BY AI ##
        """
        successful = [
            _make_scheduled_stats(
                "even",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + 2.0,
                SCHEDULE_BASE_TIME + 2.75,
            ),
            _make_scheduled_stats(
                "odd",
                SCHEDULE_BASE_TIME + 1.0,
                SCHEDULE_BASE_TIME + 4.5,
                SCHEDULE_BASE_TIME + 5.0,
            ),
        ]
        metrics = GenerativeMetrics.compile(
            _make_accumulator(successful, SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 10.0)
        )

        assert metrics.request_dispatch_delay is not None
        assert metrics.request_scheduled_latency is not None
        assert metrics.request_scheduled_latency.successful.mean == pytest.approx(
            metrics.request_dispatch_delay.successful.mean
            + metrics.request_latency.successful.mean
        )

    @pytest.mark.sanity
    def test_compile_omits_metrics_for_asap_strategies(self):
        """
        Strategies without an arrival schedule report None.

        Under throughput every target is the benchmark start time, so populated
        values would report elapsed run time rather than a delay or a latency.

        ## WRITTEN BY AI ##
        """
        successful = [
            _make_scheduled_stats(
                f"req-{index}",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + index,
                SCHEDULE_BASE_TIME + index + 0.5,
            )
            for index in range(3)
        ]
        metrics = GenerativeMetrics.compile(
            _make_accumulator(
                successful,
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + 10.0,
                strategy=ThroughputStrategy(),
            )
        )

        # None rather than a zero-filled distribution, which would read as
        # "no delay measured" instead of "not applicable".
        assert metrics.request_dispatch_delay is None
        assert metrics.request_scheduled_latency is None
        # Existing metrics are unaffected by the gating.
        assert metrics.request_latency.successful.mean == pytest.approx(0.5)

    @pytest.mark.regression
    def test_compile_skips_requests_without_a_dispatch_timestamp(self):
        """
        Requests that never reached dispatch are excluded, not counted as zero.

        A request cancelled while the scheduler was backed up has no
        request_start, so its delay is unknown. Recording it as 0.0 would drag
        the distribution toward zero for exactly the requests these metrics
        exist to describe.

        ## WRITTEN BY AI ##
        """
        dispatched = _make_scheduled_stats(
            "dispatched",
            SCHEDULE_BASE_TIME,
            SCHEDULE_BASE_TIME + 4.0,
            SCHEDULE_BASE_TIME + 4.5,
        )
        never_dispatched = GenerativeRequestStats(
            request_id="never-dispatched",
            info=RequestInfo(
                request_id="never-dispatched",
                status="completed",
                timings=RequestTimings(
                    targeted_start=SCHEDULE_BASE_TIME,
                    resolve_start=SCHEDULE_BASE_TIME + 4.0,
                    resolve_end=SCHEDULE_BASE_TIME + 4.5,
                ),
            ),
            input_metrics=UsageMetrics(text_tokens=8),
            output_metrics=UsageMetrics(text_tokens=8),
        )
        assert never_dispatched.request_dispatch_delay is None

        metrics = GenerativeMetrics.compile(
            _make_accumulator(
                [dispatched, never_dispatched],
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + 10.0,
            )
        )

        assert metrics.request_dispatch_delay is not None
        assert metrics.request_dispatch_delay.successful.count == 1
        assert metrics.request_dispatch_delay.successful.mean == pytest.approx(4.0)


def _make_latency_stats(
    request_id: str,
    request_start: float,
    first_token: float,
    request_end: float,
    output_tokens: int = 9,
) -> GenerativeRequestStats:
    """Build a completed streaming request with explicit token timings.

    ## WRITTEN BY AI ##
    """
    timings = RequestTimings(
        resolve_start=request_start,
        resolve_end=request_end,
        request_start=request_start,
        request_end=request_end,
        first_token_iteration=first_token,
        last_token_iteration=request_end,
        token_iterations=output_tokens,
    )
    return GenerativeRequestStats(
        request_id=request_id,
        info=RequestInfo(request_id=request_id, status="completed", timings=timings),
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=output_tokens),
    )


def _make_errored_stats(request_id: str, start: float) -> GenerativeRequestStats:
    """
    Build an errored request with no usable output metrics.

    ## WRITTEN BY AI ##
    """
    timings = RequestTimings(
        resolve_start=start,
        resolve_end=start + 1.0,
        request_start=start,
        request_end=start + 1.0,
    )
    return GenerativeRequestStats(
        request_id=request_id,
        info=RequestInfo(request_id=request_id, status="errored", timings=timings),
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(),
    )


def _make_goodput_accumulator(
    successful: list[GenerativeRequestStats],
    start_time: float,
    end_time: float,
    slo: GoodputSLO | None,
) -> GenerativeBenchmarkAccumulator:
    """Build an accumulator carrying latency objectives on its config.

    ## WRITTEN BY AI ##
    """
    accumulator = _make_accumulator(successful, start_time, end_time)
    accumulator.config.slo = slo

    return accumulator


class TestGoodputMetrics:
    """
    Verify goodput attainment and rate compiled against latency objectives.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_no_objectives_reports_nothing(self):
        """
        Leave goodput unset when no objectives were configured.

        ## WRITTEN BY AI ##
        """
        successful = [
            _make_latency_stats(
                "a",
                SCHEDULE_BASE_TIME,
                SCHEDULE_BASE_TIME + 0.1,
                SCHEDULE_BASE_TIME + 1.0,
            )
        ]
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                successful, SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 10.0, None
            )
        )

        assert metrics.slo_attainment is None
        assert metrics.request_goodput is None
        assert metrics.slo_determined_requests == 0

    @pytest.mark.sanity
    def test_attainment_is_the_conforming_fraction(self):
        """
        Report attainment as the share of requests meeting every objective.

        Three of four requests finish within a 2s end-to-end objective.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        successful = [
            _make_latency_stats("a", base, base + 0.1, base + 1.0),
            _make_latency_stats("b", base, base + 0.1, base + 1.5),
            _make_latency_stats("c", base, base + 0.1, base + 1.9),
            _make_latency_stats("d", base, base + 0.1, base + 5.0),
        ]
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                successful, base, base + 10.0, GoodputSLO(e2el_ms=2000)
            )
        )

        assert metrics.slo_attainment == pytest.approx(0.75)
        assert metrics.slo_determined_requests == 4

    @pytest.mark.sanity
    def test_goodput_never_exceeds_throughput(self):
        """
        Report a conforming-request rate at or below the overall request rate.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        successful = [
            _make_latency_stats("a", base, base + 0.1, base + 1.0),
            _make_latency_stats("b", base, base + 0.1, base + 5.0),
        ]
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                successful, base, base + 10.0, GoodputSLO(e2el_ms=2000)
            )
        )

        assert metrics.request_goodput is not None
        # One of two requests conforms over a 10s window. The ordering check
        # alone is a tautology: conforming is a subset of successful over the
        # same window, so it holds for any subset selection, right or wrong.
        assert metrics.request_goodput.successful.mean == pytest.approx(0.1)
        assert metrics.requests_per_second.successful.mean == pytest.approx(0.2)

    @pytest.mark.regression
    def test_unmeasurable_objective_reports_none_not_zero(self):
        """
        Report None when no request can be evaluated, rather than zero.

        A non-streaming workload has no first-token timing, so a ttft objective
        leaves every request undetermined. Reporting 0.0 would read as "nothing
        met the objectives" instead of "the objectives do not apply here".

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        successful = [
            _make_scheduled_stats("a", base, base, base + 1.0),
            _make_scheduled_stats("b", base, base, base + 9.0),
        ]
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                successful, base, base + 10.0, GoodputSLO(ttft_ms=500, e2el_ms=2000)
            )
        )

        assert metrics.slo_attainment is None
        assert metrics.request_goodput is None
        assert metrics.slo_determined_requests == 0

    @pytest.mark.smoke
    def test_goodput_fields_are_optional_for_existing_reports(self):
        """
        Carry defaults so reports written before goodput existed still validate.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                [_make_latency_stats("a", base, base + 0.1, base + 1.0)],
                base,
                base + 10.0,
                GoodputSLO(e2el_ms=2000),
            )
        )

        payload = metrics.model_dump()
        for name in ("slo_attainment", "slo_determined_requests", "request_goodput"):
            del payload[name]

        restored = GenerativeMetrics.model_validate(payload)

        assert restored.slo_attainment is None
        assert restored.request_goodput is None
        assert restored.slo_determined_requests == 0


class TestGoodputObjectiveMapping:
    """
    Verify which measured latency each objective is compared against.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_tpot_uses_inter_token_latency_not_time_per_output_token(self):
        """
        Compare tpot against inter-token latency, which excludes the first
        token, rather than against time per output token, which includes it.

        The two differ whenever time to first token differs from the steady
        inter-token interval, and the documented objective mapping depends on
        this choice.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        # 9 output tokens: slow first token at 0.5s, then 1.0s of generation.
        # inter-token latency = 1000 * 1.0 / 8 = 125ms
        # time per output token = 1000 * 1.5 / 9 = 166.7ms
        stats = _make_latency_stats("a", base, base + 0.5, base + 1.5, output_tokens=9)
        assert stats.inter_token_latency_ms == pytest.approx(125.0, abs=0.1)
        assert stats.time_per_output_token_ms == pytest.approx(166.7, abs=0.1)

        # A threshold between the two must accept the request.
        conforming = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                [stats], base, base + 10.0, GoodputSLO(tpot_ms=150)
            )
        )

        assert conforming.slo_attainment == pytest.approx(1.0)

    @pytest.mark.regression
    def test_ttft_and_tpot_objectives_are_not_swapped(self):
        """
        Compare each objective against its own measurement.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        # ttft = 500ms, inter-token latency = 125ms.
        stats = _make_latency_stats("a", base, base + 0.5, base + 1.5, output_tokens=9)

        # Thresholds that pass only when each is matched to its own metric.
        matched = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                [stats], base, base + 10.0, GoodputSLO(ttft_ms=600, tpot_ms=150)
            )
        )
        # Swapping the thresholds must fail: ttft 500 > 150, tpot 125 <= 600.
        swapped = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                [stats], base, base + 10.0, GoodputSLO(ttft_ms=150, tpot_ms=600)
            )
        )

        assert matched.slo_attainment == pytest.approx(1.0)
        assert swapped.slo_attainment == pytest.approx(0.0)

    @pytest.mark.regression
    def test_single_token_requests_are_undetermined_under_tpot(self):
        """
        Leave requests with one output token undetermined when tpot is set.

        Inter-token latency needs two tokens, so such requests are excluded
        from the population rather than counted either way.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        stats = _make_latency_stats("a", base, base + 0.1, base + 0.2, output_tokens=1)
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                [stats], base, base + 10.0, GoodputSLO(tpot_ms=1000)
            )
        )

        assert metrics.slo_determined_requests == 0
        assert metrics.slo_attainment is None


class TestGoodputPopulation:
    """
    Verify which requests the attainment fraction is averaged over.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_denominator_counts_only_determined_requests(self):
        """
        Divide by the requests that could be evaluated, not by every
        successful request.

        A mixed population is the only case that separates the two, and it is
        reachable whenever some requests carry token timings and others do not.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        # Two streaming requests that conform, two without token timings.
        successful = [
            _make_latency_stats("s1", base, base + 0.1, base + 1.0),
            _make_latency_stats("s2", base, base + 0.1, base + 1.0),
            _make_scheduled_stats("n1", base, base, base + 1.0),
            _make_scheduled_stats("n2", base, base, base + 1.0),
        ]
        metrics = GenerativeMetrics.compile(
            _make_goodput_accumulator(
                successful, base, base + 10.0, GoodputSLO(ttft_ms=500)
            )
        )

        assert metrics.slo_determined_requests == 2
        assert metrics.slo_attainment == pytest.approx(1.0)

    @pytest.mark.regression
    def test_errored_requests_count_against_attainment(self):
        """
        Score errored requests as non-conforming.

        On a saturated server overload surfaces as errors rather than slow
        successes. Averaging over successful requests alone reports a level
        that is mostly failing as fully conforming, which would make a search
        driven by attainment climb straight past the knee.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        successful = [_make_latency_stats("ok", base, base + 0.1, base + 1.0)]
        errored = [_make_errored_stats(f"err-{i}", base) for i in range(9)]
        accumulator = _make_goodput_accumulator(
            successful, base, base + 10.0, GoodputSLO(e2el_ms=5000)
        )
        accumulator.errored.requests_stats = errored
        metrics = GenerativeMetrics.compile(accumulator)

        assert metrics.slo_determined_requests == 10
        assert metrics.slo_attainment == pytest.approx(0.1)

    @pytest.mark.regression
    def test_incomplete_requests_are_excluded(self):
        """
        Ignore requests cancelled at the measurement boundary.

        Those were truncated by the run's own duration limit rather than by
        the server, so counting them would penalise every duration-limited run.

        ## WRITTEN BY AI ##
        """
        base = SCHEDULE_BASE_TIME
        successful = [_make_latency_stats("ok", base, base + 0.1, base + 1.0)]
        accumulator = _make_goodput_accumulator(
            successful, base, base + 10.0, GoodputSLO(e2el_ms=5000)
        )
        accumulator.incomplete.requests_stats = [
            _make_errored_stats(f"cancelled-{i}", base) for i in range(9)
        ]
        metrics = GenerativeMetrics.compile(accumulator)

        assert metrics.slo_determined_requests == 1
        assert metrics.slo_attainment == pytest.approx(1.0)


class TestGoodputConfigWiring:
    """
    Verify latency objectives reach the accumulator from configuration.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_objectives_reach_benchmark_config(self):
        """
        Carry the configured objectives onto BenchmarkConfig so metric
        compilation can read them.

        Without this the plumbing between the metrics arguments and the
        accumulator can be severed and every goodput metric silently becomes
        None while the run still reports success.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000)
        accumulator = _make_goodput_accumulator(
            [], SCHEDULE_BASE_TIME, SCHEDULE_BASE_TIME + 1.0, slo
        )

        assert accumulator.config.slo == slo

    @pytest.mark.regression
    def test_config_round_trips_objectives(self):
        """
        Serialize and restore objectives on BenchmarkConfig.

        ## WRITTEN BY AI ##
        """
        config = BenchmarkConfig(
            run_id="goodput",
            run_index=0,
            strategy=AsyncConstantStrategy(rate=10.0),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
            slo=GoodputSLO(ttft_ms=2000, tpot_ms=100),
        )
        restored = BenchmarkConfig.model_validate(config.model_dump())

        assert restored.slo is not None
        assert restored.slo.ttft_ms == 2000
        assert restored.slo.tpot_ms == 100

    @pytest.mark.regression
    def test_config_without_objectives_still_validates(self):
        """
        Restore a config written before the objectives field existed.

        ## WRITTEN BY AI ##
        """
        config = BenchmarkConfig(
            run_id="goodput",
            run_index=0,
            strategy=AsyncConstantStrategy(rate=10.0),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
        )
        payload = config.model_dump()
        del payload["slo"]

        assert BenchmarkConfig.model_validate(payload).slo is None
