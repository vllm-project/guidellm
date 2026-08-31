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
