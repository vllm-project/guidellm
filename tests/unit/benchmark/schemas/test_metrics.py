"""
Tests for metrics compilation, specifically verifying that tool call metrics
tables appear when all expected tool call requests errored.
"""

from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.metrics import (
    GenerativeMetricsSummary,
    GenerativeToolCallMetricsSummary,
)
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
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
