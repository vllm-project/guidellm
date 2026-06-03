from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.generative.accumulator import (
    GenerativeRequestsAccumulator,
)
from guidellm.schemas import GenerativeRequestStats, RequestInfo, UsageMetrics


def _make_stats(
    request_id: str = "req-1",
    output: str | None = "some output",
    reasoning_output: str | None = None,
    request_args: str | None = "args",
) -> GenerativeRequestStats:
    """Build a minimal GenerativeRequestStats for testing."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = 0.0
    info.timings.request_end = 1.0
    info.timings.resolve_end = 1.0
    return GenerativeRequestStats(
        request_id=request_id,
        request_args=request_args,
        output=output,
        reasoning_output=reasoning_output,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestClearStatsData:
    """
    Tests for GenerativeRequestsAccumulator.clear_stats_data.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_clears_output_and_reasoning_output(self):
        """
        clear_stats_data clears output, reasoning_output, and tool_calls
        when clear_nonsampled_outputs is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None
        assert stats.request_args == "args"

    @pytest.mark.smoke
    def test_clears_request_args(self):
        """
        clear_stats_data clears request_args when
        clear_nonsampled_request_args is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."

    @pytest.mark.smoke
    def test_clears_both(self):
        """
        clear_stats_data clears all fields when both flags are True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None

    @pytest.mark.smoke
    def test_clears_by_index(self):
        """
        clear_stats_data accepts an integer index to look up the stats
        in requests_stats.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(reasoning_output="step 1")
        acc.requests_stats.append(stats)

        acc.clear_stats_data(0)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.request_args is None

    @pytest.mark.smoke
    def test_preserves_when_both_flags_false(self):
        """
        clear_stats_data preserves all fields when both flags are False.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args == "args"
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."
