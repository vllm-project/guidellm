from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.accumulator import (
    GenerativeRequestsAccumulator,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    UsageMetrics,
)


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


class TestReservoirSampling:
    """Tests for bounded request-data retention during reservoir sampling."""

    @pytest.mark.regression
    def test_clears_request_data_when_new_request_is_not_sampled(self, monkeypatch):
        """A rejected reservoir candidate must not retain heavyweight data.

        ## WRITTEN BY AI ##
        """
        accumulator = GenerativeRequestsAccumulator(sample_size=1)

        first_request = GenerationRequest(request_id="req-1")
        first_response = GenerationResponse(
            request_id="req-1",
            request_args="args-1",
            text="output-1",
            reasoning_text="reasoning-1",
        )
        first_info = RequestInfo(request_id="req-1", status="completed")
        first_info.timings.request_start = 0.0
        first_info.timings.request_end = 1.0
        first_info.timings.resolve_end = 1.0
        accumulator.update_estimate(
            first_response,
            first_request,
            first_info,
            prefer_response_metrics=True,
        )

        monkeypatch.setattr("random.random", lambda: 1.0)
        second_request = GenerationRequest(request_id="req-2")
        second_response = GenerationResponse(
            request_id="req-2",
            request_args="args-2",
            text="output-2",
            reasoning_text="reasoning-2",
        )
        second_info = RequestInfo(request_id="req-2", status="completed")
        second_info.timings.request_start = 1.0
        second_info.timings.request_end = 2.0
        second_info.timings.resolve_end = 2.0
        accumulator.update_estimate(
            second_response,
            second_request,
            second_info,
            prefer_response_metrics=True,
        )

        assert accumulator.samples == [0]
        assert accumulator.requests_stats[0].request_args == "args-1"
        assert accumulator.requests_stats[0].output == "output-1"
        assert accumulator.requests_stats[1].request_args is None
        assert accumulator.requests_stats[1].output is None
        assert accumulator.requests_stats[1].reasoning_output is None

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
