from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.accumulator import GenerativeRequestsAccumulator
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
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


def _make_audio_update(
    request_id: str,
) -> tuple[GenerationResponse, GenerationRequest, RequestInfo]:
    """Build a complete audio response tuple for accumulator sampling tests.

    :param request_id: Stable identifier shared by response, request and info.
    :return: Response, source request and completed request info.
    """
    request = GenerationRequest(
        request_id=request_id,
        columns={"text_column": ["Speak."]},
    )
    response = GenerationResponse(
        request_id=request_id,
        request_args=None,
        use_request_output_metrics=False,
        output_media=[{"modality": "audio", "data": b"audio-payload"}],
        output_metrics=UsageMetrics(audio_seconds=0.1),
    )
    info = RequestInfo(
        request_id=request_id,
        status="completed",
        timings=RequestTimings(
            request_start=1,
            request_end=2,
            resolve_end=2,
        ),
    )
    return response, request, info


@pytest.mark.regression
@pytest.mark.parametrize("clear_outputs", [False, True])
def test_audio_payload_follows_output_retention_policy(clear_outputs):
    """Clear retained audio bytes with text while keeping measured metrics.

    :param clear_outputs: Whether unsampled output payloads should be removed.
    :return: None.
    ## WRITTEN BY AI ##
    """
    accumulator = GenerativeRequestsAccumulator(clear_nonsampled_outputs=clear_outputs)
    stats = _make_stats()
    stats.output_media = [{"modality": "audio", "data": b"\x00\x00"}]
    stats.output_metrics.audio_seconds = 0.1

    accumulator.clear_stats_data(stats)

    assert (stats.output_media is None) is clear_outputs
    assert stats.output_metrics.audio_seconds == 0.1


@pytest.mark.regression
def test_sample_size_zero_clears_audio_payload_but_keeps_measurement():
    """Apply zero-sized request sampling to audio payload retention.

    :return: None.
    ## WRITTEN BY AI ##
    """
    accumulator = GenerativeRequestsAccumulator(sample_size=0)
    response, request, info = _make_audio_update("audio-no-sample")

    stats = accumulator.update_estimate(
        response, request, info, prefer_response_metrics=True
    )

    assert stats.output_media is None
    assert stats.output_metrics.audio_seconds == 0.1


@pytest.mark.regression
def test_reservoir_eviction_clears_audio_payload(monkeypatch):
    """Release evicted audio bytes while preserving request audio metrics.

    :param monkeypatch: Pytest monkeypatch fixture for deterministic sampling.
    :return: None.
    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(
        "guidellm.benchmark.schemas.accumulator.random.random", lambda: 0.0
    )
    monkeypatch.setattr(
        "guidellm.benchmark.schemas.accumulator.random.randrange", lambda size: 0
    )
    accumulator = GenerativeRequestsAccumulator(sample_size=1)
    first_response, first_request, first_info = _make_audio_update("audio-first")
    second_response, second_request, second_info = _make_audio_update("audio-second")
    first_stats = accumulator.update_estimate(
        first_response, first_request, first_info, prefer_response_metrics=True
    )

    accumulator.update_estimate(
        second_response, second_request, second_info, prefer_response_metrics=True
    )

    assert accumulator.samples == [1]
    assert first_stats.output_media is None
    assert first_stats.output_metrics.audio_seconds == 0.1
    assert accumulator.get_sampled()[0].output_media is not None


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


def _make_timed_stats(
    request_id: str,
    request_start: float,
    request_end: float,
) -> GenerativeRequestStats:
    """Build a GenerativeRequestStats with explicit request timing bounds."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = request_start
    info.timings.request_end = request_end
    info.timings.resolve_end = request_end
    return GenerativeRequestStats(
        request_id=request_id,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestGetWithinRange:
    """
    Tests for GenerativeRequestsAccumulator.get_within_range.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_partial_overlap_includes_events_crossing_boundary(self):
        """
        Default closed=False includes events that partially overlap the window.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0)

        assert [stats.request_id for stats in in_range] == [
            "before",
            "inside",
            "after",
        ]

    @pytest.mark.sanity
    def test_closed_range_requires_full_enclosure(self):
        """
        closed=True excludes boundary-crossing events and keeps only fully
        enclosed ones.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0, closed=True)

        assert [stats.request_id for stats in in_range] == ["inside"]
