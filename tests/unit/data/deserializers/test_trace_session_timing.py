from __future__ import annotations

import pytest

from guidellm.data.deserializers.trace_session_timing import TraceSessionTiming
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)


def _graph_with_timestamps(
    timestamps: list[float | None],
) -> ConversationGraphData:
    turns = []
    for index, timestamp in enumerate(timestamps):
        columns: dict = {"text_column": [f"t{index}"]}
        if timestamp is not None:
            columns["relative_timestamp_column"] = [timestamp]
        turns.append(
            ConversationTurnData(
                node_id=f"n{index}",
                columns=columns,
            )
        )
    return ConversationGraphData(turns=turns)


def _relative_timestamps(graph: ConversationGraphData) -> list[float | None]:
    result: list[float | None] = []
    for turn in graph.turns:
        values = turn.columns.get("relative_timestamp_column")
        result.append(values[0] if values else None)
    return result


class TestTraceSessionTimingIntraSession:
    @pytest.mark.smoke
    def test_compresses_large_serial_gaps_and_shifts_later_requests(self):
        """Gaps above max_wait shrink; later timestamps shift by the trim.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 10.0, 1450.0, 1455.0])
        TraceSessionTiming(max_wait=30.0).apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 10.0, 40.0, 45.0])

    @pytest.mark.smoke
    def test_keeps_small_gaps_unchanged(self):
        """Gaps at or below max_wait are not compressed.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 5.0, 20.0])
        TraceSessionTiming(max_wait=30.0).apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 5.0, 20.0])

    @pytest.mark.smoke
    def test_parallel_equal_timestamps_stay_simultaneous(self):
        """Equal timestamps remain equal after compression.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 100.0, 100.0, 105.0])
        TraceSessionTiming(max_wait=10.0).apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 10.0, 10.0, 15.0])

    @pytest.mark.smoke
    def test_max_wait_none_is_noop(self):
        """Faithful replay leaves timestamps unchanged when max_wait is unset.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 10.0, 1450.0])
        TraceSessionTiming().apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 10.0, 1450.0])

    @pytest.mark.sanity
    def test_skips_none_timestamps(self):
        """Turns without relative_timestamp are left alone.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, None, 100.0])
        TraceSessionTiming(max_wait=10.0).apply(graph)
        result = _relative_timestamps(graph)
        assert result[0] == pytest.approx(0.0)
        assert result[1] is None
        assert result[2] == pytest.approx(10.0)

    @pytest.mark.smoke
    def test_max_wait_does_not_cross_sessions(self):
        """Intra-session compression on one graph does not rewrite another.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 10.0, 1450.0])
        second = _graph_with_timestamps([0.0, 200.0])
        timing = TraceSessionTiming(max_wait=30.0)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 10.0, 40.0])
        assert _relative_timestamps(second) == pytest.approx([0.0, 30.0])


class TestTraceSessionTimingInterSession:
    @pytest.mark.smoke
    def test_clamps_positive_gap_between_sessions(self):
        """A large idle from one session's last request to the next start is clamped.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 40.0])
        second = _graph_with_timestamps([200.0, 205.0])
        timing = TraceSessionTiming(max_session_wait=20.0)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 40.0])
        assert _relative_timestamps(second) == pytest.approx([60.0, 65.0])

    @pytest.mark.smoke
    def test_overlapping_sessions_stay_parallel(self):
        """Sessions that restart at time 0 are not forced sequential.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 40.0])
        second = _graph_with_timestamps([0.0, 10.0])
        timing = TraceSessionTiming(max_session_wait=5.0)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 40.0])
        assert _relative_timestamps(second) == pytest.approx([0.0, 10.0])

    @pytest.mark.smoke
    def test_applies_after_intra_session_compression(self):
        """Intra-session trim can widen the inter-session gap that is then clamped.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 10.0, 1450.0])
        second = _graph_with_timestamps([1500.0, 1505.0])
        timing = TraceSessionTiming(max_wait=30.0, max_session_wait=20.0)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 10.0, 40.0])
        assert _relative_timestamps(second) == pytest.approx([60.0, 65.0])

    @pytest.mark.smoke
    def test_max_session_wait_none_is_noop(self):
        """Faithful replay leaves session offsets unchanged when unset.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 20.0])
        second = _graph_with_timestamps([200.0, 205.0])
        timing = TraceSessionTiming(max_wait=30.0)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 20.0])
        assert _relative_timestamps(second) == pytest.approx([200.0, 205.0])


class TestTraceSessionTimingMinConcurrentSessions:
    @pytest.mark.smoke
    def test_packs_sequential_sessions_to_target_overlap(self):
        """Sequential sessions are shifted earlier so N overlap in steady state.

        ## WRITTEN BY AI ##
        """
        sessions = [
            _graph_with_timestamps([0.0, 10.0]),
            _graph_with_timestamps([20.0, 30.0]),
            _graph_with_timestamps([40.0, 50.0]),
        ]
        timing = TraceSessionTiming(min_concurrent_sessions=2)
        for session in sessions:
            timing.apply(session)
        assert _relative_timestamps(sessions[0]) == pytest.approx([0.0, 10.0])
        assert _relative_timestamps(sessions[1]) == pytest.approx([0.0, 10.0])
        assert _relative_timestamps(sessions[2]) == pytest.approx([10.0, 20.0])

    @pytest.mark.smoke
    def test_does_not_delay_already_overlapping_sessions(self):
        """Sessions that already start together stay at time 0.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 40.0])
        second = _graph_with_timestamps([0.0, 10.0])
        third = _graph_with_timestamps([0.0, 20.0])
        timing = TraceSessionTiming(min_concurrent_sessions=2)
        timing.apply(first)
        timing.apply(second)
        timing.apply(third)
        assert _relative_timestamps(first) == pytest.approx([0.0, 40.0])
        assert _relative_timestamps(second) == pytest.approx([0.0, 10.0])
        assert _relative_timestamps(third) == pytest.approx([0.0, 20.0])

    @pytest.mark.smoke
    def test_preserves_intra_session_spacing(self):
        """Packing shifts a session as a unit.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 10.0])
        second = _graph_with_timestamps([50.0, 55.0, 60.0])
        timing = TraceSessionTiming(min_concurrent_sessions=2)
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(second) == pytest.approx([0.0, 5.0, 10.0])

    @pytest.mark.smoke
    def test_unset_is_noop(self):
        """Faithful replay leaves session starts unchanged when unset.

        ## WRITTEN BY AI ##
        """
        first = _graph_with_timestamps([0.0, 10.0])
        second = _graph_with_timestamps([20.0, 30.0])
        timing = TraceSessionTiming()
        timing.apply(first)
        timing.apply(second)
        assert _relative_timestamps(first) == pytest.approx([0.0, 10.0])
        assert _relative_timestamps(second) == pytest.approx([20.0, 30.0])


class TestTraceSessionTimingTimeScale:
    @pytest.mark.smoke
    def test_scales_after_wait_caps(self):
        """Wait caps run in original seconds; time_scale multiplies the result.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 10.0, 1450.0])
        TraceSessionTiming(max_wait=30.0, time_scale=2.0).apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 20.0, 80.0])

    @pytest.mark.smoke
    def test_scales_without_caps(self):
        """time_scale alone multiplies original relative timestamps.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 10.0, 100.0])
        TraceSessionTiming(time_scale=2.0).apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 20.0, 200.0])

    @pytest.mark.smoke
    def test_default_time_scale_is_noop(self):
        """Default time_scale leaves timestamps unchanged.

        ## WRITTEN BY AI ##
        """
        graph = _graph_with_timestamps([0.0, 10.0, 1450.0])
        TraceSessionTiming().apply(graph)
        assert _relative_timestamps(graph) == pytest.approx([0.0, 10.0, 1450.0])
