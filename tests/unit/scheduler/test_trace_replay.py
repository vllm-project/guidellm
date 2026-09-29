from __future__ import annotations

import asyncio
from multiprocessing import get_context

import pytest

from guidellm.scheduler import SchedulingStrategy, TraceReplayStrategy
from guidellm.schemas import RequestInfo, RequestSettings

TRACE_TIMESTAMPS = [0.0, 0.0, 0.0, 0.1, 0.1, 1.5, 2.0, 2.0, 3.5, 7.0]


class TestTraceReplayStrategy:
    @pytest.mark.smoke
    def test_initialization_and_serialization(self):
        strategy = TraceReplayStrategy(time_scale=2.0)

        assert strategy.type_ == "trace"
        assert str(strategy) == "trace@2.00"
        assert strategy.processes_limit is None
        assert strategy.requests_limit is None
        restored = SchedulingStrategy.model_validate(strategy.model_dump())
        assert isinstance(restored, TraceReplayStrategy)
        assert restored.time_scale == 2.0

    @pytest.mark.smoke
    def test_resolve_dequeued_target_start_applies_trace_offset(self):
        """Dequeue resolution uses per-request settings, not provisional slot time.

        ### WRITTEN BY AI ###
        """
        strategy = TraceReplayStrategy(time_scale=2.0)
        strategy.init_processes_timings(
            worker_count=2,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            settings = RequestSettings(relative_timestamp=1.5)
            provisional = 9999.0
            resolved = await strategy.resolve_dequeued_target_start(
                1,
                provisional,
                settings,
            )
            return resolved, provisional

        resolved, provisional = asyncio.run(run())
        assert resolved == pytest.approx(1000.0 + 2.0 * 1.5, abs=1e-6)
        assert resolved != pytest.approx(provisional, abs=1e-6)

    @pytest.mark.smoke
    def test_next_request_time_returns_start_time(self):
        """Verify next_request_time schedules immediate dequeue at benchmark start.

        ### WRITTEN BY AI ###
        """
        strategy = TraceReplayStrategy(time_scale=2.0)
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            return await strategy.next_request_time(0)

        assert asyncio.run(run()) == pytest.approx(1000.0, abs=1e-6)

    @pytest.mark.smoke
    def test_resolve_dequeued_target_start_scales_timestamps(self):
        """Dequeue start is start_time plus time_scale times relative timestamp.

        ## WRITTEN BY AI ##
        """
        strategy = TraceReplayStrategy(time_scale=2.0)
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            return [
                await strategy.resolve_dequeued_target_start(
                    0,
                    1000.0,
                    RequestSettings(relative_timestamp=ts),
                )
                for ts in TRACE_TIMESTAMPS
            ]

        assert asyncio.run(run()) == pytest.approx(
            [1000.0 + 2.0 * ts for ts in TRACE_TIMESTAMPS],
            abs=1e-6,
        )

    @pytest.mark.smoke
    def test_resolve_dequeued_target_start_without_relative_timestamp(self):
        """Missing offset schedules at benchmark start, not the provisional slot.

        ### WRITTEN BY AI ###
        """
        strategy = TraceReplayStrategy(time_scale=2.0)
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            return await strategy.resolve_dequeued_target_start(
                0,
                9999.0,
                RequestSettings(),
            )

        assert asyncio.run(run()) == pytest.approx(1000.0, abs=1e-6)

    @pytest.mark.smoke
    def test_request_completed_no_op(self):
        strategy = TraceReplayStrategy(time_scale=1.0)
        info = RequestInfo(
            request_id="x",
            status="completed",
            scheduler_process_id=0,
            scheduler_start_time=0,
        )
        strategy.request_completed(info)

    @pytest.mark.sanity
    def test_relative_timing_keeps_idle_gap_after_recorded_duration(self):
        """
        Relative timing starts the next request after the recorded idle gap.

        A 1s recorded duration and a timestamp 5s later leave a 4s gap. When
        the request finishes at t=6, the next target is 10, five seconds after
        the original absolute time of 5.

        ## WRITTEN BY AI ##
        """
        strategy = _relative_strategy(time_scale=1.0)
        _complete(
            strategy,
            node_id="n0",
            relative_timestamp=0.0,
            trace_duration=1.0,
            actual_end=6.0,
        )

        resolved = asyncio.run(
            strategy.resolve_dequeued_target_start(
                0,
                0.0,
                RequestSettings(relative_timestamp=5.0, trace_duration=1.0),
                _child("n1", ["n0"], relative_timestamp=5.0),
            )
        )

        assert resolved == pytest.approx(10.0)

    @pytest.mark.sanity
    def test_relative_timing_scales_the_idle_gap(self):
        """
        Profile time_scale multiplies the idle gap, not the parent's actual end.

        ## WRITTEN BY AI ##
        """
        strategy = _relative_strategy(time_scale=2.0)
        _complete(
            strategy,
            node_id="n0",
            relative_timestamp=0.0,
            trace_duration=1.0,
            actual_end=6.0,
        )

        resolved = asyncio.run(
            strategy.resolve_dequeued_target_start(
                0,
                0.0,
                RequestSettings(relative_timestamp=5.0),
                _child("n1", ["n0"], relative_timestamp=5.0),
            )
        )

        assert resolved == pytest.approx(6.0 + 2.0 * 4.0)

    @pytest.mark.sanity
    def test_relative_timing_missing_duration_is_instantaneous(self):
        """
        A missing duration is zero, so the next request keeps the full gap.

        The same 5s timestamp offset after an actual end at t=6 targets 11.
        The loader warns when the column is absent; the scheduler does not.

        ## WRITTEN BY AI ##
        """
        strategy = _relative_strategy(time_scale=1.0)
        _complete(
            strategy,
            node_id="n0",
            relative_timestamp=0.0,
            trace_duration=None,
            actual_end=6.0,
        )
        _complete(
            strategy,
            node_id="n0b",
            relative_timestamp=1.0,
            trace_duration=None,
            actual_end=7.0,
        )
        resolved = asyncio.run(
            strategy.resolve_dequeued_target_start(
                0,
                0.0,
                RequestSettings(relative_timestamp=5.0),
                _child("n1", ["n0"], relative_timestamp=5.0),
            )
        )

        assert resolved == pytest.approx(11.0)

    @pytest.mark.sanity
    def test_relative_timing_explicit_zero_duration_is_instantaneous(self):
        """
        An explicit duration of zero keeps the full timestamp gap.

        ## WRITTEN BY AI ##
        """
        strategy = _relative_strategy(time_scale=1.0)
        _complete(
            strategy,
            node_id="n0",
            relative_timestamp=0.0,
            trace_duration=0.0,
            actual_end=6.0,
        )
        resolved = asyncio.run(
            strategy.resolve_dequeued_target_start(
                0,
                0.0,
                RequestSettings(relative_timestamp=5.0),
                _child("n1", ["n0"], relative_timestamp=5.0),
            )
        )

        assert resolved == pytest.approx(11.0)

    @pytest.mark.smoke
    def test_absolute_timing_ignores_predecessor_completion(self):
        """
        schedule_turn=timestamp still targets the trace timestamp after a late
        predecessor.

        ## WRITTEN BY AI ##
        """
        strategy = _relative_strategy(time_scale=1.0, schedule_turn="timestamp")
        _complete(
            strategy,
            node_id="n0",
            relative_timestamp=0.0,
            trace_duration=1.0,
            actual_end=6.0,
        )

        resolved = asyncio.run(
            strategy.resolve_dequeued_target_start(
                0,
                0.0,
                RequestSettings(relative_timestamp=5.0, trace_duration=1.0),
                _child("n1", ["n0"], relative_timestamp=5.0),
            )
        )

        assert resolved == pytest.approx(5.0)


def _relative_strategy(**kwargs) -> TraceReplayStrategy:
    """Start a trace strategy at wall-clock time 0.

    ## WRITTEN BY AI ##
    """
    if "schedule_turn" not in kwargs:
        kwargs["schedule_turn"] = "idle_gap"
    strategy = TraceReplayStrategy(**kwargs)
    strategy.init_processes_timings(
        worker_count=1,
        max_concurrency=10,
        mp_context=get_context(),
    )
    strategy.init_processes_start(0.0)
    return strategy


def _child(
    node_id: str, parent_node_ids: list[str], relative_timestamp: float
) -> RequestInfo:
    """Build a successor request info for relative scheduling.

    ## WRITTEN BY AI ##
    """
    return RequestInfo(
        request_id=node_id,
        conversation_id="conv",
        node_id=node_id,
        parent_node_ids=parent_node_ids,
        status="pending",
        settings=RequestSettings(relative_timestamp=relative_timestamp),
    )


def _complete(
    strategy: TraceReplayStrategy,
    node_id: str,
    relative_timestamp: float,
    trace_duration: float | None,
    actual_end: float,
) -> None:
    """Record a finished predecessor on the strategy.

    ## WRITTEN BY AI ##
    """
    info = RequestInfo(
        request_id=node_id,
        conversation_id="conv",
        node_id=node_id,
        status="completed",
        settings=RequestSettings(
            relative_timestamp=relative_timestamp,
            trace_duration=trace_duration,
        ),
    )
    info.timings.request_end = actual_end
    strategy.request_completed(info)
