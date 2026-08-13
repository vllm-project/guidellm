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
