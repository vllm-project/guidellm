from __future__ import annotations

import asyncio
from multiprocessing import get_context
from pathlib import Path

import pytest
from datasets.exceptions import DatasetGenerationError

from guidellm.data.deserializers import DataNotSupportedError
from guidellm.scheduler import SchedulingStrategy, TraceReplayStrategy
from guidellm.schemas import RequestInfo, RequestSettings
from guidellm.utils.trace_io import load_relative_timestamps


def _write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


class TestLoadRelativeTimestamps:
    @pytest.mark.smoke
    def test_loads_sorted_relative_timestamps_with_duplicates(self, tmp_path: Path):
        trace = _write_trace(
            tmp_path,
            '{"timestamp": 5.0, "input_length": 10, "output_length": 10}\n'
            '{"timestamp": 2.0, "input_length": 20, "output_length": 20}\n'
            '{"timestamp": 2.0, "input_length": 30, "output_length": 30}\n'
            '{"timestamp": 8.0, "input_length": 40, "output_length": 40}\n',
        )

        assert load_relative_timestamps(trace) == pytest.approx(
            [0.0, 0.0, 3.0, 6.0], abs=1e-9
        )

    @pytest.mark.smoke
    def test_loads_custom_timestamp_column(self, tmp_path: Path):
        trace = _write_trace(
            tmp_path,
            '{"ts": 10.0, "input_length": 10, "output_length": 10}\n'
            '{"ts": 10.25, "input_length": 20, "output_length": 20}\n',
        )

        assert load_relative_timestamps(trace, timestamp_column="ts") == pytest.approx(
            [0.0, 0.25], abs=1e-9
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("suffix", "content", "error_type", "match"),
        [
            (".jsonl", "", ValueError, "no valid rows"),
            (
                ".json",
                '[{"timestamp": 0, "input_length": 10, "output_length": 100}]',
                ValueError,
                r"Unsupported.*\.json",
            ),
            (
                ".csv",
                "timestamp,input_length,output_length\n0,10,100\n",
                ValueError,
                r"Unsupported.*\.csv",
            ),
            (
                ".jsonl",
                '{"ts": 0, "input_length": 10, "output_length": 100}\n',
                KeyError,
                "timestamp",
            ),
            (
                ".jsonl",
                '{"timestamp": "bad", "input_length": 10, "output_length": 100}\n',
                DataNotSupportedError,
                "Failed to parse string",
            ),
            (
                ".jsonl",
                '{"timestamp": 0, "input_length": 10, "output_length": 100}\n'
                "not-json\n",
                DatasetGenerationError,
                "generating the dataset",
            ),
        ],
    )
    def test_invalid_trace_inputs_raise(
        self, tmp_path: Path, suffix, content, error_type, match
    ):
        trace = _write_trace(tmp_path, content, suffix=suffix)

        with pytest.raises(error_type, match=match):
            load_relative_timestamps(trace)


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
