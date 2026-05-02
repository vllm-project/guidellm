from __future__ import annotations

import asyncio
import math
from multiprocessing import get_context
from pathlib import Path

import pytest
from datasets.exceptions import DatasetGenerationError

from guidellm.scheduler import SchedulingStrategy, TraceReplayStrategy
from guidellm.schemas import RequestInfo
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
                ValueError,
                "could not convert",
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


class TestTraceReplayStrategy:
    @pytest.mark.smoke
    def test_initialization_and_serialization(self):
        strategy = TraceReplayStrategy(
            relative_timestamps=[0.0, 0.5, 1.0],
            time_scale=2.0,
        )

        assert strategy.type_ == "trace"
        assert str(strategy) == "trace@2.00"
        assert strategy.processes_limit is None
        assert strategy.requests_limit == 3

        restored = SchedulingStrategy.model_validate(strategy.model_dump())
        assert isinstance(restored, TraceReplayStrategy)
        assert restored.relative_timestamps == [0.0, 0.5, 1.0]
        assert restored.time_scale == 2.0

    @pytest.mark.smoke
    def test_next_request_time_scales_timestamps_and_exhausts_trace(self):
        strategy = TraceReplayStrategy(
            relative_timestamps=[0.0, 0.5, 1.0],
            time_scale=2.0,
        )
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            return [await strategy.next_request_time(0) for _ in range(4)]

        assert asyncio.run(run()) == pytest.approx(
            [1000.0, 1001.0, 1002.0, math.inf], abs=1e-6
        )

    @pytest.mark.smoke
    def test_empty_trace_has_no_request_limit_and_uses_start_time(self):
        strategy = TraceReplayStrategy(relative_timestamps=[], time_scale=1.0)
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(123.0)

        assert strategy.requests_limit is None

        async def run():
            return await strategy.next_request_time(0)

        assert asyncio.run(run()) == pytest.approx(123.0)

    @pytest.mark.smoke
    def test_request_completed_no_op(self):
        strategy = TraceReplayStrategy(relative_timestamps=[0.0], time_scale=1.0)
        info = RequestInfo(
            request_id="x",
            status="completed",
            scheduler_process_id=0,
            scheduler_start_time=0,
        )
        strategy.request_completed(info)
