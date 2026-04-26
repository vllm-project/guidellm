"""
Unit tests for trace replay strategy and load_relative_timestamps.

Verifies that TraceReplayStrategy schedules requests at start_time + time_scale
* relative_timestamp[i] and that load_relative_timestamps correctly parses trace
files.
"""

from __future__ import annotations

import asyncio
import json
import math
from multiprocessing import get_context
from pathlib import Path

import pytest

from guidellm.scheduler import SchedulingStrategy, TraceReplayStrategy
from guidellm.schemas import RequestInfo
from guidellm.utils.trace_io import load_relative_timestamps


def _write_trace(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


class TestLoadRelativeTimestamps:
    """Tests for load_relative_timestamps helper."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("content", "kwargs", "expected"),
        [
            # Basic cases
            (
                '{"timestamp": 100, "input_length": 10}\n'
                '{"timestamp": 100.5, "input_length": 20}\n'
                '{"timestamp": 101.2, "input_length": 15}\n',
                {"timestamp_column": "timestamp"},
                [0.0, 0.5, 1.2],
            ),
            (
                '{"ts": 0, "input_length": 1}\n{"ts": 2.5, "input_length": 2}\n',
                {"timestamp_column": "ts"},
                [0.0, 2.5],
            ),
            # High token counts (production-like: 2K-128K contexts)
            (
                '{"timestamp": 0, "input_length": 2048, "output_length": 512}\n'
                '{"timestamp": 1.5, "input_length": 4096, "output_length": 1024}\n'
                '{"timestamp": 3.0, "input_length": 8192, "output_length": 2048}\n'
                '{"timestamp": 4.5, "input_length": 32768, "output_length": 8192}\n'
                '{"timestamp": 6.0, "input_length": 131072, "output_length": 32768}\n',
                {"timestamp_column": "timestamp"},
                [0.0, 1.5, 3.0, 4.5, 6.0],
            ),
            # Unsorted timestamps (sorted chronologically, all >= 0)
            (
                '{"timestamp": 5.0, "input_length": 10}\n'
                '{"timestamp": 2.0, "input_length": 20}\n'
                '{"timestamp": 8.0, "input_length": 30}\n',
                {"timestamp_column": "timestamp"},
                [0.0, 3.0, 6.0],  # Sorted: 2.0, 5.0, 8.0 -> 0.0, 3.0, 6.0
            ),
            # Duplicate timestamps (concurrent burst)
            (
                '{"timestamp": 1.0, "input_length": 10}\n'
                '{"timestamp": 1.0, "input_length": 20}\n'
                '{"timestamp": 1.0, "input_length": 30}\n'
                '{"timestamp": 2.5, "input_length": 40}\n',
                {"timestamp_column": "timestamp"},
                [0.0, 0.0, 0.0, 1.5],
            ),
        ],
    )
    def test_load_valid_jsonl(self, tmp_path: Path, content, kwargs, expected):
        """Load JSONL trace and get sorted relative timestamps (basic, high counts,
        unsorted, duplicates)."""
        trace = tmp_path / "trace.jsonl"
        _write_trace(trace, content)
        out = load_relative_timestamps(trace, **kwargs)
        assert out == pytest.approx(expected, abs=1e-9)

    @pytest.mark.smoke
    def test_empty_trace_raises(self, tmp_path: Path):
        """Empty trace file raises ValueError."""
        trace = tmp_path / "trace.jsonl"
        _write_trace(trace, "")
        with pytest.raises(ValueError, match="no valid rows"):
            load_relative_timestamps(trace)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("suffix", "content", "match"),
        [
            (
                "json",
                json.dumps(
                    [
                        {"timestamp": 0, "input_length": 1},
                        {"timestamp": 1.0, "input_length": 2},
                    ]
                ),
                r"Unsupported trace file format.*\.json",
            ),
            (
                "csv",
                "timestamp,input_length,output_length\n0,10,5\n0.3,20,10\n",
                r"Unsupported trace file format.*\.csv",
            ),
            ("txt", "0\n1\n", "Unsupported trace file format"),
        ],
    )
    def test_unsupported_format_raises(self, tmp_path: Path, suffix, content, match):
        """JSON array, CSV, or unknown suffix raises ValueError."""
        trace = tmp_path / f"trace.{suffix}"
        _write_trace(trace, content)
        with pytest.raises(ValueError, match=match):
            load_relative_timestamps(trace)


class TestTraceReplayStrategy:
    """Tests for TraceReplayStrategy."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("timestamps", "time_scale"),
        [
            ([0.0, 0.5, 1.0], 2.0),
            ([0.0, 1.0], 0.5),
        ],
    )
    def test_initialization_and_str(self, timestamps, time_scale):
        """Init, type_, optional str, and limits."""
        strategy = TraceReplayStrategy(
            relative_timestamps=timestamps,
            time_scale=time_scale,
        )
        assert strategy.type_ == "trace"
        assert strategy.relative_timestamps == timestamps
        assert strategy.time_scale == time_scale
        assert strategy.processes_limit is None
        assert strategy.requests_limit == len(timestamps)
        if time_scale == 0.5:
            assert str(strategy) == "trace@0.50"

    @pytest.mark.smoke
    def test_marshalling(self):
        """Pydantic dump/load and polymorphic restore."""
        strategy = TraceReplayStrategy(
            relative_timestamps=[0.0, 1.0, 2.0],
            time_scale=1.5,
        )
        data = strategy.model_dump()
        assert data["type_"] == "trace"
        assert data["relative_timestamps"] == [0.0, 1.0, 2.0]
        assert data["time_scale"] == 1.5
        reconstructed = TraceReplayStrategy.model_validate(data)
        assert reconstructed.relative_timestamps == strategy.relative_timestamps
        base = SchedulingStrategy.model_validate(data)
        assert isinstance(base, TraceReplayStrategy)

    @pytest.mark.smoke
    def test_next_request_time_scaled_timestamps(self):
        """next_request_time returns start_time + time_scale * relative_ts[i]."""
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
        expected = [1000.0, 1001.0, 1002.0]

        async def run():
            for exp in expected:
                t = await strategy.next_request_time(0)
                assert t == pytest.approx(exp, abs=1e-6)

        asyncio.run(run())

    @pytest.mark.smoke
    def test_next_request_time_beyond_trace_parks_worker(self):
        """When index > len(relative_timestamps), return math.inf to park the slot.

        Returning math.inf causes the worker to sleep indefinitely until
        constraint_reached_event cancels it, preventing it from racing the
        messaging queue with a stale target timestamp.
        """
        strategy = TraceReplayStrategy(
            relative_timestamps=[0.0, 1.0],
            time_scale=1.0,
        )
        strategy.init_processes_timings(
            worker_count=1,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(500.0)

        async def run():
            await strategy.next_request_time(0)
            await strategy.next_request_time(0)
            t3 = await strategy.next_request_time(0)
            assert t3 == math.inf

        asyncio.run(run())

    @pytest.mark.smoke
    def test_request_completed_no_op(self):
        """request_completed is a no-op."""
        strategy = TraceReplayStrategy(relative_timestamps=[0.0], time_scale=1.0)
        info = RequestInfo(
            request_id="x",
            status="completed",
            scheduler_process_id=0,
            scheduler_start_time=0,
        )
        strategy.request_completed(info)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("timestamps", "expected"),
        [
            # Concurrent burst: 3 requests at same time
            ([0.0, 0.0, 0.0, 1.0, 2.0], [1000.0, 1000.0, 1000.0, 1001.0, 1002.0]),
            # Unsorted timestamps (sorted by load_relative_timestamps, all >= 0)
            ([0.0, 3.0, 5.0, 6.0], [1000.0, 1003.0, 1005.0, 1006.0]),
            # High frequency burst (millisecond scale)
            (
                [0.0, 0.001, 0.002, 0.003, 0.004],
                [1000.0, 1000.001, 1000.002, 1000.003, 1000.004],
            ),
        ],
    )
    def test_scheduling_patterns(self, timestamps, expected):
        """Test concurrent bursts, unsorted timestamps (now sorted), and high-frequency
        patterns."""
        strategy = TraceReplayStrategy(
            relative_timestamps=timestamps,
            time_scale=1.0,
        )
        strategy.init_processes_timings(
            worker_count=3,
            max_concurrency=10,
            mp_context=get_context(),
        )
        strategy.init_processes_start(1000.0)

        async def run():
            for exp in expected:
                t = await strategy.next_request_time(0)
                assert t == pytest.approx(exp, abs=1e-6)

        asyncio.run(run())
