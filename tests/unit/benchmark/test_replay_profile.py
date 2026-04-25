## WRITTEN BY AI ##

"""
Unit tests for ReplayProfile.

Ensures replay profile loads trace timestamps and creates TraceReplayStrategy with
orrect time_scale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guidellm.benchmark.profiles import Profile, ReplayProfile
from guidellm.scheduler import TraceReplayStrategy


def _trace_path(tmp_path: Path, lines: list[str]) -> Path:
    """Write JSONL lines to a trace file and return its path."""
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines))
    return path


class TestReplayProfile:
    """Tests for ReplayProfile."""

    @pytest.mark.smoke
    def test_resolve_args_requires_data(self):
        """resolve_args raises when data is missing."""
        with pytest.raises(ValueError, match="Replay profile requires data"):
            ReplayProfile.resolve_args(
                rate_type="replay",
                rate=[1.0],
                random_seed=42,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("trace_lines", "rate", "expected_ts", "expected_scale"),
        [
            # Basic trace
            (
                [
                    '{"timestamp": 0, "input_length": 10, "output_length": 5}',
                    '{"timestamp": 0.5, "input_length": 20, "output_length": 10}',
                ],
                [2.0],
                [0.0, 0.5],
                2.0,
            ),
            # High token counts (8K-128K contexts)
            (
                [
                    '{"timestamp": 0, "input_length": 8192, "output_length": 1024}',
                    '{"timestamp": 0.5, "input_length": 32768, "output_length": 4096}',
                    '{"timestamp": 1.0, "input_length": 131072,"output_length": 16384}',
                ],
                [1.0],
                [0.0, 0.5, 1.0],
                1.0,
            ),
            # Unsorted timestamps (sorted chronologically, all >= 0)
            (
                [
                    '{"timestamp": 5.0, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 2.0, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 8.0, "input_length": 300, "output_length": 30}',
                ],
                [1.0],
                [0.0, 3.0, 6.0],  # Sorted: 2.0, 5.0, 8.0 -> 0.0, 3.0, 6.0
                1.0,
            ),
            # Duplicate timestamps (concurrent burst)
            (
                [
                    '{"timestamp": 1.0, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 1.0, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 1.0, "input_length": 300, "output_length": 30}',
                    '{"timestamp": 2.5, "input_length": 400, "output_length": 40}',
                ],
                [2.0],
                [0.0, 0.0, 0.0, 1.5],
                2.0,
            ),
            # High-frequency trace (millisecond-scale)
            (
                [
                    '{"timestamp": 0.000, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 0.001, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 0.002, "input_length": 300, "output_length": 30}',
                    '{"timestamp": 0.003, "input_length": 400, "output_length": 40}',
                ],
                [1.0],
                [0.0, 0.001, 0.002, 0.003],
                1.0,
            ),
        ],
    )
    def test_resolve_args_and_create_with_trace(
        self, tmp_path: Path, trace_lines, rate, expected_ts, expected_scale
    ):
        """resolve_args loads trace; Profile.create returns ReplayProfile with
        correct time_scale."""
        trace = _trace_path(tmp_path, trace_lines)
        out = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=rate,
            random_seed=42,
            data=[str(trace)],
        )
        assert out["relative_timestamps"] == pytest.approx(expected_ts, abs=1e-9)
        assert out["time_scale"] == expected_scale
        profile = Profile.create(
            rate_type="replay",
            rate=rate,
            random_seed=42,
            data=[str(trace)],
        )
        assert isinstance(profile, ReplayProfile)
        assert profile.relative_timestamps == pytest.approx(expected_ts, abs=1e-9)
        assert profile.time_scale == expected_scale

    @pytest.mark.smoke
    def test_next_strategy_returns_trace_then_none(self, tmp_path: Path):
        """next_strategy returns TraceReplayStrategy then None."""
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
        )
        profile = ReplayProfile(**kwargs)
        assert profile.strategy_types == ["trace"]
        s1 = profile.next_strategy(None, None)
        assert isinstance(s1, TraceReplayStrategy)
        assert s1.relative_timestamps == [0.0]
        assert s1.time_scale == 1.0
        assert profile.next_strategy(s1, None) is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("trace_lines", "data_samples", "expected_ts"),
        [
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
                    '{"timestamp": 2.0, "input_length": 3, "output_length": 3}',
                    '{"timestamp": 3.0, "input_length": 4, "output_length": 4}',
                ],
                2,
                [0.0, 1.0],
            ),
            (
                [
                    '{"timestamp": 1.0, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 1.0, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 1.0, "input_length": 300, "output_length": 30}',
                    '{"timestamp": 1.0, "input_length": 400, "output_length": 40}',
                    '{"timestamp": 1.0, "input_length": 500, "output_length": 50}',
                ],
                2,
                [0.0, 0.0],
            ),
            (
                [
                    '{"timestamp": 5.0, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 2.0, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 8.0, "input_length": 300, "output_length": 30}',
                    '{"timestamp": 1.0, "input_length": 400, "output_length": 40}',
                ],
                3,
                [0.0, 1.0, 4.0],
            ),
        ],
    )
    def test_data_samples_truncates_timestamps(
        self, tmp_path: Path, trace_lines, data_samples, expected_ts
    ):
        """data_samples truncates replay timestamps to match sampled dataset rows."""
        trace = _trace_path(tmp_path, trace_lines)
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            data_samples=data_samples,
        )
        assert kwargs["relative_timestamps"] == pytest.approx(expected_ts, abs=1e-9)

    @pytest.mark.smoke
    def test_constraints_remain_runtime_only(self, tmp_path: Path):
        """Runtime constraints are preserved and do not filter replay timestamps."""
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 0.5, "input_length": 2, "output_length": 2}',
                '{"timestamp": 1.0, "input_length": 3, "output_length": 3}',
            ],
        )
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[2.0],
            random_seed=42,
            data=[str(trace)],
            constraints={"max_requests": 2, "max_seconds": 1.5},
        )
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 0.5, 1.0], abs=1e-9)
        assert kwargs["constraints"] == {"max_requests": 2, "max_seconds": 1.5}
        assert kwargs["time_scale"] == 2.0

    @pytest.mark.smoke
    def test_data_samples_and_constraints_are_independent(self, tmp_path: Path):
        """data_samples truncates timestamps without mutating runtime constraints."""
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 2.0, "input_length": 3, "output_length": 3}',
                '{"timestamp": 3.0, "input_length": 4, "output_length": 4}',
                '{"timestamp": 4.0, "input_length": 5, "output_length": 5}',
            ],
        )
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            data_samples=3,
            constraints={"max_requests": 10, "max_seconds": 0.25},
        )
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)
        assert kwargs["constraints"] == {"max_requests": 10, "max_seconds": 0.25}
