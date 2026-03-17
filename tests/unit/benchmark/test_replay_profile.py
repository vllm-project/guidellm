## WRITTEN BY AI ##

"""
Unit tests for ReplayProfile.

Ensures replay profile loads trace timestamps and creates TraceReplayStrategy with
orrect time_scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    def test_max_requests_less_than_one_raises(self, tmp_path: Path):
        """max_requests < 1 in constraints raises ValueError."""
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )
        with pytest.raises(ValueError, match="max_requests must be >= 1"):
            ReplayProfile.resolve_args(
                rate_type="replay",
                rate=[1.0],
                random_seed=42,
                data=[str(trace)],
                constraints={"max_requests": 0},
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("trace_lines", "max_req", "expected_ts"),
        [
            # Basic truncation
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
            # Truncate concurrent burst (first 2 of 5 same-timestamp requests)
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
            # Truncate after sorting (all rows loaded, sorted, then truncated)
            # File order: 5.0, 2.0, 8.0, 1.0 -> sorted: 1.0, 2.0, 5.0, 8.0
            # Relative: 0.0, 1.0, 4.0, 7.0 -> truncated to 3: 0.0, 1.0, 4.0
            (
                [
                    '{"timestamp": 5.0, "input_length": 100, "output_length": 10}',
                    '{"timestamp": 2.0, "input_length": 200, "output_length": 20}',
                    '{"timestamp": 8.0, "input_length": 300, "output_length": 30}',
                    '{"timestamp": 1.0, "input_length": 400, "output_length": 40}',
                ],
                3,
                [0.0, 1.0, 4.0],  # 1.0->0.0, 2.0->1.0, 5.0->4.0
            ),
        ],
    )
    def test_max_requests_truncates_timestamps(
        self, tmp_path: Path, trace_lines, max_req, expected_ts
    ):
        """max_requests truncates timestamps to first N rows (handles
        duplicates/unsorted)."""
        trace = _trace_path(tmp_path, trace_lines)
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            constraints={"max_requests": max_req},
        )
        assert kwargs["relative_timestamps"] == pytest.approx(expected_ts, abs=1e-9)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("trace_lines", "rate", "max_seconds", "expected_ts"),
        [
            # Basic: time_scale=1.0, max_seconds=1.5 keeps timestamps <= 1.5
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 0.5, "input_length": 2, "output_length": 2}',
                    '{"timestamp": 1.0, "input_length": 3, "output_length": 3}',
                    '{"timestamp": 2.0, "input_length": 4, "output_length": 4}',
                ],
                [1.0],  # time_scale = 1.0
                1.5,
                [0.0, 0.5, 1.0],  # 2.0 * 1.0 = 2.0 > 1.5, so excluded
            ),
            # With time_scale=2.0: effective times are 0, 1.0, 2.0, 4.0
            # max_seconds=1.5 keeps only timestamps where ts * 2.0 <= 1.5
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 0.5, "input_length": 2, "output_length": 2}',
                    '{"timestamp": 1.0, "input_length": 3, "output_length": 3}',
                    '{"timestamp": 2.0, "input_length": 4, "output_length": 4}',
                ],
                [2.0],  # time_scale = 2.0
                1.5,
                [0.0, 0.5],  # 1.0 * 2.0 = 2.0 > 1.5, so excluded
            ),
            # With time_scale=0.5 (speedup): effective times are 0, 0.25, 0.5, 1.0
            # max_seconds=0.8 keeps only timestamps where ts * 0.5 <= 0.8
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 0.5, "input_length": 2, "output_length": 2}',
                    '{"timestamp": 1.0, "input_length": 3, "output_length": 3}',
                    '{"timestamp": 2.0, "input_length": 4, "output_length": 4}',
                ],
                [0.5],  # time_scale = 0.5
                0.8,
                [0.0, 0.5, 1.0],  # 2.0 * 0.5 = 1.0 > 0.8, so excluded
            ),
            # max_seconds larger than all timestamps: all kept
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
                ],
                [1.0],
                10.0,
                [0.0, 1.0],
            ),
            # max_seconds very small: only first timestamp kept
            (
                [
                    '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                    '{"timestamp": 0.1, "input_length": 2, "output_length": 2}',
                    '{"timestamp": 0.2, "input_length": 3, "output_length": 3}',
                ],
                [1.0],
                0.05,
                [0.0],
            ),
        ],
    )
    def test_max_seconds_filters_timestamps_with_time_scale(
        self, tmp_path: Path, trace_lines, rate, max_seconds, expected_ts
    ):
        """max_seconds filters timestamps based on effective time (ts * time_scale)."""
        trace = _trace_path(tmp_path, trace_lines)
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=rate,
            random_seed=42,
            data=[str(trace)],
            constraints={"max_seconds": max_seconds},
        )
        assert kwargs["relative_timestamps"] == pytest.approx(expected_ts, abs=1e-9)
        assert kwargs["time_scale"] == rate[0]

    @pytest.mark.smoke
    def test_max_seconds_with_max_requests_both_apply(self, tmp_path: Path):
        """Both max_seconds and max_requests constraints apply (intersection)."""
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
            constraints={"max_requests": 4, "max_seconds": 2.5},
        )
        # max_requests limits to first 4: [0, 1.0, 2.0, 3.0]
        # Then max_seconds filters to <= 2.5: [0, 1.0, 2.0]
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)

    @pytest.mark.smoke
    def test_max_seconds_filters_and_sets_max_requests(self, tmp_path: Path):
        """max_seconds filters timestamps at load time and max_requests is set to
        the actual count to synchronize the data loader and prevent benchmark hang."""
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
        constraints: dict[str, Any] = {"max_seconds": 2.5}
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            constraints=constraints,
        )
        # max_seconds=2.5 with time_scale=1.0 keeps ts <= 2.5: [0, 1.0, 2.0]
        # = 3 requests
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)
        # max_requests is always set to actual count after filtering
        assert constraints.get("max_requests") == 3
        # max_seconds is removed to avoid runtime constraint conflicts
        assert "max_seconds" not in constraints

    @pytest.mark.smoke
    def test_max_requests_always_updated_to_actual_count(self, tmp_path: Path):
        """max_requests is always set to the actual count of timestamps after
        filtering."""
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 2.0, "input_length": 3, "output_length": 3}',
                '{"timestamp": 3.0, "input_length": 4, "output_length": 4}',
            ],
        )
        constraints: dict[str, Any] = {"max_requests": 2, "max_seconds": 10.0}
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            constraints=constraints,
        )
        # max_requests=2 takes first 2 timestamps: [0, 1.0]
        # max_seconds=10.0 keeps all (ts * 1.0 <= 10.0)
        # Result: [0, 1.0] - but max_requests is always updated to actual count
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0], abs=1e-9)
        # constraints['max_requests'] is always set to actual count after filtering
        assert constraints.get("max_requests") == 2  # matches len(relative_timestamps)

    @pytest.mark.smoke
    def test_max_seconds_removed_from_constraints(self, tmp_path: Path):
        """max_seconds is removed from constraints after load-time filtering."""
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 2.0, "input_length": 3, "output_length": 3}',
            ],
        )
        constraints: dict[str, Any] = {"max_seconds": 1.5}
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            constraints=constraints,
        )
        # max_seconds should be removed to avoid runtime MaxDurationConstraint
        assert "max_seconds" not in constraints
        assert kwargs["constraints"] is constraints
        # Verify timestamps were filtered: ts <= 1.5 -> [0, 1.0]
        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0], abs=1e-9)
        # max_requests set to actual count
        assert constraints.get("max_requests") == 2
