from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import Profile, ReplayProfile
from guidellm.scheduler import TraceReplayStrategy


def _trace_path(tmp_path: Path, lines: list[str] | None = None) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines or []))
    return path


class TestReplayProfile:
    @pytest.mark.smoke
    def test_resolve_args_requires_data(self):
        with pytest.raises(ValueError, match="Replay profile requires data"):
            ReplayProfile.resolve_args(
                rate_type="replay",
                rate=[1.0],
                random_seed=42,
            )

    @pytest.mark.smoke
    def test_resolve_args_rejects_missing_or_empty_trace(self, tmp_path: Path):
        missing = tmp_path / "missing.jsonl"
        with pytest.raises(ValueError, match="not found"):
            ReplayProfile.resolve_args(
                rate_type="replay",
                rate=[1.0],
                random_seed=42,
                data=[str(missing)],
            )

        empty = _trace_path(tmp_path)
        with pytest.raises(ValueError, match="empty|No timestamps"):
            ReplayProfile.resolve_args(
                rate_type="replay",
                rate=[1.0],
                random_seed=42,
                data=[str(empty)],
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("rate", "expected_scale"),
        [
            (None, 1.0),
            ([2.0], 2.0),
        ],
    )
    def test_profile_create_resolves_timestamps_and_time_scale(
        self, tmp_path: Path, rate, expected_scale
    ):
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 5.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 2.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 8.0, "input_length": 3, "output_length": 3}',
            ],
        )

        profile = Profile.create(
            rate_type="replay",
            rate=rate,
            random_seed=42,
            data=[str(trace)],
        )

        assert isinstance(profile, ReplayProfile)
        assert profile.relative_timestamps == pytest.approx([0.0, 3.0, 6.0], abs=1e-9)
        assert profile.time_scale == expected_scale

    @pytest.mark.sanity
    def test_non_positive_time_scale_is_rejected(self, tmp_path: Path):
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )

        with pytest.raises(ValidationError):
            Profile.create(
                rate_type="replay",
                rate=[0.0],
                random_seed=42,
                data=[str(trace)],
            )

    @pytest.mark.smoke
    def test_custom_timestamp_column_via_data_args(self, tmp_path: Path):
        trace = _trace_path(
            tmp_path,
            [
                '{"ts": 5.0, "input_length": 100, "output_length": 10}',
                '{"ts": 2.0, "input_length": 200, "output_length": 20}',
                '{"ts": 8.0, "input_length": 300, "output_length": 30}',
            ],
        )

        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            data_args=[{"timestamp_column": "ts"}],
        )

        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 3.0, 6.0], abs=1e-9)

    @pytest.mark.smoke
    @pytest.mark.parametrize("invalid_value", [None, "", "   ", 123, False, []])
    def test_invalid_timestamp_column_config_falls_back_to_default(
        self, tmp_path: Path, invalid_value
    ):
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 10.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 12.0, "input_length": 2, "output_length": 2}',
            ],
        )

        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            data_args=[{"timestamp_column": invalid_value}],
        )

        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 2.0], abs=1e-9)

    @pytest.mark.smoke
    def test_data_samples_truncates_after_sorting_and_preserves_constraints(
        self, tmp_path: Path
    ):
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 5.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 2.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 8.0, "input_length": 3, "output_length": 3}',
                '{"timestamp": 1.0, "input_length": 4, "output_length": 4}',
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

        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0, 4.0], abs=1e-9)
        assert kwargs["constraints"] == {"max_requests": 10, "max_seconds": 0.25}

    @pytest.mark.smoke
    @pytest.mark.parametrize("data_samples", [0, -1])
    def test_non_positive_data_samples_do_not_truncate(
        self, tmp_path: Path, data_samples: int
    ):
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
            ],
        )

        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[1.0],
            random_seed=42,
            data=[str(trace)],
            data_samples=data_samples,
        )

        assert kwargs["relative_timestamps"] == pytest.approx([0.0, 1.0], abs=1e-9)

    @pytest.mark.smoke
    def test_resolve_profile_passes_replay_specific_kwargs(self, tmp_path: Path):
        trace = _trace_path(
            tmp_path,
            [
                '{"ts": 5.0, "input_length": 1, "output_length": 1}',
                '{"ts": 2.0, "input_length": 2, "output_length": 2}',
                '{"ts": 8.0, "input_length": 3, "output_length": 3}',
            ],
        )

        profile = asyncio.run(
            resolve_profile(
                profile="replay",
                rate=[2.0],
                random_seed=42,
                rampup=0.0,
                constraints={},
                max_seconds=None,
                max_requests=2,
                max_errors=None,
                max_error_rate=None,
                max_global_error_rate=None,
                data=[str(trace)],
                data_args=[{"timestamp_column": "ts"}],
                data_samples=2,
            )
        )

        assert isinstance(profile, ReplayProfile)
        assert profile.relative_timestamps == pytest.approx([0.0, 3.0], abs=1e-9)
        assert profile.time_scale == 2.0
        assert profile.constraints["max_requests"] == 2

    @pytest.mark.smoke
    def test_next_strategy_returns_trace_then_none(self, tmp_path: Path):
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )
        kwargs = ReplayProfile.resolve_args(
            rate_type="replay",
            rate=[2.0],
            random_seed=42,
            data=[str(trace)],
        )
        profile = ReplayProfile(**kwargs)

        strategy = profile.next_strategy(None, None)
        assert profile.strategy_types == ["trace"]
        assert isinstance(strategy, TraceReplayStrategy)
        assert strategy.relative_timestamps == [0.0]
        assert strategy.time_scale == 2.0
        assert profile.next_strategy(strategy, None) is None
