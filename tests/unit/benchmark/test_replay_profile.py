from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, ReplayProfile
from guidellm.benchmark.profiles.replay import ReplayProfileArgs
from guidellm.data.deserializers import TraceSyntheticDataArgs
from guidellm.scheduler import MaxNumberConstraint, TraceReplayStrategy


def _trace_path(tmp_path: Path, lines: list[str] | None = None) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines or []))
    return path


def _replay_args(**kwargs) -> ReplayProfileArgs:
    payload = {"kind": "replay", **kwargs}
    return ReplayProfileArgs.model_validate(payload)


def _replay_profile(
    constraints: dict[str, Any] | None = None, random_seed: int = 42, **kwargs
) -> ReplayProfile:
    data = kwargs.pop("data", None)
    data_samples = kwargs.pop("data_samples", -1)
    args = _replay_args(**kwargs)
    profile_kwargs: dict[str, Any] = {"data_samples": data_samples}
    if data is not None:
        profile_kwargs["data"] = data
    return ProfileFactory.create(
        args, random_seed, constraints=constraints, **profile_kwargs
    )


class TestReplayProfile:
    @pytest.mark.smoke
    def test_requires_data(self):
        """
        Replay profile requires a trace data source.

        ## WRITTEN BY AI ##
        """
        args = _replay_args()
        with pytest.raises(ValueError, match="exactly one data source"):
            ProfileFactory.create(args, 42)

    @pytest.mark.smoke
    def test_rejects_multiple_data_sources(self, tmp_path: Path):
        """
        Replay profile rejects more than one data source.

        ## WRITTEN BY AI ##
        """
        args = _replay_args()
        with pytest.raises(ValueError, match="exactly one data source"):
            ProfileFactory.create(
                args,
                42,
                data=[
                    TraceSyntheticDataArgs(path=tmp_path / "trace-a.jsonl"),
                    TraceSyntheticDataArgs(path=tmp_path / "trace-b.jsonl"),
                ],
            )

    @pytest.mark.smoke
    def test_rejects_missing_or_empty_trace(self, tmp_path: Path):
        """
        Replay profile rejects missing files and empty traces.

        ## WRITTEN BY AI ##
        """
        missing = tmp_path / "missing.jsonl"
        args = _replay_args()
        with pytest.raises(ValueError, match="not found"):
            ProfileFactory.create(args, 42, data=[TraceSyntheticDataArgs(path=missing)])

        empty = _trace_path(tmp_path)
        with pytest.raises(ValueError, match="empty|No timestamps"):
            ProfileFactory.create(args, 42, data=[TraceSyntheticDataArgs(path=empty)])

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("rate", "expected_scale"),
        [
            (None, 1.0),
            ([2.0], 2.0),
        ],
    )
    def test_profile_create_resolves_time_scale_and_default_max_requests(
        self, tmp_path: Path, rate, expected_scale
    ):
        """
        Profile.create resolves time scale and default max_requests from trace data.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 5.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 2.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 8.0, "input_length": 3, "output_length": 3}',
            ],
        )

        payload: dict = {
            "data": [TraceSyntheticDataArgs(path=trace)],
        }
        if rate is not None:
            payload["rate"] = rate

        profile = _replay_profile(**payload)

        assert isinstance(profile, ReplayProfile)
        assert profile.args.time_scale == expected_scale
        assert profile.constraints["max_requests"] == MaxNumberConstraint(
            type_="max_number", max_num=3, current_index=0
        )

    @pytest.mark.sanity
    def test_non_positive_time_scale_is_rejected(self, tmp_path: Path):
        """
        Non-positive time scale values are rejected during argument validation.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )

        with pytest.raises(ValidationError):
            _replay_profile(
                rate=[0.0],
                data=[TraceSyntheticDataArgs(path=trace)],
            )

    @pytest.mark.smoke
    def test_custom_timestamp_column_via_data_args(self, tmp_path: Path):
        """
        Custom timestamp columns are honored when loading trace timestamps.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"ts": 5.0, "input_length": 100, "output_length": 10}',
                '{"ts": 2.0, "input_length": 200, "output_length": 20}',
                '{"ts": 8.0, "input_length": 300, "output_length": 30}',
            ],
        )

        profile = _replay_profile(
            data=[TraceSyntheticDataArgs(path=trace, timestamp_column="ts")],
        )

        assert profile.constraints["max_requests"] == MaxNumberConstraint(
            type_="max_number", max_num=3, current_index=0
        )

    @pytest.mark.smoke
    def test_large_bursty_trace_sets_default_request_constraint(self, tmp_path: Path):
        """
        Default max_requests matches the number of loaded trace events.

        ## WRITTEN BY AI ##
        """
        prompt_lengths = [
            6755,
            7319,
            7234,
            2287,
            9013,
            6506,
            4824,
            3119,
            23090,
            3135,
            26874,
            10487,
            17448,
            6253,
            6725,
            13538,
            87162,
            6166,
            6320,
            2007,
            3174,
            3131,
            3159,
            6820,
            3154,
            9416,
            7460,
        ]
        timestamps = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
        trace = _trace_path(
            tmp_path,
            [
                (
                    f'{{"timestamp": {timestamp}, '
                    f'"input_length": {prompt_length}, "output_length": 1}}'
                )
                for timestamp, prompt_length in zip(
                    timestamps, prompt_lengths, strict=True
                )
            ],
        )

        profile = _replay_profile(data=[TraceSyntheticDataArgs(path=trace)])

        assert profile.constraints["max_requests"] == MaxNumberConstraint(
            type_="max_number", max_num=27, current_index=0
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("invalid_value", [None, 123, False, []])
    def test_non_string_timestamp_column_rejected_by_pydantic(
        self, tmp_path: Path, invalid_value
    ):
        """
        Trace data args reject invalid timestamp column types.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            TraceSyntheticDataArgs(
                path=tmp_path / "trace.jsonl",
                timestamp_column=invalid_value,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize("invalid_col", ["", "   "])
    def test_blank_timestamp_column_raises_error(self, tmp_path: Path, invalid_col):
        """
        Blank timestamp column names fail when loading trace timestamps.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 10.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 12.0, "input_length": 2, "output_length": 2}',
            ],
        )

        with pytest.raises((KeyError, ValueError)):
            _replay_profile(
                data=[TraceSyntheticDataArgs(path=trace, timestamp_column=invalid_col)],
            )

    @pytest.mark.smoke
    def test_data_samples_truncates_after_sorting_and_preserves_constraints(
        self, tmp_path: Path
    ):
        """
        data_samples truncates timestamps while preserving explicit constraints.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 5.0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 2.0, "input_length": 2, "output_length": 2}',
                '{"timestamp": 8.0, "input_length": 3, "output_length": 3}',
                '{"timestamp": 1.0, "input_length": 4, "output_length": 4}',
            ],
        )

        profile = _replay_profile(
            data=[TraceSyntheticDataArgs(path=trace)],
            data_samples=3,
            constraints={"max_requests": 10, "max_seconds": 0.25},
        )

        assert profile.constraints == {"max_requests": 10, "max_seconds": 0.25}

    @pytest.mark.smoke
    @pytest.mark.parametrize("data_samples", [0, -1])
    def test_non_positive_data_samples_do_not_truncate(
        self, tmp_path: Path, data_samples: int
    ):
        """
        Non-positive data_samples values keep the full trace.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"timestamp": 0, "input_length": 1, "output_length": 1}',
                '{"timestamp": 1.0, "input_length": 2, "output_length": 2}',
            ],
        )

        profile = _replay_profile(
            data=[TraceSyntheticDataArgs(path=trace)],
            data_samples=data_samples,
        )

        assert profile.constraints["max_requests"] == MaxNumberConstraint(
            type_="max_number", max_num=2, current_index=0
        )

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_resolve_profile_passes_replay_specific_kwargs(self, tmp_path: Path):
        """
        resolve_profile wires replay data, samples, and rate into the profile.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            [
                '{"ts": 5.0, "input_length": 1, "output_length": 1}',
                '{"ts": 2.0, "input_length": 2, "output_length": 2}',
                '{"ts": 8.0, "input_length": 3, "output_length": 3}',
            ],
        )

        profile = await resolve_profile(
            profile=ReplayProfileArgs.model_validate({"kind": "replay", "rate": [2.0]}),
            constraints={"max_requests": 2},
            random_seed=42,
            data=[TraceSyntheticDataArgs(path=trace, timestamp_column="ts")],
            data_samples=2,
        )

        assert isinstance(profile, ReplayProfile)
        assert profile.args.time_scale == 2.0
        assert profile.constraints["max_requests"] == 2

    @pytest.mark.smoke
    def test_next_strategy_returns_trace_then_none(self, tmp_path: Path):
        """
        Replay profile yields one trace strategy then completes.

        ## WRITTEN BY AI ##
        """
        trace = _trace_path(
            tmp_path,
            ['{"timestamp": 0, "input_length": 1, "output_length": 1}'],
        )
        profile = _replay_profile(
            rate=[2.0],
            data=[TraceSyntheticDataArgs(path=trace)],
        )

        strategy = profile.next_strategy(None, None)
        assert profile.strategy_types == ["trace"]
        assert isinstance(strategy, TraceReplayStrategy)
        assert strategy.time_scale == 2.0
        assert profile.next_strategy(strategy, None) is None
