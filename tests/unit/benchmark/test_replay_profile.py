from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, ReplayProfile
from guidellm.benchmark.profiles.replay import ReplayProfileArgs
from guidellm.scheduler import (
    TraceReplayStrategy,
)


def _replay_args(**kwargs) -> ReplayProfileArgs:
    payload = {"kind": "replay", **kwargs}
    return ReplayProfileArgs.model_validate(payload)


def _replay_profile(
    constraints: dict[str, Any] | None = None, random_seed: int = 42, **kwargs
) -> ReplayProfile:
    args = _replay_args(**kwargs)
    return ProfileFactory.create(
        args,
        random_seed,
        constraints=constraints,
    )


class TestReplayProfile:
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_resolve_profile_passes_replay_specific_kwargs(self, tmp_path: Path):
        """
        resolve_profile wires replay data, samples, and time_scale into the profile.

        ## WRITTEN BY AI ##
        """
        profile = await resolve_profile(
            profile=ReplayProfileArgs.model_validate(
                {"kind": "replay", "time_scale": 2.0}
            ),
            constraints={"max_requests": {"max_num": 2}},
            random_seed=42,
        )

        assert isinstance(profile, ReplayProfile)
        assert profile.args.time_scale == 2.0
        assert profile.constraints["max_requests"] == {"max_num": 2}

    @pytest.mark.smoke
    def test_next_strategy_returns_trace_then_none(self, tmp_path: Path):
        """
        Replay profile yields one trace strategy then completes.

        ## WRITTEN BY AI ##
        """
        profile = _replay_profile(time_scale=2.0)

        strategy = profile.next_strategy(None, None)
        assert profile.strategy_types == ["trace"]
        assert isinstance(strategy, TraceReplayStrategy)
        assert strategy.time_scale == 2.0
        assert profile.next_strategy(strategy, None) is None
