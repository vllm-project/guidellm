"""Trace replay benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from guidellm.scheduler import (
    ConstraintInitializer,
    SchedulingStrategy,
    TraceReplayStrategy,
)
from guidellm.schemas.benchmark.profiles import ReplayProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["ReplayProfile", "ReplayProfileArgs"]

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileFactory.register("replay")
class ReplayProfile(Profile):
    """
    Replay a trace file using per-row ``relative_timestamp`` from the dataset.

    Each request is scheduled at
    ``start_time + time_scale * relative_timestamp`` via ``RequestSettings`` on
    the dataset finalizer output. For this profile, ``rate`` is interpreted as
    ``time_scale`` (not requests per second).

    When ``data_samples`` is set, the default ``max_requests`` constraint matches
    the truncated dataset size.
    """

    args: ReplayProfileArgs

    def __init__(
        self,
        args: ReplayProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args

    @property
    def strategy_types(self) -> list[str]:
        return ["trace"]

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> TraceReplayStrategy | None:
        _ = prev_benchmark
        # Replay has a single strategy; return it once, then None
        if prev_strategy is not None:
            return None
        return TraceReplayStrategy(time_scale=self.args.time_scale)
