"""Concurrent benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from guidellm.scheduler import (
    ConcurrentStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
)
from guidellm.schemas.benchmark.profiles import ConcurrentProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["ConcurrentProfile", "ConcurrentProfileArgs"]

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileFactory.register("concurrent")
class ConcurrentProfile(Profile):
    """
    Execute strategies with fixed concurrency levels for performance testing.

    Executes requests with a fixed number of concurrent streams, useful for
    testing system performance under specific concurrency levels.
    """

    args: ConcurrentProfileArgs

    def __init__(
        self,
        args: ConcurrentProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Concurrent strategy types for each configured stream count
        """
        return [self.kind] * len(self.args.streams)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ConcurrentStrategy | None:
        """
        Generate concurrent strategy for next stream count.

        If a previous stream count was terminated by a constraint with
        stopping_scope='all', remaining stream counts are skipped.

        :param prev_strategy: Previously completed strategy
        :param prev_benchmark: Benchmark results from previous execution
        :return: ConcurrentStrategy with next stream count, or None if complete
            or escalation halted
        """
        _ = prev_strategy

        if len(self.completed_strategies) >= len(self.args.streams):
            return None

        if prev_benchmark is not None and self._should_stop_escalating(prev_benchmark):
            return None

        return ConcurrentStrategy(
            streams=self.args.streams[len(self.completed_strategies)],
            rampup_duration=self.args.rampup_duration,
        )
