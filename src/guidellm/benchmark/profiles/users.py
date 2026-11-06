"""Users benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from guidellm.scheduler import (
    ConstraintInitializer,
    SchedulingStrategy,
    UsersStrategy,
)
from guidellm.schemas.benchmark.profiles import UsersProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["UsersProfile"]

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileFactory.register("users")
class UsersProfile(Profile):
    """
    Execute strategies with fixed numbers of simulated users.

    Each user holds one dataset-defined conversation session at a time,
    waits out the dataset requeue delay between turns, and starts the
    next conversation when one finishes. Useful for answering how many
    concurrent users a server can support, unlike the concurrent profile
    which only fixes the number of parallel request streams.
    """

    args: UsersProfileArgs

    def __init__(
        self,
        args: UsersProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Users strategy types for each configured user count
        """
        return [self.kind] * len(self.args.users)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> UsersStrategy | None:
        """
        Generate users strategy for next user count.

        If a previous user count was terminated by a constraint with
        stopping_scope='all', remaining user counts are skipped.

        :param prev_strategy: Previously completed strategy
        :param prev_benchmark: Benchmark results from previous execution
        :return: UsersStrategy with next user count, or None if complete
            or escalation halted
        """
        _ = prev_strategy

        if len(self.completed_strategies) >= len(self.args.users):
            return None

        if prev_benchmark is not None and self._should_stop_escalating(prev_benchmark):
            return None

        return UsersStrategy(
            users=self.args.users[len(self.completed_strategies)],
            rampup_duration=self.args.rampup_duration,
        )
