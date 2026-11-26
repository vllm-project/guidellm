"""Users benchmark profile."""

from __future__ import annotations

from collections.abc import Sequence

from guidellm.scheduler import UsersStrategy
from guidellm.schemas.benchmark.profiles import UsersProfileArgs

from .profile import FixedCountProfile, ProfileFactory

__all__ = ["UsersProfile"]


@ProfileFactory.register("users")
class UsersProfile(FixedCountProfile[UsersStrategy]):
    """
    Execute strategies with fixed numbers of simulated users.

    Each user holds one dataset-defined conversation session at a time,
    waits out the dataset requeue delay between turns, and starts the
    next conversation when one finishes. Useful for answering how many
    concurrent users a server can support, unlike the concurrent profile
    which only fixes the number of parallel request streams.
    """

    args: UsersProfileArgs

    @property
    def _counts(self) -> Sequence[int]:
        """
        :return: Configured user counts to execute
        """
        return self.args.users

    def _build_strategy(self, count: int) -> UsersStrategy:
        """
        :param count: Number of simulated users for the next strategy
        :return: Users strategy for the count
        """
        return UsersStrategy(users=count, rampup_duration=self.args.rampup_duration)
