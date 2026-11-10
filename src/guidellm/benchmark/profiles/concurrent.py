"""Concurrent benchmark profile."""

from __future__ import annotations

from collections.abc import Sequence

from guidellm.scheduler import ConcurrentStrategy
from guidellm.schemas.benchmark.profiles import ConcurrentProfileArgs

from .profile import FixedCountProfile, ProfileFactory

__all__ = ["ConcurrentProfile"]


@ProfileFactory.register("concurrent")
class ConcurrentProfile(FixedCountProfile[ConcurrentStrategy]):
    """
    Execute strategies with fixed concurrency levels for performance testing.

    Executes requests with a fixed number of concurrent streams, useful for
    testing system performance under specific concurrency levels.
    """

    args: ConcurrentProfileArgs

    @property
    def _counts(self) -> Sequence[int]:
        """
        :return: Configured stream counts to execute
        """
        return self.args.streams

    def _build_strategy(self, count: int) -> ConcurrentStrategy:
        """
        :param count: Number of concurrent streams for the next strategy
        :return: Concurrent strategy for the count
        """
        return ConcurrentStrategy(
            streams=count, rampup_duration=self.args.rampup_duration
        )
