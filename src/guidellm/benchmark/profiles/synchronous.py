"""Synchronous benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.scheduler import (
    ConstraintInitializer,
    SchedulingStrategy,
    SynchronousStrategy,
)

from .profile import Profile, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("synchronous")
class SynchronousProfileArgs(ProfileArgs):
    """Pydantic model for synchronous profile creation arguments."""

    kind: Literal["synchronous"] = Field(
        default="synchronous",
        description="Profile type discriminator for polymorphic serialization",
    )


@ProfileFactory.register("synchronous")
class SynchronousProfile(Profile):
    """
    Execute single synchronous strategy for baseline performance metrics.

    Executes requests sequentially with one request at a time, establishing
    baseline performance metrics without concurrent execution overhead.
    """

    args: SynchronousProfileArgs

    def __init__(
        self,
        args: SynchronousProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Single synchronous strategy type
        """
        return [self.kind]

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> SynchronousStrategy | None:
        """
        Generate synchronous strategy for first execution only.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: SynchronousStrategy for first execution, None afterward
        """
        _ = (prev_strategy, prev_benchmark)  # unused
        if len(self.completed_strategies) >= 1:
            return None

        return SynchronousStrategy()
