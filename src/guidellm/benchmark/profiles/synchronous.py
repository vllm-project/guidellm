"""Synchronous benchmark profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from guidellm.scheduler import (
    SchedulingStrategy,
    SynchronousStrategy,
)

from .profile import Profile, ProfileArgs, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("synchronous")
class SynchronousProfileArgs(ProfileArgs):
    """Pydantic model for synchronous profile creation arguments."""

    kind: Literal["synchronous"] = Field(
        default="synchronous",
        description="Profile type discriminator for polymorphic serialization",
    )

    @model_validator(mode="before")
    @classmethod
    def _ensure_no_rate(cls, data: Any) -> Any:
        """Validate that user didn't provide a rate."""
        if isinstance(data, dict) and data.get("rate"):
            raise ValueError("Synchonous profile does not accept a rate parameter.")
        return data


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
        constraints: dict[str, Any] | None,
    ):
        super().__init__(args, constraints)
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
