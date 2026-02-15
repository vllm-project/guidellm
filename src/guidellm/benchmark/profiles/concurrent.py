"""Concurrent benchmark profile."""

from __future__ import annotations

import contextlib
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveInt, field_validator

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.logger import logger
from guidellm.scheduler import (
    ConcurrentStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
)
from guidellm.utils.imports import json

from .profile import Profile, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("concurrent")
class ConcurrentProfileArgs(ProfileArgs):
    """Pydantic model for concurrent profile creation arguments."""

    kind: Literal["concurrent"] = Field(
        default="concurrent",
        description="Profile type discriminator for concurrent scheduling",
    )
    streams: list[PositiveInt] = Field(
        description="Concurrent stream counts to execute",
        examples=[[1, 2, 3], 10],
    )

    @field_validator("streams", mode="before")
    @classmethod
    def _coerce_streams_to_list(cls, value: Any) -> Any:
        """Normalize streams to a list of integers.

        Allow single integer or list of integers.
        """
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                value = json.loads(value)
        if not value:
            raise ValueError("streams requires at least one value")
        if isinstance(value, list | tuple):
            streams = [int(stream) for stream in value]
            sorted_streams = sorted(streams)
            if sorted_streams != streams:
                logger.warning(
                    f"Streams reordered from {streams} to {sorted_streams} (ascending)"
                )
            return sorted_streams
        if isinstance(value, int | float):
            return [int(value)]
        raise ValueError(
            "streams must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )


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

        Stream counts are sorted ascending, so if a previous stream count was
        terminated by a failure constraint (over-saturation, errors, etc.), all
        remaining higher stream counts are skipped.

        :param prev_strategy: Previously completed strategy
        :param prev_benchmark: Benchmark results from previous execution
        :return: ConcurrentStrategy with next stream count, or None if complete
            or failure detected
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
