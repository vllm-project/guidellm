"""
Orchestrate multi-strategy benchmark execution through configurable profiles.

Provides abstractions for coordinating sequential execution of scheduling strategies
during benchmarking workflows. Profiles automatically generate strategies based on
configuration parameters, manage runtime constraints, and track completion state
across execution sequences. Each profile type implements a specific execution pattern
(synchronous, concurrent, throughput-focused, rate-based async, or adaptive sweep)
that determines how benchmark requests are scheduled and executed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_serializer,
    field_validator,
)

from guidellm import settings
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.schemas import PydanticClassRegistryMixin

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark

__all__ = [
    "AsyncProfile",
    "ConcurrentProfile",
    "Profile",
    "ProfileType",
    "SweepProfile",
    "SynchronousProfile",
    "ThroughputProfile",
]

ProfileType = Annotated[
    Literal["synchronous", "concurrent", "throughput", "async", "sweep"],
    "Profile type identifiers for polymorphic deserialization",
]


class Profile(
    PydanticClassRegistryMixin["Profile"],
    ABC,
):
    """
    Coordinate multi-strategy benchmark execution with automatic strategy generation.

    Manages sequential execution of scheduling strategies with automatic strategy
    generation, constraint management, and completion tracking. Subclasses define
    specific execution patterns like synchronous, concurrent, throughput-focused,
    rate-based async, or adaptive sweep profiles.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    schema_discriminator: ClassVar[str] = "type_"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[Profile]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base Profile class for schema validation
        """
        if cls.__name__ == "Profile":
            return cls

        return Profile

    @classmethod
    def create(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> Profile:
        """
        Create profile instances based on type identifier.

        :param rate_type: Profile type identifier to instantiate
        :param rate: Rate configuration for the profile strategy
        :param random_seed: Seed for stochastic strategy reproducibility
        :param kwargs: Additional profile-specific configuration parameters
        :return: Configured profile instance for the specified type
        :raises ValueError: If rate_type is not registered
        """
        profile_class = cls.get_registered_object(rate_type)
        if profile_class is None:
            raise ValueError(f"Profile type '{rate_type}' is not registered")

        resolved_kwargs = profile_class.resolve_args(
            rate_type=rate_type, rate=rate, random_seed=random_seed, **kwargs
        )

        return profile_class(**resolved_kwargs)

    @classmethod
    @abstractmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve and validate arguments for profile construction.

        :param rate_type: Profile type identifier
        :param rate: Rate configuration parameter
        :param random_seed: Seed for stochastic strategies
        :param kwargs: Additional arguments to resolve and validate
        :return: Resolved arguments dictionary for profile initialization
        """
        ...

    type_: Literal["profile"] = Field(
        description="Profile type discriminator for polymorphic serialization",
    )
    completed_strategies: list[SchedulingStrategy] = Field(
        default_factory=list,
        description="Strategies that completed execution in this profile",
    )
    constraints: dict[str, Any | dict[str, Any] | ConstraintInitializer] | None = Field(
        default=None,
        description="Runtime constraints applied to strategy execution",
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds to ramp up the targeted scheduling rate, if applicable"
        ),
    )

    @computed_field  # type: ignore[misc]
    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types executed or to be executed in this profile
        """
        return [strat.type_ for strat in self.completed_strategies]

    def strategies_generator(
        self,
    ) -> Generator[
        tuple[SchedulingStrategy, dict[str, Constraint] | None],
        Benchmark | None,
        None,
    ]:
        """
        Generate strategies and constraints for sequential execution.

        :return: Generator yielding (strategy, constraints) tuples and receiving
            benchmark results after each execution
        """
        prev_strategy: SchedulingStrategy | None = None
        prev_benchmark: Benchmark | None = None

        while (
            strategy := self.next_strategy(prev_strategy, prev_benchmark)
        ) is not None:
            constraints = self.next_strategy_constraints(
                strategy, prev_strategy, prev_benchmark
            )
            prev_benchmark = yield (
                strategy,
                constraints,
            )
            prev_strategy = strategy
            self.completed_strategies.append(prev_strategy)

    @abstractmethod
    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> SchedulingStrategy | None:
        """
        Generate next strategy in the profile execution sequence.

        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Next strategy to execute, or None if profile complete
        """
        ...

    def next_strategy_constraints(
        self,
        next_strategy: SchedulingStrategy | None,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> dict[str, Constraint] | None:
        """
        Generate constraints for next strategy execution.

        :param next_strategy: Strategy to be executed next
        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Constraints dictionary for next strategy, or None
        """
        _ = (prev_strategy, prev_benchmark)  # unused
        return (
            ConstraintsInitializerFactory.resolve(self.constraints)
            if next_strategy and self.constraints
            else None
        )

    @field_validator("constraints", mode="before")
    @classmethod
    def _constraints_validator(
        cls, value: Any
    ) -> dict[str, Any | dict[str, Any] | ConstraintInitializer] | None:
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError("Constraints must be a dictionary")

        return {
            key: (
                ConstraintsInitializerFactory.deserialize(initializer_dict=val)
                if isinstance(val, dict)
                and "type_" in val
                and not isinstance(val, ConstraintInitializer)
                else val
            )
            for key, val in value.items()
        }

    @field_serializer("constraints")
    def _constraints_serializer(
        self,
        constraints: dict[str, Any | dict[str, Any] | ConstraintInitializer] | None,
    ) -> dict[str, Any | dict[str, Any]] | None:
        if constraints is None:
            return None

        return {
            key: (
                val
                if not isinstance(val, ConstraintInitializer)
                else ConstraintsInitializerFactory.serialize(initializer=val)
            )
            for key, val in constraints.items()
        }


@Profile.register("synchronous")
class SynchronousProfile(Profile):
    """
    Execute single synchronous strategy for baseline performance metrics.

    Executes requests sequentially with one request at a time, establishing
    baseline performance metrics without concurrent execution overhead.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for synchronous profile construction.

        :param rate_type: Profile type identifier (ignored)
        :param rate: Rate parameter (must be None)
        :param random_seed: Random seed (ignored)
        :param kwargs: Additional arguments passed through unchanged
        :return: Resolved arguments dictionary
        :raises ValueError: If rate is not None
        """
        _ = (rate_type, random_seed)  # unused
        if rate is not None:
            raise ValueError("SynchronousProfile does not accept a rate parameter")

        return kwargs

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Single synchronous strategy type
        """
        return [self.type_]

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


@Profile.register("concurrent")
class ConcurrentProfile(Profile):
    """
    Execute strategies with fixed concurrency levels for performance testing.

    Executes requests with a fixed number of concurrent streams, useful for
    testing system performance under specific concurrency levels.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: list[PositiveInt] = Field(
        description="Concurrent stream counts for request scheduling",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for concurrent profile construction.

        :param rate_type: Profile type identifier (ignored)
        :param rate: Rate parameter remapped to streams
        :param random_seed: Random seed (ignored)
        :param kwargs: Additional arguments passed through unchanged
        :return: Resolved arguments dictionary
        :raises ValueError: If rate is None
        """
        _ = (rate_type, random_seed)  # unused
        rate = rate if isinstance(rate, list) or rate is None else [rate]
        kwargs["streams"] = [int(stream) for stream in rate] if rate else None
        return kwargs

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Concurrent strategy types for each configured stream count
        """
        return [self.type_] * len(self.streams)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ConcurrentStrategy | None:
        """
        Generate concurrent strategy for next stream count.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ConcurrentStrategy with next stream count, or None if complete
        """
        _ = (prev_strategy, prev_benchmark)  # unused

        if len(self.completed_strategies) >= len(self.streams):
            return None

        return ConcurrentStrategy(
            streams=self.streams[len(self.completed_strategies)],
            rampup_duration=self.rampup_duration,
        )


@Profile.register("throughput")
class ThroughputProfile(Profile):
    """
    Maximize system throughput with optional concurrency constraints.

    Maximizes system throughput by maintaining maximum concurrent requests,
    optionally constrained by a concurrency limit.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for throughput profile construction.

        :param rate_type: Profile type identifier (ignored)
        :param rate: Rate parameter remapped to max_concurrency
        :param random_seed: Random seed (ignored)
        :param kwargs: Additional arguments passed through unchanged
        :return: Resolved arguments dictionary
        """
        _ = (rate_type, random_seed)  # unused
        # Remap rate to max_concurrency, strip out random_seed
        kwargs.pop("random_seed", None)
        if rate is not None and len(rate) > 0:
            kwargs["max_concurrency"] = rate[0]
        else:
            # Require explicit max_concurrency; in the future max_concurrency
            # should be dynamic and rate can specify some tunable
            raise ValueError("ThroughputProfile requires a rate parameter")
        return kwargs

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Single throughput strategy type
        """
        return [self.type_]

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ThroughputStrategy | None:
        """
        Generate throughput strategy for first execution only.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ThroughputStrategy for first execution, None afterward
        """
        _ = (prev_strategy, prev_benchmark)  # unused
        if len(self.completed_strategies) >= 1:
            return None

        return ThroughputStrategy(
            max_concurrency=self.max_concurrency, rampup_duration=self.rampup_duration
        )


@Profile.register(["async", "constant", "poisson"])
class AsyncProfile(Profile):
    """
    Schedule requests at specified rates using constant or Poisson patterns.

    Schedules requests at specified rates using either constant interval or
    Poisson distribution patterns for realistic load simulation.
    """

    type_: Literal["async", "constant", "poisson"] = "async"  # type: ignore[assignment]
    strategy_type: Literal["constant", "poisson"] = Field(
        description="Asynchronous strategy pattern type",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rates in requests per second",
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for async profile construction.

        :param rate_type: Profile type identifier
        :param rate: Rate configuration for the profile
        :param random_seed: Seed for stochastic strategies
        :param kwargs: Additional arguments passed through unchanged
        :return: Resolved arguments dictionary
        :raises ValueError: If rate is None
        """
        if rate is None:
            raise ValueError("AsyncProfile requires a rate parameter")

        kwargs["type_"] = (
            rate_type
            if rate_type in ["async", "constant", "poisson"]
            else kwargs.get("type_", "async")
        )
        kwargs["strategy_type"] = (
            rate_type
            if rate_type in ["constant", "poisson"]
            else kwargs.get("strategy_type", "constant")
        )
        kwargs["rate"] = rate if isinstance(rate, list) else [rate]
        kwargs["random_seed"] = random_seed
        return kwargs

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Async strategy types for each configured rate
        """
        num_strategies = len(self.rate)
        return [self.strategy_type] * num_strategies

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> AsyncConstantStrategy | AsyncPoissonStrategy | None:
        """
        Generate async strategy for next configured rate.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: AsyncConstantStrategy or AsyncPoissonStrategy for next rate,
            or None if all rates completed
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        _ = (prev_strategy, prev_benchmark)  # unused

        if len(self.completed_strategies) >= len(self.rate):
            return None

        current_rate = self.rate[len(self.completed_strategies)]

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=current_rate,
                max_concurrency=self.max_concurrency,
                rampup_duration=self.rampup_duration,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=current_rate,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")


@Profile.register("sweep")
class SweepProfile(Profile):
    """
    Discover optimal rate range through adaptive multi-strategy execution.

    Automatically discovers optimal rate range by executing synchronous and
    throughput strategies first, then interpolating rates for async strategies
    to comprehensively sweep the performance space.
    """

    type_: Literal["sweep"] = "sweep"  # type: ignore[assignment]
    sweep_size: int = Field(
        description="Number of strategies to generate for the sweep",
        ge=2,
    )
    strategy_type: Literal["constant", "poisson"] = "constant"
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )
    synchronous_rate: float = Field(
        default=-1.0,
        description="Measured rate from synchronous strategy execution",
    )
    throughput_rate: float = Field(
        default=-1.0,
        description="Measured rate from throughput strategy execution",
    )
    async_rates: list[float] = Field(
        default_factory=list,
        description="Generated rates for async strategy sweep",
    )
    measured_rates: list[float] = Field(
        default_factory=list,
        description="Interpolated rates between synchronous and throughput",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: list[float] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for sweep profile construction.

        :param rate_type: Async strategy type for sweep execution
        :param rate: Rate parameter specifying sweep size (if provided)
        :param random_seed: Seed for stochastic strategies
        :param kwargs: Additional arguments passed through unchanged
        :return: Resolved arguments dictionary
        """
        sweep_size_from_rate = int(rate[0]) if rate else settings.default_sweep_number
        kwargs["sweep_size"] = kwargs.get("sweep_size", sweep_size_from_rate)
        kwargs["random_seed"] = random_seed
        if rate_type in ["constant", "poisson"]:
            kwargs["strategy_type"] = rate_type
        return kwargs

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types for the complete sweep sequence
        """
        types = ["synchronous", "throughput"]
        types += [self.strategy_type] * (self.sweep_size - len(types))
        return types

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> (
        AsyncConstantStrategy
        | AsyncPoissonStrategy
        | SynchronousStrategy
        | ThroughputStrategy
        | None
    ):
        """
        Generate next strategy in adaptive sweep sequence.

        Executes synchronous and throughput strategies first to measure baseline
        rates, then generates interpolated rates for async strategies.

        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Next strategy in sweep sequence, or None if complete
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        if prev_strategy is None:
            return SynchronousStrategy()

        if prev_strategy.type_ == "synchronous":
            self.synchronous_rate = prev_benchmark.request_throughput.successful.mean

            return ThroughputStrategy(
                max_concurrency=self.max_concurrency,
                rampup_duration=self.rampup_duration,
            )

        if prev_strategy.type_ == "throughput":
            self.throughput_rate = prev_benchmark.request_throughput.successful.mean
            if self.synchronous_rate <= 0 and self.throughput_rate <= 0:
                raise RuntimeError(
                    "Invalid rates in sweep; aborting. "
                    "Were there any successful requests?"
                )
            self.measured_rates = list(
                np.linspace(
                    self.synchronous_rate,
                    self.throughput_rate,
                    self.sweep_size - 1,
                )
            )[1:]  # don't rerun synchronous

        next_index = (
            len(self.completed_strategies) - 1 - 1
        )  # subtract synchronous and throughput
        next_rate = (
            self.measured_rates[next_index]
            if next_index < len(self.measured_rates)
            else None
        )

        if next_rate is None or next_rate <= 0:
            # Stop if we don't have another valid rate to run
            return None

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=next_rate, max_concurrency=self.max_concurrency
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=next_rate,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
