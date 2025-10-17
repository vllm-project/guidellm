"""
Profile configurations for orchestrating multi-strategy benchmark execution.

Provides configurable abstractions for coordinating sequential execution of
scheduling strategies during benchmarking workflows. Profiles automatically
generate strategies based on configuration parameters, manage runtime
constraints, and track completion state across the execution sequence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
    StrategyType,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.utils import PydanticClassRegistryMixin

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

ProfileType = Literal["synchronous", "concurrent", "throughput", "async", "sweep"]


class Profile(
    PydanticClassRegistryMixin["type[Profile]"],
    ABC,
):
    """
    Abstract base for coordinating multi-strategy benchmark execution.

    Manages sequential execution of scheduling strategies with automatic strategy
    generation, constraint management, and completion tracking. Subclasses define
    specific execution patterns like synchronous, concurrent, throughput-focused,
    rate-based async, or adaptive sweep profiles.

    :cvar schema_discriminator: Field name used for polymorphic deserialization
    """

    schema_discriminator: ClassVar[str] = "type_"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[Profile]:
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
        Factory method to create a profile instance based on type.

        :param rate_type: Profile type identifier to instantiate
        :param rate: Rate configuration for the profile strategy
        :param random_seed: Seed for stochastic strategy reproducibility
        :param kwargs: Additional profile-specific configuration parameters
        :return: Configured profile instance for the specified type
        :raises ValueError: If rate_type is not registered
        """
        profile_class: type[Profile] = cls.get_registered_object(rate_type)
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
        description="Strategies that have completed execution in this profile",
    )
    constraints: dict[str, Any | dict[str, Any] | ConstraintInitializer] | None = Field(
        default=None,
        description="Runtime constraints applied to strategy execution",
    )

    @computed_field  # type: ignore[misc]
    @property
    def strategy_types(self) -> list[StrategyType]:
        """
        :return: Strategy types executed or expected to execute in this profile
        """
        return [strat.type_ for strat in self.completed_strategies]

    def strategies_generator(
        self,
    ) -> Generator[
        tuple[
            SchedulingStrategy | None,
            dict[str, Any | dict[str, Any] | Constraint] | None,
        ],
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
        Generate the next strategy in the profile execution sequence.

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
    ) -> dict[str, Any | dict[str, Any] | Constraint] | None:
        """
        Generate constraints for the next strategy execution.

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
                val
                if not isinstance(val, ConstraintInitializer)
                else ConstraintsInitializerFactory.deserialize(initializer_dict=val)
            )
            for key, val in value.items()
        }

    @field_serializer
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
    """Single synchronous strategy execution profile."""

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
    def strategy_types(self) -> list[StrategyType]:
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
        Generate synchronous strategy or None if already completed.

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
    """Fixed-concurrency strategy execution profile with configurable stream counts."""

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: list[PositiveInt] = Field(
        description="Concurrent stream counts for request scheduling",
    )
    startup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "before completion-based timing"
        ),
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
    def strategy_types(self) -> list[StrategyType]:
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
        Generate concurrent strategy for the next stream count.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ConcurrentStrategy with next stream count, or None if complete
        """
        _ = (prev_strategy, prev_benchmark)  # unused

        if len(self.completed_strategies) >= len(self.streams):
            return None

        return ConcurrentStrategy(
            streams=self.streams[len(self.completed_strategies)],
            startup_duration=self.startup_duration,
        )


@Profile.register("throughput")
class ThroughputProfile(Profile):
    """
    Maximum throughput strategy execution profile with optional concurrency limits.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )
    startup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "before full throughput scheduling"
        ),
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
        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
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
        Generate throughput strategy or None if already completed.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ThroughputStrategy for first execution, None afterward
        """
        _ = (prev_strategy, prev_benchmark)  # unused
        if len(self.completed_strategies) >= 1:
            return None

        return ThroughputStrategy(
            max_concurrency=self.max_concurrency,
            startup_duration=self.startup_duration,
        )


@Profile.register(["async", "constant", "poisson"])
class AsyncProfile(Profile):
    """Rate-based asynchronous strategy execution profile with configurable patterns."""

    type_: Literal["async", "constant", "poisson"] = "async"  # type: ignore[assignment]
    strategy_type: Literal["constant", "poisson"] = Field(
        description="Asynchronous strategy pattern type to use",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rate in requests per second",
    )
    startup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "to converge quickly to desired rate"
        ),
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
    def strategy_types(self) -> list[StrategyType]:
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
        Generate async strategy for the next configured rate.

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
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=current_rate,
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")


@Profile.register("sweep")
class SweepProfile(Profile):
    """Adaptive multi-strategy sweep execution profile with rate discovery."""

    type_: Literal["sweep"] = "sweep"  # type: ignore[assignment]
    sweep_size: int = Field(
        description="Number of strategies to generate for the sweep",
        ge=2,
    )
    strategy_type: Literal["constant", "poisson"] = "constant"
    startup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "to converge quickly to desired rate"
        ),
    )
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
    def strategy_types(self) -> list[StrategyType]:
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
        | SynchronousProfile
        | ThroughputProfile
        | None
    ):
        """
        Generate the next strategy in the adaptive sweep sequence.

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
            self.synchronous_rate = prev_benchmark.get_request_metrics_sample()[
                "request_throughput"
            ]

            return ThroughputStrategy(
                max_concurrency=self.max_concurrency,
                startup_duration=self.startup_duration,
            )

        if prev_strategy.type_ == "throughput":
            self.throughput_rate = prev_benchmark.get_request_metrics_sample()[
                "request_throughput"
            ]
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

        if len(self.completed_strategies) >= self.sweep_size:
            return None

        next_rate_index = len(
            [
                strat
                for strat in self.completed_strategies
                if strat.type_ == self.strategy_type
            ]
        )
        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=self.measured_rates[next_rate_index],
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=self.measured_rates[next_rate_index],
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
