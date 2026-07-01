"""
Profile base class for multi-strategy benchmark execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, MutableMapping
from typing import Any

from guidellm.benchmark.schemas import Benchmark, ProfileArgs
from guidellm.logger import logger
from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
)
from guidellm.utils.registry import RegistryMixin


class ProfileFactory(RegistryMixin["type[Profile]"]):
    @classmethod
    def create(
        cls,
        args: ProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None = None,
        **kwargs: Any,
    ) -> Profile:
        """
        Create profile instances from validated profile arguments.

        :param args: Validated profile argument model for the target profile type
        :param random_seed: Seed for reproducible random operations in profile
            strategies.
        :param constraints: Constraints for the profile strategies.
        :param kwargs: Additional profile-specific configuration parameters
        :return: Configured profile instance for the specified type
        :raises ValueError: If the profile kind is not registered
        """
        kind = args.kind

        profile_class = cls.get_registered_object(kind)

        if profile_class is None:
            raise ValueError(
                f"Profile type '{kind}' is not registered. "
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return profile_class(args, random_seed, constraints, **kwargs)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """
        Get all registered names from the registry.
        """
        return tuple(cls.registry.keys() if cls.registry else [])


class Profile(ABC):
    """
    Coordinate multi-strategy benchmark execution with automatic strategy generation.

    Manages sequential execution of scheduling strategies with automatic strategy
    generation, constraint management, and completion tracking. Subclasses define
    specific execution patterns like synchronous, concurrent, throughput-focused,
    rate-based async, or adaptive sweep profiles.

    Example:
    ::
        @Profile.register("synchronous")
        class SynchronousProfile(Profile):
            def __init__(self, args: SynchronousProfileArgs):
                super().__init__(args)

        args = SynchronousProfileArgs(kind="synchronous")
        profile = Profile.create(args)
    """

    def __init__(
        self,
        args: ProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        """
        Initialize a profile instance.

        :param args: Validated profile argument model for this profile type
        :param random_seed: Seed for reproducible random operations in profile
            strategies.
        :param constraints: Constraints for the profile strategies.
        :param kwargs: Additional profile-specific configuration parameters
        """
        _ = kwargs  # unused
        self.kind = args.kind
        self.args = args
        self.random_seed = random_seed
        self.constraints = dict(constraints or {})
        self.completed_strategies: list[SchedulingStrategy] = []

    @property
    def info(self) -> dict[str, Any]:
        """
        Help json serialization by deferring to ProfileArgs.
        """
        return self.args.model_dump()

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types executed or to be executed in this profile
        """
        return [strat.type_ for strat in self.completed_strategies]

    @staticmethod
    def _should_stop_escalating(prev_benchmark: Benchmark) -> bool:
        """
        Check if a benchmark was terminated by a failure constraint.

        Inspects the scheduler state's end_queuing_constraints for any constraint
        that used "stop_all" for request processing, which indicates the system
        could not handle the load (over-saturation, excessive errors, etc.).
        Constraints that use "stop_local" (max duration, max requests) are normal
        completions and do not trigger escalation stops.

        :param prev_benchmark: Benchmark instance
        :return: True if a failure constraint was triggered, False otherwise
        """
        scheduler_state = getattr(prev_benchmark, "scheduler_state", None)
        if scheduler_state is None:
            return False

        for name, action in scheduler_state.end_queuing_constraints.items():
            if action.request_processing == "stop_all":
                logger.info(
                    f"Stopping rate escalation: constraint '{name}' "
                    f"triggered (request_processing=stop_all)"
                )
                return True
        return False

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
