"""
Profile base class for multi-strategy benchmark execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    computed_field,
    field_serializer,
    field_validator,
)

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
)
from guidellm.schemas import PydanticClassRegistryMixin

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark

ProfileType = Annotated[
    Literal[
        "synchronous",
        "concurrent",
        "throughput",
        "async",
        "constant",
        "poisson",
        "sweep",
        "replay",
    ],
    "Profile type identifiers for polymorphic deserialization",
]


class ProfileArgs(PydanticClassRegistryMixin["ProfileArgs"], ABC):
    """
    Base class for profile creation arguments.

    This class serves as a base for defining argument models used in the creation
    of profile instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    profile configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = ConfigDict(
        extra="ignore",
        serialize_by_alias=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[ProfileArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base ProfileArgs class for schema validation
        """
        if cls.__name__ == "ProfileArgs":
            return cls

        return ProfileArgs

    kind: Literal[
        "profile",
        "synchronous",
        "concurrent",
        "throughput",
        "async",
        "constant",
        "poisson",
        "sweep",
        "replay",
    ] = Field(
        description="Profile type discriminator for polymorphic serialization",
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

    schema_discriminator: ClassVar[str] = "kind"

    kind: Literal[
        "profile",
        "synchronous",
        "concurrent",
        "throughput",
        "async",
        "constant",
        "poisson",
        "sweep",
        "replay",
    ] = Field(
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
    def create(cls, args: ProfileArgs) -> Profile:
        """
        Create profile instances from validated profile arguments.

        :param args: Validated profile argument model for the target profile type
        :return: Configured profile instance for the specified type
        :raises ValueError: If the profile kind is not registered
        """
        profile_class = cls.get_registered_object(args.kind)

        if profile_class is None:
            raise ValueError(
                f"Profile type '{args.kind}' is not registered. "
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return profile_class.model_validate(
            args.model_dump(by_alias=True, mode="python")
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
