"""
Core constraint system protocols and base classes.

Defines the fundamental protocols and base classes that form the foundation of the
constraint system. Constraints control scheduler behavior by evaluating scheduler
state and individual requests to determine whether processing should continue or
stop based on predefined limits. The constraint system enables sophisticated
benchmark stopping criteria through composable constraint types with support for
serialization, validation, and dynamic instantiation.

The module provides:
- Protocols defining the constraint interface contract
  (Constraint, ConstraintInitializer)
- Base classes for Pydantic-based constraint initializers with serialization support
- Placeholder classes for handling unserializable constraint states

Example:
::
    from guidellm.scheduler.constraints import (
        Constraint,
        PydanticConstraintInitializer,
    )

    class MyConstraint(PydanticConstraintInitializer):
        type_: str = "my_constraint"

        def create_constraint(self) -> Constraint:
            def evaluate(state, request):
                return SchedulerUpdateAction(request_queuing="continue")
            return evaluate
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from guidellm.scheduler.schemas import SchedulerState, SchedulerUpdateAction
from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.utils import InfoMixin

__all__ = [
    "Constraint",
    "ConstraintInitializer",
    "PydanticConstraintInitializer",
    "SerializableConstraintInitializer",
    "UnserializableConstraintInitializer",
]


@runtime_checkable
class Constraint(Protocol):
    """
    Protocol for constraint evaluation functions that control scheduler behavior.

    Defines the interface that all constraint implementations must follow. Constraints
    are callable objects that evaluate scheduler state and request information to
    determine whether processing should continue or stop. The protocol enables type
    checking and runtime validation of constraint implementations while allowing
    flexible implementation approaches (functions, classes, closures).

    Example:
    ::
        def my_constraint(
            state: SchedulerState, request: RequestInfo
        ) -> SchedulerUpdateAction:
            if state.processing_requests > 100:
                return SchedulerUpdateAction(request_queuing="stop")
            return SchedulerUpdateAction(request_queuing="continue")
    """

    def __call__(
        self, state: SchedulerState, request: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against scheduler state and request information.

        :param state: Current scheduler state with metrics and timing information
        :param request: Individual request information and metadata
        :return: Action indicating whether to continue or stop scheduler operations
        """


@runtime_checkable
class ConstraintInitializer(Protocol):
    """
    Protocol for constraint initializer factory functions that create constraints.

    Defines the interface for factory objects that create constraint instances from
    configuration parameters. Constraint initializers enable dynamic constraint
    creation and configuration, supporting both simple boolean flags and complex
    parameter dictionaries. The protocol allows type checking while maintaining
    flexibility for different initialization patterns.

    Example:
    ::
        class MaxRequestsInitializer:
            def __init__(self, max_requests: int):
                self.max_requests = max_requests

            def create_constraint(self) -> Constraint:
                def evaluate(state, request):
                    if state.total_requests >= self.max_requests:
                        return SchedulerUpdateAction(request_queuing="stop")
                    return SchedulerUpdateAction(request_queuing="continue")
                return evaluate
    """

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance from configuration parameters.

        :param kwargs: Configuration parameters for constraint creation
        :return: Configured constraint evaluation function
        """


@runtime_checkable
class SerializableConstraintInitializer(Protocol):
    """
    Protocol for serializable constraint initializers supporting persistence.

    Extends ConstraintInitializer with serialization capabilities, enabling constraint
    configurations to be saved, loaded, and transmitted. Serializable initializers
    support validation, model-based configuration, and dictionary-based serialization
    for integration with configuration systems and persistence layers.

    Example:
    ::
        class SerializableInitializer:
            @classmethod
            def validated_kwargs(cls, **kwargs) -> dict[str, Any]:
                return {"max_requests": kwargs.get("max_requests", 100)}

            @classmethod
            def model_validate(cls, data: dict) -> ConstraintInitializer:
                return cls(**cls.validated_kwargs(**data))

            def model_dump(self) -> dict[str, Any]:
                return {"type_": "max_requests", "max_requests": self.max_requests}

            def create_constraint(self) -> Constraint:
                # ... create constraint
    """

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        :param args: Positional arguments for constraint configuration
        :param kwargs: Keyword arguments for constraint configuration
        :return: Validated parameter dictionary for constraint creation
        """

    @classmethod
    def model_validate(cls, **kwargs) -> ConstraintInitializer:
        """
        Create validated constraint initializer from configuration.

        :param kwargs: Configuration dictionary for initializer creation
        :return: Validated constraint initializer instance
        """

    def model_dump(self) -> dict[str, Any]:
        """
        Serialize constraint initializer to dictionary format.

        :return: Dictionary representation of constraint initializer
        """

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create constraint instance from this initializer.

        :param kwargs: Additional configuration parameters
        :return: Configured constraint evaluation function
        """


class PydanticConstraintInitializer(StandardBaseModel, ABC, InfoMixin):
    """
    Abstract base for Pydantic-based constraint initializers.

    Provides standardized serialization, validation, and metadata handling for
    constraint initializers using Pydantic models. Subclasses implement specific
    constraint creation logic while inheriting validation and persistence support.
    Integrates with the constraint factory system for dynamic instantiation and
    configuration management.

    Example:
    ::
        @ConstraintsInitializerFactory.register("max_duration")
        class MaxDurationConstraintInitializer(PydanticConstraintInitializer):
            type_: str = "max_duration"
            max_seconds: float = Field(description="Maximum duration in seconds")

            def create_constraint(self) -> Constraint:
                def evaluate(state, request):
                    if time.time() - state.start_time > self.max_seconds:
                        return SchedulerUpdateAction(request_queuing="stop")
                    return SchedulerUpdateAction(request_queuing="continue")
                return evaluate

    :cvar type_: Type identifier for the constraint initializer
    """

    type_: str = Field(description="Type identifier for the constraint initializer")

    @property
    def info(self) -> dict[str, Any]:
        """
        Extract serializable information from this constraint initializer.

        :return: Dictionary containing constraint configuration and metadata
        """
        return self.model_dump()

    @classmethod
    @abstractmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        Must be implemented by subclasses to handle their specific parameter patterns
        and validation requirements. This method processes raw input (booleans, dicts,
        etc.) and converts them into validated parameter dictionaries suitable for
        constraint initialization.

        :param args: Positional arguments passed to the constraint
        :param kwargs: Keyword arguments passed to the constraint
        :return: Validated dictionary of parameters for constraint creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @abstractmethod
    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance.

        Must be implemented by subclasses to return their specific constraint type
        with appropriate configuration and validation. The returned constraint should
        be ready for evaluation against scheduler state and requests.

        :param kwargs: Additional keyword arguments (usually unused)
        :return: Configured constraint instance
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...


class UnserializableConstraintInitializer(PydanticConstraintInitializer):
    """
    Placeholder for constraints that cannot be serialized or executed.

    Represents constraint initializers that failed serialization or contain
    non-serializable components. Cannot be executed and raises errors when
    invoked to prevent runtime failures from invalid constraint state. Used
    by the factory system to preserve constraint information even when full
    serialization is not possible.

    Example:
    ::
        # Created automatically by factory when serialization fails
        unserializable = UnserializableConstraintInitializer(
            orig_info={"type_": "custom", "data": non_serializable_object}
        )

        # Attempting to use it raises RuntimeError
        constraint = unserializable.create_constraint()  # Raises RuntimeError

    :cvar type_: Always "unserializable" to identify placeholder constraints
    :cvar orig_info: Original constraint information before serialization failure
    """

    type_: Literal["unserializable"] = "unserializable"  # type: ignore[assignment]
    orig_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Original constraint information before serialization failure",
    )

    @classmethod
    def validated_kwargs(
        cls, orig_info: dict[str, Any] | None = None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate arguments for unserializable constraint creation.

        :param orig_info: Original constraint information before serialization failure
        :param kwargs: Additional arguments (ignored)
        :return: Validated parameters for unserializable constraint creation
        """
        return {"orig_info": orig_info or {}}

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Raise error for unserializable constraint creation attempt.

        :param kwargs: Additional keyword arguments (unused)
        :raises RuntimeError: Always raised since unserializable constraints
            cannot be executed
        """
        raise RuntimeError(
            "Cannot create constraint from unserializable constraint instance. "
            "This constraint cannot be serialized and therefore cannot be executed."
        )

    def __call__(
        self, state: SchedulerState, request: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Raise error since unserializable constraints cannot be invoked.

        :param state: Current scheduler state (unused)
        :param request: Individual request information (unused)
        :raises RuntimeError: Always raised for unserializable constraints
        """
        _ = (state, request)  # Unused parameters
        raise RuntimeError(
            "Cannot invoke unserializable constraint instance. "
            "This constraint was not properly serialized and cannot be executed."
        )
