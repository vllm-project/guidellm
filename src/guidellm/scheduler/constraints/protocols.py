"""
Protocol definitions for constraint system.

Defines the core protocols that constraint classes must implement for
evaluation and initialization within the scheduler constraint system.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from guidellm.scheduler.schemas import SchedulerState, SchedulerUpdateAction
from guidellm.schemas import RequestInfo

__all__ = [
    "Constraint",
    "ConstraintInitializer",
    "SerializableConstraintInitializer",
]


@runtime_checkable
class Constraint(Protocol):
    """Protocol for constraint evaluation functions that control scheduler behavior."""

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
    """Protocol for constraint initializer factory functions that create constraints."""

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance from configuration parameters.

        :param kwargs: Configuration parameters for constraint creation
        :return: Configured constraint evaluation function
        """


@runtime_checkable
class SerializableConstraintInitializer(Protocol):
    """Protocol for serializable constraint initializers supporting persistence."""

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
