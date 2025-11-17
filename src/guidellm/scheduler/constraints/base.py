"""
Base classes for constraint initializers.

Provides abstract base classes for Pydantic-based constraint initializers
with standardized serialization, validation, and metadata handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import Field

from guidellm.scheduler.schemas import SchedulerState, SchedulerUpdateAction
from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.utils import InfoMixin

from .protocols import (
    Constraint,
)

__all__ = [
    "PydanticConstraintInitializer",
    "UnserializableConstraintInitializer",
]


class PydanticConstraintInitializer(StandardBaseModel, ABC, InfoMixin):
    """
    Abstract base for Pydantic-based constraint initializers.

    Provides standardized serialization, validation, and metadata handling for
    constraint initializers using Pydantic models. Subclasses implement specific
    constraint creation logic while inheriting validation and persistence support.
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
        and validation requirements.

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
        with appropriate configuration and validation.

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
    invoked to prevent runtime failures from invalid constraint state.
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
