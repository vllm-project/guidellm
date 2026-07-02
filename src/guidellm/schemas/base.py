"""
Pydantic utilities for polymorphic model serialization and registry integration.

Provides integration between Pydantic and the registry system, enabling
polymorphic serialization and deserialization of Pydantic models using
a discriminator field and dynamic class registry. Includes base model classes
with standardized configurations and generic status breakdown models for
structured result organization.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from disdantic import PydanticClassRegistryMixin
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BaseModelT",
    "ErroredT",
    "IncompleteT",
    "RegisterClassT",
    "StandardBaseDict",
    "StandardBaseModel",
    "StatusBreakdown",
    "SuccessfulT",
    "TotalT",
    "_PydanticClassRegistryMixin",
    "standard_model_config",
]


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
RegisterClassT = TypeVar("RegisterClassT", bound=type)
SuccessfulT = TypeVar("SuccessfulT")
ErroredT = TypeVar("ErroredT")
IncompleteT = TypeVar("IncompleteT")
TotalT = TypeVar("TotalT")


def standard_model_config(**updates: Any) -> ConfigDict:
    """
    Generate a standard Pydantic model configuration with optional updates.

    Provides a consistent base configuration for Pydantic models in the application,
    allowing for easy customization through additional keyword arguments.

    :param updates: Additional configuration settings to override defaults
    :return: ConfigDict with standard settings merged with any updates
    """
    base_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )
    return base_config | ConfigDict(**updates)  # type: ignore[typeddict-item]


class StandardBaseModel(BaseModel):
    """
    Base Pydantic model with standardized configuration for GuideLLM.

    Provides consistent validation behavior and configuration settings across
    all Pydantic models in the application, including field validation,
    attribute conversion, and default value handling.

    Example:
    ::
        class MyModel(StandardBaseModel):
            name: str
            value: int = 42

        # Access default values
        default_value = MyModel.get_default("value")  # Returns 42
    """

    model_config = standard_model_config(extra="ignore")

    @classmethod
    def get_default(cls: type[BaseModel], field: str) -> Any:
        """
        Get default value for a model field.

        :param field: Name of the field to get the default value for
        :return: Default value of the specified field
        :raises KeyError: If the field does not exist in the model
        """
        return cls.model_fields[field].default


class StandardBaseDict(StandardBaseModel):
    """
    Base Pydantic model allowing arbitrary additional fields.

    Extends StandardBaseModel to accept extra fields beyond those explicitly
    defined in the model schema. Useful for flexible data structures that
    need to accommodate varying or unknown field sets while maintaining
    type safety for known fields.
    """

    model_config = standard_model_config(extra="allow")


class StatusBreakdown(BaseModel, Generic[SuccessfulT, ErroredT, IncompleteT, TotalT]):
    """
    Generic model for organizing results by processing status.

    Provides structured categorization of results into successful, errored,
    incomplete, and total status groups. Supports flexible typing for each
    status category to accommodate different result types while maintaining
    consistent organization patterns across the application.

    Example:
    ::
        from guidellm.utils import StatusBreakdown

        # Define a breakdown for request counts
        breakdown = StatusBreakdown[int, int, int, int](
            successful=150,
            errored=5,
            incomplete=10,
            total=165
        )
    """

    successful: SuccessfulT = Field(
        description="Results or metrics for requests with successful completion status",
        default=None,  # type: ignore[assignment]
    )
    errored: ErroredT = Field(
        description="Results or metrics for requests with error completion status",
        default=None,  # type: ignore[assignment]
    )
    incomplete: IncompleteT = Field(
        description="Results or metrics for requests with incomplete processing status",
        default=None,  # type: ignore[assignment]
    )
    total: TotalT = Field(
        description="Aggregated results or metrics combining all status categories",
        default=None,  # type: ignore[assignment]
    )


class _PydanticClassRegistryMixin(PydanticClassRegistryMixin):
    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        Prevent direct instantiation of base classes that use this mixin.

        Only allows instantiation of concrete subclasses, not the base class.
        """
        base_type = cls.__pydantic_schema_base_type__()
        if cls is base_type:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return super().__new__(cls)
