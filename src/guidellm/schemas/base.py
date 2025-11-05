"""
Pydantic utilities for polymorphic model serialization and registry integration.

Provides integration between Pydantic and the registry system, enabling
polymorphic serialization and deserialization of Pydantic models using
a discriminator field and dynamic class registry. Includes base model classes
with standardized configurations and generic status breakdown models for
structured result organization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from guidellm.utils.registry import RegistryMixin

__all__ = [
    "BaseModelT",
    "ErroredT",
    "IncompleteT",
    "PydanticClassRegistryMixin",
    "RegisterClassT",
    "ReloadableBaseModel",
    "StandardBaseDict",
    "StandardBaseModel",
    "StatusBreakdown",
    "SuccessfulT",
    "TotalT",
]


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
RegisterClassT = TypeVar("RegisterClassT", bound=type)
SuccessfulT = TypeVar("SuccessfulT")
ErroredT = TypeVar("ErroredT")
IncompleteT = TypeVar("IncompleteT")
TotalT = TypeVar("TotalT")


class ReloadableBaseModel(BaseModel):
    """
    Base Pydantic model with schema reloading capabilities.

    Provides dynamic schema rebuilding functionality for models that need to
    update their validation schemas at runtime, particularly useful when
    working with registry-based polymorphic models where new types are
    registered after initial class definition.
    """

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def reload_schema(cls, parents: bool = True) -> None:
        """
        Reload the class schema with updated registry information.

        Forces a complete rebuild of the Pydantic model schema to incorporate
        any changes made to associated registries or validation rules.

        :param parents: Whether to also rebuild schemas for any pydantic parent
            types that reference this model.
        """
        cls.model_rebuild(force=True)

        if parents:
            cls.reload_parent_schemas()

    @classmethod
    def reload_parent_schemas(cls):
        """
        Recursively reload schemas for all parent Pydantic models.

        Traverses the inheritance hierarchy to find all parent classes that
        are Pydantic models and triggers schema rebuilding on each to ensure
        that any changes in child models are reflected in parent schemas.
        """
        potential_parents: set[type[BaseModel]] = {BaseModel}
        stack: list[type[BaseModel]] = [BaseModel]

        while stack:
            current = stack.pop()
            for subclass in current.__subclasses__():
                if (
                    issubclass(subclass, BaseModel)
                    and subclass is not cls
                    and subclass not in potential_parents
                ):
                    potential_parents.add(subclass)
                    stack.append(subclass)

        for check in cls.__mro__:
            if isinstance(check, type) and issubclass(check, BaseModel):
                cls._reload_schemas_depending_on(check, potential_parents)

    @classmethod
    def _reload_schemas_depending_on(cls, target: type[BaseModel], types: set[type]):
        changed = True
        while changed:
            changed = False
            for candidate in types:
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, BaseModel)
                    and any(
                        cls._uses_type(target, field_info.annotation)
                        for field_info in candidate.model_fields.values()
                        if field_info.annotation is not None
                    )
                ):
                    try:
                        before = candidate.model_json_schema()
                    except Exception:  # noqa: BLE001
                        before = None
                    candidate.model_rebuild(force=True)
                    if before is not None:
                        after = candidate.model_json_schema()
                        changed |= before != after

    @classmethod
    def _uses_type(cls, target: type, candidate: type) -> bool:
        if target is candidate:
            return True

        origin = get_origin(candidate)

        if origin is None:
            return isinstance(candidate, type) and issubclass(candidate, target)

        if isinstance(origin, type) and (
            target is origin or issubclass(origin, target)
        ):
            return True

        for arg in get_args(candidate) or []:
            if isinstance(arg, type) and cls._uses_type(target, arg):
                return True

        return False


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

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        from_attributes=True,
    )

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

    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


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


class PydanticClassRegistryMixin(
    ReloadableBaseModel, RegistryMixin[type[BaseModelT]], ABC, Generic[BaseModelT]
):
    """
    Polymorphic Pydantic model mixin enabling registry-based dynamic instantiation.

    Integrates Pydantic validation with the registry system to enable polymorphic
    serialization and deserialization based on a discriminator field. Automatically
    instantiates the correct subclass during validation based on registry mappings,
    providing a foundation for extensible plugin-style architectures.

    Example:
    ::
        from speculators.utils import PydanticClassRegistryMixin

        class BaseConfig(PydanticClassRegistryMixin["BaseConfig"]):
            schema_discriminator: ClassVar[str] = "config_type"
            config_type: str = Field(description="Configuration type identifier")

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type["BaseConfig"]:
                return BaseConfig

        @BaseConfig.register("database")
        class DatabaseConfig(BaseConfig):
            config_type: str = "database"
            connection_string: str = Field(description="Database connection string")

        # Dynamic instantiation based on discriminator
        config = BaseConfig.model_validate({
            "config_type": "database",
            "connection_string": "postgresql://localhost:5432/db"
        })

    :cvar schema_discriminator: Field name used for polymorphic type discrimination
    """

    schema_discriminator: ClassVar[str] = "model_type"

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        Prevent direct instantiation of base classes that use this mixin.

        Only allows instantiation of concrete subclasses, not the base class.
        """
        base_type = cls.__pydantic_schema_base_type__()
        if cls is base_type:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return super().__new__(cls)

    @classmethod
    def register_decorator(
        cls, clazz: RegisterClassT, name: str | list[str] | None = None
    ) -> RegisterClassT:
        """
        Register a Pydantic model class with type validation and schema reload.

        Validates that the class is a proper Pydantic BaseModel subclass before
        registering it in the class registry. Automatically triggers schema
        reload to incorporate the new type into polymorphic validation.

        :param clazz: Pydantic model class to register in the polymorphic hierarchy
        :param name: Registry identifier for the class. Uses class name if None
        :return: The registered class unchanged for decorator chaining
        :raises TypeError: If clazz is not a Pydantic BaseModel subclass
        """
        if not issubclass(clazz, BaseModel):
            raise TypeError(
                f"Cannot register {clazz.__name__} as it is not a subclass of "
                "Pydantic BaseModel"
            )

        super().register_decorator(clazz, name=name)
        cls.reload_schema()

        return cast("RegisterClassT", clazz)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate polymorphic validation schema for dynamic type instantiation.

        Creates a tagged union schema that enables Pydantic to automatically
        instantiate the correct subclass based on the discriminator field value.
        Falls back to base schema generation when no registry is available.

        :param source_type: Type being processed for schema generation
        :param handler: Pydantic core schema generation handler
        :return: Tagged union schema for polymorphic validation or base schema
        """
        if source_type == cls.__pydantic_schema_base_type__():
            if not cls.registry:
                return cls.__pydantic_generate_base_schema__(handler)

            choices = {
                name: handler(model_class) for name, model_class in cls.registry.items()
            }

            return core_schema.tagged_union_schema(
                choices=choices,
                discriminator=cls.schema_discriminator,
            )

        return handler(cls)

    @classmethod
    @abstractmethod
    def __pydantic_schema_base_type__(cls) -> type[BaseModelT]:
        """
        Define the base type for polymorphic validation hierarchy.

        Must be implemented by subclasses to specify which type serves as the
        root of the polymorphic hierarchy for schema generation and validation.

        :return: Base class type for the polymorphic model hierarchy
        """
        ...

    @classmethod
    def __pydantic_generate_base_schema__(
        cls, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate fallback schema for polymorphic models without registry.

        Provides a base schema that accepts any valid input when no registry
        is available for polymorphic validation. Used as fallback during
        schema generation when the registry has not been populated.

        :param handler: Pydantic core schema generation handler
        :return: Base CoreSchema that accepts any valid input
        """
        return core_schema.any_schema()

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Initialize registry with auto-discovery and reload validation schema.

        Triggers automatic population of the class registry through the parent
        RegistryMixin functionality and ensures the Pydantic validation schema
        is updated to include all discovered types for polymorphic validation.

        :return: True if registry was populated, False if already populated
        :raises ValueError: If called when registry_auto_discovery is disabled
        """
        populated = super().auto_populate_registry()
        cls.reload_schema()

        return populated

    @classmethod
    def registered_classes(cls) -> tuple[type[BaseModelT], ...]:
        """
        Get all registered pydantic classes from the registry.

        Automatically triggers auto-discovery if registry_auto_discovery is enabled
        to ensure all available implementations are included.

        :return: Tuple of all registered classes including auto-discovered ones
        :raises ValueError: If called before any objects have been registered
        """
        if cls.registry_auto_discovery:
            cls.auto_populate_registry()

        if cls.registry is None:
            raise ValueError(
                "ClassRegistryMixin.registered_classes() must be called after "
                "registering classes with ClassRegistryMixin.register()."
            )

        return tuple(cls.registry.values())
