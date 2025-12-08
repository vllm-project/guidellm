"""
Factory for creating and managing constraint initializers.

Provides centralized access to registered constraint types with support for
creating constraints from configuration dictionaries, simple values, or
pre-configured instances.
"""

from __future__ import annotations

from typing import Any

from guidellm.scheduler.constraints.constraint import (
    Constraint,
    ConstraintInitializer,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from guidellm.utils import InfoMixin, RegistryMixin

__all__ = ["ConstraintsInitializerFactory"]


class ConstraintsInitializerFactory(RegistryMixin[ConstraintInitializer]):
    """
    Registry factory for creating and managing constraint initializers.

    Provides centralized access to registered constraint types with support for
    creating constraints from configuration dictionaries, simple values, or
    pre-configured instances. Handles constraint resolution and type validation
    for the scheduler constraint system.

    Example:
    ::
        from guidellm.scheduler import ConstraintsInitializerFactory

        # Register new constraint type
        @ConstraintsInitializerFactory.register("new_constraint")
        class NewConstraint:
            def create_constraint(self, **kwargs) -> Constraint:
                return lambda state, request: SchedulerUpdateAction()

        # Create and use constraint
        constraint = ConstraintsInitializerFactory.create_constraint("new_constraint")
    """

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> ConstraintInitializer:
        """
        Create a constraint initializer for the specified key.

        :param key: Registered constraint initializer key
        :param args: Positional arguments for initializer creation
        :param kwargs: Keyword arguments for initializer creation
        :return: Configured constraint initializer instance
        :raises ValueError: If the key is not registered in the factory
        """
        if cls.registry is None or key not in cls.registry:
            raise ValueError(f"Unknown constraint initializer key: {key}")

        initializer_class = cls.registry[key]

        return (
            initializer_class(*args, **kwargs)  # type: ignore[operator]
            if not isinstance(initializer_class, type)
            or not issubclass(initializer_class, SerializableConstraintInitializer)
            else initializer_class(
                **initializer_class.validated_kwargs(*args, **kwargs)  # type: ignore[misc]
            )
        )

    @classmethod
    def serialize(cls, initializer: ConstraintInitializer) -> dict[str, Any]:
        """
        Serialize constraint initializer to dictionary format.

        :param initializer: Constraint initializer to serialize
        :return: Dictionary representation or unserializable placeholder
        """
        if isinstance(initializer, SerializableConstraintInitializer):
            return initializer.model_dump()
        else:
            unserializable = UnserializableConstraintInitializer(
                orig_info=InfoMixin.extract_from_obj(initializer)
            )
            return unserializable.model_dump()

    @classmethod
    def deserialize(
        cls, initializer_dict: dict[str, Any]
    ) -> SerializableConstraintInitializer | UnserializableConstraintInitializer:
        """
        Deserialize constraint initializer from dictionary format.

        :param initializer_dict: Dictionary representation of constraint initializer
        :return: Reconstructed constraint initializer instance
        :raises ValueError: If constraint type is unknown or cannot be deserialized
        """
        if initializer_dict.get("type_") == "unserializable":
            return UnserializableConstraintInitializer.model_validate(initializer_dict)

        if (
            cls.registry is not None
            and initializer_dict.get("type_")
            and initializer_dict["type_"] in cls.registry
        ):
            initializer_class = cls.registry[initializer_dict["type_"]]
            if hasattr(initializer_class, "model_validate"):
                return initializer_class.model_validate(initializer_dict)  # type: ignore[return-value]
            else:
                return initializer_class(**initializer_dict)  # type: ignore[return-value,operator]

        raise ValueError(
            f"Cannot deserialize unknown constraint initializer: "
            f"{initializer_dict.get('type_', 'unknown')}"
        )

    @classmethod
    def create_constraint(cls, key: str, *args, **kwargs) -> Constraint:
        """
        Create a constraint instance for the specified key.

        :param key: Registered constraint initializer key
        :param args: Positional arguments for constraint creation
        :param kwargs: Keyword arguments for constraint creation
        :return: Configured constraint function ready for evaluation
        :raises ValueError: If the key is not registered in the factory
        """
        return cls.create(key, *args, **kwargs).create_constraint()

    @classmethod
    def resolve(
        cls,
        initializers: dict[
            str,
            Any | dict[str, Any] | Constraint | ConstraintInitializer,
        ],
    ) -> dict[str, Constraint]:
        """
        Resolve mixed constraint specifications to callable constraints.

        :param initializers: Dictionary mapping constraint keys to specifications
        :return: Dictionary mapping constraint keys to callable functions
        :raises ValueError: If any key is not registered in the factory
        """
        constraints = {}

        for key, val in initializers.items():
            if isinstance(val, Constraint):
                constraints[key] = val
            elif isinstance(val, ConstraintInitializer):
                constraints[key] = val.create_constraint()
            elif isinstance(val, dict):
                constraints[key] = cls.create_constraint(key, **val)
            else:
                constraints[key] = cls.create_constraint(key, val)

        return constraints

    @classmethod
    def resolve_constraints(
        cls,
        constraints: dict[str, Any | dict[str, Any] | Constraint],
    ) -> dict[str, Constraint]:
        """
        Resolve constraints from mixed constraint specifications.

        :param constraints: Dictionary mapping constraint keys to specifications
        :return: Dictionary mapping constraint keys to callable functions
        :raises ValueError: If any constraint key is not registered
        """
        resolved_constraints = {}

        for key, val in constraints.items():
            if isinstance(val, Constraint):
                resolved_constraints[key] = val
            elif isinstance(val, dict):
                resolved_constraints[key] = cls.create_constraint(key, **val)
            else:
                resolved_constraints[key] = cls.create_constraint(key, val)

        return resolved_constraints
