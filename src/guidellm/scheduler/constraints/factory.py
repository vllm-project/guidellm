"""
Factory for creating and managing constraint initializers.

Provides centralized access to registered constraint types with support for
creating constraints from configuration dictionaries or pre-configured instances.
"""

from __future__ import annotations

from typing import Any

from disdantic import RegistryMixin

from guidellm.scheduler.constraints.args import ConstraintArgs
from guidellm.scheduler.constraints.constraint import (
    Constraint,
    ConstraintInitializer,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)

__all__ = ["ConstraintsInitializerFactory"]


class ConstraintsInitializerFactory(RegistryMixin[ConstraintInitializer]):
    """
    Registry factory for creating and managing constraint initializers.

    Provides centralized access to registered constraint types with support for
    creating constraints from ``ConstraintArgs`` instances or pre-configured
    initializer instances. Handles constraint resolution and type validation
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
        args = NewConstraintArgs(kind="new_constraint")
        initializer = ConstraintsInitializerFactory.create(args)
        constraint = initializer.create_constraint()
    """

    @classmethod
    def create(cls, args: ConstraintArgs) -> ConstraintInitializer:
        """
        Create a constraint initializer from a ``ConstraintArgs`` instance.

        :param args: Validated constraint arguments with kind discriminator
        :return: Configured constraint initializer instance
        :raises ValueError: If args.kind is not registered in the factory
        """
        if cls.registry is None or args.kind not in cls.registry:
            raise ValueError(f"Unknown constraint discriminator: {args.kind}")

        initializer_class = cls.registry[args.kind]
        return initializer_class(args=args)  # type: ignore[operator]

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
    def resolve(
        cls,
        initializers: dict[
            str,
            Constraint | ConstraintInitializer,
        ],
    ) -> dict[str, Constraint]:
        """
        Resolve constraint initializers to callable constraints.

        :param initializers: Dictionary mapping constraint keys to specifications.
            Values must be Constraint instances or ConstraintInitializer instances.
        :return: Dictionary mapping constraint keys to callable functions
        :raises TypeError: If a value is not a supported type
        """
        constraints = {}

        for key, val in initializers.items():
            if isinstance(val, ConstraintInitializer):
                constraints[key] = val.create_constraint()
            elif isinstance(val, Constraint):
                constraints[key] = val
            else:
                raise TypeError(
                    f"Constraint '{key}' has unsupported value type "
                    f"{type(val).__name__}. Expected a Constraint instance or "
                    f"ConstraintInitializer instance."
                )

        return constraints
