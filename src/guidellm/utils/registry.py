"""
Registry system for objects in the GuideLLM toolkit.

This module provides a flexible object registration and discovery system used
throughout the GuideLLM toolkit. It enables automatic registration of objects
and discovery of implementations through decorators.

Classes:
    RegistryMixin: Base mixin for creating object registries with decorators.
"""

from typing import Any, Callable, ClassVar, Generic, Optional, TypeVar

__all__ = ["RegistryMixin"]


RegistryObjT = TypeVar("RegistryObjT", bound=Any)


class RegistryMixin(Generic[RegistryObjT]):
    """
    A mixin class that provides a registration system for the specified object type.

    This mixin allows classes to maintain a registry of objects that can be
    dynamically discovered and instantiated. Classes that inherit from this mixin
    can use the @register decorator to add objects to the registry.

    The registry is class-specific, meaning each class that inherits from this mixin
    will have its own separate registry of implementations.

    Example:
    ::
        class BaseAlgorithm(RegistryMixin):
            pass

        @BaseAlgorithm.register()
        class ConcreteAlgorithm(BaseAlgorithm):
            pass

        @BaseAlgorithm.register("custom_name")
        class AnotherAlgorithm(BaseAlgorithm):
            pass

        # Get all registered algorithm implementations
        algorithms = BaseAlgorithm.registered_objects()

    :cvar registry: A dictionary mapping object names to objects that have been
        registered to the extending subclass through the @subclass.register() decorator
    :cvar registry_populated: A flag that tracks whether the registry has been
        populated with objects from the specified package(s).
    """

    registry: ClassVar[Optional[dict[str, RegistryObjT]]] = None
    registry_populated: ClassVar[bool] = False

    @classmethod
    def register(
        cls, name: Optional[str] = None
    ) -> Callable[[RegistryObjT], RegistryObjT]:
        """
        An invoked decorator that registers an object with the registry under
        either the provided name or the object name if no name is provided.

        Example:
        ```python
        @RegistryMixin.register()
        class ExampleClass:
            ...

        @RegistryMixin.register("custom_name")
        class AnotherExampleClass:
            ...
        ```

        :param name: Optional name to register the object under. If None, the object
            name is used as the registry key.
        :return: A decorator function that registers the decorated object.
        :raises ValueError: If name is provided but is not a string.
        """
        if name is not None and not isinstance(name, str):
            raise ValueError(
                f"RegistryMixin.register() name must be a string or None. Got {name}."
            )

        return lambda obj: cls.register_decorator(obj, name=name)

    @classmethod
    def register_decorator(
        cls, obj: RegistryObjT, name: Optional[str] = None
    ) -> RegistryObjT:
        """
        A non-invoked decorator that registers the object with the registry.
        If passed through a lambda, then name can be passed in as well.
        Otherwise, the only argument is the decorated object.

        Example:
        ```python
        @RegistryMixin.register_decorator
        class ExampleClass:
            ...
        ```

        :param obj: The object to register
        :param name: Optional name to register the object under. If None, the object
            name is used as the registry key.
        :return: The registered object.
        :raises TypeError: If the decorator is used incorrectly.
        :raises ValueError: If the object is already registered or if name is provided
            but is not a string.
        """

        if not name:
            name = getattr(obj, "__name__", str(obj))
        elif not isinstance(name, str):
            raise ValueError(
                "RegistryMixin.register_decorator must be used as a decorator "
                "and without invocation. "
                f"Got improper name arg {name}."
            )

        if cls.registry is None:
            cls.registry = {}

        if name in cls.registry:
            raise ValueError(
                f"RegistryMixin.register_decorator cannot register an object "
                f"{obj} with the name {name} because it is already registered."
            )

        cls.registry[name] = obj

        return obj

    @classmethod
    def registered_objects(cls) -> dict[str, RegistryObjT]:
        """
        :return: A dictionary mapping names to all registered objects.
        """
        if cls.registry is None:
            return {}
        return dict(cls.registry)

    @classmethod
    def get_registered_object(cls, name: str) -> RegistryObjT:
        """
        :param name: The name of the registered object.
        :return: The registred object
        """
        if cls.registry is None or name not in cls.registry:
            raise ValueError(f"Object with name {name} is not registered.")
        return cls.registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        :param name: The name to check for registration.
        :return: True if an object is registered with that name, False otherwise.
        """
        if cls.registry is None:
            return False
        return name in cls.registry

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        :param name: The name of the object to unregister.
        :return: True if the object was successfully unregistered, False if it
            wasn't registered.
        """
        if cls.registry is None:
            return False

        if name in cls.registry:
            del cls.registry[name]
            return True
        return False

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered objects from the registry.
        """
        if cls.registry is not None:
            cls.registry.clear()
