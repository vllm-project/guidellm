"""
Registry system for dynamic object registration and discovery.

Provides a flexible object registration system with optional auto-discovery
capabilities through decorators and module imports. Enables dynamic discovery
and instantiation of implementations based on configuration parameters, supporting
both manual registration and automatic package-based discovery for extensible
plugin architectures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar, cast

from guidellm.utils.auto_importer import AutoImporterMixin

__all__ = ["RegisterT", "RegistryMixin", "RegistryObjT"]


RegistryObjT = TypeVar("RegistryObjT")
"""Generic type variable for objects managed by the registry system."""
RegisterT = TypeVar(
    "RegisterT", bound=type
)  # Must be bound to type to ensure __name__ is available.
"""Generic type variable for the args and return values within the registry."""


class RegistryMixin(Generic[RegistryObjT], AutoImporterMixin):
    """
    Generic mixin for creating object registries with optional auto-discovery.

    Enables classes to maintain separate registries of objects that can be dynamically
    discovered and instantiated through decorators and module imports. Supports both
    manual registration via decorators and automatic discovery through package scanning
    for extensible plugin architectures.

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

        # Get all registered implementations
        algorithms = BaseAlgorithm.registered_objects()

    Example with auto-discovery:
    ::
        class TokenProposal(RegistryMixin):
            registry_auto_discovery = True
            auto_package = "mypackage.proposals"

        # Automatically imports and registers decorated objects
        proposals = TokenProposal.registered_objects()

    :cvar registry: Dictionary mapping names to registered objects
    :cvar registry_auto_discovery: Enable automatic package-based discovery
    :cvar registry_populated: Track whether auto-discovery has completed
    """

    registry: ClassVar[dict[str, RegistryObjT] | None] = None  # type: ignore[misc]
    registry_auto_discovery: ClassVar[bool] = False
    registry_populated: ClassVar[bool] = False

    @classmethod
    def register(
        cls, name: str | list[str] | None = None
    ) -> Callable[[RegisterT], RegisterT]:
        """
        Decorator for registering objects with the registry.

        :param name: Optional name(s) to register the object under.
            If None, uses the object's __name__ attribute
        :return: Decorator function that registers the decorated object
        :raises ValueError: If name is not a string, list of strings, or None
        """

        def _decorator(obj: RegisterT) -> RegisterT:
            cls.register_decorator(obj, name=name)
            return obj

        return _decorator

    @classmethod
    def register_decorator(
        cls, obj: RegisterT, name: str | list[str] | None = None
    ) -> RegisterT:
        """
        Register an object directly with the registry.

        :param obj: The object to register
        :param name: Optional name(s) to register the object under.
            If None, uses the object's __name__ attribute
        :return: The registered object
        :raises ValueError: If the object is already registered or name is invalid
        """

        if name is None:
            name = obj.__name__
        elif not isinstance(name, str | list):
            raise ValueError(
                "RegistryMixin.register_decorator name must be a string or "
                f"an iterable of strings. Got {name}."
            )

        if cls.registry is None:
            cls.registry = {}

        names = [name] if isinstance(name, str) else list(name)

        for register_name in names:
            if not isinstance(register_name, str):
                raise ValueError(
                    "RegistryMixin.register_decorator name must be a string or "
                    f"a list of strings. Got {register_name}."
                )

            if register_name in cls.registry:
                raise ValueError(
                    f"RegistryMixin.register_decorator cannot register an object "
                    f"{obj} with the name {register_name} because it is already "
                    "registered."
                )

            cls.registry[register_name] = cast("RegistryObjT", obj)

        return obj

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Import and register all modules from the auto_package.

        Automatically called by registered_objects when registry_auto_discovery is True
        to ensure all available implementations are discovered.

        :return: True if registry was populated, False if already populated
        :raises ValueError: If called when registry_auto_discovery is False
        """
        if not cls.registry_auto_discovery:
            raise ValueError(
                "RegistryMixin.auto_populate_registry() cannot be called "
                "because registry_auto_discovery is set to False. "
                "Set registry_auto_discovery to True to enable auto-discovery."
            )

        if cls.registry_populated:
            return False

        cls.auto_import_package_modules()
        cls.registry_populated = True

        return True

    @classmethod
    def registered_objects(cls) -> tuple[RegistryObjT, ...]:
        """
        Get all registered objects from the registry.

        Automatically triggers auto-discovery if registry_auto_discovery is enabled
        to ensure all available implementations are included.

        :return: Tuple of all registered objects including auto-discovered ones
        :raises ValueError: If called before any objects have been registered
        """
        if cls.registry_auto_discovery:
            cls.auto_populate_registry()

        if cls.registry is None:
            raise ValueError(
                "RegistryMixin.registered_objects() must be called after "
                "registering objects with RegistryMixin.register()."
            )

        return tuple(cls.registry.values())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an object is registered under the given name.
        It matches first by exact name, then by str.lower().

        :param name: The name to check for registration.
        :return: True if the object is registered, False otherwise.
        """
        if cls.registry is None:
            return False

        return name in cls.registry or name.lower() in [
            key.lower() for key in cls.registry
        ]

    @classmethod
    def get_registered_object(cls, name: str) -> RegistryObjT | None:
        """
        Get a registered object by its name. It matches first by exact name,
        then by str.lower().

        :param name: The name of the registered object.
        :return: The registered object if found, None otherwise.
        """
        if cls.registry is None:
            return None

        if name in cls.registry:
            return cls.registry[name]

        name_casefold = name.lower()
        for k, v in cls.registry.items():
            if name_casefold == k.lower():
                return v

        return None  # Not found
