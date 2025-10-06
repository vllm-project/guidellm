"""
Automatic module importing utilities for dynamic class discovery.

This module provides a mixin class for automatic module importing within a package,
enabling dynamic discovery of classes and implementations without explicit imports.
It is particularly useful for auto-registering classes in a registry pattern where
subclasses need to be discoverable at runtime.

The AutoImporterMixin can be combined with registration mechanisms to create
extensible systems where new implementations are automatically discovered and
registered when they are placed in the correct package structure.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import ClassVar

__all__ = ["AutoImporterMixin"]


class AutoImporterMixin:
    """
    Mixin class for automatic module importing within packages.

    This mixin enables dynamic discovery of classes and implementations without
    explicit imports by automatically importing all modules within specified
    packages. It is designed for use with class registration mechanisms to enable
    automatic discovery and registration of classes when they are placed in the
    correct package structure.

    Example:
    ::
        from guidellm.utils import AutoImporterMixin

        class MyRegistry(AutoImporterMixin):
            auto_package = "my_package.implementations"

        MyRegistry.auto_import_package_modules()

    :cvar auto_package: Package name or tuple of package names to import modules from
    :cvar auto_ignore_modules: Module names to ignore during import
    :cvar auto_imported_modules: List tracking which modules have been imported
    """

    auto_package: ClassVar[str | tuple[str, ...] | None] = None
    auto_ignore_modules: ClassVar[tuple[str, ...] | None] = None
    auto_imported_modules: ClassVar[list[str] | None] = None

    @classmethod
    def auto_import_package_modules(cls) -> None:
        """
        Automatically import all modules within the specified package(s).

        Scans the package(s) defined in the `auto_package` class variable and imports
        all modules found, tracking them in `auto_imported_modules`. Skips packages
        (directories) and any modules listed in `auto_ignore_modules`.

        :raises ValueError: If the `auto_package` class variable is not set
        """
        if cls.auto_package is None:
            raise ValueError(
                "The class variable 'auto_package' must be set to the package name to "
                "import modules from."
            )

        cls.auto_imported_modules = []
        packages = (
            cls.auto_package
            if isinstance(cls.auto_package, tuple)
            else (cls.auto_package,)
        )

        for package_name in packages:
            package = importlib.import_module(package_name)

            for _, module_name, is_pkg in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                if (
                    is_pkg
                    or (
                        cls.auto_ignore_modules is not None
                        and module_name in cls.auto_ignore_modules
                    )
                    or module_name in cls.auto_imported_modules
                ):
                    # Skip packages and ignored modules
                    continue

                if module_name in sys.modules:
                    # Avoid circular imports
                    cls.auto_imported_modules.append(module_name)
                else:
                    importlib.import_module(module_name)
                    cls.auto_imported_modules.append(module_name)
