"""
Registry system for dynamic object registration and discovery.

Provides a flexible object registration system with optional auto-discovery
capabilities through decorators and module imports. Enables dynamic discovery
and instantiation of implementations based on configuration parameters, supporting
both manual registration and automatic package-based discovery for extensible
plugin architectures.
"""

from __future__ import annotations

from typing import TypeVar

__all__ = ["RegisterT", "RegistryObjT"]


RegistryObjT = TypeVar("RegistryObjT")
"""Generic type variable for objects managed by the registry system."""
RegisterT = TypeVar(
    "RegisterT", bound=type
)  # Must be bound to type to ensure __name__ is available.
"""Generic type variable for the args and return values within the registry."""
