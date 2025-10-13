"""
Mixin classes for common metadata extraction and object introspection.

Provides reusable mixins for extracting structured metadata from objects,
enabling consistent information exposure across different class hierarchies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["InfoMixin"]


PYTHON_PRIMITIVES = (str, int, float, bool, list, tuple, dict)
"""Type alias for serialized object representations"""


class InfoMixin:
    """
    Mixin class providing standardized metadata extraction for introspection.

    Enables consistent object metadata extraction patterns across different
    class hierarchies for debugging, serialization, and runtime analysis.
    Provides both instance and class-level methods for extracting structured
    information from arbitrary objects with fallback handling for objects
    without built-in info capabilities.

    Example:
    ::
        from guidellm.utils.mixins import InfoMixin

        class ConfiguredClass(InfoMixin):
            def __init__(self, setting: str):
                self.setting = setting

        obj = ConfiguredClass("value")
        # Returns {'str': 'ConfiguredClass(...)', 'type': 'ConfiguredClass', ...}
        print(obj.info)
    """

    @classmethod
    def extract_from_obj(cls, obj: Any) -> dict[str, Any]:
        """
        Extract structured metadata from any object.

        Attempts to use the object's own `info` method or property if available,
        otherwise constructs metadata from object attributes and type information.
        Provides consistent metadata format across different object types.

        :param obj: Object to extract metadata from
        :return: Dictionary containing object metadata including type, class,
            module, and public attributes
        """
        if hasattr(obj, "info"):
            return obj.info() if callable(obj.info) else obj.info

        return {
            "str": str(obj),
            "type": type(obj).__name__,
            "class": obj.__class__.__name__ if hasattr(obj, "__class__") else None,
            "module": obj.__class__.__module__ if hasattr(obj, "__class__") else None,
            "attributes": (
                {
                    key: val if isinstance(val, PYTHON_PRIMITIVES) else repr(val)
                    for key, val in obj.__dict__.items()
                    if not key.startswith("_")
                }
                if hasattr(obj, "__dict__")
                else {}
            ),
        }

    @classmethod
    def create_info_dict(cls, obj: Any) -> dict[str, Any]:
        """
        Create a structured info dictionary for the given object.

        Builds standardized metadata dictionary containing object identification,
        type information, and accessible attributes. Used internally by other
        info extraction methods and available for direct metadata construction.

        :param obj: Object to extract info from
        :return: Dictionary containing structured metadata about the object
        """
        return {
            "str": str(obj),
            "type": type(obj).__name__,
            "class": obj.__class__.__name__ if hasattr(obj, "__class__") else None,
            "module": obj.__class__.__module__ if hasattr(obj, "__class__") else None,
            "attributes": (
                {
                    key: val
                    if isinstance(val, str | int | float | bool | list | dict)
                    else repr(val)
                    for key, val in obj.__dict__.items()
                    if not key.startswith("_")
                }
                if hasattr(obj, "__dict__")
                else {}
            ),
        }

    @property
    def info(self) -> dict[str, Any]:
        """
        Return structured metadata about this instance.

        Provides consistent access to object metadata for debugging, serialization,
        and introspection. Uses the create_info_dict method to generate standardized
        metadata format including class information and public attributes.

        :return: Dictionary containing class name, module, and public attributes
        """
        return self.create_info_dict(self)
