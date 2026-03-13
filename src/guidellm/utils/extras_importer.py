"""
Lazy-loading utilities for optional dependencies.

This module provides ExtrasImporter, a class for managing optional dependencies
with graceful fallback when dependencies are unavailable. It supports both eager
and lazy import strategies and creates helpful import stubs for missing packages.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
from typing import Any


class _ImportStub:
    """
    Placeholder object that raises ImportError when called or accessed.

    Allows isinstance checks and None comparisons without raising errors,
    but blocks instantiation, method calls, and attribute access.

    This enables imports to succeed even when dependencies are unavailable,
    with errors only occurring when the stub is actually used.
    """

    def __init__(
        self,
        name: str,
        import_path: str,
        extras_group: str | list[str],
    ) -> None:
        """
        Initialize an import stub.

        Args:
            name: The attribute name being imported (e.g., "SamplingParams")
            import_path: The full import path (e.g., "vllm.SamplingParams")
            extras_group: Name(s) of extras group for error messages
        """
        self._stub_name = name
        self._stub_import_path = import_path
        self._stub_extras_group = extras_group

    def _raise_import_error(self) -> None:
        """Raise helpful ImportError with installation instructions."""
        if isinstance(self._stub_extras_group, list):
            extras = " or ".join(f"guidellm[{g}]" for g in self._stub_extras_group)
        else:
            extras = f"guidellm[{self._stub_extras_group}]"

        raise ImportError(
            f"'{self._stub_name}' from '{self._stub_import_path}' requires "
            f"optional dependencies. Install with: pip install {extras}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise error when stub is called."""
        self._raise_import_error()

    def __getattr__(self, name: str) -> Any:
        """
        Raise AttributeError for attribute access.

        This allows hasattr() checks to work properly (they catch AttributeError).
        The stub itself will raise ImportError when called.
        """
        raise AttributeError(
            f"'{self._stub_name}.{name}' is not available. "
            f"Install guidellm[{self._stub_extras_group}] to use this feature."
        )

    def __bool__(self) -> bool:
        """Return False to support truthiness checks."""
        return False

    def __repr__(self) -> str:
        """Return a helpful representation."""
        return f"<ImportStub for '{self._stub_name}' from '{self._stub_import_path}'>"


class ExtrasImporter:
    """
    Lazy-loading manager for optional dependencies with graceful fallback.

    Handles optional dependency imports by creating import stubs when
    dependencies are unavailable. Supports both eager loading (import all
    at initialization) and lazy loading (import on first access).

    Example:
        vllm_extras = ExtrasImporter(
            {
                "SamplingParams": "vllm.SamplingParams",
                "AsyncEngineArgs": "vllm.engine.arg_utils.AsyncEngineArgs",
                "AsyncLLMEngine": "vllm.engine.async_llm_engine.AsyncLLMEngine",
                "RequestOutput": "vllm.outputs.RequestOutput",
            },
            extras_group="vllm",
        )

        # Access attributes (triggers lazy import)
        params = vllm_extras.SamplingParams(temperature=0.7)

        # Check availability
        if vllm_extras.is_available:
            # Use vllm features
    """

    def __init__(
        self,
        imports: dict[str, str],
        *,
        extras_group: str | list[str],
        eager: bool = True,
    ) -> None:
        """
        Initialize the extras importer.

        Args:
            imports: Mapping of attribute names to import paths.
                Examples:
                - "SamplingParams": "vllm.SamplingParams" (object import)
                - "vllm": "vllm" (module import)
            extras_group: Name(s) of extras group for error messages.
                Used to generate helpful pip install commands.
            eager: If True, attempt all imports immediately (default: True).
                If False, imports happen on first attribute access (lazy).
        """
        self._imports = imports
        self._extras_group = extras_group
        self._import_cache: dict[str, Any] = {}
        self._stub_cache: dict[str, _ImportStub] = {}
        self._lock = threading.Lock()

        if eager:
            self._import_all()

    def _import_all(self) -> None:
        """Eagerly import all registered imports."""
        for name in self._imports:
            self._import_or_stub(name)

    def _import_or_stub(self, name: str) -> Any:
        """
        Import the named attribute or create a stub.

        Args:
            name: The attribute name to import

        Returns:
            The imported object or an import stub
        """
        with self._lock:
            # Check cache first
            if name in self._import_cache:
                return self._import_cache[name]
            if name in self._stub_cache:
                return self._stub_cache[name]

            # Try to import
            import_path = self._imports[name]
            module_path, attr_name = self._parse_import_path(import_path)

            try:
                module = importlib.import_module(module_path)
                obj = getattr(module, attr_name) if attr_name else module
                self._import_cache[name] = obj
                return obj
            except (ImportError, AttributeError):
                stub = _ImportStub(name, import_path, self._extras_group)
                self._stub_cache[name] = stub
                return stub

    def __getattr__(self, name: str) -> Any:
        """
        Lazy load imports on first access.

        Args:
            name: The attribute name to access

        Returns:
            The imported object or an import stub

        Raises:
            AttributeError: If the attribute is not registered or is private
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name not in self._imports:
            raise AttributeError(f"No import registered for '{name}'")

        return self._import_or_stub(name)

    @property
    def is_available(self) -> bool:
        """
        Check if all registered imports are available without importing them.

        Uses importlib.util.find_spec to check module availability.
        Compatible with HAS_* flag pattern.

        Returns:
            True if all imports are available, False otherwise
        """
        for import_path in self._imports.values():
            module_path, _ = self._parse_import_path(import_path)
            if importlib.util.find_spec(module_path) is None:
                return False
        return True

    def _parse_import_path(self, import_path: str) -> tuple[str, str | None]:
        """
        Parse import path to distinguish module vs object import.

        Args:
            import_path: The import path to parse

        Returns:
            Tuple of (module_path, attribute_name)
            - attribute_name is None for module imports
            - attribute_name is set for object imports
        """
        if "." not in import_path:
            return import_path, None

        parts = import_path.rsplit(".", 1)
        return parts[0], parts[1]
