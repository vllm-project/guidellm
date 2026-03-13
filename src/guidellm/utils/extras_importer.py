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
    Import stub that lazily loads dependencies or raises helpful errors.

    This class handles both eager and lazy import scenarios:
    - In eager mode, the import is attempted immediately on creation
    - In lazy mode, the import is deferred until first use

    If the import fails, the stub raises helpful errors when used.
    """

    def __init__(
        self,
        name: str,
        import_path: str,
        extras_group: str | list[str],
        eager: bool = False,
    ) -> None:
        """
        Initialize an import stub.

        Args:
            name: The attribute name being imported (e.g., "SamplingParams")
            import_path: The full import path (e.g., "vllm.SamplingParams")
            extras_group: Name(s) of extras group for error messages
            eager: If True, attempt import immediately
        """
        self._stub_name = name
        self._stub_import_path = import_path
        self._stub_extras_group = extras_group
        self._stub_target: Any = None
        self._stub_failed = False
        self._stub_lock = threading.Lock()

        if eager:
            # Attempt import but don't raise on failure
            self._ensure_imported(raise_on_failure=False)

    def _parse_import_path(self) -> tuple[str, str | None]:
        """Parse import path into module and optional attribute."""
        if "." not in self._stub_import_path:
            return self._stub_import_path, None
        parts = self._stub_import_path.rsplit(".", 1)
        return parts[0], parts[1]

    def _ensure_imported(self, raise_on_failure: bool = True) -> Any:
        """
        Import the target object if not already imported.

        Args:
            raise_on_failure: If True, raise ImportError on failure.
                If False, return None.

        Returns:
            The imported object, or None if import failed
        """
        if self._stub_target is None and not self._stub_failed:
            with self._stub_lock:
                if self._stub_target is None and not self._stub_failed:
                    module_path, attr_name = self._parse_import_path()
                    try:
                        module = importlib.import_module(module_path)
                        self._stub_target = (
                            getattr(module, attr_name) if attr_name else module
                        )
                    except (ImportError, AttributeError):
                        self._stub_failed = True

        if self._stub_failed and raise_on_failure:
            self._raise_import_error()

        return self._stub_target

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
        """Forward calls to the imported object."""
        return self._ensure_imported()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the imported object."""
        if name.startswith("_stub_"):
            raise AttributeError(name)
        return getattr(self._ensure_imported(), name)

    def __bool__(self) -> bool:
        """Return truthiness of the imported object, or False if import failed."""
        if self._stub_failed:
            return False
        try:
            target = self._ensure_imported()
            return bool(target)
        except ImportError:
            return False

    def __repr__(self) -> str:
        """Return representation."""
        if self._stub_target is not None:
            return repr(self._stub_target)
        if self._stub_failed:
            return f"<ImportStub (failed) for '{self._stub_name}'>"
        return f"<ImportStub (not imported) for '{self._stub_name}'>"


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
        self._eager = eager
        self._stub_cache: dict[str, _ImportStub] = {}
        self._lock = threading.Lock()

        # Pre-create stubs if eager mode
        if eager:
            for name in imports:
                self._get_or_create_stub(name)

    def _get_or_create_stub(self, name: str) -> _ImportStub:
        """
        Get or create a stub for the named import.

        Args:
            name: The attribute name to get a stub for

        Returns:
            An import stub (eager or lazy based on configuration)
        """
        with self._lock:
            if name in self._stub_cache:
                return self._stub_cache[name]

            if name not in self._imports:
                raise AttributeError(f"No import registered for '{name}'")

            import_path = self._imports[name]
            stub = _ImportStub(name, import_path, self._extras_group, eager=self._eager)
            self._stub_cache[name] = stub
            return stub

    def __getattr__(self, name: str) -> Any:
        """
        Get a stub for the named import.

        Args:
            name: The attribute name to access

        Returns:
            An import stub

        Raises:
            AttributeError: If the attribute is not registered or is private
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        return self._get_or_create_stub(name)

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
            module_path = self._parse_module_path(import_path)
            if importlib.util.find_spec(module_path) is None:
                return False
        return True

    def _parse_module_path(self, import_path: str) -> str:
        """
        Extract the module path from an import path.

        Args:
            import_path: The import path (e.g., "vllm.SamplingParams")

        Returns:
            The module path (e.g., "vllm")
        """
        if "." not in import_path:
            return import_path
        return import_path.rsplit(".", 1)[0]
