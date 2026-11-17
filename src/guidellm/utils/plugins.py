"""
Plugin discovery and loading utilities for GuideLLM.

Discovers and imports third-party plugins via Python entry points in the
``guidellm.plugins`` group as well as module targets configured through
``settings.plugins`` (``GUIDELLM__PLUGINS``). Loading plugins at package import
time ensures registries are populated before CLI option parsing and inside
``spawn`` / ``forkserver`` worker processes.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any

from guidellm.logger import logger
from guidellm.settings import settings

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "load_plugins",
]

PLUGIN_ENTRY_POINT_GROUP = "guidellm.plugins"

_loaded_entry_points: set[tuple[str, str]] = set()
# Entry-point values and settings targets share this set so one string loads once.
_loaded_plugin_targets: set[str] = set()


def _is_duplicate_registration(err: BaseException) -> bool:
    """Return whether ``err`` means the plugin kind is already registered."""
    return isinstance(err, ValueError) and "already registered" in str(err)


def _invoke_plugin_object(loaded: Any) -> None:
    """Invoke a callable plugin hook if it is not a class object."""
    if callable(loaded) and not isinstance(loaded, type):
        loaded()


def _load_target(target: str) -> None:
    """Load a ``module`` or ``module:attr`` plugin target via EntryPoint."""
    entry_point = importlib.metadata.EntryPoint(
        name=target,
        value=target,
        group=PLUGIN_ENTRY_POINT_GROUP,
    )
    loaded = entry_point.load()
    _invoke_plugin_object(loaded)


def load_plugins() -> list[str]:
    """
    Discover and load GuideLLM plugins from entry points and settings.

    Loads plugins registered under the ``guidellm.plugins`` entry-point group
    first, followed by any module targets configured in ``settings.plugins``
    (typically via ``GUIDELLM__PLUGINS``). A target string that was already
    loaded, including an entry-point value repeated in ``settings.plugins``,
    is skipped so that hook runs once.

    An entry-point hook that raises because its kind is already registered is
    counted as loaded. A kind registered while the module was imported stays
    in place, and a colliding entry point does not replace it or warn. Other
    entry-point failures emit a warning so an unrelated broken package does
    not break all CLI invocations. Explicit ``settings.plugins`` load failures
    raise ``RuntimeError`` with the failing target name and cause.

    :return: List of newly loaded plugin identifiers
    :raises RuntimeError: If an explicitly configured plugin in
        ``settings.plugins`` fails to load
    """
    newly_loaded: list[str] = []

    discovered = sorted(
        importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP),
        key=lambda item: (item.name, item.value),
    )
    for ep in discovered:
        key = (ep.name, ep.value)
        if key in _loaded_entry_points or ep.value in _loaded_plugin_targets:
            continue
        try:
            loaded = ep.load()
            _invoke_plugin_object(loaded)
        except Exception as err:  # noqa: BLE001
            if not _is_duplicate_registration(err):
                msg = (
                    f"Failed to load GuideLLM plugin entry point "
                    f"'{ep.name}' ({ep.value}): {err}"
                )
                logger.warning(msg)
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                continue

        _loaded_entry_points.add(key)
        _loaded_plugin_targets.add(ep.value)
        newly_loaded.append(f"{ep.name}={ep.value}")

    for target in settings.plugins:
        if target in _loaded_plugin_targets:
            continue
        try:
            _load_target(target)
        except Exception as err:
            msg = f"Failed to load GuideLLM plugin '{target}': {err}"
            logger.error(msg)
            raise RuntimeError(msg) from err

        _loaded_plugin_targets.add(target)
        newly_loaded.append(target)

    return newly_loaded
