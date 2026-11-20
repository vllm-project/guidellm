"""
Plugin discovery and loading utilities for GuideLLM.

A plugin is a Python package named by an entry point in the ``guidellm.plugins``
group or listed in ``settings.plugins`` (``GUIDELLM__PLUGINS``). Plugins mirror
GuideLLM's top-level packages: when ``guidellm.<name>`` finishes importing, it
calls :func:`load_plugin_submodules`, which imports ``<plugin>.<name>`` from each
plugin that has it. ``import guidellm`` itself loads no plugin code, so a plugin
only adds import cost to the processes that import the matching GuideLLM package.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import warnings
from functools import cache

from guidellm.logger import logger
from guidellm.settings import settings

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "load_plugin_submodules",
]

PLUGIN_ENTRY_POINT_GROUP = "guidellm.plugins"

# Plugin modules already imported or already reported as failing.
_attempted_targets: set[str] = set()
# Plugin packages whose top-level import failed; their other modules are skipped.
_failed_packages: set[str] = set()


@cache
def _entry_point_packages() -> tuple[str, ...]:
    """Return the plugin package names declared by installed entry points."""
    discovered = sorted(
        importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP),
        key=lambda item: (item.name, item.value),
    )
    return tuple(dict.fromkeys(ep.module for ep in discovered))


def _plugin_packages() -> list[tuple[str, bool]]:
    """
    Return ``(package, explicit)`` pairs for every configured plugin.

    ``explicit`` is true for packages listed in ``settings.plugins``; their load
    errors are raised instead of reported as warnings.
    """
    explicit = list(dict.fromkeys(settings.plugins))
    discovered = [pkg for pkg in _entry_point_packages() if pkg not in explicit]
    return [(pkg, False) for pkg in discovered] + [(pkg, True) for pkg in explicit]


def _report_failure(target: str, package: str, explicit: bool, err: Exception) -> None:
    if explicit:
        msg = f"Failed to load GuideLLM plugin '{package}' ({target}): {err}"
        logger.error(msg)
        raise RuntimeError(msg) from err
    msg = f"Failed to load GuideLLM plugin entry point '{package}' ({target}): {err}"
    logger.warning(msg)
    warnings.warn(msg, category=UserWarning, stacklevel=3)


def load_plugin_submodules(package_name: str) -> list[str]:
    """
    Import the plugin modules that mirror the GuideLLM package ``package_name``.

    Called at the end of each top-level GuideLLM package, for example
    ``load_plugin_submodules(__name__)`` in ``guidellm.backends`` imports
    ``<plugin>.backends`` for every plugin that has that module. Plugins without
    the module are skipped. Each module is attempted at most once per process.

    :param package_name: Name of the GuideLLM package that finished importing,
        such as ``"guidellm.backends"``
    :return: Names of the plugin modules imported by this call
    :raises RuntimeError: If a module of a plugin listed in ``settings.plugins``
        fails to import. Failures of entry-point plugins are reported as warnings
        so an unrelated broken package does not break every CLI invocation.
    """
    if package_name != "guidellm" and not package_name.startswith("guidellm."):
        raise ValueError(f"Expected a GuideLLM package name, got '{package_name}'")
    suffix = package_name.removeprefix("guidellm")
    loaded: list[str] = []

    for package, explicit in _plugin_packages():
        target = f"{package}{suffix}"
        if target in _attempted_targets or package in _failed_packages:
            continue
        _attempted_targets.add(target)
        try:
            # find_spec imports the parent packages but not the target itself
            spec = importlib.util.find_spec(target)
        except Exception as err:  # noqa: BLE001
            _failed_packages.add(package)
            _report_failure(target, package, explicit, err)
            continue
        if spec is None:
            continue
        try:
            importlib.import_module(target)
        except Exception as err:  # noqa: BLE001
            _report_failure(target, package, explicit, err)
            continue
        loaded.append(target)

    return loaded
