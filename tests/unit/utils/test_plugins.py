"""Unit tests for GuideLLM plugin discovery and loading."""

from __future__ import annotations

import importlib.metadata
import sys
import warnings
from types import ModuleType

import pytest

from guidellm.backends import Backend
from guidellm.settings import Settings, settings
from guidellm.utils import plugins as plugin_utils
from guidellm.utils.env_validator import validate_env_vars


@pytest.mark.smoke
def test_guidellm_plugins_env_var_recognized(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ``GUIDELLM__PLUGINS`` should be recognized as a valid settings env var.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setenv("GUIDELLM__PLUGINS", "guidellm_plugin_example")
    invalid, valid = validate_env_vars(Settings)
    assert "GUIDELLM__PLUGINS" in valid
    assert "GUIDELLM__PLUGINS" not in invalid


@pytest.mark.sanity
def test_load_plugins_callable_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Callable ``module:func`` plugins are invoked once and repeated calls are idempotent.

    ## WRITTEN BY AI ##
    """
    calls = 0
    mod_name = "_guidellm_test_callable_plugin"
    fake_mod = ModuleType(mod_name)

    def register() -> None:
        nonlocal calls
        calls += 1

    fake_mod.register = register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            importlib.metadata.EntryPoint(
                name="callable_test",
                value=f"{mod_name}:register",
                group=group,
            )
        ],
    )
    monkeypatch.setattr(plugin_utils, "_loaded_entry_points", set())
    monkeypatch.setattr(plugin_utils, "_loaded_plugin_targets", set())
    monkeypatch.setattr(settings, "plugins", [])

    first = plugin_utils.load_plugins()
    second = plugin_utils.load_plugins()

    assert first == [f"callable_test={mod_name}:register"]
    assert second == []
    assert calls == 1


def _reset_plugin_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear loader memory so a test sees one fresh ``load_plugins`` pass."""
    monkeypatch.setattr(plugin_utils, "_loaded_entry_points", set())
    monkeypatch.setattr(plugin_utils, "_loaded_plugin_targets", set())


def _unregister_backend(kind: str) -> None:
    """Drop a test backend kind so later tests see the real registry."""
    if Backend.registry is not None and kind in Backend.registry:
        del Backend.registry[kind]


@pytest.mark.regression
def test_callable_registers_once_for_entry_point_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One ``module:callable`` discovered both ways registers its backend once.

    ## WRITTEN BY AI ##
    """
    calls = 0
    kind = "_guidellm_test_plugin_once"
    mod_name = "_guidellm_test_plugin_once_mod"
    fake_mod = ModuleType(mod_name)

    class _DummyBackend:
        pass

    def register() -> None:
        nonlocal calls
        calls += 1
        Backend.register(kind)(_DummyBackend)

    fake_mod.register = register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            importlib.metadata.EntryPoint(
                name="once_test",
                value=f"{mod_name}:register",
                group=group,
            )
        ],
    )
    _reset_plugin_loader(monkeypatch)
    monkeypatch.setattr(settings, "plugins", [f"{mod_name}:register"])

    try:
        loaded = plugin_utils.load_plugins()
        assert Backend.is_registered(kind)
    finally:
        _unregister_backend(kind)

    assert loaded == [f"once_test={mod_name}:register"]
    assert calls == 1


@pytest.mark.regression
def test_duplicate_registration_in_hook_warns_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A hook that stops on a duplicate registration is not recorded as loaded.

    ## WRITTEN BY AI ##
    """
    calls = 0
    kind = "_guidellm_test_plugin_import_time"
    mod_name = "_guidellm_test_plugin_import_time_mod"
    fake_mod = ModuleType(mod_name)

    class _DummyBackend:
        pass

    def register() -> None:
        nonlocal calls
        calls += 1
        Backend.register(kind)(_DummyBackend)

    fake_mod.register = register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    register()
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            importlib.metadata.EntryPoint(
                name="import_time_test",
                value=f"{mod_name}:register",
                group=group,
            )
        ],
    )
    _reset_plugin_loader(monkeypatch)
    monkeypatch.setattr(settings, "plugins", [])

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = plugin_utils.load_plugins()
        assert Backend.is_registered(kind)
    finally:
        _unregister_backend(kind)

    failure_warnings = [
        item for item in caught if "Failed to load GuideLLM plugin" in str(item.message)
    ]
    assert len(failure_warnings) == 1
    assert "already registered" in str(failure_warnings[0].message)
    assert loaded == []
    assert f"{mod_name}:register" not in plugin_utils._loaded_plugin_targets


@pytest.mark.regression
def test_broken_entry_point_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A broken ``guidellm.plugins`` entry point warns clearly instead of failing silently.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            importlib.metadata.EntryPoint(
                name="broken_plugin",
                value="nonexistent_guidellm_plugin_pkg_xyz",
                group=group,
            )
        ],
    )
    monkeypatch.setattr(plugin_utils, "_loaded_entry_points", set())
    monkeypatch.setattr(plugin_utils, "_loaded_plugin_targets", set())
    monkeypatch.setattr(settings, "plugins", [])

    with pytest.warns(UserWarning, match="Failed to load GuideLLM plugin entry point"):
        loaded = plugin_utils.load_plugins()

    assert loaded == []


@pytest.mark.regression
def test_broken_env_plugin_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An unimportable ``GUIDELLM__PLUGINS`` target raises a clear ``RuntimeError``.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])
    monkeypatch.setattr(plugin_utils, "_loaded_entry_points", set())
    monkeypatch.setattr(plugin_utils, "_loaded_plugin_targets", set())
    monkeypatch.setattr(settings, "plugins", ["nonexistent_guidellm_plugin_pkg_xyz"])

    with pytest.raises(
        RuntimeError,
        match="Failed to load GuideLLM plugin 'nonexistent_guidellm_plugin_pkg_xyz'",
    ) as exc_info:
        plugin_utils.load_plugins()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
