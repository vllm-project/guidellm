"""Unit tests for GuideLLM plugin discovery and loading."""

from __future__ import annotations

import importlib.metadata
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from guidellm.settings import Settings, settings
from guidellm.utils import plugins as plugin_utils
from guidellm.utils.env_validator import validate_env_vars


@pytest.fixture
def fresh_loader(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test an empty loader state, no entry points and no settings."""
    monkeypatch.setattr(plugin_utils, "_attempted_targets", set())
    monkeypatch.setattr(plugin_utils, "_failed_packages", set())
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])
    monkeypatch.setattr(settings, "plugins", [])
    plugin_utils._entry_point_packages.cache_clear()
    yield
    plugin_utils._entry_point_packages.cache_clear()


def _make_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    modules: dict[str, str],
) -> None:
    """Write a plugin package with the given ``{submodule: source}`` files."""
    pkg_dir = tmp_path / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    for module, source in modules.items():
        (pkg_dir / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    for loaded in [mod for mod in sys.modules if mod.split(".")[0] == name]:
        monkeypatch.delitem(sys.modules, loaded)


def _set_entry_points(monkeypatch: pytest.MonkeyPatch, *values: str) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            importlib.metadata.EntryPoint(name=f"ep{idx}", value=value, group=group)
            for idx, value in enumerate(values)
        ],
    )
    plugin_utils._entry_point_packages.cache_clear()


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
@pytest.mark.usefixtures("fresh_loader")
def test_loads_only_the_matching_submodule_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``guidellm.backends`` imports ``<plugin>.backends`` once and nothing else.

    ## WRITTEN BY AI ##
    """
    _make_plugin(
        tmp_path,
        monkeypatch,
        "_gl_plugin_match",
        {"backends": "", "data": "raise RuntimeError('data must stay unloaded')"},
    )
    monkeypatch.setattr(settings, "plugins", ["_gl_plugin_match"])

    first = plugin_utils.load_plugin_submodules("guidellm.backends")
    second = plugin_utils.load_plugin_submodules("guidellm.backends")

    assert first == ["_gl_plugin_match.backends"]
    assert second == []
    assert "_gl_plugin_match.data" not in sys.modules


@pytest.mark.sanity
@pytest.mark.usefixtures("fresh_loader")
def test_plugin_without_matching_submodule_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A plugin with no ``scheduler`` module adds nothing and warns about nothing.

    ## WRITTEN BY AI ##
    """
    _make_plugin(tmp_path, monkeypatch, "_gl_plugin_partial", {"backends": ""})
    _set_entry_points(monkeypatch, "_gl_plugin_partial")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = plugin_utils.load_plugin_submodules("guidellm.scheduler")

    assert loaded == []


@pytest.mark.regression
@pytest.mark.usefixtures("fresh_loader")
def test_entry_point_and_settings_package_load_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A package found both as an entry point and in settings is imported once.

    ## WRITTEN BY AI ##
    """
    _make_plugin(tmp_path, monkeypatch, "_gl_plugin_both", {"backends": ""})
    _set_entry_points(monkeypatch, "_gl_plugin_both")
    monkeypatch.setattr(settings, "plugins", ["_gl_plugin_both"])

    assert plugin_utils.load_plugin_submodules("guidellm.backends") == [
        "_gl_plugin_both.backends"
    ]


@pytest.mark.regression
@pytest.mark.usefixtures("fresh_loader")
def test_missing_entry_point_package_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An entry point naming a missing package warns once, not once per package.

    ## WRITTEN BY AI ##
    """
    _set_entry_points(monkeypatch, "nonexistent_guidellm_plugin_pkg_xyz")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugin_utils.load_plugin_submodules("guidellm.schemas")
        plugin_utils.load_plugin_submodules("guidellm.backends")

    messages = [str(item.message) for item in caught]
    assert len(messages) == 1
    assert "Failed to load GuideLLM plugin entry point" in messages[0]


@pytest.mark.regression
@pytest.mark.usefixtures("fresh_loader")
def test_broken_entry_point_submodule_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An entry-point plugin module that raises is reported as a warning.

    ## WRITTEN BY AI ##
    """
    _make_plugin(
        tmp_path, monkeypatch, "_gl_plugin_broken", {"backends": "raise ValueError"}
    )
    _set_entry_points(monkeypatch, "_gl_plugin_broken")

    with pytest.warns(UserWarning, match=r"_gl_plugin_broken\.backends"):
        loaded = plugin_utils.load_plugin_submodules("guidellm.backends")

    assert loaded == []


@pytest.mark.regression
@pytest.mark.usefixtures("fresh_loader")
def test_broken_settings_plugin_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unimportable ``GUIDELLM__PLUGINS`` package raises a clear ``RuntimeError``.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(settings, "plugins", ["nonexistent_guidellm_plugin_pkg_xyz"])

    with pytest.raises(
        RuntimeError,
        match="Failed to load GuideLLM plugin 'nonexistent_guidellm_plugin_pkg_xyz'",
    ) as exc_info:
        plugin_utils.load_plugin_submodules("guidellm.backends")

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.smoke
def test_rejects_package_outside_guidellm() -> None:
    """
    Only GuideLLM package names can be mirrored.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError, match="GuideLLM package name"):
        plugin_utils.load_plugin_submodules("other.backends")
