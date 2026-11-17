"""Unit tests for GuideLLM plugin discovery and loading."""

from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from build import ProjectBuilder

import guidellm
from guidellm.backends import Backend
from guidellm.schemas.backends import BackendArgs
from guidellm.settings import Settings, settings
from guidellm.utils import plugins as plugin_utils
from guidellm.utils.env_validator import validate_env_vars

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PKG_DIR = REPO_ROOT / "tests" / "fixtures" / "guidellm_plugin_example"


def _unpack_fixture_wheel(dest_dir: Path) -> Path:
    """Build the example plugin wheel offline and extract it into ``dest_dir``."""
    pkg_copy = dest_dir / "pkg"
    shutil.copytree(FIXTURE_PKG_DIR, pkg_copy)
    wheel_dir = dest_dir / "dist"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    builder = ProjectBuilder(str(pkg_copy))
    wheel_path = Path(builder.build("wheel", str(wheel_dir)))
    site_dir = dest_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(site_dir)
    return site_dir


def _assert_echo_plugin_registered_in_child(queue: multiprocessing.Queue) -> None:
    """Child process entrypoint checking registries after ``import guidellm``."""
    queue.put(
        (
            "echo_plugin" in (Backend.registry or {}),
            "echo_plugin" in (BackendArgs.registry or {}),
            guidellm.settings.plugins,
        )
    )


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
def test_load_plugins_from_entry_point_in_subprocess(tmp_path: Path) -> None:
    """
    Installed ``guidellm.plugins`` entry points populate registries at import and CLI.

    ## WRITTEN BY AI ##
    """
    site_dir = _unpack_fixture_wheel(tmp_path)
    env = os.environ.copy()
    env.pop("GUIDELLM__PLUGINS", None)
    env["PYTHONPATH"] = f"{site_dir}{os.pathsep}{REPO_ROOT / 'src'}"

    script = (
        "import guidellm; "
        "from guidellm.backends import Backend; "
        "from guidellm.schemas.backends import BackendArgs; "
        "assert Backend.is_registered('echo_plugin'); "
        "assert BackendArgs.is_registered('echo_plugin')"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    cli_proc = subprocess.run(
        [sys.executable, "-m", "guidellm", "run", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli_proc.returncode == 0, cli_proc.stderr
    assert "echo_plugin" in cli_proc.stdout


@pytest.mark.sanity
def test_load_plugins_from_env_in_spawn_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Spawned child processes importing ``guidellm`` load ``GUIDELLM__PLUGINS``.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setenv("GUIDELLM__PLUGINS", "guidellm_plugin_example")
    monkeypatch.syspath_prepend(str(FIXTURE_PKG_DIR))

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_assert_echo_plugin_registered_in_child, args=(queue,))
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0
    backend_ok, args_ok, plugins_cfg = queue.get(timeout=5)
    assert backend_ok is True
    assert args_ok is True
    assert plugins_cfg == ["guidellm_plugin_example"]


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
def test_import_time_registration_is_not_reported_as_entry_point_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A hook that already registered during import is still a loaded entry point.

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
    assert failure_warnings == []
    assert loaded == [f"import_time_test={mod_name}:register"]


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
