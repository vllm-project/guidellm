"""Integration tests for loading GuideLLM plugins in subprocesses."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from build import ProjectBuilder

import guidellm
from guidellm.backends import Backend
from guidellm.schemas.backends import BackendArgs

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
    """Child process entrypoint checking registries after importing GuideLLM."""
    queue.put(
        (
            "echo_plugin" in (Backend.registry or {}),
            "echo_plugin" in (BackendArgs.registry or {}),
            guidellm.settings.plugins,
        )
    )


@pytest.mark.sanity
def test_entry_point_plugin_registers_in_subprocess(tmp_path: Path) -> None:
    """
    An installed ``guidellm.plugins`` package registers its backend and CLI kind.

    ## WRITTEN BY AI ##
    """
    site_dir = _unpack_fixture_wheel(tmp_path)
    env = os.environ.copy()
    env.pop("GUIDELLM__PLUGINS", None)
    env["PYTHONPATH"] = f"{site_dir}{os.pathsep}{REPO_ROOT / 'src'}"

    script = (
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
def test_env_plugin_registers_in_spawn_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Spawned child processes importing GuideLLM packages load ``GUIDELLM__PLUGINS``.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setenv("GUIDELLM__PLUGINS", "guidellm_plugin_example")
    monkeypatch.syspath_prepend(str(FIXTURE_PKG_DIR))

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_assert_echo_plugin_registered_in_child, args=(queue,))
    proc.start()
    try:
        proc.join(timeout=30)
        assert proc.exitcode == 0
        backend_ok, args_ok, plugins_cfg = queue.get(timeout=5)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        proc.close()
        queue.close()
        queue.join_thread()
    assert backend_ok is True
    assert args_ok is True
    assert plugins_cfg == ["guidellm_plugin_example"]


_LAZY_IMPORT_SCRIPT = """
import sys

def plugin_modules():
    return sorted(m for m in sys.modules if m.startswith("guidellm_plugin_example"))

import guidellm
assert plugin_modules() == [], plugin_modules()
import guidellm.schemas
import guidellm.scheduler
assert "guidellm_plugin_example.schemas.backends" in sys.modules, plugin_modules()
assert "guidellm_plugin_example.backends" not in sys.modules, plugin_modules()
from guidellm.schemas.benchmark.entrypoints import BenchmarkArgs
args = BenchmarkArgs.model_validate({
    "backend": {"kind": "echo_plugin", "target": "http://localhost:8000"},
    "profile": {"kind": "synchronous"},
    "data": [{"kind": "synthetic_text", "prompt_tokens": 1, "output_tokens": 1}],
})
assert type(args.backend).__name__ == "EchoPluginBackendArgs", type(args.backend)
import guidellm.backends
assert "guidellm_plugin_example.backends" in sys.modules, plugin_modules()
"""


@pytest.mark.sanity
def test_plugin_modules_load_with_matching_guidellm_package(tmp_path: Path) -> None:
    """
    ``import guidellm`` loads no plugin code; each package loads only its mirror.

    ## WRITTEN BY AI ##
    """
    site_dir = _unpack_fixture_wheel(tmp_path)
    env = os.environ.copy()
    env.pop("GUIDELLM__PLUGINS", None)
    env["PYTHONPATH"] = f"{site_dir}{os.pathsep}{REPO_ROOT / 'src'}"

    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _LAZY_IMPORT_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
