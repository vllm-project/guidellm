"""E2E tests for third-party backend plugins with mock server and workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from build import ProjectBuilder

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PKG_DIR = REPO_ROOT / "tests" / "fixtures" / "guidellm_plugin_example"


@pytest.fixture(scope="module")
def plugin_mock_server() -> Iterator[E2EServer]:
    """Start a local MockServer for plugin E2E tests."""
    server = start_mock_server(ttft_ms=1.0, itl_ms=1.0, output_tokens=8)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="module")
def unpacked_plugin_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and unpack the example plugin wheel into an isolated site directory."""
    base = tmp_path_factory.mktemp("plugin_site")
    pkg_copy = base / "pkg"
    shutil.copytree(FIXTURE_PKG_DIR, pkg_copy)
    wheel_dir = base / "dist"
    site_dir = base / "site"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = Path(ProjectBuilder(str(pkg_copy)).build("wheel", str(wheel_dir)))
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(site_dir)
    return site_dir


@pytest.mark.timeout(60)
@pytest.mark.sanity
@pytest.mark.parametrize("mp_context_type", ["spawn", "forkserver"])
@pytest.mark.parametrize("discovery_mode", ["entry_point", "env"])
def test_plugin_backend_runs_end_to_end(
    plugin_mock_server: E2EServer,
    unpacked_plugin_site: Path,
    tmp_path: Path,
    mp_context_type: str,
    discovery_mode: str,
) -> None:
    """
    Custom backend plugin completes requests across spawn and forkserver workers.

    ## WRITTEN BY AI ##
    """
    report_path = tmp_path / f"report_{discovery_mode}_{mp_context_type}.json"
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "GUIDELLM__MP_CONTEXT_TYPE": mp_context_type,
        }
    )

    if discovery_mode == "entry_point":
        env.pop("GUIDELLM__PLUGINS", None)
        env["PYTHONPATH"] = f"{unpacked_plugin_site}{os.pathsep}{REPO_ROOT / 'src'}"
    else:
        env["GUIDELLM__PLUGINS"] = "guidellm_plugin_example"
        env["PYTHONPATH"] = f"{FIXTURE_PKG_DIR}{os.pathsep}{REPO_ROOT / 'src'}"

    cmd = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
        "--backend",
        f"kind=echo_plugin,target={plugin_mock_server.get_url()}",
        "--profile",
        "kind=synchronous",
        "--data",
        "kind=synthetic_text,prompt_tokens=16,output_tokens=8",
        "--tokenizer",
        f"kind=huggingface_auto,model={MINIMAL_TOKENIZER_DIR}",
        "--constraint",
        "kind=max_requests,count=5",
        "--output",
        f"kind=json,path={report_path}",
        "--disable-console",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    report = json.loads(report_path.read_text())
    successful = report["benchmarks"][0]["requests"]["successful"]
    assert len(successful) == 5
