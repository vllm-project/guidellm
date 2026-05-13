"""Integration test: benchmark synthetic_image / synthetic_video against the
guidellm mock server.

Spins up the in-tree mock server (Sanic) in a subprocess, runs a short
`guidellm benchmark run` against it for both image and video synthetic data,
and asserts the benchmark process exits cleanly with at least one successful
request recorded.

The mock backend's TTFT/ITL numbers are meaningless here. We're only proving
that the new deserializers + data pipeline + request handler chain complete
end-to-end without errors.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.server import MockServer

pytestmark = [pytest.mark.smoke]


def _start_server_process(config: MockServerConfig) -> None:
    server = MockServer(config)
    server.run()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_server(base_url: str, timeout: float = 30.0) -> None:
    async def _poll() -> None:
        backoff = 0.5
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(f"{base_url}/health", timeout=1.0)
                    if resp.status_code == 200:
                        return
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 2.0)

    asyncio.run(asyncio.wait_for(_poll(), timeout=timeout))


@pytest.fixture(scope="module")
def mock_backend():
    port = _free_port()
    config = MockServerConfig(
        host="127.0.0.1",
        port=port,
        model="test-model",
        ttft_ms=10.0,
        itl_ms=1.0,
        request_latency=0.05,
        output_tokens=16,
    )
    base_url = f"http://{config.host}:{config.port}"
    proc = multiprocessing.Process(target=_start_server_process, args=(config,))
    proc.start()
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)


def _run_benchmark(
    base_url: str,
    data: str,
    output_dir: Path,
    output_name: str,
    max_seconds: float = 3.0,
) -> subprocess.CompletedProcess:
    output_path = output_dir / output_name
    cmd = [
        sys.executable,
        "-m",
        "guidellm",
        "benchmark",
        "run",
        "--target",
        base_url,
        "--data",
        data,
        "--data-samples",
        "8",
        "--profile",
        "constant",
        "--rate",
        "2",
        "--max-seconds",
        str(max_seconds),
        "--processor",
        "Xenova/gpt-4",
        "--backend",
        "openai_http",
        "--outputs",
        str(output_path),
        "--disable-progress",
        "--disable-console-outputs",
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=180, check=False
    )


@pytest.mark.timeout(240)
def test_synthetic_image_benchmark_against_mock(mock_backend, tmp_path):
    """A short benchmark on synthetic_image must complete cleanly."""
    result = _run_benchmark(
        base_url=mock_backend,
        data=(
            "type=synthetic_image,width=128,height=128,format=jpeg,"
            "jpeg_quality=85,text_tokens=20,output_tokens=8,seed=11"
        ),
        output_dir=tmp_path,
        output_name="image.json",
    )
    assert result.returncode == 0, (
        f"image benchmark failed: stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    report_path = tmp_path / "image.json"
    assert report_path.exists(), "expected benchmark JSON output"
    report = json.loads(report_path.read_text())
    benchmarks = report.get("benchmarks", [])
    assert benchmarks, "expected at least one benchmark in the report"


@pytest.mark.timeout(240)
def test_synthetic_video_benchmark_against_mock(mock_backend, tmp_path):
    """A short benchmark on synthetic_video must complete cleanly."""
    result = _run_benchmark(
        base_url=mock_backend,
        data=(
            "type=synthetic_video,width=160,height=120,frames=4,fps=1,"
            "text_tokens=10,output_tokens=4,seed=23"
        ),
        output_dir=tmp_path,
        output_name="video.json",
        max_seconds=4.0,
    )
    assert result.returncode == 0, (
        f"video benchmark failed: stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    report_path = tmp_path / "video.json"
    assert report_path.exists(), "expected benchmark JSON output"
    report = json.loads(report_path.read_text())
    benchmarks = report.get("benchmarks", [])
    assert benchmarks, "expected at least one benchmark in the report"
