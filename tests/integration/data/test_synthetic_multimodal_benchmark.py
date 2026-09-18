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
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from guidellm.mock_server.server import MockServer
from guidellm.schemas.mock_server.config import MockServerConfig
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR

pytestmark = [pytest.mark.smoke]


def _start_server_process(config: MockServerConfig) -> None:
    server = MockServer(config)
    # Disable Sanic access logs / MOTD so ANSI formatters do not clobber pytest's TTY.
    server.run(access_log=False)


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
    data: list[str],
    output_dir: Path,
    output_name: str,
    max_seconds: float = 3.0,
) -> subprocess.CompletedProcess:
    output_path = output_dir / output_name
    cmd = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
        "--backend",
        f"kind=openai_http,target={base_url}",
        "--data-loader",
        "kind=pytorch,samples=8",
        "--profile",
        "kind=constant,rate=2",
        "--constraint",
        f"kind=max_duration,seconds={max_seconds}",
        "--tokenizer",
        f"kind=huggingface_auto,model={MINIMAL_TOKENIZER_DIR}",
        "--output",
        f"kind=json,path={output_path}",
        "--disable-console",
    ]
    for item in data:
        cmd.extend(["--data", item])
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    return subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=180, check=False, env=env
    )


def _successful_requests(report: dict) -> list[dict]:
    benchmarks = report.get("benchmarks", [])
    assert benchmarks, "expected at least one benchmark in the report"

    requests = benchmarks[0].get("requests", {}).get("successful", [])
    assert requests, "expected at least one successful request"

    return requests


def _assert_image_metrics(report: dict) -> None:

    requests = _successful_requests(report)
    static_input_val = 128 * 128

    for request in requests:
        input_metrics = request["input_metrics"]
        assert input_metrics["image_pixels"] == static_input_val
        assert input_metrics["image_bytes"] > 0

    image_metrics = report["benchmarks"][0]["metrics"]["image"]
    image_stats = image_metrics["pixels"]["input"]["successful"]

    assert image_stats["mean"] > 0
    assert image_stats["median"] > 0
    assert image_stats["total_sum"] == pytest.approx(
        image_stats["mean"] * image_stats["count"]
    )


def _assert_video_metrics(report: dict) -> None:

    requests = _successful_requests(report)

    for request in requests:
        input_metrics = request["input_metrics"]
        assert input_metrics["video_frames"] == 4
        assert input_metrics["video_seconds"] == pytest.approx(4.0)
        assert input_metrics["video_bytes"] > 0

    video_metrics = report["benchmarks"][0]["metrics"]["video"]
    video_stats = video_metrics["frames"]["input"]["successful"]
    assert video_stats["mean"] > 0
    assert video_stats["median"] > 0
    assert video_stats["total_sum"] == pytest.approx(
        video_stats["mean"] * video_stats["count"]
    )


@pytest.mark.timeout(240)
def test_synthetic_image_benchmark_against_mock(mock_backend, tmp_path):
    """A short benchmark on synthetic_image must complete cleanly.

    ## WRITTEN BY AI ##
    """
    result = _run_benchmark(
        base_url=mock_backend,
        data=[
            "kind=synthetic_text,prompt_tokens=20",
            "kind=synthetic_image,width=128,height=128,format=jpeg,"
            "jpeg_quality=85,output_tokens=8,seed=11",
        ],
        output_dir=tmp_path,
        output_name="image.json",
    )
    assert result.returncode == 0, (
        f"image benchmark failed: stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    report_path = tmp_path / "image.json"
    assert report_path.exists(), "expected benchmark JSON output"
    report = json.loads(report_path.read_text())
    _assert_image_metrics(report)


@pytest.mark.timeout(240)
def test_synthetic_video_benchmark_against_mock(mock_backend, tmp_path):
    """A short benchmark on synthetic_video must complete cleanly.

    ## WRITTEN BY AI ##
    """
    result = _run_benchmark(
        base_url=mock_backend,
        data=[
            "kind=synthetic_text,prompt_tokens=10",
            "kind=synthetic_video,width=160,height=120,frames=4,fps=1,"
            "output_tokens=4,seed=23",
        ],
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
    _assert_video_metrics(report)
