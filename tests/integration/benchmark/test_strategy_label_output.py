"""Integration test: concurrent sub-benchmark labels in the rendered console
output.

Spins up the in-tree mock server (Sanic) in a subprocess, runs a short concurrent
`guidellm run` against it with ``profile.streams`` overridden to two values, and
asserts the final summary tables label each sub-benchmark with its parameterized
strategy (``concurrent@1`` / ``concurrent@2``) rather than a bare ``concurrent``.

Unlike most benchmark integration tests, this one inspects the rendered stdout
instead of the JSON report, because the behavior under test lives in the console
renderer: the Strategy column must use ``str(strategy)`` (``concurrent@<streams>``)
and not the strategy discriminator (``strategy.type_`` == ``concurrent``). The
tables are kept with ``--disable-console-interactive`` (only the transient live
progress display is dropped).

By default the run targets the offline mock server, so the test is deterministic
and needs no GPU or network. To exercise a real vLLM / sglang server instead, set
``GUIDELLM_TEST_TARGET`` (and optionally ``GUIDELLM_TEST_MODEL`` /
``GUIDELLM_TEST_TOKENIZER``); the test then skips if that target is unreachable.
Keep the concurrency low (the default ``GUIDELLM_TEST_STREAMS=1,2``) since some
servers return 429 above a couple of concurrent requests.

Run with output visible to eyeball the tables::

    pytest -s -v tests/integration/benchmark/test_strategy_label_output.py
"""

from __future__ import annotations

import asyncio
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

DEFAULT_STREAMS = "1,2"
DEFAULT_MAX_SECONDS = "5"


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


def _external_target() -> str | None:
    target = os.environ.get("GUIDELLM_TEST_TARGET")
    return target.rstrip("/") if target else None


@pytest.fixture(scope="module")
def benchmark_backend():
    """Yield a backend base URL, preferring a real server set via the environment.

    Falls back to the in-tree mock server so the test is offline by default.

    ## WRITTEN BY AI ##
    """
    external = _external_target()
    if external is not None:
        try:
            reachable = httpx.get(f"{external}/v1/models", timeout=5.0).status_code
        except httpx.HTTPError:
            reachable = None
        if reachable != 200:
            pytest.skip(f"GUIDELLM_TEST_TARGET {external} is not reachable")
        yield external
        return

    config = MockServerConfig(
        host="127.0.0.1",
        port=_free_port(),
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


def _run_concurrent_benchmark(base_url: str, output_dir: Path) -> str:
    """Run a short concurrent benchmark via the CLI and return its stdout+stderr.

    The output is printed so ``pytest -s`` shows the rendered tables as a user
    sees them in the terminal.

    ## WRITTEN BY AI ##
    """
    streams = os.environ.get("GUIDELLM_TEST_STREAMS", DEFAULT_STREAMS)
    max_seconds = os.environ.get("GUIDELLM_TEST_MAX_SECONDS", DEFAULT_MAX_SECONDS)

    env = {**os.environ}
    if _external_target() is not None:
        backend = (
            f"kind=openai_http,target={base_url},"
            f"model={os.environ.get('GUIDELLM_TEST_MODEL', 'Qwen-3.8')}"
        )
        tokenizer = os.environ.get("GUIDELLM_TEST_TOKENIZER", "Qwen/Qwen3-8B")
    else:
        backend = f"kind=openai_http,target={base_url}"
        tokenizer = str(MINIMAL_TOKENIZER_DIR)
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )

    cmd = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
        "--backend",
        backend,
        "--profile",
        "kind=concurrent",
        "--override",
        "profile.streams",
        streams,
        "--data",
        "kind=synthetic_text,prompt_tokens=64,output_tokens=16",
        "--tokenizer",
        f"kind=huggingface_auto,model={tokenizer}",
        "--constraint",
        f"kind=max_duration,seconds={max_seconds}",
        "--output",
        f"kind=json,path={output_dir / 'benchmarks.json'}",
        # Keep the final summary tables; drop only the transient live progress.
        "--disable-console-interactive",
    ]

    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=600, check=False, env=env
    )
    output = result.stdout + result.stderr
    # Surface the rendered tables so `pytest -s` shows them for inspection.
    print(output)  # noqa: T201
    assert result.returncode == 0, f"guidellm run failed:\n{output}"

    return output


@pytest.mark.timeout(240)
def test_concurrent_output_is_human_readable(benchmark_backend, tmp_path):
    """Run the benchmark and display the complete rendered output.

    No behavioral assertions beyond a clean exit; this exists so ``pytest -s``
    dumps the final tables for manual inspection.

    ## WRITTEN BY AI ##
    """
    _run_concurrent_benchmark(benchmark_backend, tmp_path)


@pytest.mark.timeout(240)
def test_concurrent_output_keeps_parameterized_strategy_labels(
    benchmark_backend, tmp_path
):
    """The final tables label each concurrent sub-benchmark with its stream count.

    With the default ``GUIDELLM_TEST_STREAMS=1,2`` the Strategy column must show
    ``concurrent@1`` and ``concurrent@2`` so the sub-benchmark rows can be told
    apart. Rendering the discriminator (``strategy.type_``) instead drops the
    ``@<streams>`` suffix and collapses both rows to a bare ``concurrent``.

    ## WRITTEN BY AI ##
    """
    output = _run_concurrent_benchmark(benchmark_backend, tmp_path)

    streams = os.environ.get("GUIDELLM_TEST_STREAMS", DEFAULT_STREAMS)
    for value in streams.split(","):
        label = f"concurrent@{value.strip()}"
        assert label in output, (
            f"final tables are missing the {label!r} strategy label; "
            f"the Strategy column likely collapsed to a bare 'concurrent'.\n{output}"
        )

    assert "Strategy" in output
