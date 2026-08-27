"""Shared fixtures and pytest options for GuideLLM E2E tests."""

from __future__ import annotations

import multiprocessing
import socket
import time
from collections.abc import Iterator
from typing import Any, Protocol

import httpx
import pytest

from guidellm.mock_server.server import MockServer
from guidellm.schemas.mock_server import MockServerConfig
from tests.e2e.vllm_sim_server import VllmSimServer

MOCK_SERVER_HOST = "127.0.0.1"


class E2EServer(Protocol):
    """Minimal server interface used by E2E benchmarks."""

    def get_url(self) -> str: ...

    def stop(self) -> None: ...


class _MockServerHandle:
    """Handle for a MockServer subprocess started for E2E tests."""

    def __init__(self, base_url: str, process: multiprocessing.Process) -> None:
        self._base_url = base_url
        self._process = process

    def get_url(self) -> str:
        return self._base_url

    def stop(self) -> None:
        if not self._process.is_alive():
            self._process.join(timeout=5)
            return
        self._process.terminate()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)


def free_port() -> int:
    """Bind to port 0 and return the OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((MOCK_SERVER_HOST, 0))
        return int(sock.getsockname()[1])


def _start_mock_server_process(config: MockServerConfig) -> None:
    """Start the MockServer in a subprocess."""
    # Disable Sanic access logs / MOTD so ANSI formatters do not clobber pytest's TTY.
    MockServer(config).run(access_log=False)


def start_mock_server(
    *,
    ttft_ms: float = 5.0,
    itl_ms: float = 1.0,
    output_tokens: int = 32,
    request_latency: float = 0.05,
    model: str = "e2e-mock-model",
    fail_after_requests: int | None = None,
    max_concurrent_requests: int | None = None,
    **config_overrides: Any,
) -> _MockServerHandle:
    """
    Start a MockServer on an ephemeral port and wait until healthy.

    Uses an ephemeral port so a leftover listener on a fixed port cannot mask
    a failed bind.

    :return: Handle with ``get_url()`` / ``stop()`` for fixtures and tests
    """
    config = MockServerConfig(
        host=MOCK_SERVER_HOST,
        port=free_port(),
        model=model,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        output_tokens=output_tokens,
        request_latency=request_latency,
        fail_after_requests=fail_after_requests,
        max_concurrent_requests=max_concurrent_requests,
        **config_overrides,
    )
    base_url = f"http://{config.host}:{config.port}"
    process = multiprocessing.Process(
        target=_start_mock_server_process, args=(config,), daemon=True
    )
    process.start()

    # Poll until *this* process is serving /health (not a leftover listener)
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if not process.is_alive():
            process.join(timeout=5)
            pytest.fail(f"MockServer exited before ready (exitcode={process.exitcode})")
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return _MockServerHandle(base_url, process)
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.2)

    process.terminate()
    process.join(timeout=5)
    pytest.fail("MockServer failed to start within 30 seconds")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--e2e-server`` option (mock default, llm-d optional)."""
    parser.addoption(
        "--e2e-server",
        action="store",
        default="mock",
        choices=("mock", "llm-d"),
        help="Inference server for E2E benchmarks: mock (default) or llm-d",
    )


@pytest.fixture(scope="module")
def e2e_server_kind(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--e2e-server"))


@pytest.fixture(scope="module")
def server(e2e_server_kind: str) -> Iterator[E2EServer]:
    """
    Default fast E2E server (MockServer, or llm-d when ``--e2e-server=llm-d``).

    Suitable for successful / tool-call style benchmarks. Specialized modules
    define their own fixtures when they need fail_after or concurrency limits.
    """
    if e2e_server_kind == "llm-d":
        sim = VllmSimServer(
            port=free_port(),
            model="databricks/dolly-v2-12b",
            mode="random",
            time_to_first_token=1,  # 1ms TTFT
            inter_token_latency=1,  # 1ms ITL
        )
        try:
            sim.start()
            yield sim  # Yield the URL for tests to use
        finally:
            sim.stop()  # Teardown: Stop the server after tests are done
        return

    handle = start_mock_server()
    try:
        yield handle  # Yield the URL for tests to use
    finally:
        handle.stop()  # Teardown: Stop the server after tests are done
