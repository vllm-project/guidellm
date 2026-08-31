"""Multiprocessing and Rich Live integration tests for inherited loguru handlers."""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import sys
import time

import pytest
from loguru import logger as loguru_logger
from rich.console import Console
from rich.live import Live

from guidellm import configure_logger, logger, reinstall_inherited_logger
from guidellm.logger import CONSOLE_FORMAT, _stderr_sink
from guidellm.scheduler.worker import run_worker_process
from guidellm.settings import LoggingSettings
from guidellm.utils.pipe_stdout import PipeReaderThread


@pytest.fixture(autouse=True)
def reset_logger():  # noqa: PT004
    logger.remove()
    yield
    logger.remove()


def _configure_spawn_logger() -> mp.context.BaseContext:
    ctx = mp.get_context("spawn")
    configure_logger(config=LoggingSettings(console_log_level="INFO"))
    return ctx


def _inherited_log_child(parent_logger, message: str) -> None:
    reinstall_inherited_logger(parent_logger)
    logger.info(message)
    logger.complete()


def _pipe_raw_stderr_worker(stderr_conn) -> None:
    os.dup2(stderr_conn.fileno(), 2)
    stderr_conn.close()
    sys.stderr.write("pipe-raw-stderr-msg\n")
    sys.stderr.flush()


@pytest.mark.regression
@pytest.mark.parametrize("ctx_name", ["spawn", "forkserver"])
def test_inherited_worker_logging_custom_format(ctx_name, capsys):
    """
    Worker logs use inherited parent handlers with GuideLLM console format.

    ## WRITTEN BY AI ##
    """
    ctx = mp.get_context(ctx_name)
    configure_logger(config=LoggingSettings(console_log_level="INFO"))
    message = f"worker-log-{ctx_name}"
    p = ctx.Process(target=_inherited_log_child, args=(logger, message))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0

    logger.complete()
    captured = capsys.readouterr()
    assert message in captured.err
    assert "|INFO     |" in captured.err
    assert " | INFO     | " not in captured.err


@pytest.mark.sanity
def test_rich_live_inherited_worker_log():
    """
    Inherited worker logs render through Rich FileProxy in the parent process.

    ## WRITTEN BY AI ##
    """
    ctx = _configure_spawn_logger()
    console = Console(file=io.StringIO())
    message = "rich-inherited-worker-msg"
    p = ctx.Process(target=_inherited_log_child, args=(logger, message))
    with Live("", console=console, redirect_stderr=True):
        p.start()
        p.join(timeout=10)
        time.sleep(0.2)
    assert p.exitcode == 0
    out = console.file.getvalue()
    assert message in out
    assert "INFO" in out


@pytest.mark.sanity
def test_rich_live_pipe_raw_stderr():
    """
    Raw worker stderr through PipeReaderThread reaches Rich FileProxy.

    ## WRITTEN BY AI ##
    """
    ctx = mp.get_context("spawn")
    stdout_r, stdout_w = ctx.Pipe(duplex=False)
    stderr_r, stderr_w = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_pipe_raw_stderr_worker, args=(stderr_w,))
    console = Console(file=io.StringIO())
    reader = PipeReaderThread(stdout_r, stderr_r)
    reader.start()
    try:

        def run_worker() -> None:
            p.start()
            stdout_w.close()
            stderr_w.close()
            p.join(timeout=10)

        with Live("", console=console, redirect_stderr=True):
            run_worker()
            time.sleep(0.2)
        assert p.exitcode == 0
        assert "pipe-raw-stderr-msg" in console.file.getvalue()
    finally:
        reader.stop()


@pytest.mark.regression
def test_stderr_sink_works_under_rich_live():
    """
    Callable sink resolving sys.stderr at write time works with Rich FileProxy.

    ## WRITTEN BY AI ##
    """
    loguru_logger.remove()
    loguru_logger.add(
        _stderr_sink,
        format=CONSOLE_FORMAT,
        colorize=False,
        enqueue=True,
    )
    console = Console(file=io.StringIO())
    with Live("", console=console, redirect_stderr=True):
        loguru_logger.info("stderr-sink-live-msg")
        loguru_logger.complete()
        time.sleep(0.1)
    assert "stderr-sink-live-msg" in console.file.getvalue()


@pytest.mark.regression
def test_bound_stderr_write_fails_under_rich_live():
    """
    Bound sys.stderr.write bypasses Rich FileProxy and must not be used.

    ## WRITTEN BY AI ##
    """
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr.write,
        format="{message}\n",
        enqueue=True,
    )
    console = Console(file=io.StringIO())
    with Live("", console=console, redirect_stderr=True):
        loguru_logger.info("bound-write-live-msg")
        loguru_logger.complete()
        time.sleep(0.1)
    assert "bound-write-live-msg" not in console.file.getvalue()


@pytest.mark.sanity
def test_pipe_raw_stderr_without_live(capsys):
    """
    Raw worker stderr still reaches the parent when Live is not active.

    ## WRITTEN BY AI ##
    """
    ctx = mp.get_context("spawn")
    stdout_r, stdout_w = ctx.Pipe(duplex=False)
    stderr_r, stderr_w = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_pipe_raw_stderr_worker, args=(stderr_w,))
    reader = PipeReaderThread(stdout_r, stderr_r)
    reader.start()
    try:
        p.start()
        stdout_w.close()
        stderr_w.close()
        p.join(timeout=10)
        time.sleep(0.2)
        assert p.exitcode == 0
        captured = capsys.readouterr()
        assert "pipe-raw-stderr-msg" in captured.err
    finally:
        reader.stop()


class _LoggingWorkerStub:
    """Minimal worker stub for run_worker_process entrypoint tests."""

    def __init__(self, message: str) -> None:
        self._message = message
        self.ran = False

    def run(self) -> None:
        self.ran = True
        logger.info(self._message)
        logger.complete()


@pytest.mark.sanity
def test_run_worker_process_entrypoint(capsys):
    """
    run_worker_process reinstalls inherited logger before executing the worker.

    ## WRITTEN BY AI ##
    """
    ctx = mp.get_context("spawn")
    configure_logger(config=LoggingSettings(console_log_level="INFO"))
    message = "run-worker-process-msg"
    worker = _LoggingWorkerStub(message)
    p = ctx.Process(target=run_worker_process, args=(logger, worker))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0
    logger.complete()
    captured = capsys.readouterr()
    assert message in captured.err
