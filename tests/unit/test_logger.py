import importlib
import io
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger
from rich.console import Console
from rich.live import Live

from guidellm import configure_logger, logger, reinstall_inherited_logger
from guidellm.settings import LoggingSettings

_logger_module = importlib.import_module("guidellm.logger")


def _child_reinstall_for_test(parent_logger, q) -> None:
    reinstall_inherited_logger(parent_logger)
    q.put(list(logger._core.handlers.keys()))
    logger.info("reinstalled child message")
    logger.complete()


def _inherited_log_child(parent_logger, message: str) -> None:
    reinstall_inherited_logger(parent_logger)
    logger.info(message)
    logger.complete()


@pytest.fixture(autouse=True)
def reset_logger():  # noqa: PT004
    logger.remove()
    _logger_module._console_handler._sink_id = 0
    _logger_module._file_handler._sink_id = None
    yield
    logger.complete()
    logger.remove()
    _logger_module._console_handler._sink_id = 0
    _logger_module._file_handler._sink_id = None


def test_default_logger_settings(capsys):
    configure_logger(config=LoggingSettings())

    # Default settings should log to console with INFO level and no file logging
    logger.info("Info message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    logger.complete()
    captured = capsys.readouterr()
    assert captured.err.count("Warning message") == 1
    assert captured.err.count("Error message") == 1
    assert "Debug message" not in captured.err


def test_configure_logger_console_settings(capsys):
    # Test configuring the logger to change console log level
    config = LoggingSettings(console_log_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

    logger.complete()
    captured = capsys.readouterr()
    assert captured.err.count("Info message") == 1
    assert captured.err.count("Debug message") == 1


def test_configure_logger_file_settings(tmp_path):
    # Test configuring the logger to log to a file
    log_file = tmp_path / "test.log"
    config = LoggingSettings(log_file=str(log_file), log_file_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

    logger.complete()
    with Path(log_file).open() as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Debug message"') == 1


def test_configure_logger_console_and_file(capsys, tmp_path):
    # Test configuring the logger to change both console and file settings
    log_file = tmp_path / "test.log"
    config = LoggingSettings(
        console_log_level="ERROR",
        log_file=str(log_file),
        log_file_level="INFO",
    )
    configure_logger(config=config)
    logger.info("Info message")
    logger.error("Error message")

    logger.complete()
    captured = capsys.readouterr()
    assert "Info message" not in captured.err
    assert captured.err.count("Error message") == 1

    with Path(log_file).open() as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Error message"') == 1


def test_environment_variable_override(capsys, tmp_path):
    configure_logger(
        config=LoggingSettings(
            console_log_level="ERROR",
            log_file=str(tmp_path / "env_test.log"),
            log_file_level="DEBUG",
        ),
    )
    logger.info("Info message")
    logger.error("Error message")
    logger.debug("Debug message")

    logger.complete()
    captured = capsys.readouterr()
    assert "Info message" not in captured.err
    assert captured.err.count("Error message") == 1
    assert "Debug message" not in captured.err

    with Path(tmp_path / "env_test.log").open() as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Error message"') == 1
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Debug message"') == 1


def test_console_logging_disabled(capsys):
    configure_logger(config=LoggingSettings(console_log_level=None))
    logger.info("Info message")
    logger.error("Error message")

    logger.complete()
    captured = capsys.readouterr()
    assert not captured.err


def test_configure_logger_idempotent(capsys, tmp_path):
    log_file = tmp_path / "test.log"
    config = LoggingSettings(
        console_log_level="INFO",
        log_file=str(log_file),
        log_file_level="INFO",
    )
    configure_logger(config=config)
    configure_logger(config=config)

    logger.info("once")
    logger.complete()
    captured = capsys.readouterr()
    assert captured.err.count("once") == 1


@pytest.mark.parametrize(
    ("colorize", "isatty", "expect_ansi"),
    [
        ("auto", True, True),
        ("auto", False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_console_colorize(capsys, monkeypatch, colorize, isatty, expect_ansi):
    monkeypatch.setattr("sys.stderr.isatty", lambda: isatty)
    configure_logger(
        config=LoggingSettings(
            console_log_level="INFO",
            console_colorize=colorize,
        ),
    )
    logger.info("colorize test")

    logger.complete()
    captured = capsys.readouterr()
    has_ansi = "\x1b[" in captured.err
    assert has_ansi == expect_ansi


@pytest.mark.regression
def test_env_var_console_log_level_at_import():
    """
    Import-time logging config must not leave loguru's default handler active.

    ## WRITTEN BY AI ##
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GUIDELLM__")}
    env["GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL"] = "ERROR"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from guidellm import logger; "
                "logger.complete(); "
                "logger.info('info message'); "
                "logger.error('error message'); "
                "logger.complete()"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "info message" not in result.stderr
    assert "error message" in result.stderr


@pytest.mark.sanity
def test_configure_logger_passes_mp_context():
    """
    Handlers must use the same multiprocessing context as worker processes.

    ## WRITTEN BY AI ##
    """
    with patch("guidellm.logger.logger.add", wraps=logger.add) as mock_add:
        configure_logger(config=LoggingSettings(console_log_level="INFO"))
        assert mock_add.call_count >= 1
        for call in mock_add.call_args_list:
            assert call.kwargs.get("context") == mp.get_context("spawn")
            assert call.kwargs.get("enqueue") is True


@pytest.mark.sanity
def test_reinstall_inherited_logger(capsys):
    """
    Child process adopts the parent's configured handlers after reinstall.

    ## WRITTEN BY AI ##
    """
    configure_logger(config=LoggingSettings(console_log_level="INFO"))
    parent_handler_ids = set(logger._core.handlers.keys())

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_child_reinstall_for_test, args=(logger, q))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0
    assert set(q.get(timeout=5)) == parent_handler_ids

    logger.complete()
    captured = capsys.readouterr()
    assert "reinstalled child message" in captured.err


class TestInheritedLoggerMultiprocessing:
    """Multiprocessing tests for inherited loguru handler configuration.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _configure_test_logger() -> None:
        configure_logger(
            config=LoggingSettings(
                console_log_level="INFO",
                console_colorize=False,
            )
        )

    @pytest.mark.regression
    @pytest.mark.parametrize("ctx_name", ["spawn", "forkserver"])
    def test_inherited_worker_logging_custom_format(self, ctx_name, capsys):
        """
        Worker logs use inherited parent handlers with GuideLLM console format.

        ## WRITTEN BY AI ##
        """
        ctx = mp.get_context(ctx_name)
        self._configure_test_logger()
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


class TestLoggerRichLive:
    """Rich Live integration tests for console logging sinks.

    ## WRITTEN BY AI ##
    """

    @staticmethod
    def _configure_test_logger() -> None:
        configure_logger(
            config=LoggingSettings(
                console_log_level="INFO",
                console_colorize=False,
            )
        )

    @pytest.mark.sanity
    def test_logger_with_rich_live_redirect_stderr(self):
        """
        Console logs resolve sys.stderr at write time and work with Rich Live.

        ## WRITTEN BY AI ##
        """
        self._configure_test_logger()
        console = Console(file=io.StringIO())

        with Live("", console=console, redirect_stderr=True):
            logger.info("Live test message")
            logger.complete()

        assert "Live test message" in console.file.getvalue()

    @pytest.mark.sanity
    def test_rich_live_inherited_worker_log(self):
        """
        Inherited worker logs render through Rich FileProxy in the parent process.

        ## WRITTEN BY AI ##
        """
        ctx = mp.get_context("spawn")
        self._configure_test_logger()
        console = Console(file=io.StringIO())
        message = "rich-inherited-worker-msg"
        p = ctx.Process(target=_inherited_log_child, args=(logger, message))
        with Live("", console=console, redirect_stderr=True):
            p.start()
            p.join(timeout=10)
            logger.complete()
            time.sleep(0.2)
        assert p.exitcode == 0
        out = console.file.getvalue()
        assert message in out
        assert "INFO" in out

    @pytest.mark.regression
    def test_stderr_sink_works_under_rich_live(self):
        """
        Callable sink resolving sys.stderr at write time works with Rich FileProxy.

        ## WRITTEN BY AI ##
        """
        self._configure_test_logger()
        console = Console(file=io.StringIO())
        with Live("", console=console, redirect_stderr=True):
            logger.info("stderr-sink-live-msg")
            logger.complete()
            time.sleep(0.1)
        assert "stderr-sink-live-msg" in console.file.getvalue()

    @pytest.mark.regression
    def test_bound_stderr_write_fails_under_rich_live(self):
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
