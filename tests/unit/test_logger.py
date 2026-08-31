import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from guidellm import configure_logger, logger, reinstall_inherited_logger
from guidellm.settings import LoggingSettings


def _child_reinstall_for_test(parent_logger, q) -> None:
    reinstall_inherited_logger(parent_logger)
    q.put(list(logger._core.handlers.keys()))
    logger.info("reinstalled child message")
    logger.complete()


@pytest.fixture(autouse=True)
def reset_logger():  # noqa: PT004
    # Ensure logger is reset before each test
    logger.remove()
    yield
    logger.remove()

    return logger


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
