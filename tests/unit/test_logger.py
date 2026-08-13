from pathlib import Path

import pytest

from guidellm import configure_logger, logger
from guidellm.settings import LoggingSettings


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

    captured = capsys.readouterr()
    assert captured.out.count("Warning message") == 1
    assert captured.out.count("Error message") == 1
    assert "Debug message" not in captured.out


def test_configure_logger_console_settings(capsys):
    # Test configuring the logger to change console log level
    config = LoggingSettings(console_log_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

    captured = capsys.readouterr()
    assert captured.out.count("Info message") == 1
    assert captured.out.count("Debug message") == 1


def test_configure_logger_file_settings(tmp_path):
    # Test configuring the logger to log to a file
    log_file = tmp_path / "test.log"
    config = LoggingSettings(log_file=str(log_file), log_file_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

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

    captured = capsys.readouterr()
    assert "Info message" not in captured.out
    assert captured.out.count("Error message") == 1

    with Path(log_file).open() as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Error message"') == 1


def test_environment_variable_override(monkeypatch, capsys, tmp_path):
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

    captured = capsys.readouterr()
    assert "Info message" not in captured.out
    assert captured.out.count("Error message") == 1
    assert "Debug message" not in captured.out

    with Path(tmp_path / "env_test.log").open() as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Error message"') == 1
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Debug message"') == 1


def test_logging_disabled(capsys):
    configure_logger(config=LoggingSettings(disabled=True))
    logger.info("Info message")
    logger.error("Error message")

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
