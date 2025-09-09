"""
Logger configuration for GuideLLM.

This module provides a flexible logging configuration using the loguru library.
It supports console and file logging with options to configure via environment
variables or direct function calls.

Environment Variables:
    - GUIDELLM__LOGGING__DISABLED: Disable logging (default: false).
    - GUIDELLM__LOGGING__CLEAR_LOGGERS: Clear existing loggers
        from loguru (default: true).
    - GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL: Log level for console logging
        (default: none, options: DEBUG, INFO, WARNING, ERROR, CRITICAL).
    - GUIDELLM__LOGGING__LOG_FILE: Path to the log file for file logging
        (default: guidellm.log if log file level set else none)
    - GUIDELLM__LOGGING__LOG_FILE_LEVEL: Log level for file logging
        (default: INFO if log file set else none).

If logging isn't responding to the environment variables, run the `guidellm config`
command to validate that the environment variables match and are being set correctly.

Usage:
    from guidellm import logger, configure_logger, LoggerConfig

    # Configure metrics with default settings
    configure_logger(
        config=LoggingConfig
            disabled=False,
            clear_loggers=True,
            console_log_level="DEBUG",
            log_file=None,
            log_file_level=None,
        )
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
"""

import sys

from loguru import logger

from guidellm.config import LoggingSettings, settings

__all__ = ["configure_logger", "logger"]


def configure_logger(config: LoggingSettings = settings.logging):
    """
    Configure the metrics for LLM Compressor.
    This function sets up the console and file logging
    as per the specified or default parameters.

    Note: Environment variables take precedence over the function parameters.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """

    if config.disabled:
        logger.disable("guidellm")
        return

    logger.enable("guidellm")

    if config.clear_loggers:
        logger.remove()

    # log as a human readable string with the time, function, level, and message
    logger.add(
        sys.stdout,
        level=config.console_log_level.upper(),
        format="<green>{time:YY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level> \
        |<cyan>{name}:{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    if config.log_file or config.log_file_level:
        log_file = config.log_file or "guidellm.log"
        log_file_level = config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(log_file, level=log_file_level.upper(), serialize=True)


# invoke logger setup on import with default values
# enabling console logging with INFO and disabling file logging
configure_logger()
