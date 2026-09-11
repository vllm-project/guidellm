"""
Logger configuration for GuideLLM.

This module provides logging configuration using the loguru library.
Console and file handlers are configured via :func:`configure_logger`, which
is the primary runtime API. Environment variables populate fallback defaults
in :class:`~guidellm.settings.LoggingSettings` at import time only.

Environment Variables (import-time fallback defaults):
    - GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL: Console log level
        (default: INFO; set empty to disable).
    - GUIDELLM__LOGGING__CONSOLE_COLORIZE: Console ANSI colorization
        (default: auto; options: auto, true, false).
    - GUIDELLM__LOGGING__LOG_FILE: Path to the log file for file logging.
    - GUIDELLM__LOGGING__LOG_FILE_LEVEL: Log level for file logging.

Usage:
    from guidellm import logger, configure_logger
    from guidellm.settings import LoggingSettings

    configure_logger(
        config=LoggingSettings(
            console_log_level="DEBUG",
            console_colorize="auto",
        )
    )

    logger.info("This is an info message")
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from guidellm.settings import LoggingSettings, settings

if TYPE_CHECKING:
    from loguru import Logger

__all__ = [
    "configure_logger",
    "logger",
    "reinstall_inherited_logger",
]

CONSOLE_FORMAT = (
    "<green>{time:YY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level> "
    "|<cyan>{name}:{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def _stderr_sink(message: str) -> None:
    sys.stderr.write(message)


def reinstall_inherited_logger(parent_logger: Logger) -> None:
    """
    Replace the child-process module logger core with the parent's inherited core.

    Loguru's public ``reinstall()`` is not yet available in all releases; this
    helper mirrors that behavior and delegates when present.

    # TODO: Remove this shim when loguru > 0.7.3 is the minimum supported version.

    :param parent_logger: Logger instance passed from the parent process.
    """
    if hasattr(parent_logger, "reinstall"):
        parent_logger.reinstall()
        return

    logger._core = parent_logger._core  # type: ignore[attr-defined]  # noqa: SLF001


class ConsoleLogHandler:
    """Owns the console sink; removes and replaces on each configure call."""

    # loguru's preinstalled stderr handler is always id 0; claim it on first configure.
    _sink_id: int | None = 0

    def configure(
        self,
        level: str | None,
        colorize: Literal["auto"] | bool = "auto",
        *,
        mp_context: BaseContext,
    ) -> int | None:
        if self._sink_id is not None:
            with contextlib.suppress(ValueError):
                logger.remove(self._sink_id)
            self._sink_id = None
        if level is None:
            return None

        should_colorize = sys.stderr.isatty() if colorize == "auto" else colorize

        self._sink_id = logger.add(
            _stderr_sink,
            level=level.upper(),
            format=CONSOLE_FORMAT,
            colorize=should_colorize,
            enqueue=True,
            context=mp_context,
        )
        return self._sink_id


class FileLogHandler:
    """Owns the file sink; removes and replaces on each configure call."""

    _sink_id: int | None = None

    def configure(
        self,
        level: str | None,
        path: Path,
        *,
        mp_context: BaseContext,
    ) -> int | None:
        if self._sink_id is not None:
            with contextlib.suppress(ValueError):
                logger.remove(self._sink_id)
            self._sink_id = None
        if level is None:
            return None
        self._sink_id = logger.add(
            path,
            level=level.upper(),
            serialize=True,
            enqueue=True,
            context=mp_context,
        )
        return self._sink_id


_console_handler = ConsoleLogHandler()
_file_handler = FileLogHandler()


def configure_logger(config: LoggingSettings | None = None) -> None:
    """
    Configure console and file logging handlers.

    Idempotent: calling twice replaces existing sinks rather than stacking them.
    When ``config`` is ``None``, uses :data:`~guidellm.settings.settings.logging`
    as fallback defaults (import-time bootstrap only).

    :param config: Explicit logging configuration.
    """
    if config is None:
        config = settings.logging

    mp_context = mp.get_context(settings.mp_context_type)

    _console_handler.configure(
        level=config.console_log_level,
        colorize=config.console_colorize,
        mp_context=mp_context,
    )

    _file_handler.configure(
        level=config.log_file_level,
        path=config.log_file,
        mp_context=mp_context,
    )


# Logger should be configured in the main process only
if mp.parent_process() is None:
    configure_logger()
