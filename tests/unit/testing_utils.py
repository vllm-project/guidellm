"""Common test utilities for async testing and logger isolation."""

from __future__ import annotations

import asyncio
import importlib
import io
import re
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

import pytest
from rich.console import Console

from guidellm import logger

_logger_module = importlib.import_module("guidellm.logger")

# Type variables for proper typing
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def async_timeout(delay: float = 10.0, hard_fail: bool = False) -> Callable[[F], F]:
    """
    Decorator to add timeout to async test functions.

    Args:
        delay: Timeout in seconds (default: 10.0)

    Returns:
        Decorated function with timeout applied
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)
            except asyncio.TimeoutError:
                msg = f"Test {func.__name__} timed out after {delay} seconds"

                if not hard_fail:
                    pytest.xfail(msg)

                pytest.fail(msg)

        return wrapper  # type: ignore[return-value]

    return decorator


def drain_logger() -> None:
    """Drain enqueued loguru records from production sinks."""
    logger.complete()


def teardown_logger_state() -> None:
    """Remove all loguru handlers and reset GuideLLM handler tracking."""
    drain_logger()
    logger.remove()
    _logger_module._console_handler._sink_id = None
    _logger_module._file_handler._sink_id = None


def rich_console() -> Console:
    """Build a wide Rich console for Live redirect tests."""
    return Console(file=io.StringIO(), width=200, force_terminal=True)


def rich_console_output(console: Console) -> str:
    """Normalize Rich console output for stable assertions."""
    text = console.file.getvalue()
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    return " ".join(text.split())
