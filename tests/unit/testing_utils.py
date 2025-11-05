"""Common test utilities for async testing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

import pytest

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
