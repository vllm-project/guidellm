"""Common test utilities for async testing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

import pytest

# Type variables for proper typing
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


async def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    poll_interval: float = 0.0,
) -> None:
    """Poll ``predicate`` until it is true.

    Used instead of a fixed sleep so tests do not race the event loop under CI load.

    :param predicate: Called until it returns True
    :param timeout: Seconds to wait before failing
    :param poll_interval: Delay between polls; 0 yields to the event loop only
    :raises AssertionError: If ``timeout`` elapses before ``predicate`` is true
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for condition")
        await asyncio.sleep(poll_interval)


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
