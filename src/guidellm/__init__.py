"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

import asyncio
import warnings

# Configure uvloop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    warnings.warn(
        "uvloop is not installed. For improved performance, "
        "consider installing the guidellm[perf] extras group.",
        category=UserWarning,
        stacklevel=2,
    )

from .logger import configure_logger, logger
from .settings import (
    reload_settings,
    settings,
)

__all__ = [
    "configure_logger",
    "logger",
    "reload_settings",
    "settings",
]
