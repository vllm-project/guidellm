"""Shared utilities for vLLM Python backends."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

from guidellm.logger import logger

__all__ = [
    "is_scheduler_worker_process",
    "prepare_vllm_benchmark_logging",
    "reset_cpu_affinity",
    "vllm_benchmark_engine_config",
]

_DEFAULT_VLLM_BENCHMARK_LOG_LEVEL = "ERROR"
_VLLM_ROOT_LOGGER_RECONFIGURE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
)


def _set_logger_and_handler_levels(
    target_logger: logging.Logger,
    level: str,
) -> None:
    target_logger.setLevel(level)
    for handler in target_logger.handlers:
        handler.setLevel(level)


def _try_reconfigure_vllm_root_logger() -> None:
    """Re-apply vLLM's own root-logger configuration if the module is loaded.

    Calls ``vllm.logger._configure_vllm_root_logger()`` so that vLLM's
    internal Rich handler (which controls worker-process output) picks up
    the updated ``VLLM_LOGGING_LEVEL`` env var.  This is a **private API**
    that may change or be removed in future vLLM versions without notice;
    failures are swallowed so callers are never broken by vLLM API drift.
    """
    vllm_logger_module = sys.modules.get("vllm.logger")
    if vllm_logger_module is None:
        return
    if not hasattr(vllm_logger_module, "_configure_vllm_root_logger"):
        return
    try:
        vllm_logger_module._configure_vllm_root_logger()  # noqa: SLF001
    except _VLLM_ROOT_LOGGER_RECONFIGURE_ERRORS as exc:
        logger.debug(
            "Could not re-apply vLLM root logger config "
            "(vLLM API may have changed): {}",
            exc,
        )


def is_scheduler_worker_process() -> bool:
    """Return True when running inside a GuideLLM scheduler worker process."""
    return mp.parent_process() is not None


def prepare_vllm_benchmark_logging(
    level: str = _DEFAULT_VLLM_BENCHMARK_LOG_LEVEL,
) -> None:
    """Reduce vLLM log noise during in-process benchmarks.

    Three layers of silencing are applied in order:

    1. **Environment variables** (``VLLM_LOGGING_LEVEL``, ``TQDM_DISABLE``,
       etc.) — read by vLLM and its dependencies at import time and
       inherited by child processes (EngineCore, scheduler workers).
       ``setdefault`` preserves any explicit user override.

    2. **Live logger + handler levels** — needed when vLLM was already
       imported before this function ran (e.g. a worker subprocess forked
       after ``import vllm``, or a third-party library imported it first).
       In that case the env-var path has no effect because vLLM's logging
       is already initialised.  We lower the ``vllm`` root logger, all of
       its already-registered ``vllm.*`` child loggers, and each logger's
       attached handlers.  This also prevents Rich progress bars from
       corrupting the terminal during vLLM worker engine initialisation.

    3. **vLLM private reconfigure** (``_configure_vllm_root_logger``) —
       re-applies vLLM's own logging setup so its internal Rich handler
       picks up the updated ``VLLM_LOGGING_LEVEL``.  Isolated in
       ``_try_reconfigure_vllm_root_logger()`` with defensive error
       handling because this is a private API subject to vLLM API drift.

    :param level: Logging level string (e.g. ``"ERROR"``). Defaults to
        ``"ERROR"``.
    """
    level_upper = level.upper()
    os.environ.setdefault("VLLM_LOGGING_LEVEL", level_upper)
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    _set_logger_and_handler_levels(logging.getLogger("vllm"), level_upper)
    for name, candidate in logging.root.manager.loggerDict.items():
        if (name == "vllm" or name.startswith("vllm.")) and isinstance(
            candidate, logging.Logger
        ):
            _set_logger_and_handler_levels(candidate, level_upper)

    _try_reconfigure_vllm_root_logger()


def vllm_benchmark_engine_config(
    vllm_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``vllm_config`` with benchmark-friendly defaults."""
    config = dict(vllm_config)
    config.setdefault("disable_log_stats", True)
    return config


def reset_cpu_affinity() -> None:
    """Restore the full CPU set allowed by the OS/cgroup.

    When the worker process is forked from a parent that has
    already initialised an OpenMP runtime (e.g. via PyTorch),
    the child inherits a restricted CPU-affinity mask.  This
    causes vLLM's auto-bind logic to see far fewer cores than
    are actually available, destroying throughput.

    We read the effective cpuset from the cgroup filesystem
    and reset the affinity to the full set.
    """
    if sys.platform != "linux":
        return

    current = os.sched_getaffinity(0)

    # Try cgroup v2 first, then fall back to cgroup v1.
    # The `return` at the end of the loop body (outside the `if`) means
    # "stop after the first path that is readable" — OSError on a path
    # causes `continue` to the next, but a successful read always exits.
    for path_str in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ):
        try:
            raw = Path(path_str).read_text().strip()
        except OSError:
            continue

        cpus: set[int] = set()
        for part in raw.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                cpus.update(range(int(lo), int(hi) + 1))
            else:
                cpus.add(int(part))

        if cpus and current != cpus:
            os.sched_setaffinity(0, cpus)
            logger.debug(
                "Reset CPU affinity from {} to {} cores",
                len(current),
                len(cpus),
            )
        return
