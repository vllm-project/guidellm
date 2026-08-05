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


def is_scheduler_worker_process() -> bool:
    """Return True when running inside a GuideLLM scheduler worker process."""
    return mp.parent_process() is not None


def prepare_vllm_benchmark_logging(
    level: str = _DEFAULT_VLLM_BENCHMARK_LOG_LEVEL,
) -> None:
    """Reduce vLLM log noise during in-process benchmarks.

    vLLM logs through the stdlib ``logging`` module and writes to stderr.
    Scheduler workers and their EngineCore children are separate processes,
    so Rich progress redirection in the main process cannot capture them.
    Lower the log level here so engine startup does not corrupt the live UI.

    ``VLLM_LOGGING_LEVEL`` is set with ``setdefault`` so an explicit user
    setting is preserved. Child processes that import vLLM after this call
    (for example EngineCore) inherit the quieter level.
    """
    level_upper = level.upper()
    os.environ.setdefault("VLLM_LOGGING_LEVEL", level_upper)
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(level_upper)
    for handler in vllm_logger.handlers:
        handler.setLevel(level_upper)

    for name, candidate in logging.root.manager.loggerDict.items():
        if (name == "vllm" or name.startswith("vllm.")) and isinstance(
            candidate, logging.Logger
        ):
            candidate.setLevel(level_upper)
            for handler in candidate.handlers:
                handler.setLevel(level_upper)

    # vLLM configures its root logger at import time. Re-apply config when
    # vLLM was imported before this call (for example via EngineArgs).
    # _configure_vllm_root_logger is private; guard against renames on upgrade.
    if "vllm.logger" in sys.modules:
        try:
            from vllm.logger import (  # noqa: PLC0415
                _configure_vllm_root_logger,
            )

            _configure_vllm_root_logger()
        except (ImportError, AttributeError):
            pass


def vllm_benchmark_engine_config(vllm_config: dict[str, Any]) -> dict[str, Any]:
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
