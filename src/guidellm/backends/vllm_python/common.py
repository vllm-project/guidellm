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

    Sets environment variables that vLLM and its dependencies read at import
    time. Call this before the first access to any vLLM attribute so that
    child processes (EngineCore, scheduler workers) inherit the quieter
    settings. ``setdefault`` preserves any explicit user override.

    The function also reconfigures the in-process ``vllm`` logger and its
    handlers directly. This is needed as a safety net for cases where vLLM
    was imported before this function runs — for example in a worker
    subprocess that forked after ``import vllm``, or when a third-party
    library triggered the import earlier. In those scenarios the env-var
    path has no effect because vLLM's logging is already initialised, so
    we lower the live logger and handler levels explicitly. If ``vllm.logger``
    is already in ``sys.modules`` we also call its private
    ``_configure_vllm_root_logger`` to re-apply vLLM's own configuration
    with the updated level.

    :param level: Logging level string (e.g. ``"ERROR"``). Defaults to
        ``"ERROR"``.
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

    # Re-apply vLLM's own root-logger config when the module is already loaded.
    # Wrapped defensively: _configure_vllm_root_logger is a private API that
    # may change or be removed in future vLLM versions without notice.
    vllm_logger_module = sys.modules.get("vllm.logger")
    if vllm_logger_module is not None and hasattr(
        vllm_logger_module, "_configure_vllm_root_logger"
    ):
        try:
            vllm_logger_module._configure_vllm_root_logger()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Could not re-apply vLLM root logger config"
                " (vLLM API may have changed): {}",
                exc,
            )


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
