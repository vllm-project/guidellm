"""Shared utilities for vLLM Python backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from guidellm.logger import logger

__all__ = ["reset_cpu_affinity"]


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

    # Try cgroup v2 path first, then fall back to cgroup v1.
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
            logger.info(
                "Reset CPU affinity from {} to {} cores",
                len(current),
                len(cpus),
            )
        return
