"""Terminal stream utilities."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["suppress_worker_stdio"]


@contextmanager
def suppress_worker_stdio() -> Iterator[None]:
    """Redirect stdout and stderr to ``/dev/null`` at the OS fd level.

    Used in scheduler workers during heavyweight backend startup (e.g. vLLM
    engine initialisation) so that C-level writes from subprocesses and
    third-party libraries cannot corrupt the main-process Rich live display.
    Python-level ``sys.stdout``/``sys.stderr`` are also replaced so that any
    higher-level writes (HF Hub progress bars, tqdm) are silenced too.
    POSIX only; a no-op on Windows.
    """
    if sys.platform == "win32":
        yield
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)
    saved_sys_out = sys.stdout
    saved_sys_err = sys.stderr
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        yield
    finally:
        sys.stdout = saved_sys_out
        sys.stderr = saved_sys_err
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        os.close(devnull_fd)
