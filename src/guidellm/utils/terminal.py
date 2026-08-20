"""Terminal stream utilities."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["suppress_worker_stdio"]


@contextmanager
def suppress_worker_stdio() -> Iterator[None]:
    """Redirect stdout and stderr to ``/dev/null`` at the OS fd level.

    Used in scheduler workers during heavyweight backend startup (e.g. vLLM
    engine initialisation) so that C-level writes from subprocesses and
    third-party libraries cannot corrupt the main-process Rich live display.
    Python-level ``sys.stdout``/``sys.stderr`` are also replaced so that any
    higher-level writes (HF Hub progress bars, tqdm) are silenced too.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)
    saved_sys_out = sys.stdout
    saved_sys_err = sys.stderr
    null_out = Path(os.devnull).open("w")  # noqa: SIM115
    null_err = Path(os.devnull).open("w")  # noqa: SIM115
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = null_out
        sys.stderr = null_err
        yield
    finally:
        sys.stdout = saved_sys_out
        sys.stderr = saved_sys_err
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        os.close(devnull_fd)
        null_out.close()
        null_err.close()
