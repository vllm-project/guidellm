"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

import contextlib
import logging
import os

from datasets import config

with (
    open(os.devnull, "w") as devnull,  # noqa: PTH123
    contextlib.redirect_stderr(devnull),
    contextlib.redirect_stdout(devnull),
):
    from transformers.utils import logging as hf_logging  # type: ignore[import]

    # Set the log level for the transformers library to ERROR
    # to ignore None of PyTorch, TensorFlow found
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence warnings for tokenizers
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    config.USE_AUDIO_DECODE = False

from .logger import configure_logger, logger
from .settings import (
    DatasetSettings,
    Environment,
    LoggingSettings,
    Settings,
    print_config,
    reload_settings,
    settings,
)

__all__ = [
    "DatasetSettings",
    "Environment",
    "LoggingSettings",
    "Settings",
    "configure_logger",
    "logger",
    "print_config",
    "reload_settings",
    "settings",
]
