"""
VLLM Python batch backend Args schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from guidellm.schemas.backends.backend import BackendArgs
from guidellm.schemas.backends.vllm_python import VLLMPythonAsyncBackendArgs

__all__ = ["VLLMPythonBatchBackendArgs"]


@BackendArgs.register("vllm_python_batch")
class VLLMPythonBatchBackendArgs(VLLMPythonAsyncBackendArgs):
    """Pydantic model for VLLM Python batch backend creation arguments.

    Extends :class:`VLLMPythonAsyncBackendArgs` with batch-specific options
    and removes the ``stream`` field (batch generation is always
    non-streaming).
    """

    kind: Literal["vllm_python_batch"] = Field(  # type: ignore[assignment]
        default="vllm_python_batch",
        description="Backend type identifier for VLLM Python batch backend.",
    )
    batch_size: PositiveInt = Field(
        default=32,
        description=(
            "Maximum number of requests to accumulate before "
            "dispatching a single vLLM generate() call.  Full "
            "batches flush immediately; partial batches wait up "
            "to ``batch_timeout`` seconds."
        ),
    )
    batch_timeout: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Seconds to wait for more requests before flushing a "
            "partial batch.  Full batches bypass this delay."
        ),
    )

    # Hide the inherited ``stream`` field -- batch generation is never streaming.
    stream: Literal[False] = Field(  # type: ignore[assignment]
        default=False,
        exclude=True,
        description="Batch backend does not support streaming.",
    )
