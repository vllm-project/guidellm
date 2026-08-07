"""
VLLM Python backend Args schema.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import Field, model_validator

from guidellm.schemas.backends.backend import BackendArgs

logger = logging.getLogger(__name__)

__all__ = ["VLLMPythonAsyncBackendArgs"]


@BackendArgs.register(["vllm_python_async", "vllm_python"])
class VLLMPythonAsyncBackendArgs(BackendArgs):
    """Pydantic model for VLLM Python backend creation arguments."""

    kind: Literal["vllm_python_async", "vllm_python"] = Field(
        default="vllm_python_async",
        description="Backend type identifier for VLLM Python backend.",
    )
    model: str = Field(
        description="Huggingface model identifier or filesystem path for VLLM to load",
        examples=["meta-llama/Llama-2-7b-chat-hf"],
    )
    vllm_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Configuration dictionary for vLLM AsyncEngineArgs parameters. Pass "
            "any valid AsyncEngineArgs parameters here (e.g. tensor_parallel_size, "
            "gpu_memory_utilization, max_model_len). The 'model' parameter is required "
            "and can be set here or via the top-level 'model' field; if set in both "
            "places, the top-level 'model' field takes precedence."
        ),
        examples=[
            {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        ],
    )
    request_format: Literal["plain", "default-template"] | str = Field(
        default="default-template",
        description=(
            "Request format for VLLM Python backend. "
            "Valid values are 'plain' (no chat template), 'default-template' "
            "(use tokenizer default), or a path to / inline Jinja2 chat template."
        ),
        examples=[
            "/path/to/chat_template.jinja2",
        ],
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream responses from the backend.",
    )
    image_placeholder: str = Field(
        default="<image>",
        description=(
            "Placeholder string for image items in multimodal prompts. "
            "Used when injecting placeholders for multimodal data."
        ),
    )
    audio_placeholder: str = Field(
        default="<|audio|>",
        description=(
            "Placeholder string for audio items in multimodal prompts. "
            "Used when injecting placeholders for multimodal data."
        ),
    )

    @model_validator(mode="after")
    def validate_vllm_config(self):
        """Set defaults on vllm_config and ensure model is set."""

        if "model" in self.vllm_config:
            logger.warning(
                "The `model` input was passed to the vllm python backend "
                "with the `vllm_config` input. Ignoring and overwriting "
                "with the value from the `model` input."
            )
        self.vllm_config["model"] = self.model

        return self
