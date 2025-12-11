"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, concrete backend implementations, and response
handlers for standardized communication with generative AI model providers.
The backend system supports distributed execution across worker processes with
pluggable response handlers for different API formats. Key components include
the abstract Backend base class, OpenAI-compatible HTTP backend, and response
handlers for processing streaming and non-streaming API responses.
"""

from __future__ import annotations

from .backend import Backend, BackendType
from .openai import OpenAIHTTPBackend
from .response_handlers import (
    AudioResponseHandler,
    ChatCompletionsResponseHandler,
    GenerationResponseHandler,
    GenerationResponseHandlerFactory,
    TextCompletionsResponseHandler,
)

# Conditionally import VLLM backend if available
try:
    from .vllm import VLLMPythonBackend

    HAS_VLLM_BACKEND = True
except ImportError:
    VLLMPythonBackend = None  # type: ignore[assignment, misc]
    HAS_VLLM_BACKEND = False

__all__ = [
    "AudioResponseHandler",
    "Backend",
    "BackendType",
    "ChatCompletionsResponseHandler",
    "GenerationResponseHandler",
    "GenerationResponseHandlerFactory",
    "OpenAIHTTPBackend",
    "TextCompletionsResponseHandler",
]

# Conditionally add VLLM backend to exports
if HAS_VLLM_BACKEND:
    __all__.append("VLLMPythonBackend")
