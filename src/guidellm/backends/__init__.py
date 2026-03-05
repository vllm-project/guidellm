"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, concrete backend implementations, and response
handlers for standardized communication with generative AI model providers.
The backend system supports distributed execution across worker processes with
pluggable response handlers for different API formats. Key components include
the abstract Backend base class, OpenAI-compatible HTTP backend, and response
handlers for processing streaming and non-streaming API responses.
"""

from .backend import Backend, BackendType
from .openai import (
    AudioRequestHandler,
    ChatCompletionsRequestHandler,
    OpenAIHTTPBackend,
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
    TextCompletionsRequestHandler,
)
from guidellm.extras.vllm import HAS_VLLM

# Conditionally import VLLM backend if available
if HAS_VLLM:
    from .vllm_python import VLLMPythonBackend, VLLMResponseHandler
else:
    VLLMPythonBackend = None  # type: ignore[assignment, misc]
    VLLMResponseHandler = None  # type: ignore[assignment, misc]

__all__ = [
    "AudioRequestHandler",
    "Backend",
    "BackendType",
    "ChatCompletionsRequestHandler",
    "OpenAIHTTPBackend",
    "OpenAIRequestHandler",
    "OpenAIRequestHandlerFactory",
    "TextCompletionsRequestHandler",
]

# Conditionally add VLLM backend and handler to exports
if HAS_VLLM:
    __all__.extend(["VLLMPythonBackend", "VLLMResponseHandler"])
