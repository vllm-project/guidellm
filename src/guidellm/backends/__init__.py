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

# Conditionally import VLLM backend if available
try:
    from .vllm_python.vllm import VLLMPythonBackend
    from .vllm_python.vllm_response import VLLMResponseHandler

    HAS_VLLM_BACKEND = True
except ImportError:
    VLLMPythonBackend = None  # type: ignore[assignment, misc]
    VLLMResponseHandler = None  # type: ignore[assignment, misc]
    HAS_VLLM_BACKEND = False

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
if HAS_VLLM_BACKEND:
    __all__.extend(["VLLMPythonBackend", "VLLMResponseHandler"])
