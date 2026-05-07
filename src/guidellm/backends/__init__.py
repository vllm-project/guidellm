"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, concrete backend implementations, and response
handlers for standardized communication with generative AI model providers.
The backend system supports distributed execution across worker processes with
pluggable response handlers for different API formats. Key components include
the abstract Backend base class, OpenAI-compatible HTTP backend, and response
handlers for processing streaming and non-streaming API responses.
"""

from .backend import Backend, BackendArgs, BackendType
from .openai import (
    AudioRequestHandler,
    ChatCompletionsRequestHandler,
    OpenAIHTTPBackend,
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
    TextCompletionsRequestHandler,
)
from .vllm_python import VLLMPythonBackend, VLLMResponseHandler

__all__ = [
    "AudioRequestHandler",
    "Backend",
    "BackendArgs",
    "BackendType",
    "ChatCompletionsRequestHandler",
    "OpenAIHTTPBackend",
    "OpenAIRequestHandler",
    "OpenAIRequestHandlerFactory",
    "TextCompletionsRequestHandler",
    "VLLMPythonBackend",
    "VLLMResponseHandler",
]
