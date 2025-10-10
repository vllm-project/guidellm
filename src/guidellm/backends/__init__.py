"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, implemented backends, request/response objects,
and timing utilities for standardized communication with LLM providers.
"""

from .backend import (
    Backend,
    BackendType,
)
from .openai import OpenAIHTTPBackend
from .response_handlers import (
    AudioResponseHandler,
    ChatCompletionsResponseHandler,
    GenerationResponseHandler,
    GenerationResponseHandlerFactory,
    TextCompletionsResponseHandler,
)

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
