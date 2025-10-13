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
