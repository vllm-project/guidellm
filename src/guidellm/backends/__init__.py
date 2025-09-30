"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, implemented backends, request/response objects,
and timing utilities for standardized communication with LLM providers.
"""

from .backend import (
    Backend,
    BackendType,
)
from .objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from .openai import OpenAIHTTPBackend

__all__ = [
    "Backend",
    "BackendType",
    "GenerationRequest",
    "GenerationRequestTimings",
    "GenerationResponse",
    "OpenAIHTTPBackend",
]
