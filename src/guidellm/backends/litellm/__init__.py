"""
LiteLLM backend package for GuideLLM.

Provides a LiteLLM-powered backend that routes generation requests through
the LiteLLM SDK, giving access to 100+ providers (Anthropic, Gemini, Bedrock,
Groq, Cohere, Mistral, etc.) via a unified interface.
"""

from .litellm import LiteLLMBackend, LiteLLMBackendArgs

__all__ = ["LiteLLMBackend", "LiteLLMBackendArgs"]
