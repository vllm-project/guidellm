"""
VLLM Python API backend package.

Provides the VLLM Python backend and response handler for compiling
OpenAI-style response dicts into GenerationResponse.
"""

from .vllm_response import VLLMResponseHandler

__all__ = ["VLLMResponseHandler"]
