"""
Backend Args schemas for GuideLLM backend configuration.
"""

from __future__ import annotations

from guidellm.schemas.backends.backend import BackendArgs
from guidellm.schemas.backends.openai_http import OpenAIHTTPBackendArgs
from guidellm.schemas.backends.openai_websocket import OpenAIWebSocketBackendArgs
from guidellm.schemas.backends.orcarouter_http import OrcaRouterHTTPBackendArgs
from guidellm.schemas.backends.vllm_python import VLLMPythonAsyncBackendArgs
from guidellm.schemas.backends.vllm_python_batch import VLLMPythonBatchBackendArgs

__all__ = [
    "BackendArgs",
    "OpenAIHTTPBackendArgs",
    "OpenAIWebSocketBackendArgs",
    "OrcaRouterHTTPBackendArgs",
    "VLLMPythonAsyncBackendArgs",
    "VLLMPythonBatchBackendArgs",
]
