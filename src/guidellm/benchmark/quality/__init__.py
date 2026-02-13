"""
Quality validation and benchmarking tools for embeddings.

This module provides comprehensive quality validation capabilities for embeddings
including cosine similarity validation against baseline models and MTEB (Massive
Text Embedding Benchmark) integration for standardized quality evaluation.
"""

from __future__ import annotations

from .mteb_integration import DEFAULT_MTEB_TASKS, MTEBValidator
from .validators import EmbeddingsQualityValidator, compute_cosine_similarity

__all__ = [
    "DEFAULT_MTEB_TASKS",
    "EmbeddingsQualityValidator",
    "MTEBValidator",
    "compute_cosine_similarity",
]
