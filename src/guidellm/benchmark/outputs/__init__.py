"""
Output formatters for benchmark results.

Provides output formatter implementations that transform benchmark reports into
various file formats including JSON, CSV, HTML, and console display. All formatters
extend the base GenerativeBenchmarkerOutput interface, enabling dynamic resolution
and flexible output configuration for benchmark result persistence and analysis.
"""

from __future__ import annotations

from .console import GenerativeBenchmarkerConsole
from .csv import GenerativeBenchmarkerCSV
from .embeddings_console import EmbeddingsBenchmarkerConsole
from .embeddings_csv import EmbeddingsBenchmarkerCSV
from .embeddings_html import EmbeddingsBenchmarkerHTML
from .embeddings_serialized import EmbeddingsBenchmarkerSerialized
from .html import GenerativeBenchmarkerHTML
from .output import EmbeddingsBenchmarkerOutput, GenerativeBenchmarkerOutput
from .serialized import GenerativeBenchmarkerSerialized

__all__ = [
    "EmbeddingsBenchmarkerCSV",
    "EmbeddingsBenchmarkerConsole",
    "EmbeddingsBenchmarkerHTML",
    "EmbeddingsBenchmarkerOutput",
    "EmbeddingsBenchmarkerSerialized",
    "GenerativeBenchmarkerCSV",
    "GenerativeBenchmarkerConsole",
    "GenerativeBenchmarkerHTML",
    "GenerativeBenchmarkerOutput",
    "GenerativeBenchmarkerSerialized",
]
