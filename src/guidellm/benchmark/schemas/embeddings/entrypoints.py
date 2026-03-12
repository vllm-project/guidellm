"""
Configuration entrypoints for embeddings benchmark execution.

Defines parameter schemas for creating embeddings benchmark runs from scenario
files or runtime arguments. Extends standard benchmark configuration with
embeddings-specific options.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas.base_args import BaseBenchmarkArgs

__all__ = ["BenchmarkEmbeddingsArgs"]


class BenchmarkEmbeddingsArgs(BaseBenchmarkArgs):
    """
    Configuration arguments for embeddings benchmark execution.

    Extends BaseBenchmarkArgs with embeddings-specific configuration.
    Overrides defaults for embeddings workflows and adds encoding format.

    Example::

        # Basic embeddings benchmark
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000/v1",
            data=["path/to/texts.json"],
            profile="sweep"
        )
    """

    # Override defaults for embeddings
    outputs: list[str] | tuple[str] = Field(
        default_factory=lambda: ["json"],
        description="Output types to create (json, yaml)",
    )
    data_column_mapper: str | dict[str, str | list[str]] = Field(
        default="embeddings_column_mapper",
        description="Column mapping preprocessor for dataset fields",
    )
    data_finalizer: str | dict = Field(
        default="embeddings",
        description="Finalizer for preparing data samples into requests",
    )
    data_collator: str | None = Field(
        default="embeddings",
        description="Data collator for batch processing",
    )
    data_num_workers: int | None = Field(
        default=0,
        description="Number of workers for data loading",
    )

    # Embeddings uses max_duration instead of max_seconds
    max_duration: float | None = Field(
        default=None,
        description="Maximum duration in seconds",
    )

    # EMBEDDINGS-SPECIFIC: Encoding format
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Embedding encoding format (float or base64)",
    )
