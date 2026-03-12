"""
Configuration entrypoints for generative text benchmark execution.

Defines parameter schemas and construction logic for creating benchmark runs
from scenario files or runtime arguments. Provides flexible configuration
loading with support for built-in scenarios, custom YAML/JSON files, and
programmatic overrides. Handles serialization of complex types including
backends, processors, and profiles for persistent storage and reproduction of
benchmark configurations.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, NonNegativeFloat

from guidellm.benchmark.schemas.base_args import BaseBenchmarkArgs

__all__ = ["BenchmarkGenerativeTextArgs"]


class BenchmarkGenerativeTextArgs(BaseBenchmarkArgs):
    """
    Configuration arguments for generative text benchmark execution.

    Extends BaseBenchmarkArgs with generative-specific configuration.
    Includes sampling, rampup, advanced constraints, and error handling.

    Example::

        # Load from built-in scenario with overrides
        args = BenchmarkGenerativeTextArgs.create(
            scenario="chat",
            target="http://localhost:8000/v1",
            max_requests=1000
        )

        # Create from keyword arguments only
        args = BenchmarkGenerativeTextArgs(
            target="http://localhost:8000/v1",
            data=["path/to/dataset.json"],
            profile="fixed",
            rate=10.0
        )
    """

    # Override defaults for generative
    outputs: list[str] | tuple[str] = Field(
        default_factory=lambda: ["json", "csv", "html"],
        description=(
            "The aliases of the output types to create with their "
            "default filenames the file names and extensions of the "
            "output types to create"
        ),
    )
    data_preprocessors: list[str | dict[str, str | list[str]]] = Field(  # type: ignore[assignment]
        default_factory=lambda: ["encode_media"],  # type: ignore[arg-type]
        description="List of dataset preprocessors to apply in order",
    )

    # GENERATIVE-SPECIFIC: Sampling and benchmarker configuration
    sample_requests: int | None = Field(
        default=None,
        description=(
            "Number of requests to sample for detailed metrics (None for all)"
        ),
    )
    rampup: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "The time, in seconds, to ramp up the request rate over. "
            "Only applicable for Throughput/Concurrent strategies"
        ),
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description=("Whether to prefer backend response metrics over request metrics"),
    )

    # GENERATIVE-SPECIFIC: Advanced constraints
    max_seconds: int | float | None = Field(
        default=None,
        description="Maximum benchmark execution time in seconds",
    )
    max_error_rate: float | None = Field(
        default=None,
        description="Maximum error rate (0-1) before stopping",
    )
    max_global_error_rate: float | None = Field(
        default=None,
        description="Maximum global error rate (0-1) before stopping",
    )
    over_saturation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Over-saturation detection configuration. A dict with "
            "configuration parameters (enabled, min_seconds, "
            "max_window_seconds, moe_threshold, etc.)."
        ),
    )
