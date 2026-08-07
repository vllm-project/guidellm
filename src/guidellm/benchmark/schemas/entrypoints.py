"""
Configuration entrypoints for generative text benchmark execution.

Defines parameter schemas and construction logic for creating benchmark runs from
scenario files or runtime arguments. Provides flexible configuration loading with
support for built-in scenarios, custom YAML/JSON files, and programmatic overrides.
Handles serialization of complex types including backends, processors, and profiles
for persistent storage and reproduction of benchmark configurations.
"""

from guidellm.schemas.benchmark.entrypoints import (
    BenchmarkArgs,
    BenchmarkMetadata,
    BenchmarkScenario,
    GenerativeMetricsArgs,
    MetricsArgs,
)

__all__ = [
    "BenchmarkArgs",
    "BenchmarkMetadata",
    "BenchmarkScenario",
    "GenerativeMetricsArgs",
    "MetricsArgs",
]
