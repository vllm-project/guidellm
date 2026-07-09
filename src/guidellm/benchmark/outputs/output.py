"""
Base output interface for generative benchmarking results.

This module defines the abstract base class for all benchmark output formatters in
the guidellm system. Output formatters transform benchmark reports into various file
formats (JSON, CSV, HTML, etc.) enabling flexible result persistence and analysis.
The module leverages a registry pattern for dynamic format resolution and supports
both direct instantiation and configuration-based initialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from disdantic import RegistryMixin
from pydantic import BaseModel, ConfigDict

from guidellm.benchmark.schemas import BenchmarkOutputArgs, GenerativeBenchmarksReport

__all__ = ["GenerativeBenchmarkerOutput"]


class GenerativeBenchmarkerOutput(
    BaseModel, RegistryMixin[type["GenerativeBenchmarkerOutput"]], ABC
):
    """
    Abstract base for benchmark output formatters with registry support.

    Defines the interface for transforming benchmark reports into various output
    formats. Subclasses implement specific formatters (JSON, CSV, HTML) that can be
    registered and resolved dynamically.

    Example:
        ::

            output = GenerativeBenchmarkerOutput.resolve(
                JSONBenchmarkOutputArgs(path="./results.json")
            )
            await output.finalize(report)
    """

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        from_attributes=True,
        use_enum_values=True,
    )

    @classmethod
    @abstractmethod
    def from_args(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerOutput:
        """
        Create an output formatter instance from output arguments.

        :param args: Output configuration arguments
        :return: Configured output formatter instance
        """
        ...

    @classmethod
    def resolve(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerOutput:
        """
        Resolve output arguments into a formatter instance.

        Looks up the registered output class by ``args.kind`` and delegates
        construction to its :meth:`from_args` factory.

        :param args: Output configuration arguments with kind and format-specific fields
        :return: Configured output formatter instance
        :raises ValueError: If the output kind is not registered
        """
        output_class = cls.get_registered_object(args.kind)
        if output_class is None:
            available_formats = list(cls.registry.keys()) if cls.registry else []
            raise ValueError(
                f"Output format '{args.kind}' is not registered. "
                f"Available formats: {available_formats}"
            )
        return output_class.from_args(args)

    @abstractmethod
    async def finalize(self, report: GenerativeBenchmarksReport) -> Any:
        """
        Process and persist benchmark report in the formatter's output format.

        :param report: Benchmark report containing results to format and output
        :return: Format-specific output result (file path, response object, etc.)
        """
        ...
