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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.utils import RegistryMixin

__all__ = ["GenerativeBenchmarkerOutput"]


class GenerativeBenchmarkerOutput(
    BaseModel, RegistryMixin[type["GenerativeBenchmarkerOutput"]], ABC
):
    """
    Abstract base for benchmark output formatters with registry support.

    Defines the interface for transforming benchmark reports into various output
    formats. Subclasses implement specific formatters (JSON, CSV, HTML) that can be
    registered and resolved dynamically. Supports flexible initialization from string
    identifiers, file paths, or configuration dictionaries enabling declarative
    output configuration in benchmark runs.

    Example:
        ::
            # Register and resolve output formats
            outputs = GenerativeBenchmarkerOutput.resolve(
                output_formats=["json", "csv"],
                output_path="./results"
            )

            # Finalize outputs with benchmark report
            for output in outputs.values():
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
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and normalize initialization arguments for output formatter.

        Processes positional and keyword arguments into a validated parameter
        dictionary suitable for formatter instantiation. Subclasses implement
        format-specific validation logic handling their unique parameter patterns.

        :param args: Positional arguments for formatter configuration
        :param kwargs: Keyword arguments for formatter configuration
        :return: Validated dictionary of parameters for formatter creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @classmethod
    def resolve(
        cls,
        output_formats: (
            Sequence[str]
            | Mapping[str, Any | dict[str, Any] | GenerativeBenchmarkerOutput]
            | None
        ),
        output_path: str | Path | None,
    ) -> dict[str, GenerativeBenchmarkerOutput]:
        """
        Resolve output format specifications into formatter instances.

        Supports multiple input patterns: format identifiers (["json", "csv"]),
        file paths (["results.json"]), format configurations ({"json": {"indent": 2}}),
        or pre-instantiated formatters. Registered format types are resolved from the
        registry and instantiated with validated parameters.

        :param output_formats: Format specifications as sequence of identifiers/paths,
            mapping of format configurations, or None for no outputs
        :param output_path: Default output directory path for all formatters
        :return: Dictionary mapping format keys to instantiated formatter instances
        :raises TypeError: If format specification type is invalid
        :raises ValueError: If format resolution or validation fails
        """
        resolved: dict[str, GenerativeBenchmarkerOutput] = {}

        if not output_formats:
            return resolved

        if isinstance(output_formats, list | tuple):
            # convert to dict for uniform processing
            formats_list = output_formats
            output_formats = {}
            for output_format in formats_list:
                # Check for registered type, if not, then assume it's a file path
                if cls.is_registered(output_format):
                    output_formats[output_format] = {}
                else:
                    path = Path(output_format)
                    format_type = path.suffix[1:].lower()
                    output_formats[format_type] = {"output_path": path}

        for key, val in output_formats.items():  # type: ignore[union-attr]
            if isinstance(val, GenerativeBenchmarkerOutput):
                resolved[key] = val
            else:
                output_class = cls.get_registered_object(key)
                if output_class is None:
                    available_formats = (
                        list(cls.registry.keys()) if cls.registry else []
                    )
                    raise ValueError(
                        f"Output format '{key}' is not registered. "
                        f"Available formats: {available_formats}"
                    )

                kwargs: dict[str, Any] = {"output_path": output_path}

                if isinstance(val, dict):
                    kwargs.update(val)
                    kwargs = output_class.validated_kwargs(**kwargs)
                else:
                    kwargs = output_class.validated_kwargs(val, **kwargs)

                resolved[key] = output_class(**kwargs)

        return resolved

    @abstractmethod
    async def finalize(self, report: GenerativeBenchmarksReport) -> Any:
        """
        Process and persist benchmark report in the formatter's output format.

        Transforms the provided benchmark report into the target format and writes
        results to the configured output destination. Implementation details vary by
        formatter type (file writing, API calls, etc.).

        :param report: Benchmark report containing results to format and output
        :return: Format-specific output result (file path, response object, etc.)
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...
