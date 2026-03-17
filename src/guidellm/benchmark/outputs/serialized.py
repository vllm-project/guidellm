"""
Serialized output handler for generative benchmark reports.

This module provides a serialized output implementation that saves benchmark reports
to JSON or YAML file formats. It extends the base GenerativeBenchmarkerOutput to
handle file-based persistence of benchmark results, supporting both directory and
explicit file path specifications for report serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport

__all__ = ["GenerativeBenchmarkerSerialized"]


@GenerativeBenchmarkerOutput.register(["json", "yaml"])
class GenerativeBenchmarkerSerialized(GenerativeBenchmarkerOutput):
    """
    Serialized output handler for benchmark reports in JSON or YAML formats.

    This output handler persists generative benchmark reports to the file system in
    either JSON or YAML format. It supports flexible path specification, allowing
    users to provide either a directory (where a default filename will be generated)
    or an explicit file path for the serialized report output.

    Example:
    ::
        output = GenerativeBenchmarkerSerialized(output_path="/path/to/output.json")
        result_path = await output.finalize(report)
    """

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory or file path for saving the serialized report",
    )
    format_type: Literal["json", "yaml"] = Field(
        default="json",
        description="Serialization format, used to determine the default filename",
    )

    @classmethod
    def validated_kwargs(
        cls,
        output_path: str | Path | None,
        format_key: str | None = None,
        **_kwargs,
    ) -> dict[str, Any]:
        """
        Validate and normalize output path keyword arguments.

        :param output_path: Directory or file path for serialization output
        :param format_key: The registry key used to resolve this output type
        :param _kwargs: Additional keyword arguments (ignored)
        :return: Dictionary of validated keyword arguments for class initialization
        """
        validated: dict[str, Any] = {}
        if output_path is not None:
            validated["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        if format_key in ("json", "yaml"):
            validated["format_type"] = format_key
        return validated

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Serialize and save the benchmark report to the configured output path.

        :param report: The generative benchmarks report to serialize
        :return: Path to the saved report file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / f"benchmarks.{self.format_type}"
        return report.save_file(output_path)
