"""
Serialized output handler for generative benchmark reports.

This module provides a serialized output implementation that saves benchmark reports
to JSON or YAML file formats. It extends the base GenerativeBenchmarkerOutput to
handle file-based persistence of benchmark results, supporting both directory and
explicit file path specifications for report serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import BenchmarkOutputArgs, GenerativeBenchmarksReport

__all__ = [
    "GenerativeBenchmarkerSerialized",
    "JSONBenchmarkOutputArgs",
    "YAMLBenchmarkOutputArgs",
]


@BenchmarkOutputArgs.register("json")
class JSONBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for JSON benchmark output arguments."""

    kind: Literal["json"] = Field(
        default="json",
        description="The kind of output.",
        examples=["json"],
    )
    path: Path = Field(
        default=Path("./benchmarks.json"),
        description="The file to save the output to.",
        examples=["./benchmarks.json"],
    )


@BenchmarkOutputArgs.register("yaml")
class YAMLBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for YAML benchmark output arguments."""

    kind: Literal["yaml"] = Field(
        default="yaml",
        description="The kind of output.",
        examples=["yaml"],
    )
    path: Path = Field(
        default=Path("./benchmarks.yaml"),
        description="The file to save the output to.",
        examples=["./benchmarks.yaml"],
    )


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
        default_factory=Path.cwd,
        description="Directory or file path for saving the serialized report",
    )
    format_type: Literal["json", "yaml"] = Field(
        default="json",
        description="Serialization format, used to determine the default filename",
    )

    @classmethod
    def from_args(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerSerialized:
        """
        Create a serialized output formatter from output arguments.

        :param args: Output configuration with path and kind (json or yaml)
        :return: Configured serialized output formatter
        """
        if not isinstance(args, JSONBenchmarkOutputArgs | YAMLBenchmarkOutputArgs):
            raise ValueError(f"Invalid args type: {type(args)}.")

        return cls(output_path=args.path, format_type=args.kind)

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
