"""
Report container for multiple generative benchmark results with persistence.

Provides data structures for aggregating multiple benchmark executions into a single
report with file I/O capabilities. Supports loading and saving benchmark collections
in JSON and YAML formats, enabling result persistence, sharing, and analysis across
different execution sessions. Core functionality includes benchmark grouping with
shared configuration parameters and flexible file path resolution.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import Field

from guidellm.benchmark.schemas.generative.benchmark import GenerativeBenchmark
from guidellm.benchmark.schemas.generative.entrypoints import (
    BenchmarkGenerativeTextArgs,
)
from guidellm.schemas import StandardBaseModel

__all__ = ["GenerativeBenchmarkMetadata", "GenerativeBenchmarksReport"]


class GenerativeBenchmarkMetadata(StandardBaseModel):
    """
    Versioning and environment metadata for generative benchmark reports.
    """

    # Make sure to update version when making breaking changes to report schema
    version: Literal[1] = Field(
        description=(
            "Version of the benchmark report schema, increments "
            "whenever there is a breaking change to the output format"
        ),
        default=1,
    )
    guidellm_version: str = Field(
        description="Version of the guidellm package used for the benchmark",
        default_factory=lambda: version("guidellm"),
    )
    python_version: str = Field(
        description="Version of Python interpreter used during the benchmark",
        default_factory=lambda: platform.python_version(),
    )
    platform: str = Field(
        description="Operating system platform where the benchmark was executed",
        default_factory=lambda: platform.platform(),
    )


class GenerativeBenchmarksReport(StandardBaseModel):
    """
    Container for multiple benchmark results with load/save functionality.

    Aggregates multiple generative benchmark executions into a single report,
    providing persistence through JSON and YAML file formats. Enables result
    collection, storage, and retrieval across different execution sessions with
    automatic file type detection and path resolution.

    :cvar DEFAULT_FILE: Default filename used when saving to or loading from a directory
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"

    metadata: GenerativeBenchmarkMetadata = Field(
        description="Metadata about the benchmark report and execution environment",
        default_factory=GenerativeBenchmarkMetadata,
    )
    args: BenchmarkGenerativeTextArgs = Field(
        description="Benchmark arguments used for all benchmarks in the report"
    )
    benchmarks: list[GenerativeBenchmark] = Field(
        description="List of completed benchmarks in the report",
        default_factory=list,
    )

    def save_file(
        self,
        path: str | Path | None = None,
        type_: Literal["json", "yaml"] | None = None,
    ) -> Path:
        """
        Save report to file in JSON or YAML format.

        :param path: File path or directory for saving, defaults to current directory
            with DEFAULT_FILE name
        :param type_: File format override ('json' or 'yaml'), auto-detected from
            extension if None
        :return: Resolved path to the saved file
        :raises ValueError: If file type is unsupported or cannot be determined
        """
        file_path = GenerativeBenchmarksReport._resolve_path(
            path if path is not None else Path.cwd()
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_type = type_ or file_path.suffix.lower()[1:]
        model_dict = self.model_dump()

        if file_type == "json":
            save_str = json.dumps(model_dict)
        elif file_type in ["yaml", "yml"]:
            save_str = yaml.dump(model_dict)
        else:
            raise ValueError(f"Unsupported file type: {file_type} for {file_path}.")

        with file_path.open("w") as file:
            file.write(save_str)

        return file_path

    @classmethod
    def load_file(
        cls, path: str | Path, type_: Literal["json", "yaml"] | None = None
    ) -> GenerativeBenchmarksReport:
        """
        Load report from JSON or YAML file.

        :param path: File path or directory containing DEFAULT_FILE to load from
        :param type_: File format override ('json' or 'yaml'), auto-detected from
            extension if None
        :return: Loaded report instance with benchmarks and configuration
        :raises ValueError: If file type is unsupported or cannot be determined
        :raises FileNotFoundError: If specified file does not exist
        """
        file_path = GenerativeBenchmarksReport._resolve_path(path)
        file_type = type_ or file_path.suffix.lower()[1:]

        with file_path.open("r") as file:
            if file_type == "json":
                model_dict = json.loads(file.read())
            elif file_type in ["yaml", "yml"]:
                model_dict = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type} for {file_path}.")

        return GenerativeBenchmarksReport.model_validate(model_dict)

    @classmethod
    def _resolve_path(cls, path: str | Path) -> Path:
        """
        Resolve input to file path, converting directories to DEFAULT_FILE location.

        :param path: String or Path to resolve, directories append DEFAULT_FILE
        :return: Resolved file path
        """
        resolved = Path(path) if not isinstance(path, Path) else path

        if resolved.is_dir():
            resolved = resolved / GenerativeBenchmarksReport.DEFAULT_FILE

        return resolved
