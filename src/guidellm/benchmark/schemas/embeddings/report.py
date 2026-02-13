"""
Report container for multiple embeddings benchmark results with persistence.

Provides data structures for aggregating multiple embeddings benchmark executions
into a single report with file I/O capabilities. Supports loading and saving benchmark
collections in JSON and YAML formats, enabling result persistence, sharing, and analysis
across different execution sessions.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import Field

from guidellm.benchmark.schemas.embeddings.benchmark import EmbeddingsBenchmark
from guidellm.benchmark.schemas.embeddings.entrypoints import (
    BenchmarkEmbeddingsArgs,
)
from guidellm.schemas import StandardBaseModel

__all__ = ["EmbeddingsBenchmarkMetadata", "EmbeddingsBenchmarksReport"]


class EmbeddingsBenchmarkMetadata(StandardBaseModel):
    """
    Versioning and environment metadata for embeddings benchmark reports.
    """

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


class EmbeddingsBenchmarksReport(StandardBaseModel):
    """
    Container for multiple embeddings benchmark results with load/save functionality.

    Aggregates multiple embeddings benchmark executions into a single report,
    providing persistence through JSON and YAML file formats. Enables result
    collection, storage, and retrieval across different execution sessions.

    :cvar DEFAULT_FILE: Default filename used when saving to or loading from a directory
    """

    DEFAULT_FILE: ClassVar[str] = "embeddings_benchmarks.json"

    type_: Literal["embeddings_benchmarks_report"] = Field(
        description="Type identifier for embeddings benchmarks report",
        default="embeddings_benchmarks_report",
    )
    metadata: EmbeddingsBenchmarkMetadata = Field(
        description="Metadata about the benchmark report and execution environment",
        default_factory=EmbeddingsBenchmarkMetadata,
    )
    args: BenchmarkEmbeddingsArgs = Field(
        description="Benchmark arguments used for all benchmarks in the report"
    )
    benchmarks: list[EmbeddingsBenchmark] = Field(
        description="List of completed embeddings benchmarks in the report",
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
        :param type_: File format override ('json' or 'yaml'), auto-detected from extension
        :return: Resolved path to the saved file
        :raises ValueError: If file type is unsupported or cannot be determined
        """
        file_path = EmbeddingsBenchmarksReport._resolve_path(
            path if path is not None else Path.cwd()
        )

        if type_ is None:
            type_ = EmbeddingsBenchmarksReport._detect_type(file_path)

        if type_ == "json":
            file_path.write_text(
                json.dumps(
                    self.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif type_ == "yaml":
            file_path.write_text(
                yaml.dump(
                    self.model_dump(mode="json"),
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
        else:
            raise ValueError(f"Unsupported file type: {type_}")

        return file_path

    @classmethod
    def load_file(
        cls, path: str | Path, type_: Literal["json", "yaml"] | None = None
    ) -> EmbeddingsBenchmarksReport:
        """
        Load report from file in JSON or YAML format.

        :param path: File path to load from
        :param type_: File format override, auto-detected from extension if None
        :return: Loaded embeddings benchmarks report instance
        :raises ValueError: If file type is unsupported or cannot be determined
        :raises FileNotFoundError: If specified file does not exist
        """
        file_path = EmbeddingsBenchmarksReport._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if type_ is None:
            type_ = EmbeddingsBenchmarksReport._detect_type(file_path)

        content = file_path.read_text()

        if type_ == "json":
            data = json.loads(content)
        elif type_ == "yaml":
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file type: {type_}")

        return cls.model_validate(data)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """
        Resolve file path, using DEFAULT_FILE if path is a directory.

        :param path: Input path as string or Path object
        :return: Resolved absolute Path to file
        """
        file_path = Path(path) if isinstance(path, str) else path

        if file_path.is_dir():
            file_path = file_path / EmbeddingsBenchmarksReport.DEFAULT_FILE

        return file_path.resolve()

    @staticmethod
    def _detect_type(path: Path) -> Literal["json", "yaml"]:
        """
        Detect file type from path extension.

        :param path: File path to analyze
        :return: Detected file type ('json' or 'yaml')
        :raises ValueError: If extension is not recognized
        """
        suffix = path.suffix.lower()

        if suffix in {".json"}:
            return "json"
        elif suffix in {".yaml", ".yml"}:
            return "yaml"
        else:
            raise ValueError(
                f"Cannot detect file type from extension: {suffix}. "
                "Use type_ parameter to specify 'json' or 'yaml'"
            )
