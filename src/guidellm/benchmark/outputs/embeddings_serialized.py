"""
Serialized output handler for embeddings benchmark reports.

Provides a serialized output implementation that saves embeddings benchmark reports
to JSON or YAML file formats. Extends the base EmbeddingsBenchmarkerOutput to handle
file-based persistence of benchmark results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput
from guidellm.benchmark.schemas.embeddings import EmbeddingsBenchmarksReport

__all__ = ["EmbeddingsBenchmarkerSerialized"]


@EmbeddingsBenchmarkerOutput.register(["json", "yaml"])
class EmbeddingsBenchmarkerSerialized(EmbeddingsBenchmarkerOutput):
    """
    Serialized output handler for embeddings benchmark reports in JSON or YAML formats.

    Persists embeddings benchmark reports to the file system in either JSON or YAML
    format. Supports flexible path specification, allowing users to provide either
    a directory (where a default filename will be generated) or an explicit file path.

    Example:
    ::
        output = EmbeddingsBenchmarkerSerialized(
            output_path="/path/to/embeddings_output.json"
        )
        result_path = await output.finalize(report)
    """

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory or file path for saving the serialized report",
    )

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate and normalize output path keyword arguments.

        :param output_path: Directory or file path for serialization output
        :param _kwargs: Additional keyword arguments (ignored)
        :return: Dictionary of validated keyword arguments for class initialization
        """
        validated: dict[str, Any] = {}
        if output_path is not None:
            validated["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return validated

    async def finalize(self, report: EmbeddingsBenchmarksReport) -> Path:
        """
        Serialize and save the embeddings benchmark report to the configured
        output path.

        :param report: The embeddings benchmarks report to serialize
        :return: Path to the saved report file
        """
        return report.save_file(self.output_path)
