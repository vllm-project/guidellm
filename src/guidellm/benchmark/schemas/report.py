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
from typing import Any, ClassVar, Literal

import yaml
from pydantic import Field

from guidellm.benchmark.schemas.benchmark import GenerativeBenchmark
from guidellm.schemas import StandardBaseModel
from guidellm.schemas.benchmark import BenchmarkScenario

__all__ = ["GenerativeBenchmarkMetadata", "GenerativeBenchmarksReport"]

_MULTIPART_REQUIRED_FIELDS = 2
_MULTIPART_CONTENT_TYPE_FIELDS = 3
_MEDIA_FIELDS = {
    "image",
    "audio",
    "video",
    "image_url",
    "audio_url",
    "video_url",
    "image_column",
    "audio_column",
    "video_column",
    "input_image",
    "input_audio",
    "input_file",
}


def _media_metadata(payload: str, *, base64_encoded: bool = False) -> dict[str, Any]:
    """Describe an omitted serialized payload without decoding or retaining it."""
    metadata: dict[str, Any] = {
        "omitted": True,
        "serialized_characters": len(payload),
    }
    start = 0
    if payload.startswith("data:") and (separator := payload.find(",")) >= 0:
        header = payload[5:separator]
        metadata["media_type"] = header.split(";", 1)[0] or "text/plain"
        base64_encoded = "base64" in header.split(";")[1:]
        start = separator + 1
    if base64_encoded:
        encoded = payload.rstrip(" \t\r\n")
        length = len(encoded) - start
        length -= sum(encoded.count(char, start) for char in " \t\r\n")
        padding = 2 if encoded.endswith("==") else int(encoded.endswith("="))
        metadata["size_bytes"] = length * 3 // 4 - padding
    return metadata


def _omit_request_arguments(arguments: str) -> str:
    """Process structured request arguments while preserving opaque descriptions."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(_omit_media_payloads(parsed))


def _omit_uploaded_files(files: dict[str, Any]) -> None:
    """Replace multipart upload tuples with their available metadata."""
    for name, upload in files.items():
        if (
            isinstance(upload, list)
            and len(upload) >= _MULTIPART_REQUIRED_FIELDS
            and isinstance(upload[1], str)
        ):
            metadata = _media_metadata(upload[1])
            metadata["filename"] = upload[0]
            if len(upload) >= _MULTIPART_CONTENT_TYPE_FIELDS:
                metadata["media_type"] = upload[2]
            files[name] = metadata
        else:
            files[name] = _omit_media_payloads(upload)


def _omit_media_payloads(value: Any, *, media_context: bool = False) -> Any:
    """Remove recognized media from a detached JSON-compatible report tree."""
    if isinstance(value, str):
        return (
            _media_metadata(value)
            if media_context and value.startswith("data:") and "," in value
            else value
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            omitted = _omit_media_payloads(item, media_context=media_context)
            # Scalar dataset columns must remain loadable after media omission.
            value[index] = (
                json.dumps(omitted)
                if isinstance(item, str) and isinstance(omitted, dict)
                else omitted
            )
    elif isinstance(value, dict):
        media_type = value.get("type")
        is_media = media_context or (
            isinstance(media_type, str) and media_type in _MEDIA_FIELDS
        )
        column_name = value.get("column_name")
        media_column = isinstance(column_name, str) and column_name in _MEDIA_FIELDS
        for key, item in value.items():
            if key == "request_args" and isinstance(item, str):
                value[key] = _omit_request_arguments(item)
            elif key == "files" and isinstance(item, dict):
                _omit_uploaded_files(item)
            elif isinstance(item, str) and (
                key in {"file_data", "b64_json"} or (key == "data" and is_media)
            ):
                value[key] = _media_metadata(item, base64_encoded=True)
            elif (
                key == "audio"
                and isinstance(item, str)
                and ("mimetype" in value or "file_name" in value)
            ):
                value[key] = _media_metadata(item)
            else:
                value[key] = _omit_media_payloads(
                    item,
                    media_context=(
                        key in _MEDIA_FIELDS
                        or (is_media and key in {"url", "data"})
                        or (media_column and key == "data")
                    ),
                )
    return value


class GenerativeBenchmarkMetadata(StandardBaseModel):
    """
    Versioning and environment metadata for generative benchmark reports.
    """

    # Make sure to update version when making breaking changes to report schema
    version: Literal[2] = Field(
        description=(
            "Version of the benchmark report schema, increments "
            "whenever there is a breaking change to the output format"
        ),
        default=2,
    )
    guidellm_version: str = Field(
        description="Version of the guidellm package used for the benchmark",
        default_factory=lambda: version("guidellm"),
    )
    python_version: str = Field(
        description="Version of Python interpreter used during the benchmark",
        default_factory=platform.python_version,
    )
    platform: str = Field(
        description="Operating system platform where the benchmark was executed",
        default_factory=platform.platform,
    )


class GenerativeBenchmarksReport(StandardBaseModel):
    """Container for multiple benchmark results with load/save functionality.

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
    config: BenchmarkScenario = Field(
        description="Benchmark arguments used for all benchmarks in the report"
    )
    benchmarks: list[GenerativeBenchmark] = Field(
        description="List of completed benchmarks in the report",
        default_factory=list,
    )
    conclusions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "What each profile concluded once every strategy it planned has "
            "run, for profiles that answer a question rather than execute a "
            "fixed sequence. Empty when no profile in the run computed one"
        ),
    )

    def save_file(
        self,
        path: str | Path | None = None,
        type_: Literal["json", "yaml"] | None = None,
        *,
        include_media: bool = True,
    ) -> Path:
        """
        Save report to file in JSON or YAML format.

        :param path: File path or directory for saving, defaults to current directory
            with DEFAULT_FILE name
        :param type_: File format override ('json' or 'yaml'), auto-detected from
            extension if None
        :param include_media: Preserve embedded media for replay. If False, replace
            recognized media payloads with metadata without changing this report.
        :return: Resolved path to the saved file
        :raises ValueError: If file type is unsupported or cannot be determined
        """
        file_path = GenerativeBenchmarksReport._resolve_path(
            path if path is not None else Path.cwd()
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_type = type_ or file_path.suffix.lower()[1:]
        model_dict = self.model_dump(mode="json")
        if not include_media:
            _omit_media_payloads(model_dict)
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
