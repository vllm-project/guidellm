from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from guidellm.benchmark.schemas import (
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
    GenerativeMetrics,
)
from guidellm.presentation import UIDataBuilder
from guidellm.presentation.injector import create_report
from guidellm.utils import (
    Console,
    DistributionSummary,
    RegistryMixin,
    StatusDistributionSummary,
    camelize_str,
    recursive_key_update,
    safe_format_number,
    safe_format_timestamp,
)

__all__ = [
    "GenerativeBenchmarkerCSV",
    "GenerativeBenchmarkerConsole",
    "GenerativeBenchmarkerHTML",
    "GenerativeBenchmarkerOutput",
]


class GenerativeBenchmarkerOutput(
    BaseModel, RegistryMixin[type["GenerativeBenchmarkerOutput"]], ABC
):
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
        Validate and process arguments for constraint creation.

        Must be implemented by subclasses to handle their specific parameter patterns.

        :param args: Positional arguments passed to the constraint
        :param kwargs: Keyword arguments passed to the constraint
        :return: Validated dictionary of parameters for constraint creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @classmethod
    def resolve(
        cls,
        output_formats: (
            tuple[str, ...]
            | list[str]
            | dict[
                str,
                Any | dict[str, Any] | GenerativeBenchmarkerOutput,
            ]
            | None
        ),
        output_path: str | Path | None,
    ) -> dict[str, GenerativeBenchmarkerOutput]:
        if not output_formats:
            return {}

        if isinstance(output_formats, list | tuple):
            # support list of output keys: ["csv", "json"]
            # support list of files: ["path/to/file.json", "path/to/file.csv"]
            formats_list = output_formats
            output_formats = {}
            for output_format in formats_list:
                if not isinstance(output_format, str):
                    raise TypeError(
                        f"Expected string format, got {type(output_format)} for "
                        f"{output_format} in {formats_list}"
                    )
                try:
                    if cls.is_registered(output_format):
                        output_formats[output_format] = {}
                    else:
                        # treat it as a file save location
                        path = Path(output_format)
                        format_type = path.suffix[1:].lower()
                        output_formats[format_type] = {"output_path": path}

                except Exception as err:
                    raise ValueError(
                        f"Failed to resolve output format '{output_format}': {err}"
                    ) from err

        resolved = {}

        for key, val in output_formats.items():
            if isinstance(val, GenerativeBenchmarkerOutput):
                resolved[key] = val
            else:
                output_class = cls.get_registered_object(key)
                kwargs = {"output_path": output_path}

                if isinstance(val, dict):
                    kwargs.update(val)
                    kwargs = output_class.validated_kwargs(**kwargs)
                else:
                    kwargs = output_class.validated_kwargs(val, **kwargs)

                resolved[key] = output_class(**kwargs)

        return resolved

    @abstractmethod
    async def finalize(self, report: GenerativeBenchmarksReport) -> Any: ...


@GenerativeBenchmarkerOutput.register(["json", "yaml"])
class GenerativeBenchmarkerSerialized(GenerativeBenchmarkerOutput):
    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        new_kwargs = {}
        if output_path is not None:
            new_kwargs["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return new_kwargs

    output_path: Path = Field(default_factory=lambda: Path.cwd())

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        return report.save_file(self.output_path)


@dataclass
class ConsoleTableColumn:
    group: str | None = None
    name: str | None = None
    units: str | None = None
    type_: Literal["number", "text", "timestamp"] = "number"
    precision: int = 1
    values: list = field(default_factory=list)


class ConsoleTableColumnsCollection(dict[str, ConsoleTableColumn]):
    def add_value(
        self,
        value: str | float | int,
        group: str | None = None,
        name: str | None = None,
        units: str | None = None,
        type_: Literal["number", "text", "timestamp"] = "number",
        precision: int = 1,
    ):
        key = f"{group}_{name}_{units}"

        if key not in self:
            self[key] = ConsoleTableColumn(
                group=group, name=name, units=units, type_=type_, precision=precision
            )

        self[key].values.append(value)

    def add_stats(
        self,
        stats: StatusDistributionSummary,
        status: Literal["successful", "incomplete", "errored", "total"] = "successful",
        group: str | None = None,
        name: str | None = None,
        precision: int = 1,
    ):
        key = f"{group}_{name}"

        if f"{key}_median" not in self:
            self[f"{key}_mean"] = ConsoleTableColumn(
                group=group, name=name, units="Mean", precision=precision
            )
            self[f"{key}_stddev"] = ConsoleTableColumn(
                group=group, name=name, units="Std", precision=precision
            )

        status_stats: DistributionSummary = getattr(stats, status) if stats else None
        self[f"{key}_mean"].values.append(status_stats.mean if status_stats else None)
        self[f"{key}_stddev"].values.append(
            status_stats.std_dev if status_stats else None
        )

    def get_table_data(self) -> tuple[list[list[str]], list[list[str]]]:
        headers: list[list[str]] = []
        values: list[list[str]] = []

        for column in self.values():
            headers.append([column.group or "", column.name or "", column.units or ""])
            values.append(
                [
                    (
                        str(value)
                        if column.type_ == "text"
                        else safe_format_timestamp(value)
                        if column.type_ == "timestamp"
                        else safe_format_number(value, precision=column.precision)
                    )
                    for value in column.values
                ]
            )

        return headers, values


@GenerativeBenchmarkerOutput.register("console")
class GenerativeBenchmarkerConsole(GenerativeBenchmarkerOutput):
    @classmethod
    def validated_kwargs(cls, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    console: Console = Field(default_factory=Console)

    async def finalize(self, report: GenerativeBenchmarksReport) -> str:
        """
        Print the complete benchmark report to the console.

        :param report: The completed benchmark report.
        :return:
        """
        self._print_run_summary_table(report)
        self._print_text_table(report)
        self._print_image_table(report)
        self._print_video_table(report)
        self._print_audio_table(report)
        self._print_request_counts_table(report)
        self._print_request_latency_table(report)
        self._print_server_throughput_table(report)

        return "printed to console"

    def _print_run_summary_table(self, report: GenerativeBenchmarksReport):
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )
            columns.add_value(
                benchmark.start_time, group="Timings", name="Start", type_="timestamp"
            )
            columns.add_value(
                benchmark.end_time, group="Timings", name="End", type_="timestamp"
            )
            columns.add_value(
                benchmark.duration, group="Timings", name="Dur", units="Sec"
            )
            columns.add_value(
                report.args.warmup, group="Timings", name="Warm", units="Sec"
            )
            columns.add_value(
                report.args.cooldown, group="Timings", name="Cool", units="Sec"
            )
            columns.add_value(
                benchmark.metrics.prompt_token_count.successful.total_sum,
                group="Input Tokens",
                name="Comp",
                units="Tot",
            )
            columns.add_value(
                benchmark.metrics.prompt_token_count.incomplete.total_sum,
                group="Input Tokens",
                name="Inc",
                units="Tot",
            )
            columns.add_value(
                benchmark.metrics.prompt_token_count.errored.total_sum,
                group="Input Tokens",
                name="Err",
                units="Tot",
            )
            columns.add_value(
                benchmark.metrics.output_token_count.successful.total_sum,
                group="Output Tokens",
                name="Comp",
                units="Tot",
            )
            columns.add_value(
                benchmark.metrics.output_token_count.incomplete.total_sum,
                group="Output Tokens",
                name="Inc",
                units="Tot",
            )
            columns.add_value(
                benchmark.metrics.output_token_count.errored.total_sum,
                group="Output Tokens",
                name="Err",
                units="Tot",
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Run Summary Info")

    def _print_text_table(self, report: GenerativeBenchmarksReport):
        columns: dict[str, ConsoleTableColumnsCollection] = defaultdict(
            ConsoleTableColumnsCollection
        )

        for benchmark in report.benchmarks:
            columns["labels"].add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            # Add tokens columns
            tokens = benchmark.metrics.text.tokens
            columns["tokens.input"].add_stats(
                tokens.input if tokens else None,
                group="Input Tokens",
                name="Per Request",
            )
            columns["tokens.input"].add_stats(
                tokens.input_per_second if tokens else None,
                group="Input Tokens",
                name="Per Second",
            )
            columns["tokens.output"].add_stats(
                tokens.output if tokens else None,
                group="Output Tokens",
                name="Per Request",
            )
            columns["tokens.output"].add_stats(
                tokens.output_per_second if tokens else None,
                group="Output Tokens",
                name="Per Second",
            )

            # Add words columns
            words = benchmark.metrics.text.words
            columns["words.input"].add_stats(
                words.input if words else None,
                group="Input Words",
                name="Per Request",
            )
            columns["words.input"].add_stats(
                words.input_per_second if words else None,
                group="Input Words",
                name="Per Second",
            )
            columns["words.output"].add_stats(
                words.output if words else None,
                group="Output Words",
                name="Per Request",
            )
            columns["words.output"].add_stats(
                words.output_per_second if words else None,
                group="Output Words",
                name="Per Second",
            )

            # Add characters columns
            characters = benchmark.metrics.text.characters
            columns["characters.input"].add_stats(
                characters.input if characters else None,
                group="Input Characters",
                name="Per Request",
            )
            columns["characters.input"].add_stats(
                characters.input_per_second if characters else None,
                group="Input Characters",
                name="Per Second",
            )
            columns["characters.output"].add_stats(
                characters.output if characters else None,
                group="Output Characters",
                name="Per Request",
            )
            columns["characters.output"].add_stats(
                characters.output_per_second if characters else None,
                group="Output Characters",
                name="Per Second",
            )

        self._print_inp_out_tables(
            title="Text Metrics Statistics (Completed Requests)",
            labels=columns["labels"],
            groups=[
                (columns["tokens.input"], columns["tokens.output"]),
                (columns["words.input"], columns["words.output"]),
                (columns["characters.input"], columns["characters.output"]),
            ],
        )

    def _print_image_table(self, report: GenerativeBenchmarksReport):
        """Print image-specific metrics table if any image data exists."""
        columns: dict[str, ConsoleTableColumnsCollection] = defaultdict(
            ConsoleTableColumnsCollection
        )

        for benchmark in report.benchmarks:
            columns["labels"].add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            tokens = benchmark.metrics.image.tokens
            columns["tokens.input"].add_stats(
                tokens.input if tokens else None,
                group="Input Tokens",
                name="Per Request",
            )
            columns["tokens.input"].add_stats(
                tokens.input_per_second if tokens else None,
                group="Input Tokens",
                name="Per Second",
            )
            columns["tokens.output"].add_stats(
                tokens.output if tokens else None,
                group="Output Tokens",
                name="Per Request",
            )
            columns["tokens.output"].add_stats(
                tokens.output_per_second if tokens else None,
                group="Output Tokens",
                name="Per Second",
            )

            images = benchmark.metrics.image.images
            columns["images.input"].add_stats(
                images.input if images else None,
                group="Input Images",
                name="Per Request",
            )
            columns["images.input"].add_stats(
                images.input_per_second if images else None,
                group="Input Images",
                name="Per Second",
            )
            columns["images.output"].add_stats(
                images.output if images else None,
                group="Output Images",
                name="Per Request",
            )
            columns["images.output"].add_stats(
                images.output_per_second if images else None,
                group="Output Images",
                name="Per Second",
            )

            pixels = benchmark.metrics.image.pixels
            columns["pixels.input"].add_stats(
                pixels.input if pixels else None,
                group="Input Pixels",
                name="Per Request",
            )
            columns["pixels.input"].add_stats(
                pixels.input_per_second if pixels else None,
                group="Input Pixels",
                name="Per Second",
            )
            columns["pixels.output"].add_stats(
                pixels.output if pixels else None,
                group="Output Pixels",
                name="Per Request",
            )
            columns["pixels.output"].add_stats(
                pixels.output_per_second if pixels else None,
                group="Output Pixels",
                name="Per Second",
            )

            bytes_ = benchmark.metrics.image.bytes
            columns["bytes.input"].add_stats(
                bytes_.input if bytes_ else None,
                group="Input Bytes",
                name="Per Request",
            )
            columns["bytes.input"].add_stats(
                bytes_.input_per_second if bytes_ else None,
                group="Input Bytes",
                name="Per Second",
            )
            columns["bytes.output"].add_stats(
                bytes_.output if bytes_ else None,
                group="Output Bytes",
                name="Per Request",
            )
            columns["bytes.output"].add_stats(
                bytes_.output_per_second if bytes_ else None,
                group="Output Bytes",
                name="Per Second",
            )

        self._print_inp_out_tables(
            title="Image Metrics Statistics (Completed Requests)",
            labels=columns["labels"],
            groups=[
                (columns["tokens.input"], columns["tokens.output"]),
                (columns["images.input"], columns["images.output"]),
                (columns["pixels.input"], columns["pixels.output"]),
                (columns["bytes.input"], columns["bytes.output"]),
            ],
        )

    def _print_video_table(self, report: GenerativeBenchmarksReport):
        """Print video-specific metrics table if any video data exists."""
        columns: dict[str, ConsoleTableColumnsCollection] = defaultdict(
            ConsoleTableColumnsCollection
        )

        for benchmark in report.benchmarks:
            columns["labels"].add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            tokens = benchmark.metrics.video.tokens
            columns["tokens.input"].add_stats(
                tokens.input if tokens else None,
                group="Input Tokens",
                name="Per Request",
            )
            columns["tokens.input"].add_stats(
                tokens.input_per_second if tokens else None,
                group="Input Tokens",
                name="Per Second",
            )
            columns["tokens.output"].add_stats(
                tokens.output if tokens else None,
                group="Output Tokens",
                name="Per Request",
            )
            columns["tokens.output"].add_stats(
                tokens.output_per_second if tokens else None,
                group="Output Tokens",
                name="Per Second",
            )

            frames = benchmark.metrics.video.frames
            columns["frames.input"].add_stats(
                frames.input if frames else None,
                group="Input Frames",
                name="Per Request",
            )
            columns["frames.input"].add_stats(
                frames.input_per_second if frames else None,
                group="Input Frames",
                name="Per Second",
            )
            columns["frames.output"].add_stats(
                frames.output if frames else None,
                group="Output Frames",
                name="Per Request",
            )
            columns["frames.output"].add_stats(
                frames.output_per_second if frames else None,
                group="Output Frames",
                name="Per Second",
            )

            seconds = benchmark.metrics.video.seconds
            columns["seconds.input"].add_stats(
                seconds.input if seconds else None,
                group="Input Seconds",
                name="Per Request",
            )
            columns["seconds.input"].add_stats(
                seconds.input_per_second if seconds else None,
                group="Input Seconds",
                name="Per Second",
            )
            columns["seconds.output"].add_stats(
                seconds.output if seconds else None,
                group="Output Seconds",
                name="Per Request",
            )
            columns["seconds.output"].add_stats(
                seconds.output_per_second if seconds else None,
                group="Output Seconds",
                name="Per Second",
            )

            bytes_ = benchmark.metrics.video.bytes
            columns["bytes.input"].add_stats(
                bytes_.input if bytes_ else None,
                group="Input Bytes",
                name="Per Request",
            )
            columns["bytes.input"].add_stats(
                bytes_.input_per_second if bytes_ else None,
                group="Input Bytes",
                name="Per Second",
            )
            columns["bytes.output"].add_stats(
                bytes_.output if bytes_ else None,
                group="Output Bytes",
                name="Per Request",
            )
            columns["bytes.output"].add_stats(
                bytes_.output_per_second if bytes_ else None,
                group="Output Bytes",
                name="Per Second",
            )

        self._print_inp_out_tables(
            title="Video Metrics Statistics (Completed Requests)",
            labels=columns["labels"],
            groups=[
                (columns["tokens.input"], columns["tokens.output"]),
                (columns["frames.input"], columns["frames.output"]),
                (columns["seconds.input"], columns["seconds.output"]),
                (columns["bytes.input"], columns["bytes.output"]),
            ],
        )

    def _print_audio_table(self, report: GenerativeBenchmarksReport):
        """Print audio-specific metrics table if any audio data exists."""
        columns: dict[str, ConsoleTableColumnsCollection] = defaultdict(
            ConsoleTableColumnsCollection
        )

        for benchmark in report.benchmarks:
            columns["labels"].add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            tokens = benchmark.metrics.audio.tokens
            columns["tokens.input"].add_stats(
                tokens.input if tokens else None,
                group="Input Tokens",
                name="Per Request",
            )
            columns["tokens.input"].add_stats(
                tokens.input_per_second if tokens else None,
                group="Input Tokens",
                name="Per Second",
            )
            columns["tokens.output"].add_stats(
                tokens.output if tokens else None,
                group="Output Tokens",
                name="Per Request",
            )
            columns["tokens.output"].add_stats(
                tokens.output_per_second if tokens else None,
                group="Output Tokens",
                name="Per Second",
            )

            samples = benchmark.metrics.audio.samples
            columns["samples.input"].add_stats(
                samples.input if samples else None,
                group="Input Samples",
                name="Per Request",
            )
            columns["samples.input"].add_stats(
                samples.input_per_second if samples else None,
                group="Input Samples",
                name="Per Second",
            )
            columns["samples.output"].add_stats(
                samples.output if samples else None,
                group="Output Samples",
                name="Per Request",
            )
            columns["samples.output"].add_stats(
                samples.output_per_second if samples else None,
                group="Output Samples",
                name="Per Second",
            )

            seconds = benchmark.metrics.audio.seconds
            columns["seconds.input"].add_stats(
                seconds.input if seconds else None,
                group="Input Seconds",
                name="Per Request",
            )
            columns["seconds.input"].add_stats(
                seconds.input_per_second if seconds else None,
                group="Input Seconds",
                name="Per Second",
            )
            columns["seconds.output"].add_stats(
                seconds.output if seconds else None,
                group="Output Seconds",
                name="Per Request",
            )
            columns["seconds.output"].add_stats(
                seconds.output_per_second if seconds else None,
                group="Output Seconds",
                name="Per Second",
            )

            bytes_ = benchmark.metrics.audio.bytes
            columns["bytes.input"].add_stats(
                bytes_.input if bytes_ else None,
                group="Input Bytes",
                name="Per Request",
            )
            columns["bytes.input"].add_stats(
                bytes_.input_per_second if bytes_ else None,
                group="Input Bytes",
                name="Per Second",
            )
            columns["bytes.output"].add_stats(
                bytes_.output if bytes_ else None,
                group="Output Bytes",
                name="Per Request",
            )
            columns["bytes.output"].add_stats(
                bytes_.output_per_second if bytes_ else None,
                group="Output Bytes",
                name="Per Second",
            )

        self._print_inp_out_tables(
            title="Audio Metrics Statistics (Completed Requests)",
            labels=columns["labels"],
            groups=[
                (columns["tokens.input"], columns["tokens.output"]),
                (columns["samples.input"], columns["samples.output"]),
                (columns["seconds.input"], columns["seconds.output"]),
                (columns["bytes.input"], columns["bytes.output"]),
            ],
        )

    def _print_request_counts_table(self, report: GenerativeBenchmarksReport):
        """Print request latency metrics table."""
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )
            columns.add_stats(
                benchmark.metrics.prompt_token_count,
                group="Input Tok",
                name="Per Req",
            )
            columns.add_stats(
                benchmark.metrics.output_token_count,
                group="Output Tok",
                name="Per Req",
            )
            columns.add_stats(
                benchmark.metrics.total_token_count,
                group="Total Tok",
                name="Per Req",
            )
            columns.add_stats(
                benchmark.metrics.request_streaming_iterations_count,
                group="Stream Iter",
                name="Per Req",
            )
            columns.add_stats(
                benchmark.metrics.output_tokens_per_iteration,
                group="Output Tok",
                name="Per Stream Iter",
            )
            columns.add_stats(
                benchmark.metrics.output_tokens_per_iteration,
                group="Iter Tok",
                name="Per Stream Iter",
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(
            headers, values, title="Request Token Statistics (Completed Requests)"
        )

    def _print_request_latency_table(self, report: GenerativeBenchmarksReport):
        """Print request latency metrics table."""
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )
            columns.add_stats(
                benchmark.metrics.request_latency,
                group="Request Latency",
                name="Sec",
            )
            columns.add_stats(
                benchmark.metrics.time_to_first_token_ms,
                group="TTFT",
                name="ms",
            )
            columns.add_stats(
                benchmark.metrics.inter_token_latency_ms,
                group="ITL",
                name="ms",
            )
            columns.add_stats(
                benchmark.metrics.time_per_output_token_ms,
                group="TPOT",
                name="ms",
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(
            headers, values, title="Request Latency Statistics (Completed Requests)"
        )

    def _print_server_throughput_table(self, report: GenerativeBenchmarksReport):
        """Print server throughput metrics table."""
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.scheduler.strategy,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )
            columns.add_stats(
                benchmark.metrics.requests_per_second,
                group="Requests",
                name="Per Sec",
            )
            columns.add_stats(
                benchmark.metrics.request_concurrency,
                group="Requests",
                name="Concurrency",
            )
            columns.add_stats(
                benchmark.metrics.prompt_tokens_per_second,
                group="Input Tokens",
                name="Per Sec",
            )
            columns.add_stats(
                benchmark.metrics.output_tokens_per_second,
                group="Output Tokens",
                name="Per Sec",
            )
            columns.add_stats(
                benchmark.metrics.tokens_per_second,
                group="Total Tokens",
                name="Per Sec",
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Server Throughput Statistics")

    def _print_inp_out_tables(
        self,
        title: str,
        labels: ConsoleTableColumnsCollection,
        groups: list[
            tuple[ConsoleTableColumnsCollection, ConsoleTableColumnsCollection]
        ],
    ):
        """
        Print separate input and output tables for domain-specific metrics.

        :param title: Title for the table group
        :param labels: Label columns to include in both tables
        :param groups: List of (input_columns, output_columns) tuples for each
            metric type
        """
        inp_headers, inp_values = [], []
        out_headers, out_values = [], []
        inp_has_data = False
        out_has_data = False

        for inp_columns, out_columns in groups:
            # Check if columns have any non-None values
            type_inp_has_data = any(
                any(v is not None for v in col.values) for col in inp_columns.values()
            )
            type_out_has_data = any(
                any(v is not None for v in col.values) for col in out_columns.values()
            )

            if not (type_inp_has_data or type_out_has_data):
                continue

            inp_has_data = inp_has_data or type_inp_has_data
            out_has_data = out_has_data or type_out_has_data

            inp_type_headers, inp_type_columns = inp_columns.get_table_data()
            out_type_headers, out_type_columns = out_columns.get_table_data()

            inp_headers.extend(inp_type_headers)
            inp_values.extend(inp_type_columns)
            out_headers.extend(out_type_headers)
            out_values.extend(out_type_columns)

        if not (inp_has_data or out_has_data):
            return

        labels_headers, labels_values = labels.get_table_data()
        header_cols_groups = []
        value_cols_groups = []

        if inp_has_data:
            header_cols_groups.append(labels_headers + inp_headers)
            value_cols_groups.append(labels_values + inp_values)
        if out_has_data:
            header_cols_groups.append(labels_headers + out_headers)
            value_cols_groups.append(labels_values + out_values)

        if header_cols_groups and value_cols_groups:
            self.console.print("\n")
            self.console.print_tables(
                header_cols_groups=header_cols_groups,
                value_cols_groups=value_cols_groups,
                title=title,
            )


@GenerativeBenchmarkerOutput.register("csv")
class GenerativeBenchmarkerCSV(GenerativeBenchmarkerOutput):
    """CSV output formatter for benchmark results."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.csv"

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        new_kwargs = {}
        if output_path is not None:
            new_kwargs["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return new_kwargs

    output_path: Path = Field(default_factory=lambda: Path.cwd())

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as a CSV file.

        :param report: The completed benchmark report.
        :return: Path to the saved CSV file.
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / GenerativeBenchmarkerCSV.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            headers: list[str] = []
            rows: list[list[str | float | list[float]]] = []

            for benchmark in report.benchmarks:
                benchmark_headers: list[str] = []
                benchmark_values: list[str | float | list[float]] = []

                # Add basic run description info
                desc_headers, desc_values = self._get_benchmark_desc_headers_and_values(
                    benchmark
                )
                benchmark_headers.extend(desc_headers)
                benchmark_values.extend(desc_values)

                # Add status-based metrics
                for status in StatusDistributionSummary.model_fields:
                    status_headers, status_values = (
                        self._get_benchmark_status_headers_and_values(benchmark, status)
                    )
                    benchmark_headers.extend(status_headers)
                    benchmark_values.extend(status_values)

                # Add extra fields
                extras_headers, extras_values = (
                    self._get_benchmark_extras_headers_and_values(benchmark)
                )
                benchmark_headers.extend(extras_headers)
                benchmark_values.extend(extras_values)

                if not headers:
                    headers = benchmark_headers
                rows.append(benchmark_values)

            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)

        return output_path

    def _get_benchmark_desc_headers_and_values(
        self, benchmark: GenerativeBenchmark
    ) -> tuple[list[str], list[str | float]]:
        """Get description headers and values for a benchmark."""
        headers = [
            "Type",
            "Run Id",
            "Id",
            "Name",
            "Start Time",
            "End Time",
            "Duration",
        ]
        values: list[str | float] = [
            benchmark.type_,
            benchmark.run_id,
            benchmark.id_,
            str(benchmark.scheduler.strategy),
            datetime.fromtimestamp(benchmark.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(benchmark.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            benchmark.duration,
        ]
        return headers, values

    def _get_benchmark_status_headers_and_values(
        self, benchmark: GenerativeBenchmark, status: str
    ) -> tuple[list[str], list[float | list[float]]]:
        """Get status-based metrics headers and values for a benchmark."""
        headers = [f"{status.capitalize()} Requests"]
        values = [getattr(benchmark.request_totals, status)]

        for metric in GenerativeMetrics.model_fields:
            metric_headers, metric_values = self._get_benchmark_status_metrics_stats(
                benchmark, status, metric
            )
            headers.extend(metric_headers)
            values.extend(metric_values)

        return headers, values

    def _get_benchmark_status_metrics_stats(
        self, benchmark: GenerativeBenchmark, status: str, metric: str
    ) -> tuple[list[str], list[float | list[float]]]:
        """Get statistical metrics for a specific status and metric."""
        status_display = status.capitalize()
        metric_display = metric.replace("_", " ").capitalize()
        status_dist_summary: StatusDistributionSummary = getattr(
            benchmark.metrics, metric
        )
        if not hasattr(status_dist_summary, status):
            return [], []
        dist_summary: DistributionSummary = getattr(status_dist_summary, status)

        headers = [
            f"{status_display} {metric_display} mean",
            f"{status_display} {metric_display} median",
            f"{status_display} {metric_display} std dev",
            (
                f"{status_display} {metric_display} "
                "[min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]"
            ),
        ]
        values: list[float | list[float]] = [
            dist_summary.mean,
            dist_summary.median,
            dist_summary.std_dev,
            [
                dist_summary.min,
                dist_summary.percentiles.p001,
                dist_summary.percentiles.p01,
                dist_summary.percentiles.p05,
                dist_summary.percentiles.p10,
                dist_summary.percentiles.p25,
                dist_summary.percentiles.p75,
                dist_summary.percentiles.p90,
                dist_summary.percentiles.p95,
                dist_summary.percentiles.p99,
                dist_summary.max,
            ],
        ]
        return headers, values

    def _get_benchmark_extras_headers_and_values(
        self,
        benchmark: GenerativeBenchmark,
    ) -> tuple[list[str], list[str]]:
        headers = ["Profile", "Backend", "Generator Data"]
        values: list[str] = [
            benchmark.benchmarker.profile.model_dump_json(),
            json.dumps(benchmark.benchmarker.backend),
            json.dumps(benchmark.benchmarker.requests["data"]),
        ]

        if len(headers) != len(values):
            raise ValueError("Headers and values length mismatch.")

        return headers, values


@GenerativeBenchmarkerOutput.register("html")
class GenerativeBenchmarkerHTML(GenerativeBenchmarkerOutput):
    """HTML output formatter for benchmark results."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.html"

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        new_kwargs = {}
        if output_path is not None:
            new_kwargs["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return new_kwargs

    output_path: Path = Field(default_factory=lambda: Path.cwd())

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as an HTML file.

        :param report: The completed benchmark report.
        :return: Path to the saved HTML file.
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / GenerativeBenchmarkerHTML.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_builder = UIDataBuilder(report.benchmarks)
        data = data_builder.to_dict()
        camel_data = recursive_key_update(deepcopy(data), camelize_str)

        ui_api_data = {}
        for k, v in camel_data.items():
            placeholder_key = f"window.{k} = {{}};"
            replacement_value = f"window.{k} = {json.dumps(v, indent=2)};\n"
            ui_api_data[placeholder_key] = replacement_value

        create_report(ui_api_data, output_path)

        return output_path
