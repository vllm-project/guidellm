from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    SweepProfile,
    ThroughputProfile,
)
from guidellm.benchmark.schemas import (
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
    GenerativeMetrics,
)
from guidellm.presentation import UIDataBuilder
from guidellm.presentation.injector import create_report
from guidellm.settings import settings
from guidellm.utils import (
    Colors,
    DistributionSummary,
    RegistryMixin,
    StatusDistributionSummary,
    camelize_str,
    recursive_key_update,
    safe_format_timestamp,
    split_text_list_by_length,
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


@GenerativeBenchmarkerOutput.register("console")
class GenerativeBenchmarkerConsole(GenerativeBenchmarkerOutput):
    """Console output formatter for benchmark results with rich formatting."""

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
        self._print_benchmarks_metadata(report.benchmarks)
        self._print_benchmarks_info(report.benchmarks)
        self._print_benchmarks_stats(report.benchmarks)

        return "printed to console"

    def _print_benchmarks_metadata(self, benchmarks: list[GenerativeBenchmark]):
        start_time = benchmarks[0].run_stats.start_time
        end_time = benchmarks[-1].run_stats.end_time
        duration = end_time - start_time

        self._print_section_header("Benchmarks Metadata")
        self._print_labeled_line("Run id", str(benchmarks[0].run_id))
        self._print_labeled_line("Duration", f"{duration:.1f} seconds")
        self._print_labeled_line("Profile", self._get_profile_str(benchmarks[0]))

    def _print_benchmarks_info(self, benchmarks: list[GenerativeBenchmark]):
        sections = {
            "Metadata": (0, 3),
            "Requests Made": (4, 6),
            "Prompt Tok/Req": (7, 9),
            "Output Tok/Req": (10, 12),
            "Prompt Tok Total": (13, 15),
            "Output Tok Total": (16, 18),
        }
        headers = [
            "Benchmark",
            "Start Time",
            "End Time",
            "Duration (s)",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
        ]

        rows = []
        for benchmark in benchmarks:
            rows.append(
                [
                    str(benchmark.scheduler.strategy),
                    safe_format_timestamp(benchmark.start_time),
                    safe_format_timestamp(benchmark.end_time),
                    f"{(benchmark.end_time - benchmark.start_time):.1f}",
                    f"{benchmark.request_totals.successful:.0f}",
                    f"{benchmark.request_totals.incomplete:.0f}",
                    f"{benchmark.request_totals.errored:.0f}",
                    f"{benchmark.metrics.prompt_token_count.successful.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.incomplete.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.errored.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.successful.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.incomplete.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.errored.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.successful.total_sum:.0f}",
                    f"{benchmark.metrics.prompt_token_count.incomplete.total_sum:.0f}",
                    f"{benchmark.metrics.prompt_token_count.errored.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.successful.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.incomplete.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.errored.total_sum:.0f}",
                ]
            )

        self._print_table(headers, rows, "Benchmarks Info", sections)

    def _print_benchmarks_stats(self, benchmarks: list[GenerativeBenchmark]):
        sections = {
            "Metadata": (0, 0),
            "Request Stats": (1, 2),
            "Out Tok/sec": (3, 3),
            "Tot Tok/sec": (4, 4),
            "Req Latency (sec)": (5, 7),
            "TTFT (ms)": (8, 10),
            "ITL (ms)": (11, 13),
            "TPOT (ms)": (14, 16),
        }
        headers = [
            "Benchmark",
            "Per Second",
            "Concurrency",
            "mean",
            "mean",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
        ]

        rows = []
        for benchmark in benchmarks:
            rows.append(
                [
                    str(benchmark.scheduler.strategy),
                    f"{benchmark.metrics.requests_per_second.successful.mean:.2f}",
                    f"{benchmark.metrics.request_concurrency.successful.mean:.2f}",
                    f"{benchmark.metrics.output_tokens_per_second.successful.mean:.1f}",
                    f"{benchmark.metrics.tokens_per_second.successful.mean:.1f}",
                    f"{benchmark.metrics.request_latency.successful.mean:.2f}",
                    f"{benchmark.metrics.request_latency.successful.median:.2f}",
                    f"{benchmark.metrics.request_latency.successful.percentiles.p99:.2f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.median:.1f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.median:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.percentiles.p99:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.median:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.percentiles.p99:.1f}",
                ]
            )

        self._print_table(headers, rows, "Benchmarks Stats", sections)

    def _get_profile_str(self, benchmark: GenerativeBenchmark) -> str:
        profile = benchmark.benchmarker.profile
        if profile is None:
            return "None"

        profile_args = OrderedDict(
            {
                "type": profile.type_,
                "strategies": getattr(profile, "strategy_types", []),
            }
        )

        if isinstance(profile, ConcurrentProfile):
            profile_args["streams"] = str(profile.streams)
        elif isinstance(profile, ThroughputProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
        elif isinstance(profile, AsyncProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
            profile_args["rate"] = str(profile.rate)
        elif isinstance(profile, SweepProfile):
            profile_args["sweep_size"] = str(profile.sweep_size)

        return ", ".join(f"{key}={value}" for key, value in profile_args.items())

    def _print_section_header(self, title: str, indent: int = 0, new_lines: int = 2):
        self._print_line(
            f"{title}:",
            f"bold underline {Colors.info}",
            indent=indent,
            new_lines=new_lines,
        )

    def _print_labeled_line(
        self, label: str, value: str, indent: int = 4, new_lines: int = 0
    ):
        self._print_line(
            [label + ":", value],
            ["bold " + Colors.info, "italic"],
            new_lines=new_lines,
            indent=indent,
        )

    def _print_line(
        self,
        value: str | list[str],
        style: str | list[str] = "",
        indent: int = 0,
        new_lines: int = 0,
    ):
        text = Text()
        for _ in range(new_lines):
            text.append("\n")

        if not isinstance(value, list):
            value = [value]
        if not isinstance(style, list):
            style = [style for _ in range(len(value))]

        if len(value) != len(style):
            raise ValueError(
                f"Value and style length mismatch: {len(value)} vs {len(style)}"
            )

        for val, sty in zip(value, style, strict=False):
            text.append(val, style=sty)

        self.console.print(Padding.indent(text, indent))

    def _print_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: str,
        sections: dict[str, tuple[int, int]] | None = None,
        max_char_per_col: int = 1024,
        indent: int = 0,
        new_lines: int = 2,
    ):
        if rows and any(len(row) != len(headers) for row in rows):
            raise ValueError(
                "Headers and rows length mismatch: "
                f"{len(headers)} vs {len(rows[0]) if rows else 'N/A'}"
            )

        max_chars_per_column = self._calculate_max_chars_per_column(
            headers, rows, sections, max_char_per_col
        )

        self._print_section_header(title, indent=indent, new_lines=new_lines)
        self._print_table_divider(max_chars_per_column, False, indent)
        if sections:
            self._print_table_sections(sections, max_chars_per_column, indent)
        self._print_table_row(
            split_text_list_by_length(headers, max_chars_per_column),
            f"bold {Colors.info}",
            indent,
        )
        self._print_table_divider(max_chars_per_column, True, indent)
        for row in rows:
            self._print_table_row(
                split_text_list_by_length(row, max_chars_per_column),
                "italic",
                indent,
            )
        self._print_table_divider(max_chars_per_column, False, indent)

    def _calculate_max_chars_per_column(
        self,
        headers: list[str],
        rows: list[list[Any]],
        sections: dict[str, tuple[int, int]] | None,
        max_char_per_col: int,
    ) -> list[int]:
        """Calculate maximum characters per column for table formatting."""
        max_chars_per_column = []
        for ind in range(len(headers)):
            max_chars_per_column.append(min(len(headers[ind]), max_char_per_col))
            for row in rows:
                max_chars_per_column[ind] = max(
                    max_chars_per_column[ind], len(str(row[ind]))
                )

        if not sections:
            return max_chars_per_column

        for section, (start_col, end_col) in sections.items():
            min_section_len = len(section) + (end_col - start_col)
            chars_in_columns = sum(
                max_chars_per_column[start_col : end_col + 1]
            ) + 2 * (end_col - start_col)
            if min_section_len > chars_in_columns:
                add_chars_per_col = math.ceil(
                    (min_section_len - chars_in_columns) / (end_col - start_col + 1)
                )
                for col in range(start_col, end_col + 1):
                    max_chars_per_column[col] += add_chars_per_col

        return max_chars_per_column

    def _print_table_divider(
        self, max_chars_per_column: list[int], include_separators: bool, indent: int = 0
    ):
        """Print table divider line."""
        if include_separators:
            columns = [
                settings.table_headers_border_char * max_chars
                + settings.table_column_separator_char
                + settings.table_headers_border_char
                for max_chars in max_chars_per_column
            ]
        else:
            columns = [
                settings.table_border_char * (max_chars + 2)
                for max_chars in max_chars_per_column
            ]
        columns[-1] = columns[-1][:-2]
        self._print_line(columns, Colors.info, indent)

    def _print_table_sections(
        self,
        sections: dict[str, tuple[int, int]],
        max_chars_per_column: list[int],
        indent: int = 0,
    ):
        section_tuples = [(start, end, name) for name, (start, end) in sections.items()]
        section_tuples.sort(key=lambda x: x[0])

        if any(start > end for start, end, _ in section_tuples):
            raise ValueError(f"Invalid section ranges: {section_tuples}")

        if (
            any(
                section_tuples[ind][1] + 1 != section_tuples[ind + 1][0]
                for ind in range(len(section_tuples) - 1)
            )
            or section_tuples[0][0] != 0
            or section_tuples[-1][1] != len(max_chars_per_column) - 1
        ):
            raise ValueError(f"Invalid section ranges: {section_tuples}")

        line_values = []
        line_styles = []
        for section, (start_col, end_col) in sections.items():
            section_length = sum(max_chars_per_column[start_col : end_col + 1]) + 2 * (
                end_col - start_col + 1
            )
            num_separators = end_col - start_col
            line_values.extend(
                [
                    section,
                    " " * (section_length - len(section) - num_separators - 2),
                    settings.table_column_separator_char * num_separators,
                    settings.table_column_separator_char + " ",
                ]
            )
            line_styles.extend(["bold " + Colors.info, "", "", Colors.info])

        line_values = line_values[:-1]
        line_styles = line_styles[:-1]
        self._print_line(line_values, line_styles, indent)

    def _print_table_row(
        self, column_lines: list[list[str]], style: str, indent: int = 0
    ):
        for row in range(len(column_lines[0])):
            print_line = []
            print_styles = []
            for column in range(len(column_lines)):
                print_line.extend(
                    [
                        column_lines[column][row],
                        settings.table_column_separator_char,
                        " ",
                    ]
                )
                print_styles.extend([style, Colors.info, ""])
            print_line = print_line[:-2]
            print_styles = print_styles[:-2]
            self._print_line(print_line, print_styles, indent)


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
