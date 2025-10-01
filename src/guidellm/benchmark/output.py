import csv
import json
import math
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import Field
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from guidellm.benchmark.benchmark import GenerativeBenchmark, GenerativeMetrics
from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    SweepProfile,
    ThroughputProfile,
)
from guidellm.config import settings
from guidellm.objects import (
    DistributionSummary,
    StandardBaseModel,
    StatusDistributionSummary,
)
from guidellm.presentation import UIDataBuilder
from guidellm.presentation.injector import create_report
from guidellm.scheduler import strategy_display_str
from guidellm.utils import Colors, split_text_list_by_length
from guidellm.utils.dict import recursive_key_update
from guidellm.utils.text import camelize_str

__all__ = [
    "GenerativeBenchmarksConsole",
    "GenerativeBenchmarksReport",
]


class GenerativeBenchmarksReport(StandardBaseModel):
    """
    A pydantic model representing a completed benchmark report.
    Contains a list of benchmarks along with convenience methods for finalizing
    and saving the report.
    """

    @staticmethod
    def load_file(path: Union[str, Path]) -> "GenerativeBenchmarksReport":
        """
        Load a report from a file. The file type is determined by the file extension.
        If the file is a directory, it expects a file named benchmarks.json under the
        directory.

        :param path: The path to load the report from.
        :return: The loaded report.
        """
        path, type_ = GenerativeBenchmarksReport._file_setup(path)

        if type_ == "json":
            with path.open("r") as file:
                model_dict = json.load(file)

            return GenerativeBenchmarksReport.model_validate(model_dict)

        if type_ == "yaml":
            with path.open("r") as file:
                model_dict = yaml.safe_load(file)

            return GenerativeBenchmarksReport.model_validate(model_dict)

        if type_ == "csv":
            raise ValueError(f"CSV file type is not supported for loading: {path}.")

        if type_ == "html":
            raise ValueError(f"HTML file type is not supported for loading: {path}.")

        raise ValueError(f"Unsupported file type: {type_} for {path}.")

    benchmarks: list[GenerativeBenchmark] = Field(
        description="The list of completed benchmarks contained within the report.",
        default_factory=list,
    )

    def set_sample_size(
        self, sample_size: Optional[int]
    ) -> "GenerativeBenchmarksReport":
        """
        Set the sample size for each benchmark in the report. In doing this, it will
        reduce the contained requests of each benchmark to the sample size.
        If sample size is None, it will return the report as is.

        :param sample_size: The sample size to set for each benchmark.
            If None, the report will be returned as is.
        :return: The report with the sample size set for each benchmark.
        """

        if sample_size is not None:
            for benchmark in self.benchmarks:
                benchmark.set_sample_size(sample_size)

        return self

    def save_file(self, path: Union[str, Path]) -> Path:
        """
        Save the report to a file. The file type is determined by the file extension.
        If the file is a directory, it will save the report to a file named
        benchmarks.json under the directory.

        :param path: The path to save the report to.
        :return: The path to the saved report.
        """
        path, type_ = GenerativeBenchmarksReport._file_setup(path)

        if type_ == "json":
            return self.save_json(path)

        if type_ == "yaml":
            return self.save_yaml(path)

        if type_ == "csv":
            return self.save_csv(path)

        if type_ == "html":
            return self.save_html(path)

        raise ValueError(f"Unsupported file type: {type_} for {path}.")

    def save_json(self, path: Union[str, Path]) -> Path:
        """
        Save the report to a JSON file containing all of the report data which is
        reloadable using the pydantic model. If the file is a directory, it will save
        the report to a file named benchmarks.json under the directory.

        :param path: The path to save the report to.
        :return: The path to the saved report.
        """
        path, type_ = GenerativeBenchmarksReport._file_setup(path, "json")

        if type_ != "json":
            raise ValueError(
                f"Unsupported file type for saving a JSON: {type_} for {path}."
            )

        model_dict = self.model_dump()

        with path.open("w", encoding="utf-8") as file:
            json.dump(model_dict, file, ensure_ascii=False, indent=4)

        return path

    def save_yaml(self, path: Union[str, Path]) -> Path:
        """
        Save the report to a YAML file containing all of the report data which is
        reloadable using the pydantic model. If the file is a directory, it will save
        the report to a file named benchmarks.yaml under the directory.

        :param path: The path to save the report to.
        :return: The path to the saved report.
        """

        path, type_ = GenerativeBenchmarksReport._file_setup(path, "yaml")

        if type_ != "yaml":
            raise ValueError(
                f"Unsupported file type for saving a YAML: {type_} for {path}."
            )

        model_dict = self.model_dump()
        model_yaml = yaml.dump(model_dict)

        with path.open("w") as file:
            file.write(model_yaml)

        return path

    def save_csv(self, path: Union[str, Path]) -> Path:
        """
        Save the report to a CSV file containing the summarized statistics and values
        for each report. Note, this data is not reloadable using the pydantic model.
        If the file is a directory, it will save the report to a file named
        benchmarks.csv under the directory.

        :param path: The path to save the report to.
        :return: The path to the saved report.
        """
        path, type_ = GenerativeBenchmarksReport._file_setup(path, "csv")

        if type_ != "csv":
            raise ValueError(
                f"Unsupported file type for saving a CSV: {type_} for {path}."
            )

        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            headers: list[str] = []
            rows: list[list[Union[str, float, list[float]]]] = []

            for benchmark in self.benchmarks:
                benchmark_headers: list[str] = []
                benchmark_values: list[Union[str, float, list[float]]] = []

                desc_headers, desc_values = self._benchmark_desc_headers_and_values(
                    benchmark
                )
                benchmark_headers += desc_headers
                benchmark_values += desc_values

                for status in StatusDistributionSummary.model_fields:
                    status_headers, status_values = (
                        self._benchmark_status_headers_and_values(benchmark, status)
                    )
                    benchmark_headers += status_headers
                    benchmark_values += status_values

                benchmark_extra_headers, benchmark_extra_values = (
                    self._benchmark_extras_headers_and_values(benchmark)
                )
                benchmark_headers += benchmark_extra_headers
                benchmark_values += benchmark_extra_values

                if not headers:
                    headers = benchmark_headers
                rows.append(benchmark_values)

            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)

        return path

    def save_html(self, path: Union[str, Path]) -> Path:
        """
        Download html, inject report data and save to a file.

        :param path: The path to create the report at.
        :return: The path to the report.
        """

        data_builder = UIDataBuilder(self.benchmarks)
        data = data_builder.to_dict()
        camel_data = recursive_key_update(deepcopy(data), camelize_str)
        ui_api_data = {}
        for k, v in camel_data.items():
            key = f"window.{k} = {{}};"
            value = f"window.{k} = {json.dumps(v, indent=2)};\n"
            ui_api_data[key] = value
        return create_report(ui_api_data, path)

    @staticmethod
    def _file_setup(
        path: Union[str, Path],
        default_file_type: Literal["json", "yaml", "csv", "html"] = "json",
    ) -> tuple[Path, Literal["json", "yaml", "csv", "html"]]:
        path = Path(path) if not isinstance(path, Path) else path

        if path.is_dir():
            path = path / f"benchmarks.{default_file_type}"

        path.parent.mkdir(parents=True, exist_ok=True)
        path_suffix = path.suffix.lower()

        if path_suffix == ".json":
            return path, "json"

        if path_suffix in [".yaml", ".yml"]:
            return path, "yaml"

        if path_suffix in [".csv"]:
            return path, "csv"

        if path_suffix in [".html"]:
            return path, "html"

        raise ValueError(
            f"Unsupported file extension: {path_suffix} for {path}; "
            "expected json, yaml, csv, or html."
        )

    @staticmethod
    def _benchmark_desc_headers_and_values(
        benchmark: GenerativeBenchmark,
    ) -> tuple[list[str], list[Union[str, float]]]:
        headers = [
            "Type",
            "Run Id",
            "Id",
            "Name",
            "Start Time",
            "End Time",
            "Duration",
        ]
        values: list[Union[str, float]] = [
            benchmark.type_,
            benchmark.run_id,
            benchmark.id_,
            strategy_display_str(benchmark.args.strategy),
            datetime.fromtimestamp(benchmark.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(benchmark.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            benchmark.duration,
        ]

        if len(headers) != len(values):
            raise ValueError("Headers and values length mismatch.")

        return headers, values

    @staticmethod
    def _benchmark_extras_headers_and_values(
        benchmark: GenerativeBenchmark,
    ) -> tuple[list[str], list[str]]:
        headers = ["Args", "Worker", "Request Loader", "Extras"]
        values: list[str] = [
            json.dumps(benchmark.args.model_dump()),
            json.dumps(benchmark.worker.model_dump()),
            json.dumps(benchmark.request_loader.model_dump()),
            json.dumps(benchmark.extras),
        ]

        if len(headers) != len(values):
            raise ValueError("Headers and values length mismatch.")

        return headers, values

    @staticmethod
    def _benchmark_status_headers_and_values(
        benchmark: GenerativeBenchmark, status: str
    ) -> tuple[list[str], list[Union[float, list[float]]]]:
        headers = [
            f"{status.capitalize()} Requests",
        ]
        values = [
            getattr(benchmark.request_totals, status),
        ]

        for metric in GenerativeMetrics.model_fields:
            metric_headers, metric_values = (
                GenerativeBenchmarksReport._benchmark_status_metrics_stats(
                    benchmark, status, metric
                )
            )
            headers += metric_headers
            values += metric_values

        if len(headers) != len(values):
            raise ValueError("Headers and values length mismatch.")

        return headers, values

    @staticmethod
    def _benchmark_status_metrics_stats(
        benchmark: GenerativeBenchmark,
        status: str,
        metric: str,
    ) -> tuple[list[str], list[Union[float, list[float]]]]:
        status_display = status.capitalize()
        metric_display = metric.replace("_", " ").capitalize()
        status_dist_summary: StatusDistributionSummary = getattr(
            benchmark.metrics, metric
        )
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
        values: list[Union[float, list[float]]] = [
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

        if len(headers) != len(values):
            raise ValueError("Headers and values length mismatch.")

        return headers, values


class GenerativeBenchmarksConsole:
    """
    A class for outputting progress and benchmark results to the console.
    Utilizes the rich library for formatting, enabling colored and styled output.
    """

    def __init__(self, enabled: bool = True):
        """
        :param enabled: Whether to enable console output. Defaults to True.
            If False, all console output will be suppressed.
        """
        self.enabled = enabled
        self.benchmarks: Optional[list[GenerativeBenchmark]] = None
        self.console = Console()

    @property
    def benchmarks_profile_str(self) -> str:
        """
        :return: A string representation of the profile used for the benchmarks.
        """
        profile = self.benchmarks[0].args.profile if self.benchmarks else None

        if profile is None:
            return "None"

        profile_args = OrderedDict(
            {
                "type": profile.type_,
                "strategies": profile.strategy_types,
            }
        )

        if isinstance(profile, ConcurrentProfile):
            profile_args["streams"] = str(profile.streams)
        elif isinstance(profile, ThroughputProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
        elif isinstance(profile, AsyncProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
            profile_args["rate"] = str(profile.rate)
            profile_args["initial_burst"] = str(profile.initial_burst)
        elif isinstance(profile, SweepProfile):
            profile_args["sweep_size"] = str(profile.sweep_size)

        return ", ".join(f"{key}={value}" for key, value in profile_args.items())

    @property
    def benchmarks_args_str(self) -> str:
        """
        :return: A string representation of the arguments used for the benchmarks.
        """
        args = self.benchmarks[0].args if self.benchmarks else None

        if args is None:
            return "None"

        args_dict = OrderedDict(
            {
                "max_number": args.max_number,
                "max_duration": args.max_duration,
                "warmup_number": args.warmup_number,
                "warmup_duration": args.warmup_duration,
                "cooldown_number": args.cooldown_number,
                "cooldown_duration": args.cooldown_duration,
            }
        )

        return ", ".join(f"{key}={value}" for key, value in args_dict.items())

    @property
    def benchmarks_worker_desc_str(self) -> str:
        """
        :return: A string representation of the worker used for the benchmarks.
        """
        return str(self.benchmarks[0].worker) if self.benchmarks else "None"

    @property
    def benchmarks_request_loader_desc_str(self) -> str:
        """
        :return: A string representation of the request loader used for the benchmarks.
        """
        return str(self.benchmarks[0].request_loader) if self.benchmarks else "None"

    @property
    def benchmarks_extras_str(self) -> str:
        """
        :return: A string representation of the extras used for the benchmarks.
        """
        extras = self.benchmarks[0].extras if self.benchmarks else None

        if not extras:
            return "None"

        return ", ".join(f"{key}={value}" for key, value in extras.items())

    def print_section_header(self, title: str, indent: int = 0, new_lines: int = 2):
        """
        Print out a styled section header to the console.
        The title is underlined, bolded, and colored with the INFO color.

        :param title: The title of the section.
        :param indent: The number of spaces to indent the title.
            Defaults to 0.
        :param new_lines: The number of new lines to print before the title.
            Defaults to 2.
        """
        self.print_line(
            value=f"{title}:",
            style=f"bold underline {Colors.INFO}",
            indent=indent,
            new_lines=new_lines,
        )

    def print_labeled_line(
        self, label: str, value: str, indent: int = 4, new_lines: int = 0
    ):
        """
        Print out a styled, labeled line (label: value) to the console.
        The label is bolded and colored with the INFO color,
        and the value is italicized.

        :param label: The label of the line.
        :param value: The value of the line.
        :param indent: The number of spaces to indent the line.
            Defaults to 4.
        :param new_lines: The number of new lines to print before the line.
            Defaults to 0.
        """
        self.print_line(
            value=[label + ":", value],
            style=["bold " + Colors.INFO, "italic"],
            new_lines=new_lines,
            indent=indent,
        )

    def print_line(
        self,
        value: Union[str, list[str]],
        style: Union[str, list[str]] = "",
        indent: int = 0,
        new_lines: int = 0,
    ):
        """
        Print out a a value to the console as a line with optional indentation.

        :param value: The value to print.
        :param style: The style to apply to the value.
            Defaults to none.
        :param indent: The number of spaces to indent the line.
            Defaults to 0.
        :param new_lines: The number of new lines to print before the value.
            Defaults to 0.
        """
        if not self.enabled:
            return

        text = Text()

        for _ in range(new_lines):
            text.append("\n")

        if not isinstance(value, list):
            value = [value]

        if not isinstance(style, list):
            style = [style for _ in range(len(value))]

        if len(value) != len(style):
            raise ValueError(
                f"Value and style length mismatch. Value length: {len(value)}, "
                f"Style length: {len(style)}."
            )

        for val, sty in zip(value, style):
            text.append(val, style=sty)

        self.console.print(Padding.indent(text, indent))

    def print_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: str,
        sections: Optional[dict[str, tuple[int, int]]] = None,
        max_char_per_col: int = 2**10,
        indent: int = 0,
        new_lines: int = 2,
    ):
        """
        Print a table to the console with the given headers and rows.

        :param headers: The headers of the table.
        :param rows: The rows of the table.
        :param title: The title of the table.
        :param sections: The sections of the table grouping columns together.
            This is a mapping of the section display name to a tuple of the start and
            end column indices. If None, no sections are added (default).
        :param max_char_per_col: The maximum number of characters per column.
        :param indent: The number of spaces to indent the table.
            Defaults to 0.
        :param new_lines: The number of new lines to print before the table.
            Defaults to 0.
        """

        if rows and any(len(row) != len(headers) for row in rows):
            raise ValueError(
                f"Headers and rows length mismatch. Headers length: {len(headers)}, "
                f"Row length: {len(rows[0]) if rows else 'N/A'}."
            )

        max_characters_per_column = self.calculate_max_chars_per_column(
            headers, rows, sections, max_char_per_col
        )

        self.print_section_header(title, indent=indent, new_lines=new_lines)
        self.print_table_divider(
            max_characters_per_column, include_separators=False, indent=indent
        )
        if sections:
            self.print_table_sections(
                sections, max_characters_per_column, indent=indent
            )
        self.print_table_row(
            split_text_list_by_length(headers, max_characters_per_column),
            style=f"bold {Colors.INFO}",
            indent=indent,
        )
        self.print_table_divider(
            max_characters_per_column, include_separators=True, indent=indent
        )
        for row in rows:
            self.print_table_row(
                split_text_list_by_length(row, max_characters_per_column),
                style="italic",
                indent=indent,
            )
        self.print_table_divider(
            max_characters_per_column, include_separators=False, indent=indent
        )

    def calculate_max_chars_per_column(
        self,
        headers: list[str],
        rows: list[list[Any]],
        sections: Optional[dict[str, tuple[int, int]]],
        max_char_per_col: int,
    ) -> list[int]:
        """
        Calculate the maximum number of characters per column in the table.
        This is done by checking the length of the headers, rows, and optional sections
        to ensure all columns are accounted for and spaced correctly.

        :param headers: The headers of the table.
        :param rows: The rows of the table.
        :param sections: The sections of the table grouping columns together.
            This is a mapping of the section display name to a tuple of the start and
            end column indices. If None, no sections are added (default).
        :param max_char_per_col: The maximum number of characters per column.
        :return: A list of the maximum number of characters per column.
        """
        max_characters_per_column = []
        for ind in range(len(headers)):
            max_characters_per_column.append(min(len(headers[ind]), max_char_per_col))

            for row in rows:
                max_characters_per_column[ind] = max(
                    max_characters_per_column[ind], len(str(row[ind]))
                )

        if not sections:
            return max_characters_per_column

        for section in sections:
            start_col, end_col = sections[section]
            min_section_len = len(section) + (
                end_col - start_col
            )  # ensure we have enough space for separators
            chars_in_columns = sum(
                max_characters_per_column[start_col : end_col + 1]
            ) + 2 * (end_col - start_col)
            if min_section_len > chars_in_columns:
                add_chars_per_col = math.ceil(
                    (min_section_len - chars_in_columns) / (end_col - start_col + 1)
                )
                for col in range(start_col, end_col + 1):
                    max_characters_per_column[col] += add_chars_per_col

        return max_characters_per_column

    def print_table_divider(
        self, max_chars_per_column: list[int], include_separators: bool, indent: int = 0
    ):
        """
        Print a divider line for the table (top and bottom of table with '=' characters)

        :param max_chars_per_column: The maximum number of characters per column.
        :param include_separators: Whether to include separators between columns.
        :param indent: The number of spaces to indent the line.
            Defaults to 0.
        """
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
        self.print_line(value=columns, style=Colors.INFO, indent=indent)

    def print_table_sections(
        self,
        sections: dict[str, tuple[int, int]],
        max_chars_per_column: list[int],
        indent: int = 0,
    ):
        """
        Print the sections of the table with corresponding separators to the columns
        the sections are mapped to to ensure it is compliant with a CSV format.
        For example, a section named "Metadata" with columns 0-3 will print this:
        Metadata               ,,,,
        Where the spaces plus the separators at the end will span the columns 0-3.
        All columns must be accounted for in the sections.

        :param sections: The sections of the table.
        :param max_chars_per_column: The maximum number of characters per column.
        :param indent: The number of spaces to indent the line.
            Defaults to 0.
        """
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
            line_values.append(section)
            line_styles.append("bold " + Colors.INFO)
            line_values.append(
                " " * (section_length - len(section) - num_separators - 2)
            )
            line_styles.append("")
            line_values.append(settings.table_column_separator_char * num_separators)
            line_styles.append("")
            line_values.append(settings.table_column_separator_char + " ")
            line_styles.append(Colors.INFO)
        line_values = line_values[:-1]
        line_styles = line_styles[:-1]
        self.print_line(value=line_values, style=line_styles, indent=indent)

    def print_table_row(
        self, column_lines: list[list[str]], style: str, indent: int = 0
    ):
        """
        Print a single row of a table to the console.

        :param column_lines: The lines of text to print for each column.
        :param indent: The number of spaces to indent the line.
            Defaults to 0.
        """
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
                print_styles.extend([style, Colors.INFO, ""])
            print_line = print_line[:-2]
            print_styles = print_styles[:-2]
            self.print_line(value=print_line, style=print_styles, indent=indent)

    def print_benchmarks_metadata(self):
        """
        Print out the metadata of the benchmarks to the console including the run id,
        duration, profile, args, worker, request loader, and extras.
        """

        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print metadata for. Please set benchmarks first."
            )

        start_time = self.benchmarks[0].run_stats.start_time
        end_time = self.benchmarks[-1].run_stats.end_time
        duration = end_time - start_time

        self.print_section_header(title="Benchmarks Metadata")
        self.print_labeled_line(
            label="Run id",
            value=str(self.benchmarks[0].run_id),
        )
        self.print_labeled_line(
            label="Duration",
            value=f"{duration:.1f} seconds",
        )
        self.print_labeled_line(
            label="Profile",
            value=self.benchmarks_profile_str,
        )
        self.print_labeled_line(
            label="Args",
            value=self.benchmarks_args_str,
        )
        self.print_labeled_line(
            label="Worker",
            value=self.benchmarks_worker_desc_str,
        )
        self.print_labeled_line(
            label="Request Loader",
            value=self.benchmarks_request_loader_desc_str,
        )
        self.print_labeled_line(
            label="Extras",
            value=self.benchmarks_extras_str,
        )

    def print_benchmarks_info(self):
        """
        Print out the benchmark information to the console including the start time,
        end time, duration, request totals, and token totals for each benchmark.
        """
        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print info for. Please set benchmarks first."
            )

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

        for benchmark in self.benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.args.strategy),
                    f"{datetime.fromtimestamp(benchmark.start_time).strftime('%H:%M:%S')}",
                    f"{datetime.fromtimestamp(benchmark.end_time).strftime('%H:%M:%S')}",
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

        self.print_table(
            headers=headers, rows=rows, title="Benchmarks Info", sections=sections
        )

    def print_benchmarks_stats(self):
        """
        Print out the benchmark statistics to the console including the requests per
        second, request concurrency, output tokens per second, total tokens per second,
        request latency, time to first token, inter token latency, and time per output
        token for each benchmark.
        """
        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print stats for. Please set benchmarks first."
            )

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

        for benchmark in self.benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.args.strategy),
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

        self.print_table(
            headers=headers,
            rows=rows,
            title="Benchmarks Stats",
            sections=sections,
        )

    def print_full_report(self):
        """
        Print out the benchmark statistics to the console.
        Temporarily enables the console if it's disabled.

        Format:
        - Metadata
        - Info
        - Stats
        """
        orig_enabled = self.enabled
        self.enabled = True
        self.print_benchmarks_metadata()
        self.print_benchmarks_info()
        self.print_benchmarks_stats()
        self.enabled = orig_enabled
