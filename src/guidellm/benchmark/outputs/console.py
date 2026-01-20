"""
Console output formatter for generative benchmarker results.

This module provides console-based output formatting for benchmark reports, organizing
metrics into structured tables that display request statistics, latency measurements,
throughput data, and modality-specific metrics (text, image, video, audio). It uses
the Console utility to render multi-column tables with proper alignment and formatting
for terminal display.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.schemas import DistributionSummary, StatusDistributionSummary
from guidellm.utils import Console, safe_format_number, safe_format_timestamp

__all__ = ["GenerativeBenchmarkerConsole"]


StatTypesAlias = Literal["mean", "median", "p95"]


@dataclass
class ConsoleTableColumn:
    """
    Data structure for a single console table column.

    Stores column metadata (group, name, units, type) and accumulated values for
    rendering formatted table output with proper type-specific formatting and precision.

    :cvar group: Optional group header for related columns
    :cvar name: Column name displayed in header
    :cvar units: Optional unit label for numeric values
    :cvar type_: Data type determining formatting (number, text, timestamp)
    :cvar precision: Decimal precision for numeric formatting
    :cvar values: Accumulated values for this column across rows
    """

    group: str | None = None
    name: str | None = None
    units: str | None = None
    type_: Literal["number", "text", "timestamp"] = "number"
    precision: int = 1
    values: list[str | float | int | None] = field(default_factory=list)


class ConsoleTableColumnsCollection(dict[str, ConsoleTableColumn]):
    """
    Collection manager for console table columns.

    Extends dict to provide specialized methods for adding values and statistics to
    columns, automatically creating columns as needed and organizing them by composite
    keys for consistent table rendering.
    """

    def add_value(
        self,
        value: str | float | int | None,
        group: str | None = None,
        name: str | None = None,
        units: str | None = None,
        type_: Literal["number", "text", "timestamp"] = "number",
        precision: int = 1,
    ):
        """
        Add a value to a column, creating the column if it doesn't exist.

        :param value: The value to add to the column
        :param group: Optional group header for the column
        :param name: Column name for display
        :param units: Optional unit label
        :param type_: Data type for formatting
        :param precision: Decimal precision for numbers
        """
        key = f"{group}_{name}_{units}"

        if key not in self:
            self[key] = ConsoleTableColumn(
                group=group, name=name, units=units, type_=type_, precision=precision
            )

        self[key].values.append(value)

    def add_stats(
        self,
        stats: StatusDistributionSummary | None,
        status: Literal["successful", "incomplete", "errored", "total"] = "successful",
        group: str | None = None,
        name: str | None = None,
        precision: int = 1,
        types: Sequence[StatTypesAlias] = ("median", "p95"),
    ):
        """
        Add statistical summary columns (mean and p95) for a metric.

        Creates paired mean/p95 columns automatically and appends values from the
        specified status category of the distribution summary.

        :param stats: Distribution summary containing status-specific statistics
        :param status: Status category to extract statistics from
        :param group: Optional group header for the columns
        :param name: Column name for display
        :param precision: Decimal precision for numbers
        """
        key = f"{group}_{name}"
        status_stats: DistributionSummary | None = (
            getattr(stats, status) if stats else None
        )

        for stat_type in types:
            col_key = f"{key}_{stat_type}"
            col_name, col_value = self._get_stat_type_name_val(stat_type, status_stats)
            if col_key not in self:
                self[col_key] = ConsoleTableColumn(
                    group=group,
                    name=name,
                    units=col_name,
                    precision=precision,
                )
            self[col_key].values.append(col_value)

    def get_table_data(self) -> tuple[list[list[str]], list[list[str]]]:
        """
        Convert column collection to formatted table data.

        Transforms stored columns and values into header and value lists suitable for
        console table rendering, applying type-specific formatting.

        :return: Tuple of (headers, values) where each is a list of column string lists
        """
        headers: list[list[str]] = []
        values: list[list[str]] = []

        for column in self.values():
            headers.append([column.group or "", column.name or "", column.units or ""])
            formatted_values: list[str] = []
            for value in column.values:
                if column.type_ == "text":
                    formatted_values.append(str(value))
                    continue

                if not isinstance(value, float | int) and value is not None:
                    raise ValueError(
                        f"Expected numeric value for column '{column.name}', "
                        f"got: {value}"
                    )

                if column.type_ == "timestamp":
                    formatted_values.append(
                        safe_format_timestamp(cast("float | None", value))
                    )
                elif column.type_ == "number":
                    formatted_values.append(
                        safe_format_number(
                            value,
                            precision=column.precision,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported column type: {column.type_}")
            values.append(formatted_values)

        return headers, values

    @classmethod
    def _get_stat_type_name_val(
        cls, stat_type: StatTypesAlias, stats: DistributionSummary | None
    ) -> tuple[str, float | None]:
        if stat_type == "mean":
            return "Mean", stats.mean if stats else None
        elif stat_type == "median":
            return "Mdn", stats.median if stats else None
        elif stat_type == "p95":
            return "p95", stats.percentiles.p95 if stats else None
        else:
            raise ValueError(f"Unsupported stat type: {stat_type}")


@GenerativeBenchmarkerOutput.register("console")
class GenerativeBenchmarkerConsole(GenerativeBenchmarkerOutput):
    """
    Console output formatter for benchmark reports.

    Renders benchmark results as formatted tables in the terminal, organizing metrics
    by category (run summary, request counts, latency, throughput, modality-specific)
    with proper alignment and type-specific formatting for readability.
    """

    @classmethod
    def validated_kwargs(cls, *_args, **_kwargs) -> dict[str, Any]:
        """
        Validate and return keyword arguments for initialization.

        :return: Empty dict as no additional kwargs are required
        """
        return {}

    console: Console = Field(
        default_factory=Console,
        description="Console utility for rendering formatted tables",
    )

    async def finalize(self, report: GenerativeBenchmarksReport) -> str:
        """
        Print the complete benchmark report to the console.

        Renders all metric tables including run summary, request counts, latency,
        throughput, and modality-specific statistics to the console.

        :param report: The completed benchmark report
        :return: Status message indicating output location
        """
        self.print_run_summary_table(report)
        self.print_text_table(report)
        self.print_image_table(report)
        self.print_video_table(report)
        self.print_audio_table(report)
        self.print_request_counts_table(report)
        self.print_request_latency_table(report)
        self.print_server_throughput_table(report)

        return "printed to console"

    def print_run_summary_table(self, report: GenerativeBenchmarksReport):
        """
        Print the run summary table with timing and token information.

        :param report: The benchmark report containing run metadata
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
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
                benchmark.warmup_duration, group="Timings", name="Warm", units="Sec"
            )
            columns.add_value(
                benchmark.cooldown_duration, group="Timings", name="Cool", units="Sec"
            )

            for token_metrics, group in [
                (benchmark.metrics.prompt_token_count, "Input Tokens"),
                (benchmark.metrics.output_token_count, "Output Tokens"),
            ]:
                columns.add_value(
                    token_metrics.successful.total_sum,
                    group=group,
                    name="Comp",
                    units="Tot",
                )
                columns.add_value(
                    token_metrics.incomplete.total_sum,
                    group=group,
                    name="Inc",
                    units="Tot",
                )
                columns.add_value(
                    token_metrics.errored.total_sum,
                    group=group,
                    name="Err",
                    units="Tot",
                )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Run Summary Info")

    def print_text_table(self, report: GenerativeBenchmarksReport):
        """
        Print text-specific metrics table if any text data exists.

        :param report: The benchmark report containing text metrics
        """
        self._print_modality_table(
            report=report,
            modality="text",
            title="Text Metrics Statistics (Completed Requests)",
            metric_groups=[
                ("tokens", "Tokens"),
                ("words", "Words"),
                ("characters", "Characters"),
            ],
        )

    def print_image_table(self, report: GenerativeBenchmarksReport):
        """
        Print image-specific metrics table if any image data exists.

        :param report: The benchmark report containing image metrics
        """
        self._print_modality_table(
            report=report,
            modality="image",
            title="Image Metrics Statistics (Completed Requests)",
            metric_groups=[
                ("tokens", "Tokens"),
                ("images", "Images"),
                ("pixels", "Pixels"),
                ("bytes", "Bytes"),
            ],
        )

    def print_video_table(self, report: GenerativeBenchmarksReport):
        """
        Print video-specific metrics table if any video data exists.

        :param report: The benchmark report containing video metrics
        """
        self._print_modality_table(
            report=report,
            modality="video",
            title="Video Metrics Statistics (Completed Requests)",
            metric_groups=[
                ("tokens", "Tokens"),
                ("frames", "Frames"),
                ("seconds", "Seconds"),
                ("bytes", "Bytes"),
            ],
        )

    def print_audio_table(self, report: GenerativeBenchmarksReport):
        """
        Print audio-specific metrics table if any audio data exists.

        :param report: The benchmark report containing audio metrics
        """
        self._print_modality_table(
            report=report,
            modality="audio",
            title="Audio Metrics Statistics (Completed Requests)",
            metric_groups=[
                ("tokens", "Tokens"),
                ("samples", "Samples"),
                ("seconds", "Seconds"),
                ("bytes", "Bytes"),
            ],
        )

    def print_request_counts_table(self, report: GenerativeBenchmarksReport):
        """
        Print request token count statistics table.

        :param report: The benchmark report containing request count metrics
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
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

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(
            headers,
            values,
            title="Request Token Statistics (Completed Requests)",
        )

    def print_request_latency_table(self, report: GenerativeBenchmarksReport):
        """
        Print request latency metrics table.

        :param report: The benchmark report containing latency metrics
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
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
            headers,
            values,
            title="Request Latency Statistics (Completed Requests)",
        )

    def print_server_throughput_table(self, report: GenerativeBenchmarksReport):
        """
        Print server throughput metrics table.

        :param report: The benchmark report containing throughput metrics
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )
            columns.add_stats(
                benchmark.metrics.requests_per_second,
                status="total",
                group="Requests",
                name="Per Sec",
                types=("median", "mean"),
            )
            columns.add_stats(
                benchmark.metrics.request_concurrency,
                status="total",
                group="Requests",
                name="Concurrency",
                types=("median", "mean"),
            )
            columns.add_stats(
                benchmark.metrics.prompt_tokens_per_second,
                status="total",
                group="Input Tokens",
                name="Per Sec",
                types=("median", "mean"),
            )
            columns.add_stats(
                benchmark.metrics.output_tokens_per_second,
                status="total",
                group="Output Tokens",
                name="Per Sec",
                types=("median", "mean"),
            )
            columns.add_stats(
                benchmark.metrics.tokens_per_second,
                status="total",
                group="Total Tokens",
                name="Per Sec",
                types=("median", "mean"),
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(
            headers, values, title="Server Throughput Statistics (All Requests)"
        )

    def _print_modality_table(
        self,
        report: GenerativeBenchmarksReport,
        modality: Literal["text", "image", "video", "audio"],
        title: str,
        metric_groups: list[tuple[str, str]],
    ):
        columns: dict[str, ConsoleTableColumnsCollection] = defaultdict(
            ConsoleTableColumnsCollection
        )

        for benchmark in report.benchmarks:
            columns["labels"].add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            modality_metrics = getattr(benchmark.metrics, modality)

            for metric_attr, display_name in metric_groups:
                metric_obj = getattr(modality_metrics, metric_attr, None)
                input_stats: StatusDistributionSummary | None = (
                    getattr(metric_obj, "input", None) if metric_obj else None
                )
                columns[f"{metric_attr}.input"].add_stats(
                    input_stats,
                    group=f"Input {display_name}",
                    name="Per Request",
                )
                input_per_second_stats: StatusDistributionSummary | None = (
                    getattr(metric_obj, "input_per_second", None)
                    if metric_obj
                    else None
                )
                columns[f"{metric_attr}.input"].add_stats(
                    input_per_second_stats,
                    group=f"Input {display_name}",
                    name="Per Second",
                    types=("median", "mean"),
                )
                output_stats: StatusDistributionSummary | None = (
                    getattr(metric_obj, "output", None) if metric_obj else None
                )
                columns[f"{metric_attr}.output"].add_stats(
                    output_stats,
                    group=f"Output {display_name}",
                    name="Per Request",
                )
                output_per_second_stats: StatusDistributionSummary | None = (
                    getattr(metric_obj, "output_per_second", None)
                    if metric_obj
                    else None
                )
                columns[f"{metric_attr}.output"].add_stats(
                    output_per_second_stats,
                    group=f"Output {display_name}",
                    name="Per Second",
                    types=("median", "mean"),
                )

        self._print_inp_out_tables(
            title=title,
            labels=columns["labels"],
            groups=[
                (columns[f"{metric_attr}.input"], columns[f"{metric_attr}.output"])
                for metric_attr, _ in metric_groups
            ],
        )

    def _print_inp_out_tables(
        self,
        title: str,
        labels: ConsoleTableColumnsCollection,
        groups: list[
            tuple[ConsoleTableColumnsCollection, ConsoleTableColumnsCollection]
        ],
    ):
        input_headers, input_values = [], []
        output_headers, output_values = [], []
        input_has_data = False
        output_has_data = False

        for input_columns, output_columns in groups:
            # Check if columns have any non-None values
            type_input_has_data = any(
                any(value is not None for value in column.values)
                for column in input_columns.values()
            )
            type_output_has_data = any(
                any(value is not None for value in column.values)
                for column in output_columns.values()
            )

            if not (type_input_has_data or type_output_has_data):
                continue

            input_has_data = input_has_data or type_input_has_data
            output_has_data = output_has_data or type_output_has_data

            input_type_headers, input_type_columns = input_columns.get_table_data()
            output_type_headers, output_type_columns = output_columns.get_table_data()

            input_headers.extend(input_type_headers)
            input_values.extend(input_type_columns)
            output_headers.extend(output_type_headers)
            output_values.extend(output_type_columns)

        if not (input_has_data or output_has_data):
            return

        labels_headers, labels_values = labels.get_table_data()
        header_cols_groups = []
        value_cols_groups = []

        if input_has_data:
            header_cols_groups.append(labels_headers + input_headers)
            value_cols_groups.append(labels_values + input_values)
        if output_has_data:
            header_cols_groups.append(labels_headers + output_headers)
            value_cols_groups.append(labels_values + output_values)

        if header_cols_groups and value_cols_groups:
            self.console.print("\n")
            self.console.print_tables(
                header_cols_groups=header_cols_groups,
                value_cols_groups=value_cols_groups,
                title=title,
            )
