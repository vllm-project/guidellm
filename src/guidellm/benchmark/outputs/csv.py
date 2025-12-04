"""
CSV output formatter for benchmark results.

This module provides the GenerativeBenchmarkerCSV class which exports benchmark
reports to CSV format with comprehensive metrics including timing, throughput,
latency, modality data, and scheduler information. The CSV output uses multi-row
headers to organize metrics hierarchically and includes both summary statistics
and distribution percentiles.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.schemas import DistributionSummary, StatusDistributionSummary
from guidellm.utils import safe_format_timestamp

__all__ = ["GenerativeBenchmarkerCSV"]

TIMESTAMP_FORMAT: Annotated[str, "Format string for timestamp output in CSV files"] = (
    "%Y-%m-%d %H:%M:%S"
)
MODALITY_METRICS: Annotated[
    dict[str, list[tuple[str, str]]],
    "Mapping of modality types to their metric names and display labels",
] = {
    "text": [
        ("tokens", "Tokens"),
        ("words", "Words"),
        ("characters", "Characters"),
    ],
    "image": [
        ("tokens", "Tokens"),
        ("images", "Images"),
        ("pixels", "Pixels"),
        ("bytes", "Bytes"),
    ],
    "video": [
        ("tokens", "Tokens"),
        ("frames", "Frames"),
        ("seconds", "Seconds"),
        ("bytes", "Bytes"),
    ],
    "audio": [
        ("tokens", "Tokens"),
        ("samples", "Samples"),
        ("seconds", "Seconds"),
        ("bytes", "Bytes"),
    ],
}


@GenerativeBenchmarkerOutput.register("csv")
class GenerativeBenchmarkerCSV(GenerativeBenchmarkerOutput):
    """
    CSV output formatter for benchmark results.

    Exports comprehensive benchmark data to CSV format with multi-row headers
    organizing metrics into categories including run information, timing, request
    counts, latency, throughput, modality-specific data, and scheduler state. Each
    benchmark run becomes a row with statistical distributions represented as
    mean, median, standard deviation, and percentiles.

    :cvar DEFAULT_FILE: Default filename for CSV output
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.csv"

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate and normalize constructor keyword arguments.

        :param output_path: Path for CSV output file or directory
        :param _kwargs: Additional keyword arguments (ignored)
        :return: Normalized keyword arguments dictionary
        """
        new_kwargs = {}
        if output_path is not None:
            new_kwargs["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return new_kwargs

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description=(
            "Path where the CSV file will be saved, defaults to current directory"
        ),
    )

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as a CSV file.

        :param report: The completed benchmark report
        :return: Path to the saved CSV file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / GenerativeBenchmarkerCSV.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            headers: list[list[str]] = []
            rows: list[list[str | int | float]] = []

            for benchmark in report.benchmarks:
                benchmark_headers: list[list[str]] = []
                benchmark_values: list[str | int | float] = []

                self._add_run_info(benchmark, benchmark_headers, benchmark_values)
                self._add_benchmark_info(benchmark, benchmark_headers, benchmark_values)
                self._add_timing_info(benchmark, benchmark_headers, benchmark_values)
                self._add_request_counts(benchmark, benchmark_headers, benchmark_values)
                self._add_request_latency_metrics(
                    benchmark, benchmark_headers, benchmark_values
                )
                self._add_server_throughput_metrics(
                    benchmark, benchmark_headers, benchmark_values
                )
                for modality_name in ["text", "image", "video", "audio"]:
                    self._add_modality_metrics(
                        benchmark,
                        modality_name,  # type: ignore[arg-type]
                        benchmark_headers,
                        benchmark_values,
                    )
                self._add_scheduler_info(benchmark, benchmark_headers, benchmark_values)
                self._add_runtime_info(report, benchmark_headers, benchmark_values)

                if not headers:
                    headers = benchmark_headers
                rows.append(benchmark_values)

            self._write_multirow_header(writer, headers)
            for row in rows:
                writer.writerow(row)

        return output_path

    def _write_multirow_header(self, writer: Any, headers: list[list[str]]) -> None:
        """
        Write multi-row header to CSV for hierarchical metric organization.

        :param writer: CSV writer instance
        :param headers: List of column header hierarchies as string lists
        """
        max_rows = max((len(col) for col in headers), default=0)
        for row_idx in range(max_rows):
            row = [col[row_idx] if row_idx < len(col) else "" for col in headers]
            writer.writerow(row)

    def _add_field(
        self,
        headers: list[list[str]],
        values: list[str | int | float],
        group: str,
        field_name: str,
        value: Any,
        units: str = "",
    ) -> None:
        """
        Add a single field to headers and values lists.

        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        :param group: Top-level category for the field
        :param field_name: Name of the field
        :param value: Value for the field
        :param units: Optional units for the field
        """
        headers.append([group, field_name, units])
        values.append(value)

    def _add_runtime_info(
        self,
        report: GenerativeBenchmarksReport,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add global metadata and environment information.

        :param report: Benchmark report to extract global info from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_field(
            headers,
            values,
            "Runtime Info",
            "Metadata",
            report.metadata.model_dump_json(),
        )
        self._add_field(
            headers,
            values,
            "Runtime Info",
            "Arguments",
            report.args.model_dump_json(),
        )

    def _add_run_info(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add overall run identification and configuration information.

        :param benchmark: Benchmark data to extract run info from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_field(headers, values, "Run Info", "Run ID", benchmark.config.run_id)
        self._add_field(
            headers, values, "Run Info", "Run Index", benchmark.config.run_index
        )
        self._add_field(
            headers,
            values,
            "Run Info",
            "Profile",
            benchmark.config.profile.model_dump_json(),
        )
        self._add_field(
            headers,
            values,
            "Run Info",
            "Requests",
            json.dumps(benchmark.config.requests),
        )
        self._add_field(
            headers, values, "Run Info", "Backend", json.dumps(benchmark.config.backend)
        )
        self._add_field(
            headers,
            values,
            "Run Info",
            "Environment",
            json.dumps(benchmark.config.environment),
        )

    def _add_benchmark_info(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add individual benchmark configuration details.

        :param benchmark: Benchmark data to extract configuration from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_field(headers, values, "Benchmark", "Type", benchmark.type_)
        self._add_field(headers, values, "Benchmark", "ID", benchmark.config.id_)
        self._add_field(
            headers, values, "Benchmark", "Strategy", benchmark.config.strategy.type_
        )
        self._add_field(
            headers,
            values,
            "Benchmark",
            "Constraints",
            json.dumps(benchmark.config.constraints),
        )

    def _add_timing_info(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add timing information including start, end, duration, warmup, and cooldown.

        :param benchmark: Benchmark data to extract timing from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        timing_fields: list[tuple[str, Any]] = [
            ("Start Time", benchmark.scheduler_metrics.start_time),
            ("Request Start Time", benchmark.scheduler_metrics.request_start_time),
            ("Measure Start Time", benchmark.scheduler_metrics.measure_start_time),
            ("Measure End Time", benchmark.scheduler_metrics.measure_end_time),
            ("Request End Time", benchmark.scheduler_metrics.request_end_time),
            ("End Time", benchmark.scheduler_metrics.end_time),
        ]
        for field_name, timestamp in timing_fields:
            self._add_field(
                headers,
                values,
                "Timings",
                field_name,
                safe_format_timestamp(timestamp, TIMESTAMP_FORMAT),
            )

        duration_fields: list[tuple[str, float | str]] = [
            ("Duration", benchmark.duration),
            ("Warmup", benchmark.warmup_duration),
            ("Cooldown", benchmark.cooldown_duration),
        ]
        for field_name, duration_value in duration_fields:
            self._add_field(
                headers, values, "Timings", field_name, duration_value, "Sec"
            )

    def _add_request_counts(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add request count totals by status.

        :param benchmark: Benchmark data to extract request counts from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        for status in ["successful", "incomplete", "errored", "total"]:
            self._add_field(
                headers,
                values,
                "Request Counts",
                status.capitalize(),
                getattr(benchmark.metrics.request_totals, status),
            )

    def _add_request_latency_metrics(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add request latency and streaming metrics.

        :param benchmark: Benchmark data to extract latency metrics from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_stats_for_metric(
            headers, values, benchmark.metrics.request_latency, "Request Latency", "Sec"
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.request_streaming_iterations_count,
            "Streaming Iterations",
            "Count",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.time_to_first_token_ms,
            "Time to First Token",
            "ms",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.time_per_output_token_ms,
            "Time per Output Token",
            "ms",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.inter_token_latency_ms,
            "Inter Token Latency",
            "ms",
        )

    def _add_server_throughput_metrics(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add server throughput metrics including requests, tokens, and concurrency.

        :param benchmark: Benchmark data to extract throughput metrics from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.requests_per_second,
            "Server Throughput",
            "Requests/Sec",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.request_concurrency,
            "Server Throughput",
            "Concurrency",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.prompt_token_count,
            "Token Metrics",
            "Input Tokens",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.output_token_count,
            "Token Metrics",
            "Output Tokens",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.total_token_count,
            "Token Metrics",
            "Total Tokens",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.prompt_tokens_per_second,
            "Token Throughput",
            "Input Tokens/Sec",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.output_tokens_per_second,
            "Token Throughput",
            "Output Tokens/Sec",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.tokens_per_second,
            "Token Throughput",
            "Total Tokens/Sec",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.output_tokens_per_iteration,
            "Token Streaming",
            "Output Tokens/Iter",
        )
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.iter_tokens_per_iteration,
            "Token Streaming",
            "Iter Tokens/Iter",
        )

    def _add_modality_metrics(
        self,
        benchmark: GenerativeBenchmark,
        modality: Literal["text", "image", "video", "audio"],
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add modality-specific metrics for text, image, video, or audio data.

        :param benchmark: Benchmark data to extract modality metrics from
        :param modality: Type of modality to extract metrics for
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        modality_summary = getattr(benchmark.metrics, modality)
        metric_definitions = MODALITY_METRICS[modality]

        for metric_name, display_name in metric_definitions:
            metric_obj = getattr(modality_summary, metric_name, None)
            if metric_obj is None:
                continue

            for io_type in ["input", "output", "total"]:
                dist_summary = getattr(metric_obj, io_type, None)
                if dist_summary is None:
                    continue

                if not self._has_distribution_data(dist_summary):
                    continue

                self._add_stats_for_metric(
                    headers,
                    values,
                    dist_summary,
                    f"{modality.capitalize()} {display_name}",
                    io_type.capitalize(),
                )

    def _has_distribution_data(self, dist_summary: StatusDistributionSummary) -> bool:
        """
        Check if distribution summary contains any data.

        :param dist_summary: Distribution summary to check
        :return: True if summary contains data, False otherwise
        """
        return any(
            getattr(dist_summary, status, None) is not None
            and getattr(dist_summary, status).total_sum > 0.0
            for status in ["successful", "incomplete", "errored"]
        )

    def _add_scheduler_info(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add scheduler state and performance information.

        :param benchmark: Benchmark data to extract scheduler info from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        self._add_scheduler_state(benchmark, headers, values)
        self._add_scheduler_metrics(benchmark, headers, values)

    def _add_scheduler_state(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add scheduler state information including request counts and timing.

        :param benchmark: Benchmark data to extract scheduler state from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        state = benchmark.scheduler_state

        state_fields: list[tuple[str, Any]] = [
            ("Node ID", state.node_id),
            ("Num Processes", state.num_processes),
            ("Created Requests", state.created_requests),
            ("Processed Requests", state.processed_requests),
            ("Successful Requests", state.successful_requests),
            ("Errored Requests", state.errored_requests),
            ("Cancelled Requests", state.cancelled_requests),
        ]

        for field_name, value in state_fields:
            self._add_field(headers, values, "Scheduler State", field_name, value)

        if state.end_queuing_time:
            self._add_field(
                headers,
                values,
                "Scheduler State",
                "End Queuing Time",
                safe_format_timestamp(state.end_queuing_time, TIMESTAMP_FORMAT),
            )
            end_queuing_constraints_dict = {
                key: constraint.model_dump()
                for key, constraint in state.end_queuing_constraints.items()
            }
            self._add_field(
                headers,
                values,
                "Scheduler State",
                "End Queuing Constraints",
                json.dumps(end_queuing_constraints_dict),
            )

        if state.end_processing_time:
            self._add_field(
                headers,
                values,
                "Scheduler State",
                "End Processing Time",
                safe_format_timestamp(state.end_processing_time, TIMESTAMP_FORMAT),
            )
            end_processing_constraints_dict = {
                key: constraint.model_dump()
                for key, constraint in state.end_processing_constraints.items()
            }
            self._add_field(
                headers,
                values,
                "Scheduler State",
                "End Processing Constraints",
                json.dumps(end_processing_constraints_dict),
            )

    def _add_scheduler_metrics(
        self,
        benchmark: GenerativeBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """
        Add scheduler performance metrics including delays and processing times.

        :param benchmark: Benchmark data to extract scheduler metrics from
        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        """
        metrics = benchmark.scheduler_metrics

        requests_made_fields: list[tuple[str, int]] = [
            ("Requests Made Successful", metrics.requests_made.successful),
            ("Requests Made Incomplete", metrics.requests_made.incomplete),
            ("Requests Made Errored", metrics.requests_made.errored),
            ("Requests Made Total", metrics.requests_made.total),
        ]
        for field_name, value in requests_made_fields:
            self._add_field(headers, values, "Scheduler Metrics", field_name, value)

        timing_metrics: list[tuple[str, float]] = [
            ("Queued Time Avg", metrics.queued_time_avg),
            ("Resolve Start Delay Avg", metrics.resolve_start_delay_avg),
            (
                "Resolve Targeted Start Delay Avg",
                metrics.resolve_targeted_start_delay_avg,
            ),
            ("Request Start Delay Avg", metrics.request_start_delay_avg),
            (
                "Request Targeted Start Delay Avg",
                metrics.request_targeted_start_delay_avg,
            ),
            ("Request Time Avg", metrics.request_time_avg),
            ("Resolve End Delay Avg", metrics.resolve_end_delay_avg),
            ("Resolve Time Avg", metrics.resolve_time_avg),
            ("Finalized Delay Avg", metrics.finalized_delay_avg),
            ("Processed Delay Avg", metrics.processed_delay_avg),
        ]
        for field_name, timing in timing_metrics:
            self._add_field(
                headers, values, "Scheduler Metrics", field_name, timing, "Sec"
            )

    def _add_stats_for_metric(
        self,
        headers: list[list[str]],
        values: list[str | int | float],
        metric: StatusDistributionSummary | DistributionSummary,
        group: str,
        units: str,
    ) -> None:
        """
        Add statistical summaries for a metric across all statuses.

        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        :param metric: Distribution summary to extract statistics from
        :param group: Top-level category for the metric
        :param units: Units for the metric values
        """
        if isinstance(metric, StatusDistributionSummary):
            for status in ["successful", "incomplete", "errored"]:
                dist = getattr(metric, status, None)
                if dist is None or dist.total_sum == 0.0:
                    continue
                self._add_distribution_stats(
                    headers, values, dist, group, units, status
                )
        else:
            self._add_distribution_stats(headers, values, metric, group, units, None)

    def _add_distribution_stats(
        self,
        headers: list[list[str]],
        values: list[str | int | float],
        dist: DistributionSummary,
        group: str,
        units: str,
        status: str | None,
    ) -> None:
        """
        Add distribution statistics including mean, median, and percentiles.

        :param headers: List of header hierarchies to append to
        :param values: List of values to append to
        :param dist: Distribution summary with statistical data
        :param group: Top-level category for the metric
        :param units: Units for the metric values
        :param status: Request status (successful, incomplete, errored) or None
        """
        status_prefix = f"{status.capitalize()} " if status else ""

        headers.append([group, f"{status_prefix}{units}", "Mean"])
        values.append(dist.mean)

        headers.append([group, f"{status_prefix}{units}", "Median"])
        values.append(dist.median)

        headers.append([group, f"{status_prefix}{units}", "Std Dev"])
        values.append(dist.std_dev)

        headers.append([group, f"{status_prefix}{units}", "Percentiles"])
        percentiles_str = (
            f"[{dist.min}, {dist.percentiles.p001}, {dist.percentiles.p01}, "
            f"{dist.percentiles.p05}, {dist.percentiles.p10}, {dist.percentiles.p25}, "
            f"{dist.percentiles.p75}, {dist.percentiles.p90}, {dist.percentiles.p95}, "
            f"{dist.percentiles.p99}, {dist.max}]"
        )
        values.append(percentiles_str)
