"""
CSV output formatter for embeddings benchmark results.

Provides CSV export functionality for embeddings benchmark reports with comprehensive
metrics including timing, throughput, latency, input token data, and optional quality
validation metrics (cosine similarity, MTEB scores). Uses multi-row headers to organize
metrics hierarchically without output tokens or streaming behavior.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import Field

if TYPE_CHECKING:
    from _csv import _writer

from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput
from guidellm.benchmark.schemas.embeddings import (
    EmbeddingsBenchmark,
    EmbeddingsBenchmarksReport,
)
from guidellm.schemas import DistributionSummary, StatusDistributionSummary
from guidellm.utils import safe_format_timestamp

__all__ = ["EmbeddingsBenchmarkerCSV"]

TIMESTAMP_FORMAT: Annotated[str, "Format string for timestamp output in CSV files"] = (
    "%Y-%m-%d %H:%M:%S"
)


@EmbeddingsBenchmarkerOutput.register("csv")
class EmbeddingsBenchmarkerCSV(EmbeddingsBenchmarkerOutput):
    """
    CSV output formatter for embeddings benchmark results.

    Exports comprehensive embeddings benchmark data to CSV format with
    multi-row headers organizing metrics into categories including run
    information, timing, request counts, latency, throughput, input token
    data, quality validation metrics, and scheduler state. Each benchmark run
    becomes a row with statistical distributions represented as mean, median,
    standard deviation, and percentiles.

    :cvar DEFAULT_FILE: Default filename for CSV output
    """

    DEFAULT_FILE: ClassVar[str] = "embeddings_benchmarks.csv"

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
            "Path where the CSV file will be saved, defaults to current "
            "directory"
        ),
    )

    async def finalize(self, report: EmbeddingsBenchmarksReport) -> Path:
        """
        Save the embeddings benchmark report as a CSV file.

        :param report: The completed embeddings benchmark report
        :return: Path to the saved CSV file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / EmbeddingsBenchmarkerCSV.DEFAULT_FILE
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
                self._add_input_token_metrics(
                    benchmark, benchmark_headers, benchmark_values
                )
                self._add_quality_metrics(
                    benchmark, benchmark_headers, benchmark_values
                )
                self._add_scheduler_info(
                    benchmark, benchmark_headers, benchmark_values
                )
                self._add_runtime_info(report, benchmark_headers, benchmark_values)

                if not headers:
                    headers = benchmark_headers
                rows.append(benchmark_values)

            self._write_multirow_header(writer, headers)
            for row in rows:
                writer.writerow(row)

        return output_path

    def _write_multirow_header(
        self, writer: _writer, headers: list[list[str]]
    ) -> None:
        """
        Write multi-row header to CSV file.

        Transposes column-wise headers into row-wise header rows with proper
        alignment for hierarchical metric organization.

        :param writer: CSV writer instance
        :param headers: List of header columns, each column is [group, name, units]
        """
        if not headers:
            return

        num_rows = max(len(header) for header in headers)
        header_rows: list[list[str]] = [[] for _ in range(num_rows)]

        for header in headers:
            for i in range(num_rows):
                header_rows[i].append(header[i] if i < len(header) else "")

        for row in header_rows:
            writer.writerow(row)

    def _add_run_info(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add run identification information."""
        headers.append(["Run Info", "Model", ""])
        model = (
            benchmark.config.requests.get("model", "N/A")
            if isinstance(benchmark.config.requests, dict)
            else "N/A"
        )
        values.append(model)

        headers.append(["Run Info", "Backend", ""])
        backend = (
            benchmark.config.backend.get("type", "N/A")
            if isinstance(benchmark.config.backend, dict)
            else "N/A"
        )
        values.append(backend)

    def _add_benchmark_info(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add benchmark configuration information."""
        headers.append(["Benchmark", "Strategy", ""])
        values.append(benchmark.config.strategy.type_)

        if hasattr(benchmark.config.strategy, "rate"):
            headers.append(["Benchmark", "Rate", "Req/s"])
            values.append(benchmark.config.strategy.rate or 0)

    def _add_timing_info(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add timing information."""
        headers.append(["Timings", "Start", ""])
        values.append(safe_format_timestamp(benchmark.start_time, TIMESTAMP_FORMAT))

        headers.append(["Timings", "End", ""])
        values.append(safe_format_timestamp(benchmark.end_time, TIMESTAMP_FORMAT))

        headers.append(["Timings", "Duration", "Sec"])
        values.append(benchmark.duration)

        headers.append(["Timings", "Warmup", "Sec"])
        values.append(benchmark.warmup_duration)

        headers.append(["Timings", "Cooldown", "Sec"])
        values.append(benchmark.cooldown_duration)

    def _add_request_counts(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add request count information."""
        for status in ["successful", "incomplete", "errored", "total"]:
            count = getattr(benchmark.metrics.request_totals, status)
            headers.append(["Request Counts", status.capitalize(), "Reqs"])
            values.append(count)

    def _add_request_latency_metrics(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add request latency metrics."""
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.request_latency,
            "Request Latency",
            "Latency (s)",
        )

        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.request_concurrency,
            "Concurrency",
            "Concurrent Reqs",
        )

    def _add_server_throughput_metrics(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add server throughput metrics."""
        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.requests_per_second,
            "Request Throughput",
            "Reqs/s",
        )

        self._add_stats_for_metric(
            headers,
            values,
            benchmark.metrics.input_tokens_per_second,
            "Token Throughput",
            "Input Tok/s",
        )

    def _add_input_token_metrics(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add input token count metrics (no output tokens for embeddings)."""
        for status in ["successful", "incomplete", "errored", "total"]:
            count = getattr(benchmark.metrics.input_tokens_count, status)
            headers.append(["Input Tokens", status.capitalize(), "Tokens"])
            values.append(count)

    def _add_quality_metrics(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add quality validation metrics if available."""
        if not benchmark.metrics.quality:
            return

        # Cosine similarity
        if benchmark.metrics.quality.baseline_cosine_similarity:
            self._add_stats_for_metric(
                headers,
                values,
                benchmark.metrics.quality.baseline_cosine_similarity,
                "Quality Validation",
                "Cosine Sim",
            )

        # Self-consistency
        if benchmark.metrics.quality.self_consistency_score:
            self._add_stats_for_metric(
                headers,
                values,
                benchmark.metrics.quality.self_consistency_score,
                "Quality Validation",
                "Consistency",
            )

        # MTEB main score
        if benchmark.metrics.quality.mteb_main_score is not None:
            headers.append(["MTEB", "Main Score", ""])
            values.append(benchmark.metrics.quality.mteb_main_score)

        # MTEB task scores
        if benchmark.metrics.quality.mteb_task_scores:
            for task, score in benchmark.metrics.quality.mteb_task_scores.items():
                headers.append(["MTEB Tasks", task, "Score"])
                values.append(score)

    def _add_scheduler_info(
        self,
        benchmark: EmbeddingsBenchmark,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add scheduler state information."""
        headers.append(["Scheduler", "Queued Avg", "Sec"])
        values.append(benchmark.scheduler_metrics.queued_time_avg)

        headers.append(["Scheduler", "Resolve Avg", "Sec"])
        values.append(benchmark.scheduler_metrics.resolve_time_avg)

    def _add_runtime_info(
        self,
        report: EmbeddingsBenchmarksReport,
        headers: list[list[str]],
        values: list[str | int | float],
    ) -> None:
        """Add runtime environment information."""
        headers.append(["Runtime", "GuideLLM Ver", ""])
        values.append(report.metadata.guidellm_version)

        headers.append(["Runtime", "Python Ver", ""])
        values.append(report.metadata.python_version)

    def _add_stats_for_metric(
        self,
        headers: list[list[str]],
        values: list[str | int | float],
        stats: StatusDistributionSummary,
        group: str,
        metric_name: str,
    ) -> None:
        """
        Add statistical columns for a metric with mean, median, stddev, and percentiles.

        :param headers: Headers list to append to
        :param values: Values list to append to
        :param stats: Status distribution summary containing statistics
        :param group: Metric group name for header
        :param metric_name: Metric display name
        """
        successful_stats: DistributionSummary | None = stats.successful

        # Mean
        headers.append([group, metric_name, "Mean"])
        values.append(successful_stats.mean if successful_stats else 0)

        # Median
        headers.append([group, metric_name, "Median"])
        values.append(successful_stats.median if successful_stats else 0)

        # Std Dev
        headers.append([group, metric_name, "StdDev"])
        values.append(successful_stats.std_dev if successful_stats else 0)

        # P95
        headers.append([group, metric_name, "P95"])
        values.append(
            successful_stats.percentiles.p95 if successful_stats else 0
        )

        # P99
        headers.append([group, metric_name, "P99"])
        values.append(
            successful_stats.percentiles.p99 if successful_stats else 0
        )
