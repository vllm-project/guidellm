"""
Console output formatter for embeddings benchmarker results.

Provides console-based output formatting for embeddings benchmark reports,
organizing metrics into structured tables that display request statistics,
latency measurements, throughput data, and optional quality validation metrics
(cosine similarity, MTEB scores). Simplified compared to generative output since
embeddings don't have output tokens or streaming behavior.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from guidellm.benchmark.outputs.console import ConsoleTableColumnsCollection
from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput
from guidellm.benchmark.schemas.embeddings import EmbeddingsBenchmarksReport
from guidellm.utils import Console

__all__ = ["EmbeddingsBenchmarkerConsole"]


@EmbeddingsBenchmarkerOutput.register(["console"])
class EmbeddingsBenchmarkerConsole(EmbeddingsBenchmarkerOutput):
    """
    Console output formatter for embeddings benchmark reports.

    Renders embeddings benchmark results as formatted tables in the terminal,
    organizing metrics by category (run summary, request counts, latency,
    throughput, quality validation) with proper alignment and type-specific
    formatting for readability.
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

    async def finalize(self, report: EmbeddingsBenchmarksReport) -> None:
        """
        Print the complete embeddings benchmark report to the console.

        Renders all metric tables including run summary, request counts, latency,
        throughput, and quality metrics to the console.

        :param report: The completed embeddings benchmark report
        :return: None (console output only)
        """
        self.print_run_summary_table(report)
        self.print_request_counts_table(report)
        self.print_request_latency_table(report)
        self.print_server_throughput_table(report)
        self.print_quality_metrics_table(report)

    def print_run_summary_table(self, report: EmbeddingsBenchmarksReport):
        """
        Print the run summary table with timing and token information.

        :param report: The embeddings benchmark report containing run metadata
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

            # Only input tokens for embeddings (no output tokens)
            token_metrics = benchmark.metrics.input_tokens_count
            columns.add_value(
                token_metrics.successful,
                group="Input Tokens",
                name="Comp",
                units="Tot",
            )
            columns.add_value(
                token_metrics.incomplete,
                group="Input Tokens",
                name="Inc",
                units="Tot",
            )
            columns.add_value(
                token_metrics.errored,
                group="Input Tokens",
                name="Err",
                units="Tot",
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Run Summary")

    def print_request_counts_table(self, report: EmbeddingsBenchmarksReport):
        """
        Print the request counts table.

        :param report: The embeddings benchmark report
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            for status in ["successful", "incomplete", "errored", "total"]:
                count = getattr(benchmark.metrics.request_totals, status)
                columns.add_value(
                    count,
                    group="Request Counts",
                    name=status.capitalize(),
                    units="Reqs",
                )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Request Counts")

    def print_request_latency_table(self, report: EmbeddingsBenchmarksReport):
        """
        Print the request latency table.

        :param report: The embeddings benchmark report
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            # Request latency stats
            columns.add_stats(
                benchmark.metrics.request_latency,
                status="successful",
                group="Request Latency",
                name="Latency",
                precision=3,
            )

            # Request concurrency
            columns.add_stats(
                benchmark.metrics.request_concurrency,
                status="successful",
                group="Concurrency",
                name="Concurrent",
                precision=1,
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Request Latency")

    def print_server_throughput_table(self, report: EmbeddingsBenchmarksReport):
        """
        Print the server throughput table.

        :param report: The embeddings benchmark report
        """
        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            # Requests per second
            columns.add_stats(
                benchmark.metrics.requests_per_second,
                status="successful",
                group="Request Throughput",
                name="Reqs",
                precision=2,
            )

            # Input tokens per second
            columns.add_stats(
                benchmark.metrics.input_tokens_per_second,
                status="successful",
                group="Token Throughput",
                name="Input Tok",
                precision=1,
            )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Server Throughput")

    def print_quality_metrics_table(self, report: EmbeddingsBenchmarksReport):
        """
        Print the quality metrics table (if quality validation was enabled).

        :param report: The embeddings benchmark report
        """
        # Check if any benchmark has quality metrics
        has_quality = any(
            benchmark.metrics.quality is not None for benchmark in report.benchmarks
        )

        if not has_quality:
            return

        columns = ConsoleTableColumnsCollection()

        for benchmark in report.benchmarks:
            columns.add_value(
                benchmark.config.strategy.type_,
                group="Benchmark",
                name="Strategy",
                type_="text",
            )

            if benchmark.metrics.quality:
                # Cosine similarity
                if benchmark.metrics.quality.baseline_cosine_similarity:
                    columns.add_stats(
                        benchmark.metrics.quality.baseline_cosine_similarity,
                        status="successful",
                        group="Cosine Similarity",
                        name="Baseline",
                        precision=4,
                    )

                # Self-consistency
                if benchmark.metrics.quality.self_consistency_score:
                    columns.add_stats(
                        benchmark.metrics.quality.self_consistency_score,
                        status="successful",
                        group="Consistency",
                        name="Self",
                        precision=4,
                    )

                # MTEB main score
                if benchmark.metrics.quality.mteb_main_score is not None:
                    columns.add_value(
                        benchmark.metrics.quality.mteb_main_score,
                        group="MTEB",
                        name="Main",
                        units="Score",
                        precision=4,
                    )

        headers, values = columns.get_table_data()
        self.console.print("\n")
        self.console.print_table(headers, values, title="Quality Metrics")
