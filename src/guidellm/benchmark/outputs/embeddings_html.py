"""
HTML output formatter for embeddings benchmark results.

Transforms embeddings benchmark data into interactive web-based reports by building
UI data structures, converting keys to camelCase for JavaScript compatibility, and
injecting formatted data into HTML templates. Simplified compared to generative output
since embeddings don't have output tokens, streaming behavior, or multi-modality support.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput
from guidellm.benchmark.schemas.embeddings import (
    BenchmarkEmbeddingsArgs,
    EmbeddingsBenchmark,
    EmbeddingsBenchmarksReport,
)
from guidellm.utils import camelize_str, recursive_key_update
from guidellm.utils.text import load_text

__all__ = ["EmbeddingsBenchmarkerHTML"]


@EmbeddingsBenchmarkerOutput.register("html")
class EmbeddingsBenchmarkerHTML(EmbeddingsBenchmarkerOutput):
    """
    HTML output formatter for embeddings benchmark results.

    Generates interactive HTML reports from embeddings benchmark data by transforming
    results into camelCase JSON structures and injecting them into HTML templates.
    The formatter processes benchmark metrics, creates distribution visualizations,
    and embeds all data into a pre-built HTML template for browser-based display.

    :cvar DEFAULT_FILE: Default filename for HTML output when a directory is provided
    """

    DEFAULT_FILE: ClassVar[str] = "embeddings_benchmarks.html"

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory or file path for saving the HTML report",
    )

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate and normalize output path argument.

        :param output_path: Output file or directory path for the HTML report
        :return: Dictionary containing validated output_path if provided
        """
        validated: dict[str, Any] = {}
        if output_path is not None:
            validated["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return validated

    async def finalize(self, report: EmbeddingsBenchmarksReport) -> Path:
        """
        Generate and save the HTML embeddings benchmark report.

        :param report: Completed embeddings benchmark report
        :return: Path to the saved HTML report file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / self.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._build_ui_data(report.benchmarks, report.args)
        camel_data = recursive_key_update(deepcopy(data), camelize_str)

        ui_api_data = {
            "data": camel_data,
            "guidelLmVersion": report.metadata.guidellm_version,
        }

        # Load HTML template from package resources
        import importlib.resources
        template_content = (
            importlib.resources.files("guidellm.benchmark.outputs")
            .joinpath("html_outputs/embeddings_template.html")
            .read_text()
        )

        # Inject data into template
        html_content = template_content.replace(
            "const uiApiData = {};",
            f"const uiApiData = {json.dumps(ui_api_data, indent=2)};",
        )

        output_path.write_text(html_content)
        return output_path

    def _build_ui_data(
        self,
        benchmarks: list[EmbeddingsBenchmark],
        args: BenchmarkEmbeddingsArgs,
    ) -> dict[str, Any]:
        """
        Build UI data structure from benchmarks and arguments.

        :param benchmarks: List of completed benchmarks
        :param args: Benchmark arguments
        :return: Dictionary containing all UI data
        """
        return {
            "run_info": {
                "model": args.model or "N/A",
                "backend": str(args.backend),
                "task": "embeddings",
                "target": args.target,
            },
            "workload_details": self._build_workload_details(benchmarks),
            "benchmarks": self._build_benchmarks_data(benchmarks),
        }

    def _build_workload_details(
        self, benchmarks: list[EmbeddingsBenchmark]
    ) -> dict[str, Any]:
        """
        Build workload details section.

        :param benchmarks: List of completed benchmarks
        :return: Workload details dictionary
        """
        if not benchmarks:
            return {}

        # Sample from first benchmark
        first_benchmark = benchmarks[0]

        # Build input text statistics
        input_texts = []
        for req in first_benchmark.requests.successful[:10]:  # Sample first 10
            if req.input_metrics.text_tokens:
                input_texts.append(
                    {
                        "tokens": req.input_metrics.text_tokens,
                        "sample": f"Sample request {req.request_id[:8]}...",
                    }
                )

        return {
            "prompts": {
                "samples": input_texts,
                "token_statistics": {
                    "mean": (
                        first_benchmark.metrics.input_tokens_count.successful
                        / first_benchmark.metrics.request_totals.successful
                        if first_benchmark.metrics.request_totals.successful > 0
                        else 0
                    ),
                },
            },
            "quality_validation": self._build_quality_section(first_benchmark)
            if first_benchmark.metrics.quality
            else None,
        }

    def _build_quality_section(
        self, benchmark: EmbeddingsBenchmark
    ) -> dict[str, Any] | None:
        """
        Build quality validation section.

        :param benchmark: Benchmark with quality metrics
        :return: Quality section dictionary or None
        """
        if not benchmark.metrics.quality:
            return None

        quality = benchmark.metrics.quality
        section: dict[str, Any] = {}

        # Cosine similarity distribution
        if quality.baseline_cosine_similarity and quality.baseline_cosine_similarity.successful:
            section["cosine_similarity"] = {
                "mean": quality.baseline_cosine_similarity.successful.mean,
                "median": quality.baseline_cosine_similarity.successful.median,
                "std_dev": quality.baseline_cosine_similarity.successful.std_dev,
                "p95": quality.baseline_cosine_similarity.successful.percentiles.p95,
            }

        # MTEB scores
        if quality.mteb_main_score is not None:
            section["mteb"] = {
                "main_score": quality.mteb_main_score,
                "task_scores": quality.mteb_task_scores or {},
            }

        return section if section else None

    def _build_benchmarks_data(
        self, benchmarks: list[EmbeddingsBenchmark]
    ) -> list[dict[str, Any]]:
        """
        Build benchmarks data for visualization.

        :param benchmarks: List of completed benchmarks
        :return: List of benchmark data dictionaries
        """
        results = []

        for benchmark in benchmarks:
            metrics = benchmark.metrics

            benchmark_data = {
                "strategy": benchmark.config.strategy.type_,
                "rate": getattr(benchmark.config.strategy, "rate", None),
                "duration": benchmark.duration,
                "warmup_duration": benchmark.warmup_duration,
                "cooldown_duration": benchmark.cooldown_duration,
                # Request counts
                "request_counts": {
                    "successful": metrics.request_totals.successful,
                    "incomplete": metrics.request_totals.incomplete,
                    "errored": metrics.request_totals.errored,
                    "total": metrics.request_totals.total,
                },
                # Request metrics
                "request_latency": self._distribution_to_dict(
                    metrics.request_latency.successful
                ),
                "request_concurrency": self._distribution_to_dict(
                    metrics.request_concurrency.successful
                ),
                "requests_per_second": self._distribution_to_dict(
                    metrics.requests_per_second.successful
                ),
                # Token metrics (input only)
                "input_tokens": {
                    "total": metrics.input_tokens_count.successful,
                    "per_second": self._distribution_to_dict(
                        metrics.input_tokens_per_second.successful
                    ),
                },
                # Quality metrics (if available)
                "quality": self._build_quality_data(benchmark)
                if metrics.quality
                else None,
            }

            results.append(benchmark_data)

        return results

    def _build_quality_data(self, benchmark: EmbeddingsBenchmark) -> dict[str, Any] | None:
        """
        Build quality metrics data.

        :param benchmark: Benchmark with quality metrics
        :return: Quality data dictionary or None
        """
        if not benchmark.metrics.quality:
            return None

        quality = benchmark.metrics.quality
        data: dict[str, Any] = {}

        if quality.baseline_cosine_similarity and quality.baseline_cosine_similarity.successful:
            data["cosine_similarity"] = self._distribution_to_dict(
                quality.baseline_cosine_similarity.successful
            )

        if quality.self_consistency_score and quality.self_consistency_score.successful:
            data["self_consistency"] = self._distribution_to_dict(
                quality.self_consistency_score.successful
            )

        if quality.mteb_main_score is not None:
            data["mteb_main_score"] = quality.mteb_main_score

        if quality.mteb_task_scores:
            data["mteb_task_scores"] = quality.mteb_task_scores

        return data if data else None

    def _distribution_to_dict(
        self, dist: Any
    ) -> dict[str, float | None]:
        """
        Convert distribution summary to dictionary.

        :param dist: Distribution summary object
        :return: Dictionary with mean, median, std_dev, and percentiles
        """
        if dist is None:
            return {
                "mean": None,
                "median": None,
                "std_dev": None,
                "p50": None,
                "p95": None,
                "p99": None,
            }

        return {
            "mean": dist.mean,
            "median": dist.median,
            "std_dev": dist.std_dev,
            "p50": dist.percentiles.p50 if hasattr(dist, "percentiles") else dist.median,
            "p95": dist.percentiles.p95 if hasattr(dist, "percentiles") else None,
            "p99": dist.percentiles.p99 if hasattr(dist, "percentiles") else None,
        }
