"""
HTML output formatter for benchmark results.

Transforms benchmark data into interactive web-based reports by building UI data
structures, converting keys to camelCase for JavaScript compatibility, and injecting
formatted data into HTML templates. The formatter processes GenerativeBenchmark
instances and their associated metrics, creating histogram buckets for distributions,
formatting percentile statistics for tabular display, and embedding all data as
JavaScript objects within an HTML template for client-side rendering and visualization.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field, computed_field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import (
    BenchmarkGenerativeTextArgs,
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
)
from guidellm.schemas import DistributionSummary, Percentiles
from guidellm.settings import settings
from guidellm.utils import camelize_str, recursive_key_update
from guidellm.utils.text import load_text

__all__ = ["GenerativeBenchmarkerHTML"]


@GenerativeBenchmarkerOutput.register("html")
class GenerativeBenchmarkerHTML(GenerativeBenchmarkerOutput):
    """
    HTML output formatter for benchmark results.

    Generates interactive HTML reports from benchmark data by transforming results
    into camelCase JSON structures and injecting them into HTML templates. The
    formatter processes benchmark metrics, creates histogram distributions, and
    embeds all data into a pre-built HTML template for browser-based visualization.
    Reports are saved to the specified output path or current working directory.

    :cvar DEFAULT_FILE: Default filename for HTML output when a directory is provided
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.html"

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description=(
            "Directory or file path for saving the HTML report, "
            "defaults to current working directory"
        ),
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

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Generate and save the HTML benchmark report.

        Transforms benchmark data into camelCase JSON format, injects it into the
        HTML template, and writes the resulting report to the output path. Creates
        parent directories if they don't exist.

        :param report: Completed benchmark report containing all results
        :return: Path to the saved HTML report file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / self.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = _build_ui_data(report.benchmarks, report.args)
        camel_data = recursive_key_update(deepcopy(data), camelize_str)

        ui_api_data = {
            f"window.{key} = {{}};": f"window.{key} = {json.dumps(value, indent=2)};\n"
            for key, value in camel_data.items()
        }

        _create_html_report(ui_api_data, output_path)

        return output_path


class _Bucket(BaseModel):
    """
    Histogram bucket for data distribution visualization.

    Represents a single bucket in a histogram with its starting value and count
    of data points falling within the bucket range. Used to create distribution
    histograms for metrics like token counts and request timings.
    """

    value: float | int = Field(description="Starting value of the bucket range")
    count: int = Field(description="Number of data points falling within this bucket")

    @staticmethod
    def from_data(
        data: list[float] | list[int],
        bucket_width: float | None = None,
        n_buckets: int | None = None,
    ) -> tuple[list[_Bucket], float]:
        """
        Create histogram buckets from numeric data values.

        Divides the data range into equal-width buckets and counts values within
        each bucket. Either bucket_width or n_buckets can be specified; if neither
        is provided, defaults to 10 buckets.

        :param data: Numeric values to bucket
        :param bucket_width: Width of each bucket, computed if None
        :param n_buckets: Number of buckets, defaults to 10 if width not specified
        :return: Tuple of bucket list and computed bucket width
        """
        if not data:
            return [], 1.0

        min_v = min(data)
        max_v = max(data)
        range_v = (1 + max_v) - min_v

        if bucket_width is None:
            if n_buckets is None:
                n_buckets = 10
            bucket_width = range_v / n_buckets
        else:
            n_buckets = ceil(range_v / bucket_width)

        bucket_counts: defaultdict[float | int, int] = defaultdict(int)
        for val in data:
            idx = int((val - min_v) // bucket_width)
            if idx >= n_buckets:
                idx = n_buckets - 1
            bucket_start = min_v + idx * bucket_width
            bucket_counts[bucket_start] += 1

        buckets = [
            _Bucket(value=start, count=count)
            for start, count in sorted(bucket_counts.items())
        ]
        return buckets, bucket_width


class _TabularDistributionSummary(DistributionSummary):
    """
    Distribution summary with tabular percentile representation.

    Extends DistributionSummary to provide percentile data formatted for table
    display in the HTML report. Filters to show only key percentiles (p50, p90,
    p95, p99) for concise presentation.
    """

    @computed_field
    def percentile_rows(self) -> list[dict[str, str | float]]:
        """
        Format percentiles as table rows for UI display.

        :return: List of dictionaries with percentile names and values
        """
        rows = [
            {"percentile": name, "value": value}
            for name, value in self.percentiles.model_dump().items()
        ]
        return list(
            filter(lambda row: row["percentile"] in ["p50", "p90", "p95", "p99"], rows)
        )

    def model_dump(self, **kwargs) -> dict:
        """
        Override model_dump to filter duplicate consecutive percentile values.

        This prevents visualization errors when distributions have limited data
        points causing multiple percentiles to collapse to the same value.

        :param kwargs: Arguments to pass to parent model_dump
        :return: Dictionary with filtered percentiles
        """
        data = super().model_dump(**kwargs)

        if "percentiles" in data and data["percentiles"]:
            filtered_percentiles = _filter_duplicate_percentiles(data["percentiles"])
            data["percentiles"] = filtered_percentiles

        return data

    @classmethod
    def from_distribution_summary(
        cls, distribution: DistributionSummary
    ) -> _TabularDistributionSummary:
        """
        Convert standard DistributionSummary to tabular format.

        :param distribution: Source distribution summary to convert
        :return: Tabular distribution summary with formatted percentile rows
        """
        return cls(**distribution.model_dump())


def _create_html_report(js_data: dict[str, str], output_path: Path) -> Path:
    """
    Create HTML report by injecting JavaScript data into template.

    Loads the HTML template, injects JavaScript data into the head section, and
    writes the final report to the specified output path.

    :param js_data: Dictionary mapping placeholder strings to JavaScript code
    :param output_path: Path where HTML report will be saved
    :return: Path to the saved report file
    """
    html_content = load_text(settings.report_generation.source)
    report_content = _inject_data(js_data, html_content)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)
    return output_path


def _filter_duplicate_percentiles(percentiles: dict[str, float]) -> dict[str, float]:
    """
    Filter out consecutive duplicate percentile values.

    When distributions have very few data points, multiple percentiles can have
    the same value, which causes visualization libraries to fail. This function
    keeps only the largest percentile for consecutive duplicate values, which is
    more mathematically accurate as higher percentiles have greater statistical
    significance.

    :param percentiles: Dictionary of percentile names to values
    :return: Filtered percentiles dictionary with no consecutive duplicates
    """
    if not percentiles:
        return percentiles

    percentile_order = list(Percentiles.model_fields.keys())

    # Iterate in reverse to keep the largest percentile for each value
    filtered = {}
    previous_value = None

    for key in reversed(percentile_order):
        if key in percentiles:
            current_value = percentiles[key]
            if previous_value is None or current_value != previous_value:
                filtered[key] = current_value
                previous_value = current_value

    # Restore original order
    return {key: filtered[key] for key in percentile_order if key in filtered}


def _inject_data(js_data: dict[str, str], html: str) -> str:
    """
    Inject JavaScript data into HTML head section.

    Replaces placeholder strings in the HTML head section with actual JavaScript
    code containing benchmark data. Returns original HTML if no head section found.

    :param js_data: Dictionary mapping placeholder strings to JavaScript code
    :param html: HTML template content
    :return: HTML with injected JavaScript data
    """
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.DOTALL | re.IGNORECASE)
    if not head_match:
        logger.warning("<head> section missing, returning original HTML.")
        return html

    head_content = head_match.group(1)

    for placeholder, script in js_data.items():
        head_content = head_content.replace(placeholder, script)

    new_head = f"<head>{head_content}</head>"
    return html[: head_match.start()] + new_head + html[head_match.end() :]


def _build_ui_data(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build complete UI data structure from benchmarks.

    Aggregates benchmark results into a structured format for the HTML UI,
    including run metadata, workload details, and per-benchmark metrics.

    :param benchmarks: List of completed benchmark results
    :param args: Benchmark configuration arguments
    :return: Dictionary with run_info, workload_details, and benchmarks sections
    """
    return {
        "run_info": _build_run_info(benchmarks, args),
        "workload_details": _build_workload_details(benchmarks, args),
        "benchmarks": _build_benchmarks(benchmarks),
    }


def _build_run_info(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build run metadata from benchmarks.

    Extracts model name, timestamp, and dataset information from the benchmark
    configuration and results.

    :param benchmarks: List of completed benchmark results
    :param args: Benchmark configuration arguments
    :return: Dictionary with model, task, timestamp, and dataset information
    """
    model = args.model or "N/A"
    timestamp = max(bm.start_time for bm in benchmarks if bm.start_time is not None)
    return {
        "model": {"name": model, "size": 0},
        "task": "N/A",
        "timestamp": timestamp,
        "dataset": {"name": "N/A"},
    }


def _build_workload_details(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build workload details from benchmarks.

    Aggregates prompt and generation samples, token distribution statistics,
    request timing histograms, and server configuration. Samples up to 5 random
    prompts and outputs for display.

    :param benchmarks: List of completed benchmark results
    :param args: Benchmark configuration arguments
    :return: Dictionary with prompts, generations, request timing, and server info
    """
    target = args.target
    rate_type = benchmarks[0].config.strategy.type_
    successful_requests = [req for bm in benchmarks for req in bm.requests.successful]

    sample_indices = random.sample(
        range(len(successful_requests)), min(5, len(successful_requests))
    )
    sample_prompts = [
        req.request_args.replace("\n", " ").replace('"', "'")
        if (req := successful_requests[i]).request_args
        else ""
        for i in sample_indices
    ]
    sample_outputs = [
        req.output.replace("\n", " ").replace('"', "'")
        if (req := successful_requests[i]).output
        else ""
        for i in sample_indices
    ]

    prompt_tokens = [
        float(req.prompt_tokens) if req.prompt_tokens is not None else -1
        for bm in benchmarks
        for req in bm.requests.successful
    ]
    output_tokens = [
        float(req.output_tokens) if req.output_tokens is not None else -1
        for bm in benchmarks
        for req in bm.requests.successful
    ]

    prompt_token_buckets, _prompt_bucket_width = _Bucket.from_data(prompt_tokens, 1)
    output_token_buckets, _output_bucket_width = _Bucket.from_data(output_tokens, 1)

    prompt_token_stats = DistributionSummary.from_values(prompt_tokens)
    output_token_stats = DistributionSummary.from_values(output_tokens)

    min_start_time = benchmarks[0].start_time
    all_req_times = [
        req.info.timings.request_start - min_start_time
        for bm in benchmarks
        for req in bm.requests.successful
        if req.info.timings.request_start is not None
    ]

    number_of_buckets = len(benchmarks)
    request_buckets, bucket_width = _Bucket.from_data(
        all_req_times, None, number_of_buckets
    )

    return {
        "prompts": {
            "samples": sample_prompts,
            "token_distributions": {
                "statistics": prompt_token_stats.model_dump()
                if prompt_token_stats
                else None,
                "buckets": [b.model_dump() for b in prompt_token_buckets],
                "bucket_width": 1,
            },
        },
        "generations": {
            "samples": sample_outputs,
            "token_distributions": {
                "statistics": output_token_stats.model_dump()
                if output_token_stats
                else None,
                "buckets": [b.model_dump() for b in output_token_buckets],
                "bucket_width": 1,
            },
        },
        "requests_over_time": {
            "requests_over_time": {
                "buckets": [b.model_dump() for b in request_buckets],
                "bucket_width": bucket_width,
            },
            "num_benchmarks": number_of_buckets,
        },
        "rate_type": rate_type,
        "server": {"target": target},
    }


def _build_benchmarks(benchmarks: list[GenerativeBenchmark]) -> list[dict[str, Any]]:
    """
    Build benchmark metrics data for UI display.

    Extracts key performance metrics from each benchmark including requests per
    second, inter-token latency, time to first token, throughput, and request
    latency. Formats distribution summaries for tabular display.

    :param benchmarks: List of completed benchmark results
    :return: List of dictionaries with formatted benchmark metrics
    """
    result = []
    for bm in benchmarks:
        result.append(
            {
                "requests_per_second": bm.metrics.requests_per_second.successful.mean,
                "itl": _TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.inter_token_latency_ms.successful
                ).model_dump(),
                "ttft": _TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.time_to_first_token_ms.successful
                ).model_dump(),
                "throughput": _TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.output_tokens_per_second.successful
                ).model_dump(),
                "time_per_request": (
                    _TabularDistributionSummary.from_distribution_summary(
                        bm.metrics.request_latency.successful
                    ).model_dump()
                ),
            }
        )
    return result
