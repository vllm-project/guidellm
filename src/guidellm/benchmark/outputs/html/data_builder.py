"""Data building functions for HTML reports.

These functions transform benchmark results into data structures suitable for
HTML report generation. They are shared between the legacy Next.js HTML output
and the new Plotly-based HTML output.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import ceil
from typing import Any

from pydantic import BaseModel, Field, computed_field

from guidellm.benchmark.schemas import (
    BenchmarkGenerativeTextArgs,
    GenerativeBenchmark,
)
from guidellm.schemas import DistributionSummary

__all__ = [
    "Bucket",
    "TabularDistributionSummary",
    "build_benchmarks",
    "build_run_info",
    "build_ui_data",
    "build_workload_details",
]


class Bucket(BaseModel):
    """
    Histogram bucket for data distribution visualization.

    Represents a single bucket in a histogram with its starting value and count
    of data points falling within the bucket range.
    """

    value: float | int = Field(description="Starting value of the bucket range")
    count: int = Field(description="Number of data points falling within this bucket")

    @staticmethod
    def from_data(
        data: list[float] | list[int],
        bucket_width: float | None = None,
        n_buckets: int | None = None,
    ) -> tuple[list[Bucket], float]:
        """
        Create histogram buckets from numeric data values.

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
            Bucket(value=start, count=count)
            for start, count in sorted(bucket_counts.items())
        ]
        return buckets, bucket_width


def _filter_duplicate_percentiles(percentiles: dict[str, float]) -> dict[str, float]:
    """
    Filter out duplicate consecutive percentile values.

    Keeps the highest percentile from each group of duplicates for
    mathematical accuracy.

    :param percentiles: Dictionary of percentile names to values
    :return: Filtered percentiles dictionary with highest percentile from
        each duplicate group
    """
    if not percentiles:
        return percentiles

    # First pass: identify which percentiles to keep (last of each duplicate group)
    items = list(percentiles.items())
    to_keep = set()

    for i, (name, value) in enumerate(items):
        # Check if this is the last occurrence of this value
        is_last = i == len(items) - 1 or items[i + 1][1] != value
        if is_last:
            to_keep.add(name)

    # Build result maintaining order
    return {name: value for name, value in percentiles.items() if name in to_keep}


class TabularDistributionSummary(DistributionSummary):
    """
    Distribution summary with tabular percentile representation.

    Extends DistributionSummary to provide percentile data formatted for table
    display in the HTML report.
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
    ) -> TabularDistributionSummary:
        """
        Convert standard DistributionSummary to tabular format.

        :param distribution: Source distribution summary to convert
        :return: Tabular distribution summary with formatted percentile rows
        """
        return cls(**distribution.model_dump())


def build_ui_data(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build complete UI data structure from benchmarks.

    :param benchmarks: List of completed benchmark results
    :param args: Benchmark configuration arguments
    :return: Dictionary with run_info, workload_details, and benchmarks sections
    """
    return {
        "run_info": build_run_info(benchmarks, args),
        "workload_details": build_workload_details(benchmarks, args),
        "benchmarks": build_benchmarks(benchmarks),
    }


def build_run_info(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build run metadata from benchmarks.

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


def build_workload_details(
    benchmarks: list[GenerativeBenchmark], args: BenchmarkGenerativeTextArgs
) -> dict[str, Any]:
    """
    Build workload details from benchmarks.

    :param benchmarks: List of completed benchmark results
    :param args: Benchmark configuration arguments
    :return: Dictionary with prompts, generations, request timing, and server info
    """
    target = args.target

    # Collect all rate types in execution order, keeping first occurrence only
    rate_types_raw = [bm.config.strategy.type_ for bm in benchmarks]
    seen = set()
    rate_types = []
    for rt in rate_types_raw:
        if rt not in seen:
            seen.add(rt)
            rate_types.append(rt)

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

    # Token counts for successful requests only
    prompt_tokens = [
        float(req.prompt_tokens)
        for bm in benchmarks
        for req in bm.requests.successful
        if req.prompt_tokens is not None
    ]
    output_tokens = [
        float(req.output_tokens)
        for bm in benchmarks
        for req in bm.requests.successful
        if req.output_tokens is not None
    ]

    prompt_token_buckets, _prompt_bucket_width = Bucket.from_data(prompt_tokens, 1)
    output_token_buckets, _output_bucket_width = Bucket.from_data(output_tokens, 1)

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
    request_buckets, bucket_width = Bucket.from_data(
        all_req_times, None, number_of_buckets
    )

    # Calculate requests per benchmark (successful, incomplete, errored)
    requests_per_benchmark = {
        "successful": [len(bm.requests.successful) for bm in benchmarks],
        "incomplete": [len(bm.requests.incomplete) for bm in benchmarks],
        "errored": [len(bm.requests.errored) for bm in benchmarks],
    }
    total_requests = sum(
        len(bm.requests.successful)
        + len(bm.requests.incomplete)
        + len(bm.requests.errored)
        for bm in benchmarks
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
            "requests_per_benchmark": requests_per_benchmark,
            "total_requests": total_requests,
        },
        "rate_types": rate_types,
        "server": {"target": target},
    }


def build_benchmarks(benchmarks: list[GenerativeBenchmark]) -> list[dict[str, Any]]:
    """
    Build benchmark metrics data for UI display.

    :param benchmarks: List of completed benchmark results
    :return: List of dictionaries with formatted benchmark metrics
    """
    result = []
    for bm in benchmarks:
        result.append(
            {
                "requests_per_second": bm.metrics.requests_per_second.successful.mean,
                "itl": TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.inter_token_latency_ms.successful
                ).model_dump(),
                "ttft": TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.time_to_first_token_ms.successful
                ).model_dump(),
                "throughput": TabularDistributionSummary.from_distribution_summary(
                    bm.metrics.output_tokens_per_second.total
                ).model_dump(),
                "time_per_request": (
                    TabularDistributionSummary.from_distribution_summary(
                        bm.metrics.request_latency.successful
                    ).model_dump()
                ),
            }
        )
    return result
