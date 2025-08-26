"""
Statistical analysis utilities for distribution calculations and running metrics.

Provides comprehensive statistical computation tools for analyzing numerical
distributions, percentiles, and streaming data. Includes specialized support for
request timing analysis, concurrency measurement, and rate calculations. Integrates
with Pydantic for serializable statistical models and supports both weighted and
unweighted distributions with cumulative distribution function (CDF) generation.
"""

from __future__ import annotations

import math
import time as timer
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from pydantic import Field, computed_field

from guidellm.utils.pydantic_utils import StandardBaseModel, StatusBreakdown

__all__ = [
    "DistributionSummary",
    "Percentiles",
    "RunningStats",
    "StatusDistributionSummary",
    "TimeRunningStats",
]


class Percentiles(StandardBaseModel):
    """
    Standard percentiles model for statistical distribution analysis.

    Provides complete percentile coverage from 0.1th to 99.9th percentiles for
    statistical distribution characterization. Used as a component within
    DistributionSummary to provide detailed distribution shape analysis.
    """

    p001: float = Field(
        description="The 0.1th percentile of the distribution.",
    )
    p01: float = Field(
        description="The 1st percentile of the distribution.",
    )
    p05: float = Field(
        description="The 5th percentile of the distribution.",
    )
    p10: float = Field(
        description="The 10th percentile of the distribution.",
    )
    p25: float = Field(
        description="The 25th percentile of the distribution.",
    )
    p50: float = Field(
        description="The 50th percentile of the distribution.",
    )
    p75: float = Field(
        description="The 75th percentile of the distribution.",
    )
    p90: float = Field(
        description="The 90th percentile of the distribution.",
    )
    p95: float = Field(
        description="The 95th percentile of the distribution.",
    )
    p99: float = Field(
        description="The 99th percentile of the distribution.",
    )
    p999: float = Field(
        description="The 99.9th percentile of the distribution.",
    )


class DistributionSummary(StandardBaseModel):
    """
    Comprehensive statistical summary for numerical value distributions.

    Calculates and stores complete statistical metrics including central tendency,
    dispersion, extremes, and percentiles for any numerical distribution. Supports
    both weighted and unweighted data with optional cumulative distribution function
    generation. Primary statistical analysis tool for request timing, performance
    metrics, and benchmark result characterization.

    Example:
    ::
        # Create from simple values
        summary = DistributionSummary.from_values([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"Mean: {summary.mean}, P95: {summary.percentiles.p95}")

        # Create from request timings for concurrency analysis
        requests = [(0.0, 1.0), (0.5, 2.0), (1.0, 2.5)]
        concurrency = DistributionSummary.from_request_times(
            requests, "concurrency"
        )
    """

    mean: float = Field(
        description="The mean/average of the distribution.",
    )
    median: float = Field(
        description="The median of the distribution.",
    )
    mode: float = Field(
        description="The mode of the distribution.",
    )
    variance: float = Field(
        description="The variance of the distribution.",
    )
    std_dev: float = Field(
        description="The standard deviation of the distribution.",
    )
    min: float = Field(
        description="The minimum value of the distribution.",
    )
    max: float = Field(
        description="The maximum value of the distribution.",
    )
    count: int = Field(
        description="The number of values in the distribution.",
    )
    total_sum: float = Field(
        description="The total sum of the values in the distribution.",
    )
    percentiles: Percentiles = Field(
        description="The percentiles of the distribution.",
    )
    cumulative_distribution_function: list[tuple[float, float]] | None = Field(
        description="The cumulative distribution function (CDF) of the distribution.",
        default=None,
    )

    @staticmethod
    def from_distribution_function(
        distribution: list[tuple[float, float]],
        include_cdf: bool = False,
    ) -> DistributionSummary:
        """
        Create statistical summary from weighted distribution or probability function.

        Converts weighted numerical values or probability distribution function (PDF)
        into comprehensive statistical summary. Normalizes weights to probabilities
        and calculates all statistical metrics including percentiles.

        :param distribution: List of (value, weight) or (value, probability) tuples
            representing the distribution
        :param include_cdf: Whether to include cumulative distribution function
            in the output
        :return: DistributionSummary instance with calculated statistical metrics
        """
        values, weights = zip(*distribution) if distribution else ([], [])
        values = np.array(values)  # type: ignore[assignment]
        weights = np.array(weights)  # type: ignore[assignment]

        # create the PDF
        probabilities = weights / np.sum(weights)  # type: ignore[operator]
        pdf = np.column_stack((values, probabilities))
        pdf = pdf[np.argsort(pdf[:, 0])]
        values = pdf[:, 0]  # type: ignore[assignment]
        probabilities = pdf[:, 1]

        # calculate the CDF
        cumulative_probabilities = np.cumsum(probabilities)
        cdf = np.column_stack((values, cumulative_probabilities))

        # calculate statistics
        mean = np.sum(values * probabilities).item()  # type: ignore[attr-defined]
        median = cdf[np.argmax(cdf[:, 1] >= 0.5), 0].item() if len(cdf) > 0 else 0  # noqa: PLR2004
        mode = values[np.argmax(probabilities)].item() if len(values) > 0 else 0  # type: ignore[call-overload]
        variance = np.sum((values - mean) ** 2 * probabilities).item()  # type: ignore[attr-defined]
        std_dev = math.sqrt(variance)
        minimum = values[0].item() if len(values) > 0 else 0
        maximum = values[-1].item() if len(values) > 0 else 0
        count = len(values)
        total_sum = np.sum(values).item()  # type: ignore[attr-defined]

        return DistributionSummary(
            mean=mean,
            median=median,
            mode=mode,
            variance=variance,
            std_dev=std_dev,
            min=minimum,
            max=maximum,
            count=count,
            total_sum=total_sum,
            percentiles=(
                Percentiles(
                    p001=cdf[np.argmax(cdf[:, 1] >= 0.001), 0].item(),  # noqa: PLR2004
                    p01=cdf[np.argmax(cdf[:, 1] >= 0.01), 0].item(),  # noqa: PLR2004
                    p05=cdf[np.argmax(cdf[:, 1] >= 0.05), 0].item(),  # noqa: PLR2004
                    p10=cdf[np.argmax(cdf[:, 1] >= 0.1), 0].item(),  # noqa: PLR2004
                    p25=cdf[np.argmax(cdf[:, 1] >= 0.25), 0].item(),  # noqa: PLR2004
                    p50=cdf[np.argmax(cdf[:, 1] >= 0.50), 0].item(),  # noqa: PLR2004
                    p75=cdf[np.argmax(cdf[:, 1] >= 0.75), 0].item(),  # noqa: PLR2004
                    p90=cdf[np.argmax(cdf[:, 1] >= 0.9), 0].item(),  # noqa: PLR2004
                    p95=cdf[np.argmax(cdf[:, 1] >= 0.95), 0].item(),  # noqa: PLR2004
                    p99=cdf[np.argmax(cdf[:, 1] >= 0.99), 0].item(),  # noqa: PLR2004
                    p999=cdf[np.argmax(cdf[:, 1] >= 0.999), 0].item(),  # noqa: PLR2004
                )
                if len(cdf) > 0
                else Percentiles(
                    p001=0,
                    p01=0,
                    p05=0,
                    p10=0,
                    p25=0,
                    p50=0,
                    p75=0,
                    p90=0,
                    p95=0,
                    p99=0,
                    p999=0,
                )
            ),
            cumulative_distribution_function=cdf.tolist() if include_cdf else None,
        )

    @staticmethod
    def from_values(
        values: list[float],
        weights: list[float] | None = None,
        include_cdf: bool = False,
    ) -> DistributionSummary:
        """
        Create statistical summary from numerical values with optional weights.

        Wrapper around from_distribution_function for simple value lists. If weights
        are not provided, all values are equally weighted. Enables statistical
        analysis of any numerical dataset.

        :param values: Numerical values representing the distribution
        :param weights: Optional weights for each value. If not provided, all values
            are equally weighted
        :param include_cdf: Whether to include cumulative distribution function in
            the output DistributionSummary
        :return: DistributionSummary instance with calculated statistical metrics
        :raises ValueError: If values and weights lists have different lengths
        """
        if weights is None:
            weights = [1.0] * len(values)

        if len(values) != len(weights):
            raise ValueError(
                "The length of values and weights must be the same.",
            )

        return DistributionSummary.from_distribution_function(
            distribution=list(zip(values, weights)),
            include_cdf=include_cdf,
        )

    @staticmethod
    def from_request_times(
        requests: list[tuple[float, float]],
        distribution_type: Literal["concurrency", "rate"],
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> DistributionSummary:
        """
        Create statistical summary from request timing data.

        Analyzes request start/end times to calculate concurrency or rate
        distributions. Converts timing events into statistical metrics for
        performance analysis and load characterization.

        :param requests: List of (start_time, end_time) tuples for each request
        :param distribution_type: Type of analysis - "concurrency" for simultaneous
            requests or "rate" for completion rates
        :param include_cdf: Whether to include cumulative distribution function
        :param epsilon: Threshold for merging close timing events
        :return: DistributionSummary with timing-based statistical metrics
        :raises ValueError: If distribution_type is not "concurrency" or "rate"
        """
        if distribution_type == "concurrency":
            # convert to delta changes based on when requests were running
            time_deltas: dict[float, int] = defaultdict(int)
            for start, end in requests:
                time_deltas[start] += 1
                time_deltas[end] -= 1

            # convert to the events over time measuring concurrency changes
            events = []
            active = 0

            for time, delta in sorted(time_deltas.items()):
                active += delta
                events.append((time, active))
        elif distribution_type == "rate":
            # convert to events for when requests finished
            global_start = min(start for start, _ in requests) if requests else 0
            events = [(global_start, 1)] + [(end, 1) for _, end in requests]
        else:
            raise ValueError(
                f"Invalid distribution_type '{distribution_type}'. "
                "Must be 'concurrency' or 'rate'."
            )

        # combine any events that are very close together
        flattened_events: list[tuple[float, float]] = []
        for time, val in sorted(events):
            last_time, last_val = (
                flattened_events[-1] if flattened_events else (None, None)
            )

            if (
                last_time is not None
                and last_val is not None
                and abs(last_time - time) <= epsilon
            ):
                flattened_events[-1] = (last_time, last_val + val)
            else:
                flattened_events.append((time, val))

        # convert to value distribution function
        distribution: dict[float, float] = defaultdict(float)

        for ind in range(len(flattened_events) - 1):
            start_time, value = flattened_events[ind]
            end_time, _ = flattened_events[ind + 1]
            duration = end_time - start_time

            if distribution_type == "concurrency":
                # weight the concurrency value by the duration
                distribution[value] += duration
            elif distribution_type == "rate":
                # weight the rate value by the duration
                rate = value / duration
                distribution[rate] += duration

        distribution_list: list[tuple[float, float]] = sorted(distribution.items())

        return DistributionSummary.from_distribution_function(
            distribution=distribution_list,
            include_cdf=include_cdf,
        )

    @staticmethod
    def from_iterable_request_times(
        requests: list[tuple[float, float]],
        first_iter_times: list[float],
        iter_counts: list[int],
        first_iter_counts: list[int] | None = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> DistributionSummary:
        """
        Create statistical summary from iterative request timing data.

        Analyzes autoregressive or streaming requests with multiple iterations
        between start and end times. Calculates rate distributions based on
        iteration timing patterns for LLM token generation analysis.

        :param requests: List of (start_time, end_time) tuples for each request
        :param first_iter_times: Times when first iteration was received for
            each request
        :param iter_counts: Total iteration counts for each request from first
            iteration to end
        :param first_iter_counts: Iteration counts for first iteration (defaults
            to 1 for each request)
        :param include_cdf: Whether to include cumulative distribution function
        :param epsilon: Threshold for merging close timing events
        :return: DistributionSummary with iteration rate statistical metrics
        :raises ValueError: If input lists have mismatched lengths
        """

        if first_iter_counts is None:
            first_iter_counts = [1] * len(requests)

        if (
            len(requests) != len(first_iter_times)
            or len(requests) != len(iter_counts)
            or len(requests) != len(first_iter_counts)
        ):
            raise ValueError(
                "requests, first_iter_times, iter_counts, and first_iter_counts must"
                "be the same length."
                f"Given {len(requests)}, {len(first_iter_times)}, {len(iter_counts)}, "
                f"{len(first_iter_counts)}",
            )

        # first break up the requests into individual iterable events
        events = defaultdict(int)
        global_start = min(start for start, _ in requests) if requests else 0
        global_end = max(end for _, end in requests) if requests else 0
        events[global_start] = 0
        events[global_end] = 0

        for (_, end), first_iter, first_iter_count, total_count in zip(
            requests, first_iter_times, first_iter_counts, iter_counts
        ):
            events[first_iter] += first_iter_count

            if total_count > 1:
                iter_latency = (end - first_iter) / (total_count - 1)
                for ind in range(1, total_count):
                    events[first_iter + ind * iter_latency] += 1

        # combine any events that are very close together
        flattened_events: list[tuple[float, int]] = []

        for time, count in sorted(events.items()):
            last_time, last_count = (
                flattened_events[-1] if flattened_events else (None, None)
            )

            if (
                last_time is not None
                and last_count is not None
                and abs(last_time - time) <= epsilon
            ):
                flattened_events[-1] = (last_time, last_count + count)
            else:
                flattened_events.append((time, count))

        # convert to value distribution function
        distribution: dict[float, float] = defaultdict(float)

        for ind in range(len(flattened_events) - 1):
            start_time, count = flattened_events[ind]
            end_time, _ = flattened_events[ind + 1]
            duration = end_time - start_time
            rate = count / duration
            distribution[rate] += duration

        distribution_list = sorted(distribution.items())

        return DistributionSummary.from_distribution_function(
            distribution=distribution_list,
            include_cdf=include_cdf,
        )


class StatusDistributionSummary(
    StatusBreakdown[
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
    ]
):
    """
    Status-grouped statistical summary for request processing analysis.

    Provides comprehensive statistical analysis grouped by request status (total,
    successful, incomplete, errored). Enables performance analysis across different
    request outcomes for benchmarking and monitoring applications. Each status
    category maintains complete DistributionSummary metrics.

    Example:
    ::
        status_summary = StatusDistributionSummary.from_values(
            value_types=["successful", "error", "successful"],
            values=[1.5, 10.0, 2.1]
        )
        print(f"Success mean: {status_summary.successful.mean}")
        print(f"Error rate: {status_summary.errored.count}")
    """

    @staticmethod
    def from_values(
        value_types: list[Literal["successful", "incomplete", "error"]],
        values: list[float],
        weights: list[float] | None = None,
        include_cdf: bool = False,
    ) -> StatusDistributionSummary:
        """
        Create status-grouped statistical summary from values and status types.

        Groups numerical values by request status and calculates complete
        statistical summaries for each category. Enables performance analysis
        across different request outcomes.

        :param value_types: Status type for each value ("successful", "incomplete",
            or "error")
        :param values: Numerical values representing the distribution
        :param weights: Optional weights for each value (defaults to equal weighting)
        :param include_cdf: Whether to include cumulative distribution functions
        :return: StatusDistributionSummary with statistics grouped by status
        :raises ValueError: If input lists have mismatched lengths or invalid
            status types
        """
        if any(
            type_ not in {"successful", "incomplete", "error"} for type_ in value_types
        ):
            raise ValueError(
                "value_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {value_types} instead.",
            )

        if weights is None:
            weights = [1.0] * len(values)

        if len(value_types) != len(values) or len(value_types) != len(weights):
            raise ValueError(
                "The length of value_types, values, and weights must be the same.",
            )

        _, successful_values, successful_weights = (
            zip(*successful)
            if (
                successful := list(
                    filter(
                        lambda val: val[0] == "successful",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )
        _, incomplete_values, incomplete_weights = (
            zip(*incomplete)
            if (
                incomplete := list(
                    filter(
                        lambda val: val[0] == "incomplete",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )
        _, errored_values, errored_weights = (
            zip(*errored)
            if (
                errored := list(
                    filter(
                        lambda val: val[0] == "error",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_values(
                values,
                weights,
                include_cdf=include_cdf,
            ),
            successful=DistributionSummary.from_values(
                successful_values,  # type: ignore[arg-type]
                successful_weights,  # type: ignore[arg-type]
                include_cdf=include_cdf,
            ),
            incomplete=DistributionSummary.from_values(
                incomplete_values,  # type: ignore[arg-type]
                incomplete_weights,  # type: ignore[arg-type]
                include_cdf=include_cdf,
            ),
            errored=DistributionSummary.from_values(
                errored_values,  # type: ignore[arg-type]
                errored_weights,  # type: ignore[arg-type]
                include_cdf=include_cdf,
            ),
        )

    @staticmethod
    def from_request_times(
        request_types: list[Literal["successful", "incomplete", "error"]],
        requests: list[tuple[float, float]],
        distribution_type: Literal["concurrency", "rate"],
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create status-grouped statistical summary from request timing data.

        Analyzes request timings grouped by status to calculate concurrency or
        rate distributions for each outcome category. Enables comparative
        performance analysis across successful, incomplete, and errored requests.

        :param request_types: Status type for each request ("successful",
            "incomplete", or "error")
        :param requests: List of (start_time, end_time) tuples for each request
        :param distribution_type: Analysis type - "concurrency" or "rate"
        :param include_cdf: Whether to include cumulative distribution functions
        :param epsilon: Threshold for merging close timing events
        :return: StatusDistributionSummary with timing statistics by status
        :raises ValueError: If input lists have mismatched lengths or invalid types
        """
        if distribution_type not in {"concurrency", "rate"}:
            raise ValueError(
                f"Invalid distribution_type '{distribution_type}'. "
                "Must be 'concurrency' or 'rate'."
            )

        if any(
            type_ not in {"successful", "incomplete", "error"}
            for type_ in request_types
        ):
            raise ValueError(
                "request_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {request_types} instead.",
            )

        if len(request_types) != len(requests):
            raise ValueError(
                "The length of request_types and requests must be the same. "
                f"Got {len(request_types)} and {len(requests)} instead.",
            )

        _, successful_requests = (
            zip(*successful)
            if (
                successful := list(
                    filter(
                        lambda val: val[0] == "successful",
                        zip(request_types, requests),
                    )
                )
            )
            else ([], [])
        )
        _, incomplete_requests = (
            zip(*incomplete)
            if (
                incomplete := list(
                    filter(
                        lambda val: val[0] == "incomplete",
                        zip(request_types, requests),
                    )
                )
            )
            else ([], [])
        )
        _, errored_requests = (
            zip(*errored)
            if (
                errored := list(
                    filter(
                        lambda val: val[0] == "error",
                        zip(request_types, requests),
                    )
                )
            )
            else ([], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_request_times(
                requests,
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.from_request_times(
                successful_requests,  # type: ignore[arg-type]
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.from_request_times(
                incomplete_requests,  # type: ignore[arg-type]
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.from_request_times(
                errored_requests,  # type: ignore[arg-type]
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
        )

    @staticmethod
    def from_iterable_request_times(
        request_types: list[Literal["successful", "incomplete", "error"]],
        requests: list[tuple[float, float]],
        first_iter_times: list[float],
        iter_counts: list[int] | None = None,
        first_iter_counts: list[int] | None = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create status-grouped statistical summary from iterative request timing data.

        Analyzes autoregressive request timings grouped by status to calculate
        iteration rate distributions for each outcome category. Enables comparative
        analysis of token generation or streaming response performance across
        different request statuses.

        :param request_types: Status type for each request ("successful",
            "incomplete", or "error")
        :param requests: List of (start_time, end_time) tuples for each request
        :param first_iter_times: Times when first iteration was received for
            each request
        :param iter_counts: Total iteration counts for each request (defaults to 1)
        :param first_iter_counts: Iteration counts for first iteration (defaults
            to 1)
        :param include_cdf: Whether to include cumulative distribution functions
        :param epsilon: Threshold for merging close timing events
        :return: StatusDistributionSummary with iteration statistics by status
        :raises ValueError: If input lists have mismatched lengths or invalid types
        """
        if any(
            type_ not in {"successful", "incomplete", "error"}
            for type_ in request_types
        ):
            raise ValueError(
                "request_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {request_types} instead.",
            )

        if iter_counts is None:
            iter_counts = [1] * len(requests)

        if first_iter_counts is None:
            first_iter_counts = [1] * len(requests)

        if (
            len(request_types) != len(requests)
            or len(requests) != len(first_iter_times)
            or len(requests) != len(iter_counts)
            or len(requests) != len(first_iter_counts)
        ):
            raise ValueError(
                "request_types, requests, first_iter_times, iter_counts, and "
                "first_iter_counts must be the same length."
                f"Given {len(request_types)}, {len(requests)}, "
                f"{len(first_iter_times)}, {len(iter_counts)}, "
                f"{len(first_iter_counts)}",
            )

        (
            _,
            successful_requests,
            successful_first_iter_times,
            successful_iter_counts,
            successful_first_iter_counts,
        ) = (
            zip(*successful)
            if (
                successful := list(
                    filter(
                        lambda val: val[0] == "successful",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )
        (
            _,
            incomplete_requests,
            incomplete_first_iter_times,
            incomplete_iter_counts,
            incomplete_first_iter_counts,
        ) = (
            zip(*incomplete)
            if (
                incomplete := list(
                    filter(
                        lambda val: val[0] == "incomplete",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )
        (
            _,
            errored_requests,
            errored_first_iter_times,
            errored_iter_counts,
            errored_first_iter_counts,
        ) = (
            zip(*errored)
            if (
                errored := list(
                    filter(
                        lambda val: val[0] == "error",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_iterable_request_times(
                requests,
                first_iter_times,
                iter_counts,
                first_iter_counts,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.from_iterable_request_times(
                successful_requests,  # type: ignore[arg-type]
                successful_first_iter_times,  # type: ignore[arg-type]
                successful_iter_counts,  # type: ignore[arg-type]
                successful_first_iter_counts,  # type: ignore[arg-type]
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.from_iterable_request_times(
                incomplete_requests,  # type: ignore[arg-type]
                incomplete_first_iter_times,  # type: ignore[arg-type]
                incomplete_iter_counts,  # type: ignore[arg-type]
                incomplete_first_iter_counts,  # type: ignore[arg-type]
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.from_iterable_request_times(
                errored_requests,  # type: ignore[arg-type]
                errored_first_iter_times,  # type: ignore[arg-type]
                errored_iter_counts,  # type: ignore[arg-type]
                errored_first_iter_counts,  # type: ignore[arg-type]
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
        )


class RunningStats(StandardBaseModel):
    """
    Real-time statistics tracking for streaming numerical data.

    Maintains mean, rate, and cumulative statistics for continuous data streams
    without storing individual values. Optimized for memory efficiency in
    long-running monitoring applications. Supports arithmetic operators for
    convenient value addition and provides computed properties for derived metrics.

    Example:
    ::
        stats = RunningStats()
        stats += 10.5  # Add value using operator
        stats.update(20.0, count=3)  # Add value with custom count
        print(f"Mean: {stats.mean}, Rate: {stats.rate}")
    """

    start_time: float = Field(
        default_factory=timer.time,
        description=(
            "The time the running statistics object was created. "
            "This is used to calculate the rate of the statistics."
        ),
    )
    count: int = Field(
        default=0,
        description="The number of values added to the running statistics.",
    )
    total: float = Field(
        default=0.0,
        description="The total sum of the values added to the running statistics.",
    )
    last: float = Field(
        default=0.0,
        description="The last value added to the running statistics.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def mean(self) -> float:
        """
        :return: The mean of the running statistics (total / count).
            If count is 0, return 0.0.
        """
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @computed_field  # type: ignore[misc]
    @property
    def rate(self) -> float:
        """
        :return: The rate of the running statistics
            (total / (time.time() - start_time)).
            If count is 0, return 0.0.
        """
        if self.count == 0:
            return 0.0
        return self.total / (timer.time() - self.start_time)

    def __add__(self, value: Any) -> float:
        """
        Add value using + operator and return current mean.

        :param value: Numerical value to add to the running statistics
        :return: Updated mean after adding the value
        :raises ValueError: If value is not numeric (int or float)
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self.mean

    def __iadd__(self, value: Any) -> RunningStats:
        """
        Add value using += operator and return updated instance.

        :param value: Numerical value to add to the running statistics
        :return: Self reference for method chaining
        :raises ValueError: If value is not numeric (int or float)
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self

    def update(self, value: float, count: int = 1) -> None:
        """
        Update running statistics with new value and count.

        :param value: Numerical value to add to the running statistics
        :param count: Number of occurrences to count for this value (defaults to 1)
        """
        self.count += count
        self.total += value
        self.last = value


class TimeRunningStats(RunningStats):
    """
    Specialized running statistics for time-based measurements.

    Extends RunningStats with time-specific computed properties for millisecond
    conversions. Designed for tracking latency, duration, and timing metrics in
    performance monitoring applications.

    Example:
    ::
        time_stats = TimeRunningStats()
        time_stats += 0.125  # Add 125ms in seconds
        print(f"Mean: {time_stats.mean_ms}ms, Total: {time_stats.total_ms}ms")
    """

    @computed_field  # type: ignore[misc]
    @property
    def total_ms(self) -> float:
        """
        :return: The total time multiplied by 1000.0 to convert to milliseconds.
        """
        return self.total * 1000.0

    @computed_field  # type: ignore[misc]
    @property
    def last_ms(self) -> float:
        """
        :return: The last time multiplied by 1000.0 to convert to milliseconds.
        """
        return self.last * 1000.0

    @computed_field  # type: ignore[misc]
    @property
    def mean_ms(self) -> float:
        """
        :return: The mean time multiplied by 1000.0 to convert to milliseconds.
        """
        return self.mean * 1000.0

    @computed_field  # type: ignore[misc]
    @property
    def rate_ms(self) -> float:
        """
        :return: The rate of the running statistics multiplied by 1000.0
            to convert to milliseconds.
        """
        return self.rate * 1000.0
