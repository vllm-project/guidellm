import math
import time as timer
from collections import defaultdict
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field, computed_field

from guidellm.objects.pydantic import StandardBaseModel, StatusBreakdown

__all__ = [
    "DistributionSummary",
    "Percentiles",
    "RunningStats",
    "StatusDistributionSummary",
    "TimeRunningStats",
]


class Percentiles(StandardBaseModel):
    """
    A pydantic model representing the standard percentiles of a distribution.
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
    A pydantic model representing a statistical summary for a given
    distribution of numerical values.
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
    cumulative_distribution_function: Optional[list[tuple[float, float]]] = Field(
        description="The cumulative distribution function (CDF) of the distribution.",
        default=None,
    )

    @staticmethod
    def from_distribution_function(
        distribution: list[tuple[float, float]],
        include_cdf: bool = False,
    ) -> "DistributionSummary":
        """
        Create a statistical summary for a given distribution of weighted numerical
        values or a probability distribution function (PDF).
        1.  If the distribution is a PDF, it is expected to be a list of tuples
            where each tuple contains (value, probability). The sum of the
            probabilities should be 1. If it is not, it will be normalized.
        2.  If the distribution is a values distribution function, it is expected
            to be a list of tuples where each tuple contains (value, weight).
            The weights are normalized to a probability distribution function.

        :param distribution: A list of tuples representing the distribution.
            Each tuple contains (value, weight) or (value, probability).
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output DistributionSummary.
        :return: An instance of DistributionSummary with calculated values.
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
        weights: Optional[list[float]] = None,
        include_cdf: bool = False,
    ) -> "DistributionSummary":
        """
        Create a statistical summary for a given distribution of numerical values.
        This is a wrapper around from_distribution_function to handle the optional case
        of including weights for the values. If weights are not provided, they are
        automatically set to 1.0 for each value, so each value is equally weighted.

        :param values: A list of numerical values representing the distribution.
        :param weights: A list of weights for each value in the distribution.
            If not provided, all values are equally weighted.
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output DistributionSummary.
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
    ) -> "DistributionSummary":
        """
        Create a statistical summary for a given distribution of request times.
        Specifically, this is used to measure concurrency or rate of requests
        given an input list containing the start and end time of each request.
        This will first convert the request times into a distribution function
        and then calculate the statistics with from_distribution_function.

        :param requests: A list of tuples representing the start and end times of
            each request. Example: [(start_1, end_1), (start_2, end_2), ...]
        :param distribution_type: The type of distribution to calculate.
            Either "concurrency" or "rate".
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output DistributionSummary.
        :param epsilon: The epsilon value for merging close events.
        :return: An instance of DistributionSummary with calculated values.
        """
        if distribution_type == "concurrency":
            # convert to delta changes based on when requests were running
            events = [(start, 1) for start, _ in requests] + [
                (end, -1) for _, end in requests
            ]
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

        if distribution_type == "concurrency":
            # convert to the events over time measuring concurrency changes
            events_over_time: list[tuple[float, float]] = []
            active = 0
            for time, delta in flattened_events:
                active += delta  # type: ignore [assignment]
                events_over_time.append((time, active))

            flattened_events = events_over_time

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
        first_iter_counts: Optional[list[int]] = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "DistributionSummary":
        """
        Create a statistical summary for a given distribution of request times
        for a request with iterable responses between the start and end.
        For example, this is used to measure auto regressive requests where
        a request is started and at some later point, iterative responses are
        received. This will convert the request times and iterable values into
        a distribution function and then calculate the statistics with
        from_distribution_function.

        :param requests: A list of tuples representing the start and end times of
            each request. Example: [(start_1, end_1), (start_2, end_2), ...]
        :param first_iter_times: A list of times when the first iteration of
            each request was received. Must be the same length as requests.
        :param iter_counts: A list of the total number of iterations for each
            request that occurred starting at the first iteration and ending
            at the request end time. Must be the same length as requests.
        :param first_iter_counts: A list of the number of iterations to log
            for the first iteration of each request. For example, when calculating
            total number of tokens processed, this is set to the prompt tokens number.
            If not provided, defaults to 1 for each request.
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output DistributionSummary.
        :param epsilon: The epsilon value for merging close events.
        :return: An instance of DistributionSummary with calculated values.
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
    A pydantic model representing a statistical summary for a given
    distribution of numerical values grouped by status.
    Specifically used to represent the total, successful, incomplete,
    and errored values for a benchmark or other statistical summary.
    """

    @staticmethod
    def from_values(
        value_types: list[Literal["successful", "incomplete", "error"]],
        values: list[float],
        weights: Optional[list[float]] = None,
        include_cdf: bool = False,
    ) -> "StatusDistributionSummary":
        """
        Create a statistical summary by status for a given distribution of numerical
        values. This is used to measure the distribution of values for different
        statuses (e.g., successful, incomplete, error) and calculate the statistics
        for each status. Weights are optional to weight the probability distribution
        for each value by. If not provided, all values are equally weighted.

        :param value_types: A list of status types for each value in the distribution.
            Must be one of 'successful', 'incomplete', or 'error'.
        :param values: A list of numerical values representing the distribution.
            Must be the same length as value_types.
        :param weights: A list of weights for each value in the distribution.
            If not provided, all values are equally weighted (set to 1).
            Must be the same length as value_types.
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output StatusDistributionSummary.
        :return: An instance of StatusDistributionSummary with calculated values.
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
    ) -> "StatusDistributionSummary":
        """
        Create a statistical summary by status for given distribution of request times.
        This is used to measure the distribution of request times for different statuses
        (e.g., successful, incomplete, error) for concurrency and rates.
        This will call into DistributionSummary.from_request_times to calculate
        the statistics for each status.

        :param request_types: List of status types for each request in the distribution.
            Must be one of 'successful', 'incomplete', or 'error'.
        :param requests: A list of tuples representing the start and end times of
            each request. Example: [(start_1, end_1), (start_2, end_2), ...].
            Must be the same length as request_types.
        :param distribution_type: The type of distribution to calculate.
            Either "concurrency" or "rate".
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output StatusDistributionSummary.
        :param epsilon: The epsilon value for merging close events.
        :return: An instance of StatusDistributionSummary with calculated values.
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
        iter_counts: Optional[list[int]] = None,
        first_iter_counts: Optional[list[int]] = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "StatusDistributionSummary":
        """
        Create a statistical summary by status for given distribution of request times
        for a request with iterable responses between the start and end.
        For example, this is used to measure auto regressive requests where
        a request is started and at some later point, iterative responses are
        received. This will call into DistributionSummary.from_iterable_request_times
        to calculate the statistics for each status.

        :param request_types: List of status types for each request in the distribution.
            Must be one of 'successful', 'incomplete', or 'error'.
        :param requests: A list of tuples representing the start and end times of
            each request. Example: [(start_1, end_1), (start_2, end_2), ...].
            Must be the same length as request_types.
        :param first_iter_times: A list of times when the first iteration of
            each request was received. Must be the same length as requests.
        :param iter_counts: A list of the total number of iterations for each
            request that occurred starting at the first iteration and ending
            at the request end time. Must be the same length as requests.
            If not provided, defaults to 1 for each request.
        :param first_iter_counts: A list of the number of iterations to log
            for the first iteration of each request. For example, when calculating
            total number of tokens processed, this is set to the prompt tokens number.
            If not provided, defaults to 1 for each request.
        :param include_cdf: Whether to include the calculated cumulative distribution
            function (CDF) in the output StatusDistributionSummary.
        :param epsilon: The epsilon value for merging close events.
        :return: An instance of StatusDistributionSummary with calculated values.
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
    Create a running statistics object to track the mean, rate, and other
    statistics of a stream of values.
    1.  The start time is set to the time the object is created.
    2.  The count is set to 0.
    3.  The total is set to 0.
    4.  The last value is set to 0.
    5.  The mean is calculated as the total / count.
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
        Enable the use of the + operator to add a value to the running statistics.

        :param value: The value to add to the running statistics.
        :return: The mean of the running statistics.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self.mean

    def __iadd__(self, value: Any) -> "RunningStats":
        """
        Enable the use of the += operator to add a value to the running statistics.

        :param value: The value to add to the running statistics.
        :return: The running statistics object.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self

    def update(self, value: float, count: int = 1) -> None:
        """
        Update the running statistics with a new value.

        :param value: The new value to add to the running statistics.
        :param count: The number of times to 'count' for the value.
            If not provided, defaults to 1.
        """
        self.count += count
        self.total += value
        self.last = value


class TimeRunningStats(RunningStats):
    """
    Create a running statistics object to track the mean, rate, and other
    statistics of a stream of time values. This is used to track time values
    in milliseconds and seconds.

    Adds time specific computed_fields such as measurements in milliseconds and seconds.
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
