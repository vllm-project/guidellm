"""
Statistical distribution analysis and summary calculations for benchmark metrics.

Provides comprehensive statistical analysis tools including percentile calculations,
summary statistics, and status-based distributions. Supports value distributions,
time-based rate and concurrency distributions with weighted sampling, and probability
density functions for analyzing benchmark performance metrics and request patterns
across different status categories (successful, incomplete, errored).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeVar

import numpy as np
from pydantic import Field

from guidellm.schemas.base import StandardBaseModel, StatusBreakdown

__all__ = [
    "DistributionSummary",
    "FunctionObjT",
    "Percentiles",
    "StatusDistributionSummary",
]

FunctionObjT = TypeVar("FunctionObjT")


class Percentiles(StandardBaseModel):
    """
    Standard percentile values for probability distributions.

    Captures key percentile points from 0.1th to 99.9th percentile for comprehensive
    distribution analysis, enabling assessment of central tendency, spread, and tail
    behavior in benchmark metrics.
    """

    p001: float = Field(description="0.1th percentile value")
    p01: float = Field(description="1st percentile value")
    p05: float = Field(description="5th percentile value")
    p10: float = Field(description="10th percentile value")
    p25: float = Field(description="25th percentile value")
    p50: float = Field(description="50th percentile (median) value")
    p75: float = Field(description="75th percentile value")
    p90: float = Field(description="90th percentile value")
    p95: float = Field(description="95th percentile value")
    p99: float = Field(description="99th percentile value")
    p999: float = Field(description="99.9th percentile value")

    @classmethod
    def from_pdf(
        cls, pdf: np.ndarray, epsilon: float = 1e-6, validate: bool = True
    ) -> Percentiles:
        """
        Create percentiles from a probability density function.

        :param pdf: 2D array (N, 2) with values in column 0 and probabilities in
            column 1
        :param epsilon: Tolerance for probability sum validation
        :param validate: Whether to validate probabilities sum to 1 and are
            non-negative
        :return: Percentiles object with computed values
        :raises ValueError: If PDF shape is invalid, probabilities are negative,
            or probabilities don't sum to 1
        """
        expected_shape = (None, 2)

        if len(pdf.shape) != len(expected_shape) or pdf.shape[1] != expected_shape[1]:
            raise ValueError(
                "PDF must be a 2D array of shape (N, 2) where first column is values "
                f"and second column is probabilities. Got {pdf.shape} instead."
            )

        percentile_probs = {
            "p001": 0.001,
            "p01": 0.01,
            "p05": 0.05,
            "p10": 0.1,
            "p25": 0.25,
            "p50": 0.5,
            "p75": 0.75,
            "p90": 0.9,
            "p95": 0.95,
            "p99": 0.99,
            "p999": 0.999,
        }

        if pdf.shape[0] == 0:
            return Percentiles(**dict.fromkeys(percentile_probs.keys(), 0.0))

        probabilities = pdf[:, 1]

        if validate:
            if np.any(probabilities < 0):
                raise ValueError("Probabilities must be non-negative.")

            prob_sum = np.sum(probabilities)
            if abs(prob_sum - 1.0) > epsilon:
                raise ValueError(f"Probabilities must sum to 1, got {prob_sum}.")

        cdf_probs = np.cumsum(probabilities)

        return Percentiles(
            **{
                key: pdf[np.searchsorted(cdf_probs, value, side="left"), 0].item()
                for key, value in percentile_probs.items()
            }
        )


class DistributionSummary(StandardBaseModel):
    """
    Comprehensive statistical summary of a probability distribution.

    Captures central tendency (mean, median, mode), spread (variance, std_dev),
    extrema (min, max), and percentile information with optional probability density
    function. Supports creation from raw values, PDFs, or time-based event data for
    rate and concurrency analysis in benchmark metrics.
    """

    mean: float = Field(description="Mean/average value")
    median: float = Field(description="Median (50th percentile) value")
    mode: float = Field(description="Mode (most probable) value")
    variance: float = Field(description="Variance of the distribution")
    std_dev: float = Field(description="Standard deviation")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    count: int = Field(description="Number of observations")
    total_sum: float = Field(description="Sum of all values")
    percentiles: Percentiles = Field(description="Standard percentile values")
    pdf: list[tuple[float, float]] | None = Field(
        description="Probability density function as (value, probability) pairs",
        default=None,
    )

    @classmethod
    def from_pdf(
        cls,
        pdf: np.ndarray,
        count: int | None = None,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
        validate: bool = True,
    ) -> DistributionSummary:
        """
        Create distribution summary from a probability density function.

        :param pdf: 2D array (N, 2) with values in column 0 and probabilities in
            column 1
        :param count: Number of original observations; defaults to PDF length
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :param validate: Whether to validate probabilities sum to 1 and are non-negative
        :return: Complete distribution summary with statistics
        :raises ValueError: If PDF shape is invalid or probabilities are invalid
        """
        expected_shape = (None, 2)

        if len(pdf.shape) != len(expected_shape) or pdf.shape[1] != expected_shape[1]:
            raise ValueError(
                "PDF must be a 2D array of shape (N, 2) where first column is values "
                f"and second column is probabilities. Got {pdf.shape} instead."
            )

        if pdf.shape[0] == 0:
            return DistributionSummary(
                mean=0.0,
                median=0.0,
                mode=0.0,
                variance=0.0,
                std_dev=0.0,
                min=0.0,
                max=0.0,
                count=0 if count is None else count,
                total_sum=0.0,
                percentiles=Percentiles.from_pdf(pdf, epsilon=epsilon),
                pdf=None if include_pdf is False else [],
            )

        # Calculate stats
        values = pdf[:, 0]
        probabilities = pdf[:, 1]

        if validate:
            # Fail if probabilities don't sum to 1 or are negative
            if np.any(probabilities < 0):
                raise ValueError("Probabilities must be non-negative.")

            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, atol=epsilon):
                raise ValueError(f"Probabilities must sum to 1.0 (sum={prob_sum}).")

            # Fail if values are not sorted
            if not np.all(values[:-1] <= values[1:]):
                raise ValueError("Values in PDF must be sorted in ascending order.")

        percentiles = Percentiles.from_pdf(pdf, epsilon=epsilon, validate=False)
        median = percentiles.p50
        mean = np.sum(values * probabilities).item()
        mode = values[np.argmax(probabilities)].item()
        variance = np.sum((values - mean) ** 2 * probabilities).item()
        std_dev = math.sqrt(variance)
        minimum = values[0].item()
        maximum = values[-1].item()

        if count is None:
            count = len(pdf)

        total_sum = mean * count
        sampled_pdf = cls._sample_pdf(pdf, include_pdf)

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
            percentiles=percentiles,
            pdf=sampled_pdf,
        )

    @classmethod
    def _sample_pdf(
        cls, pdf: np.ndarray, include_pdf: bool | int
    ) -> list[tuple[float, float]] | None:
        """
        Sample PDF based on include_pdf parameter.

        :param pdf: PDF array to sample
        :param include_pdf: False for None, True for full, int for sampled size
        :return: Sampled PDF as list of tuples or None
        """
        if include_pdf is False:
            return None
        if include_pdf is True:
            return pdf.tolist()
        if isinstance(include_pdf, int) and include_pdf > 0:
            if len(pdf) <= include_pdf:
                return pdf.tolist()
            sample_indices = np.linspace(0, len(pdf) - 1, include_pdf, dtype=int)
            return pdf[sample_indices].tolist()
        return []

    @classmethod
    def from_values(
        cls,
        values: Sequence[float | tuple[float, float]] | np.ndarray,
        count: int | None = None,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> DistributionSummary:
        """
        Create distribution summary from raw values with optional weights.

        :param values: Values or (value, weight) tuples, or numpy array
        :param count: Number of original observations; defaults to sum of weights
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Distribution summary computed from the values
        :raises ValueError: If total weight is zero or invalid
        """
        np_values = cls._to_weighted_ndarray(values, num_values_per_item=2)

        if np_values.shape[0] == 0:
            return DistributionSummary.from_pdf(
                pdf=np.empty((0, 2)), count=0, include_pdf=include_pdf, epsilon=epsilon
            )

        if count is None:
            count = round(np.sum(np_values[:, 1]).item())

        # Sort values and weights by values
        sort_ind = np.argsort(np_values[:, 0])
        sorted_values = np_values[sort_ind, 0]
        sorted_weights = np_values[sort_ind, 1]

        # Combine any duplicate values by summing their weights
        unique_values, inverse_indices = np.unique(sorted_values, return_inverse=True)
        combined_weights = np.zeros_like(unique_values, dtype=float)
        np.add.at(combined_weights, inverse_indices, sorted_weights)

        # Remove any values with zero weight
        nonzero_mask = combined_weights > 0
        final_values = unique_values[nonzero_mask]
        final_weights = combined_weights[nonzero_mask]

        # Create PDF by normalizing weights and stacking
        total_weight = np.sum(final_weights)
        if total_weight <= epsilon:
            # No valid weights to create PDF, overwrite to uniform distribution
            final_weights = np.ones_like(final_values)
            total_weight = np.sum(final_weights)

        probabilities = final_weights / total_weight
        pdf = np.column_stack((final_values, probabilities))

        return DistributionSummary.from_pdf(
            pdf=pdf,
            count=count,
            include_pdf=include_pdf,
            epsilon=epsilon,
            validate=False,
        )

    @classmethod
    def rate_distribution_from_timings(
        cls,
        event_times: Sequence[float | tuple[float, float]] | np.ndarray,
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,  # 1/10th of a millisecond
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> DistributionSummary:
        """
        Create rate distribution from event timestamps.

        Computes event rates over time intervals weighted by interval duration for
        analyzing request throughput patterns.

        :param event_times: Event timestamps or (timestamp, weight) tuples
        :param start_time: Analysis window start; filters earlier events
        :param end_time: Analysis window end; filters later events
        :param threshold: Time threshold for merging nearby events; 1/10th millisecond
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Distribution summary of event rates over time
        """
        weighted_times = cls._to_weighted_ndarray(event_times, num_values_per_item=2)

        if start_time is not None:
            # Filter out any times before start, insert start time with 0 weight
            weighted_times = np.insert(
                weighted_times[weighted_times[:, 0] >= start_time],
                0,
                [start_time, 0.0],
                axis=0,
            )

        if end_time is not None:
            # Filter out any times after end, insert end time with 0 weight
            weighted_times = np.append(
                weighted_times[weighted_times[:, 0] <= end_time],
                [[end_time, 0.0]],
                axis=0,
            )

        # Sort by time for merging, merge any times within threshold
        sort_ind = np.argsort(weighted_times[:, 0])
        weighted_times = weighted_times[sort_ind]
        weighted_times = cls._merge_sorted_times_with_weights(weighted_times, threshold)

        if len(weighted_times) <= 1:
            # No data to calculate rates from (need at least two times)
            return cls.from_values(
                [],
                count=len(weighted_times),
                include_pdf=include_pdf,
                epsilon=epsilon,
            )

        times = weighted_times[:, 0]
        occurrences = weighted_times[:, 1]

        # Calculate local duration for each event: ((times[i+1] - times[i-1])) / 2
        midpoints = (times[1:] + times[:-1]) / 2
        durations = np.empty_like(times)
        durations[0] = midpoints[0] - times[0]
        durations[1:-1] = midpoints[1:] - midpoints[:-1]
        durations[-1] = np.clip(times[-1] - midpoints[-1], epsilon, None)

        # Calculate rate at each interval: occurences[i] / duration[i]
        rates = occurrences / durations
        count = round(np.sum(occurrences).item())

        return cls.from_values(
            np.column_stack((rates, durations)),
            count=count,
            include_pdf=include_pdf,
            epsilon=epsilon,
        )

    @classmethod
    def concurrency_distribution_from_timings(
        cls,
        event_intervals: (
            Sequence[tuple[float, float] | tuple[float, float, float]] | np.ndarray
        ),
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,  # 1/10th of a millisecond
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> DistributionSummary:
        """
        Create concurrency distribution from event time intervals.

        Tracks overlapping events to compute concurrency levels over time for analyzing
        request processing patterns and resource utilization.

        :param event_intervals: Event (start, end) or (start, end, weight) tuples
        :param start_time: Analysis window start
        :param end_time: Analysis window end
        :param threshold: Time threshold for merging nearby transitions;
            1/10th millisecond
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Distribution summary of concurrency levels over time
        """
        weighted_intervals = cls._to_weighted_ndarray(
            event_intervals, num_values_per_item=3
        )

        # If start_time, filter any intervals that end before start_time
        if start_time is not None:
            keep_mask = weighted_intervals[:, 1] >= start_time
            weighted_intervals = weighted_intervals[keep_mask]

        # If end_time, filter any intervals that start after end_time
        if end_time is not None:
            keep_mask = weighted_intervals[:, 0] <= end_time
            weighted_intervals = weighted_intervals[keep_mask]

        count = len(weighted_intervals)

        # Convert to concurrency changes at each time
        add_occurences = (
            np.stack(
                (
                    weighted_intervals[:, 0],
                    weighted_intervals[:, 2],
                ),
                axis=1,
            )
            if len(weighted_intervals) > 0
            else np.empty((0, 2))
        )
        remove_occurences = (
            np.stack(
                (
                    weighted_intervals[:, 1],
                    -1 * weighted_intervals[:, 2],
                ),
                axis=1,
            )
            if len(weighted_intervals) > 0
            else np.empty((0, 2))
        )

        # Combine add and remove occurences into weighted times
        weighted_times = np.vstack((add_occurences, remove_occurences))

        # Sort by the times and merge any times within threshold
        weighted_times = weighted_times[np.argsort(weighted_times[:, 0])]
        weighted_times = cls._merge_sorted_times_with_weights(weighted_times, threshold)

        # If start_time, ensure included (if any before, add final concurrency at start)
        if start_time is not None and len(weighted_times) > 0:
            start_ind = np.searchsorted(weighted_times[:, 0], start_time, side="left")
            prior_delta = (
                np.sum(weighted_times[:start_ind, 1]) if start_ind > 0 else 0.0
            )
            weighted_times = np.insert(
                weighted_times[start_ind:], 0, [start_time, prior_delta], axis=0
            )

        # If end_time, ensure included (if any after, filter out)
        if end_time is not None and len(weighted_times) > 0:
            end_ind = np.searchsorted(weighted_times[:, 0], end_time, side="right")
            weighted_times = np.append(
                weighted_times[:end_ind], [[end_time, 0.0]], axis=0
            )

        # Calculate concurrency from cumulative sum of changes over time
        concurrencies = np.clip(np.cumsum(weighted_times[:, 1]), 0, None)

        if len(concurrencies) <= 1:
            # No data to calculate concurrency from
            return cls.from_values(
                [] if count == 0 else [concurrencies[0].item()],
                include_pdf=include_pdf,
                epsilon=epsilon,
            )

        # Calculate durations equal to times[i+1] - times[i]
        # The last concurrency level is not used since no following time point
        durations = np.clip(np.diff(weighted_times[:, 0]), 0, None)
        values = np.column_stack((concurrencies[:-1], durations))

        return (
            cls.from_values(
                values,
                count=count,
                include_pdf=include_pdf,
                epsilon=epsilon,
            )
            if np.any(durations > 0)
            else cls.from_values(
                [],
                count=count,
                include_pdf=include_pdf,
                epsilon=epsilon,
            )
        )

    @classmethod
    def _to_weighted_ndarray(
        cls,
        inputs: (
            Sequence[float | tuple[float, float] | tuple[float, float, float]]
            | np.ndarray
        ),
        num_values_per_item: Literal[2, 3],
    ) -> np.ndarray:
        if not isinstance(inputs, np.ndarray):
            # Convert list to structured numpy array with dims (N, num_dimensions)
            # Fill in missing weights with 1.0
            return cls._sequence_to_weighted_ndarray(inputs, num_values_per_item)

        if len(inputs.shape) == 1:
            # 1D array: reshape to (N, 1) and add weights column
            inputs = inputs.reshape(-1, 1)
            weights = np.ones((inputs.shape[0], 1), dtype=float)

            return (
                np.hstack((inputs, weights))
                if num_values_per_item == 2  # noqa: PLR2004
                else np.hstack((inputs, inputs, weights))
            )

        if len(inputs.shape) == 2 and inputs.shape[1] == num_values_per_item - 1:  # noqa: PLR2004
            # Add weights column of 1.0
            weights = np.ones((inputs.shape[0], 1), dtype=float)

            return np.hstack((inputs, weights))

        if len(inputs.shape) == 2 and inputs.shape[1] == num_values_per_item:  # noqa: PLR2004
            return inputs

        raise ValueError(
            "inputs must be a numpy array of shape (N,), "
            f"(N, {num_values_per_item - 1}), or (N, {num_values_per_item}). "
            f"Got shape {inputs.shape}."
        )

    @classmethod
    def _sequence_to_weighted_ndarray(
        cls,
        inputs: Sequence[float | tuple[float, float] | tuple[float, float, float]],
        num_values_per_item: Literal[2, 3],
    ) -> np.ndarray:
        ndarray = np.empty((len(inputs), num_values_per_item), dtype=float)
        scalar_types: tuple[type, ...] = (int, float, np.integer, np.floating)

        for ind, val in enumerate(inputs):
            if isinstance(val, scalar_types):
                ndarray[ind, :] = (
                    (val, 1.0) if num_values_per_item == 2 else (val, val, 1.0)  # noqa: PLR2004
                )
            elif isinstance(val, tuple) and len(val) == num_values_per_item:
                ndarray[ind, :] = val
            elif isinstance(val, tuple) and len(val) == num_values_per_item - 1:
                ndarray[ind, :] = (
                    (val[0], 1.0) if num_values_per_item == 2 else (val[0], val[1], 1.0)  # noqa: PLR2004
                )
            else:
                raise ValueError(
                    "Each item must be a float or a tuple of "
                    f"{num_values_per_item} or {num_values_per_item - 1} "
                    "elements."
                )

        return ndarray

    @classmethod
    def _merge_sorted_times_with_weights(
        cls, weighted_times: np.ndarray, threshold: float | None
    ) -> np.ndarray:
        # First remove any exact duplicate times and sum their weights
        unique_times, inverse = np.unique(weighted_times[:, 0], return_inverse=True)
        unique_weights = np.zeros_like(unique_times, dtype=float)
        np.add.at(unique_weights, inverse, weighted_times[:, 1])
        weighted_times = np.column_stack((unique_times, unique_weights))

        if threshold is None or threshold <= 0.0:
            return weighted_times

        # Loop to merge times within threshold until no more merges possible
        # (loop due to possible overlapping merge groups)
        while weighted_times.shape[0] > 1:
            times = weighted_times[:, 0]
            weights = weighted_times[:, 1]

            # Find diffs between consecutive times, create mask for within-threshold
            diffs = np.diff(times)
            within = diffs <= threshold
            if not np.any(within):
                break

            # Start indices are marked by the transition from 0 to 1 in the mask
            # End indices found by searching for last time within threshold from start
            starts = np.where(np.diff(np.insert(within.astype(int), 0, 0)) == 1)[0]
            start_end_times = times[starts] + threshold
            ends = np.searchsorted(times, start_end_times, side="right") - 1

            # Collapse overlapping or chained merge groups
            if len(starts) > 1:
                valid_mask = np.concatenate([[True], starts[1:] > ends[:-1]])
                starts, ends = starts[valid_mask], ends[valid_mask]

            # Update weights at start indices to sum of merged weights
            cumsum = np.concatenate(([0.0], np.cumsum(weights)))
            weighted_times[starts, 1] = cumsum[ends + 1] - cumsum[starts]

            # Calculate vectorized mask for removing merged entries
            merged_events = np.zeros(len(weighted_times) + 1, dtype=int)
            np.add.at(merged_events, starts, 1)
            np.add.at(merged_events, ends + 1, -1)
            remove_mask = np.cumsum(merged_events[:-1]) > 0
            remove_mask[starts] = False  # Keep start indices

            # Remove merged entries, update weighted_times
            weights = weights[~remove_mask]
            times = times[~remove_mask]
            weighted_times = np.column_stack((times, weights))

        return weighted_times


class StatusDistributionSummary(
    StatusBreakdown[
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
    ]
):
    """
    Distribution summaries broken down by request status categories.

    Provides separate statistical analysis for successful, incomplete, and errored
    requests with total aggregate statistics. Enables status-aware performance analysis
    and SLO validation across different request outcomes in benchmark results.
    """

    @property
    def count(self) -> int:
        """
        :return: Total count of samples across all status categories
        """
        return self.total.count

    @property
    def total_sum(self) -> float:
        """
        :return: Total sum of values across all status categories
        """
        return self.total.total_sum

    @classmethod
    def from_values(
        cls,
        successful: Sequence[float | tuple[float, float]] | np.ndarray,
        incomplete: Sequence[float | tuple[float, float]] | np.ndarray,
        errored: Sequence[float | tuple[float, float]] | np.ndarray,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create status-broken-down distribution from values by status category.

        :param successful: Values or (value, weight) tuples for successful requests
        :param incomplete: Values or (value, weight) tuples for incomplete requests
        :param errored: Values or (value, weight) tuples for errored requests
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of distribution summaries
        """
        total, successful_arr, incomplete_arr, errored_arr = cls._combine_status_arrays(
            successful, incomplete, errored, num_values_per_item=2
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_values(
                total, include_pdf=include_pdf, epsilon=epsilon
            ),
            successful=DistributionSummary.from_values(
                successful_arr, include_pdf=include_pdf, epsilon=epsilon
            ),
            incomplete=DistributionSummary.from_values(
                incomplete_arr, include_pdf=include_pdf, epsilon=epsilon
            ),
            errored=DistributionSummary.from_values(
                errored_arr, include_pdf=include_pdf, epsilon=epsilon
            ),
        )

    @classmethod
    def from_values_function(
        cls,
        function: Callable[
            [FunctionObjT],
            float | tuple[float, float] | Sequence[float | tuple[float, float]] | None,
        ],
        successful: Sequence[FunctionObjT],
        incomplete: Sequence[FunctionObjT],
        errored: Sequence[FunctionObjT],
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create distribution summary by extracting values from objects via function.

        :param function: Function to extract value(s) from each object
        :param successful: Successful request objects
        :param incomplete: Incomplete request objects
        :param errored: Errored request objects
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of distribution summaries
        """

        def _extract_values(
            _objs: Sequence[FunctionObjT],
        ) -> Sequence[float | tuple[float, float]]:
            _outputs: list[float | tuple[float, float]] = []
            for _obj in _objs:
                if (_result := function(_obj)) is None:
                    continue
                if isinstance(_result, Sequence) and not isinstance(_result, tuple):
                    _outputs.extend(_result)
                else:
                    _outputs.append(_result)
            return _outputs

        return cls.from_values(
            successful=_extract_values(successful),
            incomplete=_extract_values(incomplete),
            errored=_extract_values(errored),
            include_pdf=include_pdf,
            epsilon=epsilon,
        )

    @classmethod
    def rate_distribution_from_timings(
        cls,
        successful: Sequence[float | tuple[float, float]] | np.ndarray,
        incomplete: Sequence[float | tuple[float, float]] | np.ndarray,
        errored: Sequence[float | tuple[float, float]] | np.ndarray,
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create status-broken-down rate distribution from event timestamps.

        :param successful: Timestamps for successful request events
        :param incomplete: Timestamps for incomplete request events
        :param errored: Timestamps for errored request events
        :param start_time: Analysis window start
        :param end_time: Analysis window end
        :param threshold: Time threshold for merging nearby events
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of rate distribution summaries
        """
        total, successful_arr, incomplete_arr, errored_arr = cls._combine_status_arrays(
            successful, incomplete, errored, num_values_per_item=2
        )

        return StatusDistributionSummary(
            total=DistributionSummary.rate_distribution_from_timings(
                total,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.rate_distribution_from_timings(
                successful_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.rate_distribution_from_timings(
                incomplete_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.rate_distribution_from_timings(
                errored_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
        )

    @classmethod
    def rate_distribution_from_timings_function(
        cls,
        function: Callable[
            [FunctionObjT],
            float | tuple[float, float] | Sequence[float | tuple[float, float]] | None,
        ],
        successful: Sequence[FunctionObjT],
        incomplete: Sequence[FunctionObjT],
        errored: Sequence[FunctionObjT],
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create rate distribution by extracting timestamps from objects via function.

        :param function: Function to extract timestamp(s) from each object
        :param successful: Successful request objects
        :param incomplete: Incomplete request objects
        :param errored: Errored request objects
        :param start_time: Analysis window start
        :param end_time: Analysis window end
        :param threshold: Time threshold for merging nearby events
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of rate distribution summaries
        """

        def _extract_values(
            _objs: Sequence[FunctionObjT],
        ) -> Sequence[float | tuple[float, float]]:
            _outputs: list[float | tuple[float, float]] = []
            for _obj in _objs:
                if (_result := function(_obj)) is None:
                    continue
                if isinstance(_result, Sequence) and not isinstance(_result, tuple):
                    _outputs.extend(_result)
                else:
                    _outputs.append(_result)
            return _outputs

        return cls.rate_distribution_from_timings(
            successful=_extract_values(successful),
            incomplete=_extract_values(incomplete),
            errored=_extract_values(errored),
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            include_pdf=include_pdf,
            epsilon=epsilon,
        )

    @classmethod
    def concurrency_distribution_from_timings(
        cls,
        successful: Sequence[tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        incomplete: Sequence[tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        errored: Sequence[tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create status-broken-down concurrency distribution from event intervals.

        :param successful: Event intervals for successful requests
        :param incomplete: Event intervals for incomplete requests
        :param errored: Event intervals for errored requests
        :param start_time: Analysis window start
        :param end_time: Analysis window end
        :param threshold: Time threshold for merging nearby transitions
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of concurrency distribution summaries
        """
        total, successful_arr, incomplete_arr, errored_arr = cls._combine_status_arrays(
            successful, incomplete, errored, num_values_per_item=3
        )

        return StatusDistributionSummary(
            total=DistributionSummary.concurrency_distribution_from_timings(
                total,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.concurrency_distribution_from_timings(
                successful_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.concurrency_distribution_from_timings(
                incomplete_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.concurrency_distribution_from_timings(
                errored_arr,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold,
                include_pdf=include_pdf,
                epsilon=epsilon,
            ),
        )

    @classmethod
    def concurrency_distribution_from_timings_function(
        cls,
        function: Callable[
            [FunctionObjT],
            tuple[float, float]
            | tuple[float, float, float]
            | Sequence[tuple[float, float] | tuple[float, float, float]]
            | None,
        ],
        successful: Sequence[FunctionObjT],
        incomplete: Sequence[FunctionObjT],
        errored: Sequence[FunctionObjT],
        start_time: float | None = None,
        end_time: float | None = None,
        threshold: float | None = 1e-4,
        include_pdf: bool | int = False,
        epsilon: float = 1e-6,
    ) -> StatusDistributionSummary:
        """
        Create concurrency distribution by extracting intervals from objects.

        :param function: Function to extract time interval(s) from each object
        :param successful: Successful request objects
        :param incomplete: Incomplete request objects
        :param errored: Errored request objects
        :param start_time: Analysis window start
        :param end_time: Analysis window end
        :param threshold: Time threshold for merging nearby transitions
        :param include_pdf: Whether to include PDF; True for full, int for sampled size
        :param epsilon: Tolerance for probability validation
        :return: Status breakdown of concurrency distribution summaries
        """

        def _extract_values(
            _objs: Sequence[FunctionObjT],
        ) -> Sequence[tuple[float, float] | tuple[float, float, float]]:
            _outputs: list[tuple[float, float] | tuple[float, float, float]] = []
            for _obj in _objs:
                if (_result := function(_obj)) is None:
                    continue
                if isinstance(_result, Sequence) and not isinstance(_result, tuple):
                    _outputs.extend(_result)
                else:
                    _outputs.append(_result)
            return _outputs

        return cls.concurrency_distribution_from_timings(
            successful=_extract_values(successful),
            incomplete=_extract_values(incomplete),
            errored=_extract_values(errored),
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            include_pdf=include_pdf,
            epsilon=epsilon,
        )

    @classmethod
    def _combine_status_arrays(
        cls,
        successful: Sequence[float | tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        incomplete: Sequence[float | tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        errored: Sequence[float | tuple[float, float] | tuple[float, float, float]]
        | np.ndarray,
        num_values_per_item: Literal[2, 3],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        successful_array = DistributionSummary._to_weighted_ndarray(  # noqa: SLF001
            successful, num_values_per_item=num_values_per_item
        )
        incomplete_array = DistributionSummary._to_weighted_ndarray(  # noqa: SLF001
            incomplete, num_values_per_item=num_values_per_item
        )
        errored_array = DistributionSummary._to_weighted_ndarray(  # noqa: SLF001
            errored, num_values_per_item=num_values_per_item
        )
        total_array = np.concatenate(
            (successful_array, incomplete_array, errored_array), axis=0
        )
        return total_array, successful_array, incomplete_array, errored_array
