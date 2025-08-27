import math
import time
from typing import Literal

import numpy as np
import pytest

from guidellm.utils import (
    DistributionSummary,
    Percentiles,
    RunningStats,
    StatusDistributionSummary,
    TimeRunningStats,
)


def create_default_percentiles() -> Percentiles:
    return Percentiles(
        p001=0.1,
        p01=1.0,
        p05=5.0,
        p10=10.0,
        p25=25.0,
        p50=50.0,
        p75=75.0,
        p90=90.0,
        p95=95.0,
        p99=99.0,
        p999=99.9,
    )


def create_default_distribution_summary() -> DistributionSummary:
    return DistributionSummary(
        mean=50.0,
        median=50.0,
        mode=50.0,
        variance=835,
        std_dev=math.sqrt(835),
        min=0.0,
        max=100.0,
        count=1001,
        total_sum=50050.0,
        percentiles=create_default_percentiles(),
    )


@pytest.mark.smoke
def test_percentiles_initialization():
    percentiles = create_default_percentiles()
    assert percentiles.p001 == 0.1
    assert percentiles.p01 == 1.0
    assert percentiles.p05 == 5.0
    assert percentiles.p10 == 10.0
    assert percentiles.p25 == 25.0
    assert percentiles.p50 == 50.0
    assert percentiles.p75 == 75.0
    assert percentiles.p90 == 90.0
    assert percentiles.p95 == 95.0
    assert percentiles.p99 == 99.0
    assert percentiles.p999 == 99.9


@pytest.mark.smoke
def test_percentiles_invalid_initialization():
    test_kwargs = {
        "p001": 0.1,
        "p01": 1.0,
        "p05": 5.0,
        "p10": 10.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "p999": 99.9,
    }
    test_missing_keys = list(test_kwargs.keys())

    for missing_key in test_missing_keys:
        kwargs = {key: val for key, val in test_kwargs.items() if key != missing_key}
        with pytest.raises(ValueError):
            Percentiles(**kwargs)


@pytest.mark.smoke
def test_percentiles_marshalling():
    percentiles = create_default_percentiles()
    serialized = percentiles.model_dump()
    deserialized = Percentiles.model_validate(serialized)

    for key, value in vars(percentiles).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke
def test_distribution_summary_initilaization():
    distribution_summary = create_default_distribution_summary()
    assert distribution_summary.mean == 50.0
    assert distribution_summary.median == 50.0
    assert distribution_summary.mode == 50.0
    assert distribution_summary.variance == 835
    assert distribution_summary.std_dev == math.sqrt(835)
    assert distribution_summary.min == 0.0
    assert distribution_summary.max == 100.0
    assert distribution_summary.count == 1001
    assert distribution_summary.total_sum == 50050.0
    assert distribution_summary.percentiles.p001 == 0.1
    assert distribution_summary.percentiles.p01 == 1.0
    assert distribution_summary.percentiles.p05 == 5.0
    assert distribution_summary.percentiles.p10 == 10.0
    assert distribution_summary.percentiles.p25 == 25.0
    assert distribution_summary.percentiles.p50 == 50.0
    assert distribution_summary.percentiles.p75 == 75.0
    assert distribution_summary.percentiles.p90 == 90.0
    assert distribution_summary.percentiles.p95 == 95.0
    assert distribution_summary.percentiles.p99 == 99.0
    assert distribution_summary.percentiles.p999 == 99.9


@pytest.mark.smoke
def test_distribution_summary_invalid_initialization():
    test_kwargs = {
        "mean": 50.0,
        "median": 50.0,
        "mode": 50.0,
        "variance": 835,
        "std_dev": math.sqrt(835),
        "min": 0.0,
        "max": 100.0,
        "count": 1001,
        "total_sum": 50050.0,
        "percentiles": create_default_percentiles(),
    }
    test_missing_keys = list(test_kwargs.keys())
    for missing_key in test_missing_keys:
        kwargs = {key: val for key, val in test_kwargs.items() if key != missing_key}
        with pytest.raises(ValueError):
            DistributionSummary(**kwargs)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_distribution_summary_marshalling():
    distribution_summary = create_default_distribution_summary()
    serialized = distribution_summary.model_dump()
    deserialized = DistributionSummary.model_validate(serialized)

    for key, value in vars(distribution_summary).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke
def test_distribution_summary_from_distribution_function():
    values = [val / 10.0 for val in range(1001)]
    distribution = [(val, 1.0) for val in values]
    distribution_summary = DistributionSummary.from_distribution_function(distribution)
    assert distribution_summary.mean == pytest.approx(np.mean(values))
    assert distribution_summary.median == pytest.approx(np.median(values))
    assert distribution_summary.mode == 0.0
    assert distribution_summary.variance == pytest.approx(np.var(values, ddof=0))
    assert distribution_summary.std_dev == pytest.approx(np.std(values, ddof=0))
    assert distribution_summary.min == min(values)
    assert distribution_summary.max == max(values)
    assert distribution_summary.count == len(values)
    assert distribution_summary.total_sum == sum(values)
    assert distribution_summary.percentiles.p001 == pytest.approx(
        np.percentile(values, 0.1)
    )
    assert distribution_summary.percentiles.p01 == pytest.approx(
        np.percentile(values, 1.0)
    )
    assert distribution_summary.percentiles.p05 == pytest.approx(
        np.percentile(values, 5.0)
    )
    assert distribution_summary.percentiles.p10 == pytest.approx(
        np.percentile(values, 10.0)
    )
    assert distribution_summary.percentiles.p25 == pytest.approx(
        np.percentile(values, 25.0)
    )
    assert distribution_summary.percentiles.p50 == pytest.approx(
        np.percentile(values, 50.0)
    )
    assert distribution_summary.percentiles.p75 == pytest.approx(
        np.percentile(values, 75.0)
    )
    assert distribution_summary.percentiles.p90 == pytest.approx(
        np.percentile(values, 90.0)
    )
    assert distribution_summary.percentiles.p95 == pytest.approx(
        np.percentile(values, 95.0)
    )
    assert distribution_summary.percentiles.p99 == pytest.approx(
        np.percentile(values, 99.0)
    )
    assert distribution_summary.percentiles.p999 == pytest.approx(
        np.percentile(values, 99.9)
    )
    assert distribution_summary.cumulative_distribution_function is None

    distribution_summary_cdf = DistributionSummary.from_distribution_function(
        distribution, include_cdf=True
    )
    assert distribution_summary_cdf.cumulative_distribution_function is not None
    assert len(distribution_summary_cdf.cumulative_distribution_function) == len(values)


def test_distribution_summary_from_values():
    values = [val / 10 for val in range(1001)]
    distribution_summary = DistributionSummary.from_values(values)
    assert distribution_summary.mean == pytest.approx(np.mean(values))
    assert distribution_summary.median == pytest.approx(np.median(values))
    assert distribution_summary.mode == 0.0
    assert distribution_summary.variance == pytest.approx(np.var(values, ddof=0))
    assert distribution_summary.std_dev == pytest.approx(np.std(values, ddof=0))
    assert distribution_summary.min == min(values)
    assert distribution_summary.max == max(values)
    assert distribution_summary.count == len(values)
    assert distribution_summary.total_sum == sum(values)
    assert distribution_summary.percentiles.p001 == pytest.approx(
        np.percentile(values, 0.1)
    )
    assert distribution_summary.percentiles.p01 == pytest.approx(
        np.percentile(values, 1.0)
    )
    assert distribution_summary.percentiles.p05 == pytest.approx(
        np.percentile(values, 5.0)
    )
    assert distribution_summary.percentiles.p10 == pytest.approx(
        np.percentile(values, 10.0)
    )
    assert distribution_summary.percentiles.p25 == pytest.approx(
        np.percentile(values, 25.0)
    )
    assert distribution_summary.percentiles.p50 == pytest.approx(
        np.percentile(values, 50.0)
    )
    assert distribution_summary.percentiles.p75 == pytest.approx(
        np.percentile(values, 75.0)
    )
    assert distribution_summary.percentiles.p90 == pytest.approx(
        np.percentile(values, 90.0)
    )
    assert distribution_summary.percentiles.p95 == pytest.approx(
        np.percentile(values, 95.0)
    )
    assert distribution_summary.percentiles.p99 == pytest.approx(
        np.percentile(values, 99.0)
    )
    assert distribution_summary.percentiles.p999 == pytest.approx(
        np.percentile(values, 99.9)
    )
    assert distribution_summary.cumulative_distribution_function is None

    distribution_summary_weights = DistributionSummary.from_values(
        values, weights=[2] * len(values)
    )
    assert distribution_summary_weights.mean == pytest.approx(np.mean(values))
    assert distribution_summary_weights.median == pytest.approx(np.median(values))
    assert distribution_summary_weights.mode == 0.0
    assert distribution_summary_weights.variance == pytest.approx(
        np.var(values, ddof=0)
    )
    assert distribution_summary_weights.std_dev == pytest.approx(np.std(values, ddof=0))
    assert distribution_summary_weights.min == min(values)
    assert distribution_summary_weights.max == max(values)
    assert distribution_summary_weights.count == len(values)
    assert distribution_summary_weights.total_sum == sum(values)
    assert distribution_summary_weights.cumulative_distribution_function is None

    distribution_summary_cdf = DistributionSummary.from_values(values, include_cdf=True)
    assert distribution_summary_cdf.cumulative_distribution_function is not None
    assert len(distribution_summary_cdf.cumulative_distribution_function) == len(values)


def test_distribution_summary_from_request_times_concurrency():
    # create consistent timestamped values matching a rate of 10 per second
    requests = [(val / 10, val / 10 + 1) for val in range(10001)]
    distribution_summary = DistributionSummary.from_request_times(
        requests, distribution_type="concurrency"
    )
    assert distribution_summary.mean == pytest.approx(10.0, abs=0.01)
    assert distribution_summary.median == pytest.approx(10.0)
    assert distribution_summary.mode == 10.0
    assert distribution_summary.variance == pytest.approx(0, abs=0.1)
    assert distribution_summary.std_dev == pytest.approx(0, abs=0.3)
    assert distribution_summary.min == pytest.approx(1)
    assert distribution_summary.max == pytest.approx(10.0)
    assert distribution_summary.count == 10
    assert distribution_summary.total_sum == pytest.approx(55.0)
    assert distribution_summary.percentiles.p001 == pytest.approx(10, abs=5)
    assert distribution_summary.percentiles.p01 == pytest.approx(10)
    assert distribution_summary.percentiles.p05 == pytest.approx(10)
    assert distribution_summary.percentiles.p10 == pytest.approx(10)
    assert distribution_summary.percentiles.p25 == pytest.approx(10)
    assert distribution_summary.percentiles.p50 == pytest.approx(10)
    assert distribution_summary.percentiles.p75 == pytest.approx(10)
    assert distribution_summary.percentiles.p90 == pytest.approx(10)
    assert distribution_summary.percentiles.p95 == pytest.approx(10)
    assert distribution_summary.percentiles.p99 == pytest.approx(10)
    assert distribution_summary.percentiles.p999 == pytest.approx(10)
    assert distribution_summary.cumulative_distribution_function is None

    distribution_summary_cdf = DistributionSummary.from_request_times(
        requests, distribution_type="concurrency", include_cdf=True
    )
    assert distribution_summary_cdf.cumulative_distribution_function is not None
    assert len(distribution_summary_cdf.cumulative_distribution_function) == 10


def test_distribution_summary_from_request_times_rate():
    # create consistent timestamped values matching a rate of 10 per second
    requests = [(val / 10, val / 10 + 1) for val in range(10001)]
    distribution_summary = DistributionSummary.from_request_times(
        requests, distribution_type="rate"
    )
    assert distribution_summary.mean == pytest.approx(10.0, abs=0.01)
    assert distribution_summary.median == pytest.approx(10.0)
    assert distribution_summary.mode == pytest.approx(10.0)
    assert distribution_summary.variance == pytest.approx(0, abs=0.1)
    assert distribution_summary.std_dev == pytest.approx(0, abs=0.3)
    assert distribution_summary.min == pytest.approx(1.0)
    assert distribution_summary.max == pytest.approx(10.0)
    assert distribution_summary.count == 12
    assert distribution_summary.total_sum == pytest.approx(111.0)
    assert distribution_summary.percentiles.p001 == pytest.approx(10.0, abs=0.5)
    assert distribution_summary.percentiles.p01 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p05 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p10 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p25 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p50 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p75 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p90 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p95 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p99 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p999 == pytest.approx(10.0)
    assert distribution_summary.cumulative_distribution_function is None

    distribution_summary_cdf = DistributionSummary.from_request_times(
        requests, distribution_type="rate", include_cdf=True
    )
    assert distribution_summary_cdf.cumulative_distribution_function is not None
    assert len(distribution_summary_cdf.cumulative_distribution_function) == 12


def test_distribution_summary_from_iterable_request_times():
    # create consistent timestamped values matching a rate of 10 per second
    requests = [(val / 10, val / 10 + 1) for val in range(10001)]
    # create 9 iterations for each request with first iter at start + 0.1
    # and spaced at 0.1 seconds apart
    first_iter_times = [val / 10 + 0.1 for val in range(10001)]
    iter_counts = [9 for _ in range(10001)]
    first_iter_counts = [1 for _ in range(10001)]

    distribution_summary = DistributionSummary.from_iterable_request_times(
        requests, first_iter_times, iter_counts, first_iter_counts
    )
    assert distribution_summary.mean == pytest.approx(90.0, abs=0.1)
    assert distribution_summary.median == pytest.approx(80.0)
    assert distribution_summary.mode == pytest.approx(80.0)
    assert distribution_summary.variance == pytest.approx(704.463, abs=0.001)
    assert distribution_summary.std_dev == pytest.approx(26.541, abs=0.001)
    assert distribution_summary.min == pytest.approx(0.0)
    assert distribution_summary.max == pytest.approx(160.0)
    assert distribution_summary.count == 44
    assert distribution_summary.total_sum == pytest.approx(3538.85, abs=0.01)
    assert distribution_summary.percentiles.p001 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p01 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p05 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p10 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p25 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p50 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p75 == pytest.approx(80.0)
    assert distribution_summary.percentiles.p90 == pytest.approx(160.0)
    assert distribution_summary.percentiles.p95 == pytest.approx(160.0)
    assert distribution_summary.percentiles.p99 == pytest.approx(160.0)
    assert distribution_summary.percentiles.p999 == pytest.approx(160.0)
    assert distribution_summary.cumulative_distribution_function is None

    distribution_summary_cdf = DistributionSummary.from_iterable_request_times(
        requests, first_iter_times, iter_counts, first_iter_counts, include_cdf=True
    )
    assert distribution_summary_cdf.cumulative_distribution_function is not None
    assert len(distribution_summary_cdf.cumulative_distribution_function) == 44


def test_status_distribution_summary_initialization():
    status_distribution_summary = StatusDistributionSummary(
        total=create_default_distribution_summary(),
        successful=create_default_distribution_summary(),
        incomplete=create_default_distribution_summary(),
        errored=create_default_distribution_summary(),
    )
    assert status_distribution_summary.total.mean == 50.0
    assert status_distribution_summary.successful.mean == 50.0
    assert status_distribution_summary.incomplete.mean == 50.0
    assert status_distribution_summary.errored.mean == 50.0


def test_status_distribution_summary_marshalling():
    status_distribution_summary = StatusDistributionSummary(
        total=create_default_distribution_summary(),
        successful=create_default_distribution_summary(),
        incomplete=create_default_distribution_summary(),
        errored=create_default_distribution_summary(),
    )
    serialized = status_distribution_summary.model_dump()
    deserialized = StatusDistributionSummary.model_validate(serialized)

    for key, value in vars(status_distribution_summary).items():
        for child_key, child_value in vars(value).items():
            assert getattr(getattr(deserialized, key), child_key) == child_value


def test_status_distribution_summary_from_values():
    value_types: list[Literal["successful", "incomplete", "error"]] = [
        "successful",
        "incomplete",
        "error",
    ] * 1000
    values = [float(val % 3) for val in range(3000)]
    status_distribution_summary = StatusDistributionSummary.from_values(
        value_types, values
    )
    assert status_distribution_summary.total.count == len(values)
    assert status_distribution_summary.total.mean == pytest.approx(np.mean(values))
    assert status_distribution_summary.total.cumulative_distribution_function is None
    assert status_distribution_summary.successful.mean == pytest.approx(
        np.mean(
            [val for ind, val in enumerate(values) if value_types[ind] == "successful"]
        )
    )
    assert status_distribution_summary.successful.count == len(
        [val for ind, val in enumerate(values) if value_types[ind] == "successful"]
    )
    assert (
        status_distribution_summary.successful.cumulative_distribution_function is None
    )
    assert status_distribution_summary.incomplete.mean == pytest.approx(
        np.mean(
            [val for ind, val in enumerate(values) if value_types[ind] == "incomplete"]
        )
    )
    assert status_distribution_summary.incomplete.count == len(
        [val for ind, val in enumerate(values) if value_types[ind] == "incomplete"]
    )
    assert (
        status_distribution_summary.incomplete.cumulative_distribution_function is None
    )
    assert status_distribution_summary.errored.mean == pytest.approx(
        np.mean([val for ind, val in enumerate(values) if value_types[ind] == "error"])
    )
    assert status_distribution_summary.errored.count == len(
        [val for ind, val in enumerate(values) if value_types[ind] == "error"]
    )
    assert status_distribution_summary.errored.cumulative_distribution_function is None

    status_distribution_summary_cdf = StatusDistributionSummary.from_values(
        value_types, values, include_cdf=True
    )
    assert (
        status_distribution_summary_cdf.total.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.successful.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.incomplete.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.errored.cumulative_distribution_function
        is not None
    )


def test_status_distribution_summary_from_request_times():
    request_types: list[Literal["successful", "incomplete", "error"]] = [
        "successful",
        "incomplete",
        "error",
    ] * 1000
    requests = [((val % 3) / 10, (val % 3) / 10 + 1) for val in range(3000)]
    status_distribution_summary = StatusDistributionSummary.from_request_times(
        request_types, requests, distribution_type="concurrency"
    )
    assert status_distribution_summary.total.mean == pytest.approx(2500.0, abs=0.01)
    assert status_distribution_summary.total.cumulative_distribution_function is None
    assert status_distribution_summary.successful.mean == pytest.approx(
        1000.0, abs=0.01
    )
    assert (
        status_distribution_summary.successful.cumulative_distribution_function is None
    )
    assert status_distribution_summary.incomplete.mean == pytest.approx(
        1000.0, abs=0.01
    )
    assert (
        status_distribution_summary.incomplete.cumulative_distribution_function is None
    )
    assert status_distribution_summary.errored.mean == pytest.approx(1000.0, abs=0.01)
    assert status_distribution_summary.errored.cumulative_distribution_function is None

    status_distribution_summary_cdf = StatusDistributionSummary.from_request_times(
        request_types, requests, distribution_type="concurrency", include_cdf=True
    )
    assert (
        status_distribution_summary_cdf.total.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.successful.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.incomplete.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.errored.cumulative_distribution_function
        is not None
    )


def test_status_distribution_summary_from_iterable_request_times():
    request_types: list[Literal["successful", "incomplete", "error"]] = [
        "successful",
        "incomplete",
        "error",
    ] * 1000
    requests = [(val % 3 / 10, val % 3 / 10 + 1) for val in range(3000)]
    first_iter_times = [val % 3 / 10 + 0.1 for val in range(3000)]
    iter_counts = [9 for _ in range(3000)]
    first_iter_counts = [1 for _ in range(3000)]
    status_distribution_summary = StatusDistributionSummary.from_iterable_request_times(
        request_types,
        requests,
        first_iter_times,
        iter_counts,
        first_iter_counts,
    )
    assert status_distribution_summary.total.mean == pytest.approx(21666.66, abs=0.01)
    assert status_distribution_summary.total.cumulative_distribution_function is None
    assert status_distribution_summary.successful.mean == pytest.approx(
        8000.0, abs=0.01
    )
    assert (
        status_distribution_summary.successful.cumulative_distribution_function is None
    )
    assert status_distribution_summary.incomplete.mean == pytest.approx(
        8000.0, abs=0.01
    )
    assert (
        status_distribution_summary.incomplete.cumulative_distribution_function is None
    )
    assert status_distribution_summary.errored.mean == pytest.approx(8000.0, abs=0.01)
    assert status_distribution_summary.errored.cumulative_distribution_function is None

    status_distribution_summary_cdf = (
        StatusDistributionSummary.from_iterable_request_times(
            request_types,
            requests,
            first_iter_times,
            iter_counts,
            first_iter_counts,
            include_cdf=True,
        )
    )
    assert (
        status_distribution_summary_cdf.total.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.successful.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.incomplete.cumulative_distribution_function
        is not None
    )
    assert (
        status_distribution_summary_cdf.errored.cumulative_distribution_function
        is not None
    )


def test_running_stats_initialization():
    running_stats = RunningStats()
    assert running_stats.start_time == pytest.approx(time.time(), abs=0.01)
    assert running_stats.count == 0
    assert running_stats.total == 0
    assert running_stats.last == 0
    assert running_stats.mean == 0
    assert running_stats.rate == 0


def test_running_stats_marshalling():
    running_stats = RunningStats()
    serialized = running_stats.model_dump()
    deserialized = RunningStats.model_validate(serialized)

    for key, value in vars(running_stats).items():
        assert getattr(deserialized, key) == value


def test_running_stats_update():
    running_stats = RunningStats()
    running_stats.update(1)
    assert running_stats.count == 1
    assert running_stats.total == 1
    assert running_stats.last == 1
    assert running_stats.mean == 1
    time.sleep(1.0)
    assert running_stats.rate == pytest.approx(
        1.0 / (time.time() - running_stats.start_time), abs=0.1
    )

    running_stats.update(2)
    assert running_stats.count == 2
    assert running_stats.total == 3
    assert running_stats.last == 2
    assert running_stats.mean == 1.5
    time.sleep(1)
    assert running_stats.rate == pytest.approx(
        3 / (time.time() - running_stats.start_time), abs=0.1
    )


def test_running_stats_add():
    running_stats = RunningStats()
    mean = running_stats + 1
    assert mean == 1
    assert mean == running_stats.mean
    assert running_stats.count == 1
    assert running_stats.total == 1
    assert running_stats.last == 1


def test_running_stats_iadd():
    running_stats = RunningStats()
    running_stats += 1
    assert running_stats.count == 1
    assert running_stats.total == 1
    assert running_stats.last == 1
    assert running_stats.mean == 1


def test_time_running_stats_initialization():
    time_running_stats = TimeRunningStats()
    assert time_running_stats.start_time == pytest.approx(time.time(), abs=0.01)
    assert time_running_stats.count == 0
    assert time_running_stats.total == 0
    assert time_running_stats.last == 0
    assert time_running_stats.mean == 0
    assert time_running_stats.rate == 0
    assert time_running_stats.total_ms == 0
    assert time_running_stats.last_ms == 0
    assert time_running_stats.mean_ms == 0
    assert time_running_stats.rate_ms == 0


def test_time_running_stats_marshalling():
    time_running_stats = TimeRunningStats()
    serialized = time_running_stats.model_dump()
    deserialized = TimeRunningStats.model_validate(serialized)

    for key, value in vars(time_running_stats).items():
        assert getattr(deserialized, key) == value


def test_time_running_stats_update():
    time_running_stats = TimeRunningStats()
    time_running_stats.update(1)
    assert time_running_stats.count == 1
    assert time_running_stats.total == 1
    assert time_running_stats.last == 1
    assert time_running_stats.mean == 1
    assert time_running_stats.total_ms == 1000
    assert time_running_stats.last_ms == 1000
    assert time_running_stats.mean_ms == 1000
    time.sleep(1.0)
    assert time_running_stats.rate == pytest.approx(
        1.0 / (time.time() - time_running_stats.start_time), abs=0.1
    )
    assert time_running_stats.rate_ms == pytest.approx(
        1000 / (time.time() - time_running_stats.start_time), abs=0.1
    )

    time_running_stats.update(2)
    assert time_running_stats.count == 2
    assert time_running_stats.total == 3
    assert time_running_stats.last == 2
    assert time_running_stats.mean == 1.5
    assert time_running_stats.total_ms == 3000
    assert time_running_stats.last_ms == 2000
    assert time_running_stats.mean_ms == 1500
    time.sleep(1)
    assert time_running_stats.rate == pytest.approx(
        3 / (time.time() - time_running_stats.start_time), abs=0.1
    )
    assert time_running_stats.rate_ms == pytest.approx(
        3000 / (time.time() - time_running_stats.start_time), abs=0.1
    )
