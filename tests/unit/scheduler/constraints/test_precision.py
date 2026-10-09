"""Unit tests for the target margin of error constraint."""

from __future__ import annotations

import math
import random

import pytest

from guidellm.scheduler import (
    Constraint,
    ConstraintsInitializerFactory,
    SchedulerState,
    TargetMoeConstraint,
    TargetMoeConstraintInitializer,
)
from guidellm.schemas import RequestInfo, RequestTimings
from guidellm.schemas.base.statistics import DistributionSummary
from guidellm.schemas.scheduler import TargetMoeConstraintArgs
from guidellm.utils.statistics import (
    mean_confidence_interval,
    quantile_confidence_interval,
)


def _request(
    request_id: int,
    ttft_seconds: float | None = 0.1,
    latency_seconds: float = 1.0,
    status: str = "completed",
    start: float = 1000.0,
) -> RequestInfo:
    return RequestInfo(
        request_id=str(request_id),
        status=status,  # type: ignore[arg-type]
        timings=RequestTimings(
            request_start=start,
            first_token_iteration=(
                None if ttft_seconds is None else start + ttft_seconds
            ),
            request_end=start + latency_seconds,
        ),
    )


def _lognormal_ttfts(count: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.lognormvariate(-2.3, 0.4) for _ in range(count)]


def _feed(constraint: TargetMoeConstraint, ttfts: list[float], start_id: int = 0):
    state = SchedulerState()
    action = None
    for offset, ttft in enumerate(ttfts):
        action = constraint(state, _request(start_id + offset, ttft_seconds=ttft))
    return action


class TestTargetMoeConstraintInitializer:
    """Test suite for TargetMoeConstraintInitializer."""

    @pytest.mark.smoke
    def test_factory_creates_configured_constraint(self):
        """
        Test that the factory builds the constraint from target_moe args.

        ## WRITTEN BY AI ##
        """
        args = TargetMoeConstraintArgs(
            metric="request_latency",
            statistic="p90",
            moe=0.1,
            confidence=0.9,
            min_samples=50,
            check_interval=5,
            stopping_scope="all",
        )

        initializer = ConstraintsInitializerFactory.create(args)
        constraint = initializer.create_constraint()

        assert isinstance(initializer, TargetMoeConstraintInitializer)
        assert isinstance(constraint, TargetMoeConstraint)
        assert isinstance(constraint, Constraint)
        assert constraint.info == {
            "type_": "target_moe",
            "metric": "request_latency",
            "statistic": "p90",
            "moe": 0.1,
            "confidence": 0.9,
            "min_samples": 50,
            "check_interval": 5,
            "stopping_scope": "all",
        }

    @pytest.mark.sanity
    def test_constraints_do_not_share_samples(self):
        """
        Test that each created constraint starts without samples, so each
        strategy in a sweep measures its own precision.

        ## WRITTEN BY AI ##
        """
        initializer = TargetMoeConstraintInitializer(
            args=TargetMoeConstraintArgs(moe=0.05)
        )

        first = initializer.create_constraint()
        second = initializer.create_constraint()
        assert isinstance(first, TargetMoeConstraint)
        assert isinstance(second, TargetMoeConstraint)

        _feed(first, [0.1] * 20)

        assert first is not second
        assert first.count == 20
        assert second.count == 0


class TestTargetMoeConstraint:
    """Test suite for TargetMoeConstraint."""

    @pytest.mark.smoke
    def test_continues_before_min_samples(self):
        """
        Test that no check is made before min_samples, even with zero spread.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.05, min_samples=30, check_interval=1)

        action = _feed(constraint, [0.1] * 29)

        assert action is not None
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"
        assert action.metadata["relative_moe"] is None
        assert action.metadata["target_moe_reached"] is False

    @pytest.mark.smoke
    def test_stops_once_target_reached(self):
        """
        Test that queuing and local processing stop at the first due check that
        meets the target, and that stop_time is reported.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(
            moe=0.05, min_samples=30, check_interval=1, stopping_scope="all"
        )

        action = _feed(constraint, [0.1] * 30)

        assert action is not None
        assert action.request_queuing == "stop"
        assert action.request_processing == "stop_local"
        assert action.stopping_scope == "all"
        assert action.metadata["target_moe_reached"] is True
        assert action.metadata["relative_moe"] == 0.0
        assert action.metadata["stop_time"] is not None

    @pytest.mark.sanity
    def test_stop_decision_is_kept(self):
        """
        Test that later requests and polls keep the stop decision and add no
        samples.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.05, min_samples=30, check_interval=1)
        _feed(constraint, [0.1] * 30)

        after_request = constraint(SchedulerState(), _request(99, ttft_seconds=5.0))
        after_poll = constraint(SchedulerState(), None)

        assert constraint.count == 30
        assert after_request.request_queuing == "stop"
        assert after_poll.request_queuing == "stop"
        assert after_poll.request_processing == "stop_local"

    @pytest.mark.sanity
    def test_checks_follow_interval(self):
        """
        Test that the interval is recomputed only every check_interval samples
        once min_samples is reached.

        ## WRITTEN BY AI ##
        """
        ttfts = _lognormal_ttfts(60)
        constraint = TargetMoeConstraint(moe=0.001, min_samples=30, check_interval=10)
        checked_counts = []

        for index, ttft in enumerate(ttfts):
            before = constraint.last_checked_count
            constraint(SchedulerState(), _request(index, ttft_seconds=ttft))
            if constraint.last_checked_count != before:
                checked_counts.append(constraint.count)

        assert checked_counts == [30, 40, 50, 60]
        assert constraint.target_reached is False

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "info",
        [
            _request(1, status="in_progress"),
            _request(2, status="first_token"),
            _request(3, status="errored"),
            _request(4, status="cancelled"),
            _request(5, ttft_seconds=None),
        ],
    )
    def test_ignores_requests_without_a_sample(self, info):
        """
        Test that only completed requests with the metric available are sampled.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.05)

        constraint(SchedulerState(), info)

        assert constraint.count == 0

    @pytest.mark.sanity
    def test_ignores_repeated_updates_and_polls(self):
        """
        Test that a request is sampled once and polls add nothing.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.05)
        info = _request(1)

        constraint(SchedulerState(), info)
        constraint(SchedulerState(), info)
        constraint(SchedulerState(), None)

        assert constraint.count == 1

    @pytest.mark.regression
    def test_mean_interval_matches_report_estimator(self):
        """
        Test that the mean and its relative margin come from the same estimator
        the report uses for its mean confidence interval.

        ## WRITTEN BY AI ##
        """
        ttfts = _lognormal_ttfts(30)
        constraint = TargetMoeConstraint(moe=0.001, min_samples=30, check_interval=30)

        action = _feed(constraint, ttfts)
        summary = DistributionSummary.from_values([1000 * ttft for ttft in ttfts])
        expected = mean_confidence_interval(
            count=summary.count,
            mean=summary.mean,
            std_dev=summary.std_dev,
            confidence=0.95,
        )

        assert expected is not None
        assert action is not None
        assert action.metadata["estimate"] == pytest.approx(summary.mean)
        assert action.metadata["lower"] == pytest.approx(expected[0])
        assert action.metadata["upper"] == pytest.approx(expected[1])
        assert action.metadata["relative_moe"] == pytest.approx(
            (expected[1] - summary.mean) / summary.mean
        )

    @pytest.mark.regression
    @pytest.mark.parametrize("statistic", ["p50", "p90", "p95"])
    def test_percentile_estimate_matches_report(self, statistic):
        """
        Test that the percentile estimate equals the report's percentile.

        ## WRITTEN BY AI ##
        """
        ttfts = _lognormal_ttfts(500, seed=1)
        constraint = TargetMoeConstraint(
            moe=0.001, statistic=statistic, min_samples=500, check_interval=500
        )

        action = _feed(constraint, ttfts)
        summary = DistributionSummary.from_values([1000 * ttft for ttft in ttfts])

        assert action is not None
        assert action.metadata["estimate"] == pytest.approx(
            summary.percentiles.model_dump()[statistic]
        )
        assert action.metadata["lower"] <= action.metadata["estimate"]
        assert action.metadata["upper"] >= action.metadata["estimate"]

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("statistic", "quantile", "lower_side_wider"),
        [("p25", 0.25, True), ("p95", 0.95, False)],
    )
    def test_percentile_margin_uses_wider_side(
        self, statistic, quantile, lower_side_wider
    ):
        """
        Test that the margin of an asymmetric percentile interval is the wider
        side, whichever side that is.

        ## WRITTEN BY AI ##
        """
        ttfts = _lognormal_ttfts(500, seed=2)
        constraint = TargetMoeConstraint(
            moe=0.001, statistic=statistic, min_samples=500, check_interval=500
        )

        action = _feed(constraint, ttfts)
        expected = quantile_confidence_interval(
            sorted(1000 * ttft for ttft in ttfts), quantile, 0.95
        )

        assert expected is not None
        assert action is not None
        estimate = action.metadata["estimate"]
        lower_side = estimate - expected[0]
        upper_side = expected[1] - estimate
        assert (lower_side > upper_side) is lower_side_wider
        assert action.metadata["relative_moe"] == pytest.approx(
            max(lower_side, upper_side) / estimate
        )

    @pytest.mark.sanity
    def test_percentile_waits_for_a_bounded_interval(self):
        """
        Test that p99 is not checked against the target until the sample can
        bound it, which takes 368 samples at 95% confidence.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(
            moe=0.5, statistic="p99", min_samples=30, check_interval=10
        )

        action = _feed(constraint, _lognormal_ttfts(300))

        assert action is not None
        assert action.request_queuing == "continue"
        assert action.metadata["lower"] is None
        assert action.metadata["relative_moe"] is None
        assert action.metadata["required_samples"] == 368
        assert action.progress.total_requests == 368
        assert action.progress.remaining_requests == 68

    @pytest.mark.sanity
    def test_progress_extrapolates_required_samples(self):
        """
        Test that the remaining sample estimate scales with the squared ratio
        of the current to the target margin.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.01, min_samples=100, check_interval=100)

        action = _feed(constraint, _lognormal_ttfts(100))

        assert action is not None
        relative_moe = action.metadata["relative_moe"]
        expected = math.ceil(100 * (relative_moe / 0.01) ** 2)
        assert action.metadata["required_samples"] == expected
        assert action.progress.total_requests == expected
        assert action.progress.remaining_requests == expected - 100

    @pytest.mark.sanity
    def test_estimated_remaining_seconds(self):
        """
        Test that the remaining time divides the samples still needed by the
        rate at which samples have completed so far.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.01, min_samples=100, check_interval=100)
        state = SchedulerState()
        action = None

        # One completion every 0.1 seconds, a rate of 10 samples per second.
        for index, ttft in enumerate(_lognormal_ttfts(100)):
            info = _request(index, ttft_seconds=ttft, start=1000.0 + 0.1 * index)
            action = constraint(state, info)

        assert action is not None
        remaining = action.metadata["required_samples"] - 100
        assert remaining > 0
        assert action.metadata["estimated_remaining_seconds"] == pytest.approx(
            remaining / 10.0
        )

    @pytest.mark.sanity
    def test_estimated_remaining_seconds_edges(self):
        """
        Test that the remaining time is zero once the target is reached and
        unknown while no completion rate can be measured.

        ## WRITTEN BY AI ##
        """
        reached = TargetMoeConstraint(moe=0.05, min_samples=30, check_interval=30)
        unmeasured = TargetMoeConstraint(moe=0.001, min_samples=30, check_interval=30)

        reached_action = _feed(reached, [0.1] * 30)
        # Every sample completes at the same instant, so no rate is available.
        unmeasured_action = _feed(unmeasured, _lognormal_ttfts(30))

        assert reached_action is not None
        assert unmeasured_action is not None
        assert reached_action.metadata["estimated_remaining_seconds"] == 0.0
        assert unmeasured_action.metadata["required_samples"] > 30
        assert unmeasured_action.metadata["estimated_remaining_seconds"] is None

    @pytest.mark.sanity
    def test_request_latency_metric(self):
        """
        Test that request_latency samples request_end minus request_start.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(
            moe=0.05, metric="request_latency", min_samples=30, check_interval=30
        )
        state = SchedulerState()
        action = None

        for index in range(30):
            action = constraint(state, _request(index, latency_seconds=2.5))

        assert action is not None
        assert action.metadata["estimate"] == pytest.approx(2.5)
        assert action.request_queuing == "stop"

    @pytest.mark.sanity
    def test_zero_estimate_never_stops(self):
        """
        Test that a zero estimate, where a relative margin is undefined, does not
        stop the run.

        ## WRITTEN BY AI ##
        """
        constraint = TargetMoeConstraint(moe=0.05, min_samples=30, check_interval=1)

        action = _feed(constraint, [0.0] * 40)

        assert action is not None
        assert action.request_queuing == "continue"
        assert action.metadata["relative_moe"] is None
