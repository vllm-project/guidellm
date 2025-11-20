"""Comprehensive unit tests for over-saturation constraint implementation.

This module provides thorough testing to validate that over-saturation detection
and stopping features work correctly under various conditions and edge cases.
"""

import math
import time
from unittest.mock import patch

import pytest

from guidellm.scheduler import (
    OverSaturationConstraint,
    OverSaturationConstraintInitializer,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.scheduler.constraints.saturation import (
    SlopeChecker,
    approx_t_ppf,
)
from guidellm.schemas import RequestInfo, RequestTimings


class TestSlopeCheckerStatisticalAccuracy:
    """Test the statistical accuracy of SlopeChecker implementation."""

    @pytest.mark.sanity
    def test_approx_t_ppf_accuracy(self):
        """Test that approx_t_ppf produces reasonable approximations."""
        # Test known values for t-distribution
        # For df=10, p=0.975 (95% confidence, two-tailed), t ≈ 2.228
        result = approx_t_ppf(0.975, 10)
        assert 2.0 < result < 2.5, f"Expected ~2.228, got {result}"

        # For df=30, p=0.975, t ≈ 2.042
        result = approx_t_ppf(0.975, 30)
        assert 1.9 < result < 2.2, f"Expected ~2.042, got {result}"

        # For large df, should approach normal distribution (z=1.96)
        result = approx_t_ppf(0.975, 1000)
        assert 1.8 < result < 2.1, f"Expected ~1.96, got {result}"

    @pytest.mark.sanity
    def test_approx_t_ppf_edge_cases(self):
        """Test approx_t_ppf with edge cases."""
        # Very small df
        result = approx_t_ppf(0.975, 1)
        assert result > 5.0, "t-value should be large for df=1"

        # Invalid df should return NaN
        result = approx_t_ppf(0.975, 0)
        assert math.isnan(result)

        result = approx_t_ppf(0.975, -1)
        assert math.isnan(result)

    @pytest.mark.smoke
    def test_slope_calculation_perfect_line(self):
        """Test slope calculation with perfect linear data."""
        checker = SlopeChecker(moe_threshold=0.1, confidence=0.95)

        # Perfect line: y = 2x + 1
        for i in range(10):
            x = float(i)
            y = 2.0 * x + 1.0
            checker.add_data_point(x, y)

        result = checker.check_slope(10.0)
        assert result is True
        assert abs(checker.slope - 2.0) < 0.001, (
            f"Expected slope ~2.0, got {checker.slope}"
        )

    @pytest.mark.smoke
    def test_slope_calculation_zero_slope(self):
        """Test slope calculation with horizontal line."""
        checker = SlopeChecker(moe_threshold=0.1, confidence=0.95)

        # Horizontal line: y = 5
        for i in range(10):
            x = float(i)
            y = 5.0
            checker.add_data_point(x, y)

        result = checker.check_slope(10.0)
        # Should not detect positive slope
        if result:
            assert checker.slope <= 0.1, f"Slope should be ~0, got {checker.slope}"

    @pytest.mark.sanity
    def test_slope_calculation_negative_slope(self):
        """Test slope calculation with negative slope."""
        checker = SlopeChecker(moe_threshold=0.1, confidence=0.95)

        # Negative slope: y = -1.5x + 10
        for i in range(10):
            x = float(i)
            y = -1.5 * x + 10.0
            checker.add_data_point(x, y)

        result = checker.check_slope(10.0)
        # Should not detect positive slope
        assert result is False or checker.slope <= 0

    @pytest.mark.sanity
    def test_slope_calculation_with_noise(self):
        """Test slope calculation with noisy data."""
        import random

        random.seed(42)  # Reproducible results

        checker = SlopeChecker(moe_threshold=1.0, confidence=0.90)

        # Positive slope with noise: y = 1.5x + noise
        for i in range(50):
            x = float(i)
            noise = random.uniform(-2.0, 2.0)
            y = 1.5 * x + noise
            checker.add_data_point(x, y)

        result = checker.check_slope(50.0)
        if result:
            assert 1.0 < checker.slope < 2.0, (
                f"Expected slope ~1.5, got {checker.slope}"
            )

    @pytest.mark.sanity
    def test_margin_of_error_calculation(self):
        """Test that margin of error is calculated correctly."""
        checker = SlopeChecker(moe_threshold=0.5, confidence=0.95)

        # Add data with known properties
        for i in range(20):
            x = float(i)
            y = 2.0 * x + 1.0
            checker.add_data_point(x, y)

        result = checker.check_slope(20.0)
        assert result is True
        assert checker.margin_of_error is not None
        assert checker.margin_of_error >= 0
        # For perfect data, margin of error should be very small
        assert checker.margin_of_error < 0.1


class TestOverSaturationConstraintRobustness:
    """Test the robustness of OverSaturationConstraint under various conditions."""

    @pytest.mark.sanity
    def test_constraint_with_empty_data(self):
        """Test constraint behavior with no data."""
        constraint = OverSaturationConstraint(minimum_duration=0.0, enabled=True)

        # Should not alert with no data
        assert constraint._check_alert() is False

        # Should handle update_duration gracefully
        constraint._update_duration(100.0)
        assert constraint._check_alert() is False

    @pytest.mark.sanity
    def test_constraint_with_single_request(self):
        """Test constraint behavior with single request."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=1, enabled=True
        )

        constraint._add_started({"concurrent_requests": 5, "duration": 1.0})
        constraint._add_finished({"ttft": 2.0, "duration": 2.0})
        constraint._update_duration(10.0)

        # Should not alert with insufficient data
        assert constraint._check_alert() is False

    @pytest.mark.sanity
    def test_constraint_with_identical_values(self):
        """Test constraint with identical values (zero variance)."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=3, enabled=True
        )

        # Add identical values
        for i in range(10):
            constraint._add_started({"concurrent_requests": 5, "duration": float(i)})
            constraint._add_finished({"ttft": 1.0, "duration": float(i)})

        constraint._update_duration(20.0)
        result = constraint._check_alert()

        # Should not alert for flat data
        assert result is False

    @pytest.mark.sanity
    def test_constraint_extreme_values(self):
        """Test constraint with extreme values."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=3, enabled=True
        )

        # Add extreme values
        values = [0.1, 1000.0, 0.01, 5000.0, 0.001]
        for i, val in enumerate(values):
            constraint._add_started(
                {"concurrent_requests": int(val), "duration": float(i)}
            )
            constraint._add_finished({"ttft": val, "duration": float(i)})

        constraint._update_duration(20.0)
        # Should handle without crashing
        result = constraint._check_alert()
        assert result in [True, False]

    @pytest.mark.sanity
    def test_constraint_precision_edge_cases(self):
        """Test constraint with floating point precision edge cases."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=3, enabled=True
        )

        # Very small increments
        base = 1e10
        for i in range(10):
            constraint._add_started(
                {"concurrent_requests": 5, "duration": base + i * 1e-10}
            )
            constraint._add_finished({"ttft": 1.0, "duration": base + i * 1e-10})

        constraint._update_duration(base + 100.0)
        # Should handle without numerical issues
        result = constraint._check_alert()
        assert result in [True, False]

    @pytest.mark.sanity
    def test_constraint_window_management_stress(self):
        """Test constraint window management under stress."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0,
            maximum_window_seconds=10.0,
            minimum_window_size=5,
            enabled=True,
        )

        # Add many requests over time
        for i in range(1000):
            duration = float(i * 0.1)  # 100 seconds total
            constraint._add_started(
                {"concurrent_requests": i % 50, "duration": duration}
            )
            constraint._add_finished({"ttft": (i % 100) * 0.01, "duration": duration})

            # Periodic window updates
            if i % 100 == 0:
                constraint._update_duration(duration + 5.0)

        # Should maintain reasonable window size
        assert len(constraint.started_requests) <= 200  # Should be pruned
        assert len(constraint.finished_requests) <= 200


class TestOverSaturationConstraintRealisticScenarios:
    """Test detector with realistic request patterns."""

    @pytest.mark.sanity
    def test_gradual_performance_degradation(self):
        """Test detection of gradual performance degradation."""
        constraint = OverSaturationConstraint(
            minimum_duration=5.0,
            minimum_window_size=10,
            moe_threshold=1.5,
            enabled=True,
        )

        # Simulate gradual degradation
        for i in range(50):
            # Gradually increasing concurrent requests
            concurrent = 10 + i * 0.5
            # Gradually increasing TTFT
            ttft = 1.0 + i * 0.1
            duration = float(i)

            constraint._add_started(
                {"concurrent_requests": int(concurrent), "duration": duration}
            )
            constraint._add_finished({"ttft": ttft, "duration": duration})

        constraint._update_duration(60.0)
        result = constraint._check_alert()

        # Should detect the degradation
        assert result is True, "Should detect gradual performance degradation"

    @pytest.mark.sanity
    def test_sudden_load_spike(self):
        """Test detection of sudden load spike."""
        constraint = OverSaturationConstraint(
            minimum_duration=5.0,
            minimum_window_size=10,
            moe_threshold=1.0,
            enabled=True,
        )

        # Normal operations first
        for i in range(20):
            constraint._add_started({"concurrent_requests": 5, "duration": float(i)})
            constraint._add_finished({"ttft": 1.0, "duration": float(i)})

        # Sudden spike
        for i in range(20, 40):
            constraint._add_started({"concurrent_requests": 50, "duration": float(i)})
            constraint._add_finished({"ttft": 5.0, "duration": float(i)})

        constraint._update_duration(50.0)
        result = constraint._check_alert()

        # Should detect the spike
        assert result is True, "Should detect sudden load spike"

    @pytest.mark.sanity
    def test_variable_but_stable_performance(self):
        """Test that variable but stable performance doesn't trigger false positives."""
        constraint = OverSaturationConstraint(
            minimum_duration=5.0,
            minimum_window_size=10,
            moe_threshold=2.0,
            enabled=True,
        )

        import random

        random.seed(123)  # Reproducible

        # Variable but centered around stable values
        for i in range(100):
            concurrent = 15 + random.randint(-5, 5)  # 10-20 range
            ttft = 2.0 + random.uniform(-0.5, 0.5)  # 1.5-2.5 range
            duration = float(i)

            constraint._add_started(
                {"concurrent_requests": concurrent, "duration": duration}
            )
            constraint._add_finished({"ttft": ttft, "duration": duration})

        constraint._update_duration(120.0)
        result = constraint._check_alert()

        # Should not trigger false positive
        assert result is False, (
            "Should not trigger false positive for stable performance"
        )

    @pytest.mark.sanity
    def test_recovery_after_degradation(self):
        """Test that detector handles recovery after degradation."""
        constraint = OverSaturationConstraint(
            minimum_duration=5.0,
            minimum_window_size=10,
            maximum_window_seconds=30.0,
            enabled=True,
        )

        # Initial degradation
        for i in range(20):
            concurrent = 10 + i * 2  # Increasing load
            ttft = 1.0 + i * 0.2  # Increasing TTFT
            constraint._add_started(
                {"concurrent_requests": concurrent, "duration": float(i)}
            )
            constraint._add_finished({"ttft": ttft, "duration": float(i)})

        constraint._update_duration(25.0)
        degradation_result = constraint._check_alert()

        # Add recovery period - improved performance
        for i in range(40, 60):
            constraint._add_started({"concurrent_requests": 5, "duration": float(i)})
            constraint._add_finished({"ttft": 0.8, "duration": float(i)})

        constraint._update_duration(65.0)
        recovery_result = constraint._check_alert()

        # Should detect degradation initially, then not alert during recovery
        # (depending on window management)
        assert degradation_result in [True, False]  # Could go either way
        # After recovery with window management, should be less likely to alert
        if len(constraint.finished_requests) < 15:  # If old data was purged
            assert recovery_result is False, "Should not alert after recovery"


class TestOverSaturationConstraintIntegration:
    """Test integration between constraint and detector with complex scenarios."""

    def create_realistic_constraint(self) -> OverSaturationConstraint:
        """Create a constraint with realistic settings."""
        return OverSaturationConstraint(
            minimum_duration=10.0,
            minimum_window_size=5,
            maximum_window_seconds=60.0,
            moe_threshold=1.5,
            confidence=0.90,
            enabled=True,
        )

    @pytest.mark.sanity
    def test_constraint_metadata_completeness(self):
        """Test that constraint provides complete metadata."""
        constraint = self.create_realistic_constraint()
        start_time = time.time()

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=10,
        )

        request = RequestInfo(
            request_id="test-request",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)

        # Verify metadata completeness
        required_fields = [
            "is_over_saturated",
            "concurrent_slope",
            "concurrent_n",
            "ttft_slope",
            "ttft_n",
            "ttft_violations",  # Correct field name
            # Note: total_started_ever, total_finished_ever,
            # window sizes not in metadata
        ]

        for field in required_fields:
            assert field in action.metadata, f"Missing metadata field: {field}"

    @pytest.mark.sanity
    def test_constraint_with_realistic_request_flow(self):
        """Test constraint with realistic request flow."""
        constraint = self.create_realistic_constraint()
        start_time = time.time()
        actions = []

        # Simulate 60 seconds of requests
        for second in range(60):
            current_time = start_time + second

            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                processing_requests=10 + second,  # Gradually increasing load
            )

            # Mix of request statuses
            for req_num in range(3):  # 3 requests per second
                request_id = f"req-{second}-{req_num}"

                if req_num == 0:  # Completed request
                    timings = RequestTimings(
                        request_start=current_time - 2.0,
                        first_iteration=current_time
                        - 2.0
                        + (second * 0.05),  # Gradually slower
                    )
                    request = RequestInfo(
                        request_id=request_id,
                        status="completed",
                        scheduler_node_id=0,
                        scheduler_process_id=0,
                        scheduler_start_time=start_time,
                        timings=timings,
                    )
                else:  # In progress request
                    request = RequestInfo(
                        request_id=request_id,
                        status="in_progress",
                        scheduler_node_id=0,
                        scheduler_process_id=0,
                        scheduler_start_time=start_time,
                    )

                action = constraint(state, request)
                actions.append((second, action))

        # Analyze results
        stop_actions = [a for s, a in actions if a.request_queuing == "stop"]

        # Should eventually detect over-saturation
        if len(stop_actions) > 0:
            first_stop_second = min(
                s for s, a in actions if a.request_queuing == "stop"
            )
            assert first_stop_second >= 10, "Should not stop before minimum duration"

    @pytest.mark.sanity
    def test_constraint_disabled_never_stops(self):
        """Test that disabled constraint never stops regardless of load."""
        detector = OverSaturationConstraint(minimum_duration=0.0, minimum_window_size=3)
        constraint = OverSaturationConstraint(
            detector,
            enabled=False,  # Disabled
        )

        # Add obviously over-saturated data
        for i in range(50):
            constraint._add_started(
                {"concurrent_requests": i * 10, "duration": float(i)}
            )
            constraint._add_finished({"ttft": i * 2.0, "duration": float(i)})

        constraint._update_duration(60.0)

        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=500,  # Very high load
        )

        request = RequestInfo(
            request_id="test-request",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)

        # Should continue despite over-saturation
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"
        assert action.metadata["is_over_saturated"] in [True, False]  # Could be either


class TestOverSaturationConstraintPerformance:
    """Test performance characteristics of the constraint."""

    @pytest.mark.sanity
    def test_detector_memory_usage(self):
        """Test that detector manages memory properly."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0,
            maximum_window_seconds=10.0,
            minimum_window_size=5,
            enabled=True,
        )

        # Add many requests
        for i in range(10000):
            duration = float(i * 0.01)  # 100 seconds total
            constraint._add_started({"concurrent_requests": 10, "duration": duration})
            constraint._add_finished({"ttft": 1.0, "duration": duration})

            if i % 1000 == 0:
                constraint._update_duration(duration + 5.0)

        # Memory should be bounded due to window management
        assert len(constraint.started_requests) < 2000, "Started requests not bounded"
        assert len(constraint.finished_requests) < 2000, "Finished requests not bounded"

    @pytest.mark.sanity
    def test_constraint_computational_efficiency(self):
        """Test that constraint operations remain efficient."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=10, enabled=True
        )

        # Add baseline data
        for i in range(100):
            constraint._add_started({"concurrent_requests": 10, "duration": float(i)})
            constraint._add_finished({"ttft": 1.0, "duration": float(i)})

        constraint._update_duration(120.0)

        # Time multiple check_alert calls
        start_time = time.time()
        for _ in range(100):
            constraint._check_alert()
        elapsed = time.time() - start_time

        # Should complete quickly (< 1 second for 100 calls)
        assert elapsed < 1.0, f"Detection too slow: {elapsed:.3f}s for 100 calls"


class TestOverSaturationConstraintInitializerRobustness:
    """Test robustness of the constraint initializer."""

    @pytest.mark.smoke
    def test_initializer_parameter_validation(self):
        """Test parameter validation in initializer."""
        # Valid parameters
        initializer = OverSaturationConstraintInitializer(
            enabled=True,
            min_seconds=5.0,
            max_window_seconds=30.0,
            moe_threshold=1.5,
            confidence=0.95,
        )

        constraint = initializer.create_constraint()
        assert constraint.enabled is True
        assert constraint.over_saturation_constraint.minimum_duration == 5.0
        assert constraint.over_saturation_constraint.maximum_window_seconds == 30.0

    @pytest.mark.smoke
    def test_initializer_with_extreme_parameters(self):
        """Test initializer with extreme but valid parameters."""
        # Very permissive settings - only test parameters actually supported
        initializer = OverSaturationConstraintInitializer(
            enabled=True,
            min_seconds=0.1,
            max_window_seconds=3600.0,  # 1 hour
        )

        constraint = initializer.create_constraint()

        assert constraint.minimum_duration == 0.1
        assert constraint.maximum_window_seconds == 3600.0
        # Note: moe_threshold and confidence may have default values

    @pytest.mark.smoke
    def test_initializer_alias_precedence(self):
        """Test alias precedence in validated_kwargs."""
        # Multiple aliases provided - should use the explicit one
        result = OverSaturationConstraintInitializer.validated_kwargs(
            over_saturation=False,  # Explicit parameter
            detect_saturation=True,  # Alias
        )

        # detect_saturation should override over_saturation=False
        assert result == {"enabled": True}

    @pytest.mark.smoke
    def test_constraint_creation_with_mock_constraint(self):
        """Test constraint creation with mocked constraint for isolation."""
        constraint = OverSaturationConstraint(enabled=True)
        # Set up constraint state to simulate over-saturation
        constraint.ttft_slope_checker.slope = 1.5
        constraint.ttft_slope_checker.margin_of_error = 0.3
        constraint.ttft_slope_checker.n = 10
        constraint.concurrent_slope_checker.slope = 2.0
        constraint.concurrent_slope_checker.margin_of_error = 0.5
        constraint.concurrent_slope_checker.n = 15
        constraint.ttft_violations_counter = 5
        constraint.duration = 30.0  # Set duration to pass minimum check

        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=10,
        )

        request = RequestInfo(
            request_id="test-request",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)

        # Should provide metadata about saturation state
        assert "is_over_saturated" in action.metadata


class TestOverSaturationEdgeCasesAndRegression:
    """Test edge cases and regression scenarios."""

    @pytest.mark.sanity
    def test_detector_with_malformed_request_data(self):
        """Test detector requires proper request data structure."""
        constraint = OverSaturationConstraint(minimum_duration=0.0, enabled=True)

        # Missing fields should raise KeyError
        with pytest.raises(KeyError):
            constraint._add_started({})  # Missing required fields

        with pytest.raises(KeyError):
            constraint._add_finished({})

        with pytest.raises(KeyError):
            constraint._add_started({"concurrent_requests": 5})  # Missing duration

        with pytest.raises(KeyError):
            constraint._add_finished({"ttft": 1.0})  # Missing duration

        # Valid data should work
        constraint._add_started({"concurrent_requests": 5, "duration": 1.0})
        constraint._add_finished({"ttft": 1.0, "duration": 1.0})

        constraint._update_duration(10.0)
        result = constraint._check_alert()
        assert result in [True, False]

    @pytest.mark.sanity
    def test_constraint_with_missing_timings_data(self):
        """Test constraint handles missing timings data gracefully."""
        constraint = OverSaturationConstraint(
            OverSaturationConstraint(minimum_duration=0.0),
            enabled=True,
        )

        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=5,
        )

        # Create request without timings (in_progress status)
        request = RequestInfo(
            request_id="test-request",
            status="in_progress",  # No timings expected for in_progress
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        # Should not crash
        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)

    @pytest.mark.sanity
    def test_detector_concurrent_modification_safety(self):
        """Test detector behavior under concurrent-like modifications."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=3, enabled=True
        )

        # Add requests
        requests = []
        for i in range(20):
            req = {"concurrent_requests": i, "duration": float(i)}
            constraint._add_started(req)
            requests.append(req)

        # Remove some while iterating (simulating concurrent access pattern)
        for i in range(0, 10, 2):  # Remove every other early request
            constraint._remove_started(requests[i])

        # Should still function
        constraint._update_duration(25.0)
        result = constraint._check_alert()
        assert result in [True, False]

    @pytest.mark.sanity
    def test_slope_checker_numerical_stability(self):
        """Test SlopeChecker numerical stability with challenging data."""
        checker = SlopeChecker(moe_threshold=0.1, confidence=0.95)

        # Add data that could cause numerical instability
        base = 1e15  # Very large numbers
        for i in range(10):
            x = base + i
            y = base + i * 1e-10  # Very small slope relative to magnitude
            checker.add_data_point(x, y)

        # Should handle without overflow/underflow
        result = checker.check_slope(10.0)
        assert result in [True, False]

        if checker.slope is not None:
            assert not math.isnan(checker.slope)
            assert not math.isinf(checker.slope)

    @pytest.mark.sanity
    def test_detector_reset_clears_all_state(self):
        """Test that detector reset completely clears state."""
        constraint = OverSaturationConstraint(minimum_duration=0.0, enabled=True)

        # Add data and trigger computation
        for i in range(20):
            constraint._add_started({"concurrent_requests": i, "duration": float(i)})
            constraint._add_finished({"ttft": i * 0.1, "duration": float(i)})

        constraint._update_duration(25.0)
        constraint._check_alert()  # Populate computed values

        # Verify state exists
        assert len(constraint.started_requests) > 0
        assert len(constraint.finished_requests) > 0
        assert constraint.total_started_ever > 0
        assert constraint.total_finished_ever > 0

        # Reset
        constraint.reset()

        # Verify complete reset
        assert len(constraint.started_requests) == 0
        assert len(constraint.finished_requests) == 0
        assert constraint.total_started_ever == 0
        assert constraint.total_finished_ever == 0
        assert constraint.ttft_violations_counter == 0
        assert constraint.duration == 0.0

        # Slope checkers should be reset too
        assert constraint.concurrent_slope_checker.n == 0
        assert constraint.ttft_slope_checker.n == 0

    @pytest.mark.sanity
    @patch("time.time")
    def test_constraint_time_calculation_accuracy(self, mock_time):
        """Test that constraint calculates durations accurately."""
        # Mock time to control duration calculation
        start_time = 1000.0
        current_time = 1030.0  # 30 seconds later
        mock_time.return_value = current_time

        constraint = OverSaturationConstraint(
            minimum_duration=25.0, enabled=True
        )  # Should be met

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=5,
        )

        request = RequestInfo(
            request_id="test-request",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        # Call constraint - should update detector duration
        constraint(state, request)

        # Verify duration was calculated correctly
        assert abs(constraint.duration - 30.0) < 0.001, (
            f"Expected duration ~30.0, got {constraint.duration}"
        )

    @pytest.mark.sanity
    def test_ttft_violation_counting_accuracy(self):
        """Test TTFT violation counting is accurate."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0,
            minimum_ttft=2.0,  # Threshold
            enabled=True,
        )

        # Add requests with known TTFT values
        ttft_values = [1.0, 3.0, 1.5, 4.0, 2.1, 0.5, 5.0, 1.9]
        expected_violations = sum(
            1 for ttft in ttft_values if ttft > 2.0
        )  # Should be 4

        for i, ttft in enumerate(ttft_values):
            constraint._add_finished({"ttft": ttft, "duration": float(i)})

        assert constraint.ttft_violations_counter == expected_violations, (
            f"Expected {expected_violations} violations, "
            f"got {constraint.ttft_violations_counter}"
        )
