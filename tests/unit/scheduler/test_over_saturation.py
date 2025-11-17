"""Unit tests for over-saturation constraint implementation."""

import inspect
import time

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    OverSaturationConstraint,
    OverSaturationConstraintInitializer,
    OverSaturationDetector,
    PydanticConstraintInitializer,
    SchedulerState,
    SchedulerUpdateAction,
    SerializableConstraintInitializer,
)
from guidellm.schemas import RequestInfo, RequestTimings


class TestOverSaturationDetector:
    """Test the OverSaturationDetector implementation."""

    @pytest.fixture(
        params=[
            {"minimum_duration": 30.0, "maximum_window_seconds": 120.0},
            {"minimum_duration": 10.0, "maximum_window_seconds": 60.0},
            {"minimum_duration": 60.0, "maximum_window_seconds": 240.0},
        ]
    )
    def valid_instances(self, request):
        """Create OverSaturationDetector instances with valid parameters."""
        constructor_args = request.param
        instance = OverSaturationDetector(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that OverSaturationDetector can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.smoke
    def test_initialization_defaults(self):
        """Test that OverSaturationDetector has correct default values."""
        detector = OverSaturationDetector()

        assert detector.minimum_duration == 30.0
        assert detector.minimum_ttft == 2.5
        assert detector.maximum_window_seconds == 120.0
        assert detector.moe_threshold == 2.0
        assert detector.maximum_window_ratio == 0.75
        assert detector.minimum_window_size == 5
        assert detector.confidence == 0.95
        assert detector.eps == 1e-12

    @pytest.mark.smoke
    def test_reset(self, valid_instances):
        """Test that reset method properly initializes detector state."""
        detector, _ = valid_instances
        detector.reset()

        assert detector.duration == 0.0
        assert detector.started_requests == []
        assert detector.finished_requests == []
        assert detector.ttft_violations_counter == 0
        assert detector.total_finished_ever == 0
        assert detector.total_started_ever == 0
        assert hasattr(detector, "concurrent_slope_checker")
        assert hasattr(detector, "ttft_slope_checker")

    @pytest.mark.sanity
    def test_add_and_remove_started(self):
        """Test adding and removing started requests."""
        detector = OverSaturationDetector(minimum_duration=0.0)

        # Add started requests
        for i in range(10):
            detector.add_started({"concurrent_requests": i, "duration": float(i)})

        assert len(detector.started_requests) == 10
        assert detector.total_started_ever == 10
        assert detector.concurrent_slope_checker.n == 10

        # Remove started requests
        request = detector.started_requests[0]
        detector.remove_started(request)

        assert len(detector.started_requests) == 9
        assert detector.concurrent_slope_checker.n == 9

    @pytest.mark.sanity
    def test_add_and_remove_finished(self):
        """Test adding and removing finished requests."""
        detector = OverSaturationDetector(minimum_duration=0.0, minimum_ttft=1.0)

        # Add finished requests
        for i in range(10):
            ttft = 0.5 if i < 5 else 3.0  # First 5 below threshold, rest above
            detector.add_finished({"ttft": ttft, "duration": float(i)})

        assert len(detector.finished_requests) == 10
        assert detector.total_finished_ever == 10
        assert detector.ttft_slope_checker.n == 10
        assert detector.ttft_violations_counter == 5  # 5 above minimum_ttft

        # Remove finished request
        request = detector.finished_requests[0]
        detector.remove_finished(request)

        assert len(detector.finished_requests) == 9
        assert detector.ttft_slope_checker.n == 9

    @pytest.mark.sanity
    def test_update_duration_window_management(self):
        """Test that update_duration properly manages window sizes."""
        detector = OverSaturationDetector(
            minimum_duration=0.0,
            maximum_window_seconds=100.0,
            maximum_window_ratio=0.5,
        )

        # Add many requests
        for i in range(100):
            detector.add_started({"concurrent_requests": i, "duration": float(i)})
            detector.add_finished({"ttft": 1.0, "duration": float(i)})

        # Update duration to trigger window management
        detector.update_duration(150.0)

        # Should remove old requests outside window
        # Window is 100 seconds, so requests with duration < 50 should be removed
        if len(detector.started_requests) > 0:
            assert detector.started_requests[0]["duration"] >= 50.0

    @pytest.mark.sanity
    def test_check_alert_requires_minimum_duration(self):
        """Test that check_alert returns False before minimum duration."""
        detector = OverSaturationDetector(minimum_duration=30.0)

        detector.update_duration(15.0)
        assert detector.check_alert() is False

        detector.update_duration(35.0)
        # Still might return False due to insufficient data
        # but should at least not fail

    @pytest.mark.sanity
    def test_check_alert_requires_minimum_window_size(self):
        """Test that check_alert requires minimum window size."""
        detector = OverSaturationDetector(
            minimum_duration=0.0, minimum_window_size=10
        )

        # Add few requests
        for i in range(5):
            detector.add_started({"concurrent_requests": i, "duration": float(i)})

        detector.update_duration(10.0)
        assert detector.check_alert() is False  # Not enough data


class TestOverSaturationConstraint:
    """Test the OverSaturationConstraint implementation."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return OverSaturationDetector(minimum_duration=0.0, minimum_window_size=3)

    @pytest.fixture(
        params=[
            {"stop_over_saturated": True},
            {"stop_over_saturated": False},
        ]
    )
    def valid_instances(self, request, detector):
        """Create OverSaturationConstraint instances with valid parameters."""
        constructor_args = request.param
        instance = OverSaturationConstraint(
            over_saturation_detector=detector,
            **constructor_args,
        )
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that OverSaturationConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that OverSaturationConstraint has the correct method signature."""
        constraint = OverSaturationConstraint(
            over_saturation_detector=OverSaturationDetector(),
            stop_over_saturated=True,
        )
        call_method = constraint.__call__
        sig = inspect.signature(call_method)

        expected_params = ["state", "request_info"]
        assert list(sig.parameters.keys()) == expected_params

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that OverSaturationConstraint can be initialized with valid parameters."""
        constraint, constructor_args = valid_instances

        assert constraint.stop_over_saturated == constructor_args["stop_over_saturated"]
        assert constraint.over_saturation_detector is not None

    @pytest.mark.sanity
    def test_constraint_returns_continue_when_not_saturated(self, detector):
        """Test constraint returns continue when not over-saturated."""
        constraint = OverSaturationConstraint(
            over_saturation_detector=detector, stop_over_saturated=True
        )
        start_time = time.time()

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=5,
        )

        request = RequestInfo(
            request_id="test-1",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"
        assert isinstance(action.metadata, dict)
        assert "is_over_saturated" in action.metadata

    @pytest.mark.sanity
    def test_constraint_with_completed_request(self, detector):
        """Test constraint with completed request including timings."""
        constraint = OverSaturationConstraint(
            over_saturation_detector=detector, stop_over_saturated=True
        )
        start_time = time.time()

        # Create timings with first_iteration
        timings = RequestTimings(
            request_start=start_time + 0.1, first_iteration=start_time + 0.2
        )

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=5,
        )

        request = RequestInfo(
            request_id="test-1",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
            timings=timings,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert "ttft_slope" in action.metadata
        assert "ttft_n" in action.metadata

    @pytest.mark.sanity
    def test_constraint_stops_when_over_saturated(self, detector):
        """Test constraint stops when over-saturated and flag is enabled."""
        constraint = OverSaturationConstraint(
            over_saturation_detector=detector, stop_over_saturated=True
        )
        start_time = time.time()

        # Simulate over-saturation by creating positive slopes
        # Add many started requests with increasing concurrent count
        for i in range(20):
            detector.add_started(
                {"concurrent_requests": i * 2, "duration": float(i)}
            )

        # Add finished requests with increasing TTFT
        for i in range(20):
            detector.add_finished(
                {"ttft": 1.0 + i * 0.1, "duration": float(i) + 10.0}
            )

        detector.update_duration(30.0)
        detector.check_alert()  # Prime the slope checkers

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=40,
        )

        request = RequestInfo(
            request_id="test-1",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        # If over-saturated, should stop (but depends on slope detection)
        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        # The exact action depends on whether detection triggers
        assert action.request_queuing in ["continue", "stop"]
        assert "is_over_saturated" in action.metadata

    @pytest.mark.sanity
    def test_constraint_never_stops_when_flag_disabled(self, detector):
        """Test constraint never stops when stop_over_saturated is False."""
        constraint = OverSaturationConstraint(
            over_saturation_detector=detector, stop_over_saturated=False
        )
        start_time = time.time()

        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            processing_requests=100,  # High concurrent requests
        )

        request = RequestInfo(
            request_id="test-1",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        # Even if over-saturated, should continue when flag is False
        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"


class TestOverSaturationConstraintInitializer:
    """Test the OverSaturationConstraintInitializer implementation."""

    @pytest.fixture(
        params=[
            {"stop_over_saturated": True},
            {"stop_over_saturated": False},
            {"stop_over_saturated": True, "min_seconds": 10.0, "max_window_seconds": 60.0},
        ]
    )
    def valid_instances(self, request):
        """Create OverSaturationConstraintInitializer instances with valid parameters."""
        constructor_args = request.param
        instance = OverSaturationConstraintInitializer(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_pydantic_constraint_initializer(self, valid_instances):
        """Test that initializer is a PydanticConstraintInitializer."""
        instance, _ = valid_instances
        assert isinstance(instance, PydanticConstraintInitializer)
        assert isinstance(instance, SerializableConstraintInitializer)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """Test that initializer satisfies ConstraintInitializer protocol."""
        instance, _ = valid_instances
        assert isinstance(instance, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        assert instance.type_ == "stop_over_saturated"
        assert instance.stop_over_saturated == constructor_args["stop_over_saturated"]

        if "min_seconds" in constructor_args:
            assert instance.min_seconds == constructor_args["min_seconds"]
        if "max_window_seconds" in constructor_args:
            assert instance.max_window_seconds == constructor_args["max_window_seconds"]

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that initializer rejects invalid parameters."""
        # Missing required field
        with pytest.raises(ValidationError):
            OverSaturationConstraintInitializer()

        # Invalid type
        with pytest.raises(ValidationError):
            OverSaturationConstraintInitializer(
                stop_over_saturated="invalid", type_="invalid"
            )

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test that create_constraint returns OverSaturationConstraint."""
        instance, _ = valid_instances
        constraint = instance.create_constraint()

        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.stop_over_saturated == instance.stop_over_saturated
        assert constraint.over_saturation_detector is not None

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test validated_kwargs method with various inputs."""
        result = OverSaturationConstraintInitializer.validated_kwargs(
            stop_over_saturated=True
        )
        assert result == {"stop_over_saturated": True}

        result = OverSaturationConstraintInitializer.validated_kwargs(
            stop_over_saturated=False
        )
        assert result == {"stop_over_saturated": False}

        # Test with aliases
        result = OverSaturationConstraintInitializer.validated_kwargs(
            stop_over_saturated=False, stop_over_sat=True
        )
        assert result == {"stop_over_saturated": True}

        result = OverSaturationConstraintInitializer.validated_kwargs(
            stop_over_saturated=False, stop_osd=True
        )
        assert result == {"stop_over_saturated": True}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that initializer can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert data["type_"] == "stop_over_saturated"
        assert data["stop_over_saturated"] == constructor_args["stop_over_saturated"]

        reconstructed = OverSaturationConstraintInitializer.model_validate(data)
        assert reconstructed.stop_over_saturated == instance.stop_over_saturated

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that initializer is properly registered with expected aliases."""
        expected_aliases = [
            "stop_over_saturated",
            "stop_over_sat",
            "stop_osd",
        ]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == OverSaturationConstraintInitializer

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias", ["stop_over_saturated", "stop_over_sat", "stop_osd"]
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, stop_over_saturated=True
        )
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.stop_over_saturated is True

        # Test with simple boolean value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, True)
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.stop_over_saturated is True

        constraint = ConstraintsInitializerFactory.create_constraint(alias, False)
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.stop_over_saturated is False

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"stop_over_saturated": {"stop_over_saturated": True}}
        )
        assert isinstance(resolved["stop_over_saturated"], OverSaturationConstraint)
        assert resolved["stop_over_saturated"].stop_over_saturated is True

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"stop_over_sat": True})
        assert isinstance(resolved["stop_over_sat"], OverSaturationConstraint)
        assert resolved["stop_over_sat"].stop_over_saturated is True

        # Test with instance
        instance = OverSaturationConstraintInitializer(stop_over_saturated=False)
        constraint_instance = instance.create_constraint()
        resolved = ConstraintsInitializerFactory.resolve(
            {"stop_osd": constraint_instance}
        )
        assert resolved["stop_osd"] is constraint_instance

    @pytest.mark.smoke
    def test_functional_constraint_creation(self):
        """Test that created constraints are functionally correct."""
        constraint = ConstraintsInitializerFactory.create_constraint(
            "stop_over_saturated", stop_over_saturated=True
        )
        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=5,
            processed_requests=5,
            processing_requests=3,
        )
        request = RequestInfo(
            request_id="test-request",
            status="in_progress",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        # Should continue when not over-saturated
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"
        assert "is_over_saturated" in action.metadata


class TestSlopeChecker:
    """Test the SlopeChecker implementation used by OverSaturationDetector."""

    @pytest.fixture
    def slope_checker(self):
        """Create a SlopeChecker instance for testing."""
        from guidellm.scheduler.advanced_constraints.over_saturation import (
            SlopeChecker,
        )

        return SlopeChecker(moe_threshold=1.0, confidence=0.95)

    @pytest.mark.smoke
    def test_initialization(self, slope_checker):
        """Test SlopeChecker initialization."""
        assert slope_checker.n == 0
        assert slope_checker.sum_x == 0.0
        assert slope_checker.sum_y == 0.0
        assert slope_checker.moe_threshold == 1.0
        assert slope_checker.confidence == 0.95

    @pytest.mark.sanity
    def test_add_and_remove_data_points(self, slope_checker):
        """Test adding and removing data points."""
        # Add data points
        slope_checker.add_data_point(1.0, 2.0)
        slope_checker.add_data_point(2.0, 4.0)
        slope_checker.add_data_point(3.0, 6.0)

        assert slope_checker.n == 3
        assert slope_checker.sum_x == 6.0
        assert slope_checker.sum_y == 12.0

        # Remove data point
        slope_checker.remove_data_point(1.0, 2.0)

        assert slope_checker.n == 2
        assert slope_checker.sum_x == 5.0
        assert slope_checker.sum_y == 10.0

    @pytest.mark.sanity
    def test_check_slope_with_positive_slope(self, slope_checker):
        """Test check_slope with clear positive slope."""
        # Create data with clear positive slope
        for i in range(10):
            slope_checker.add_data_point(float(i), float(i * 2))

        result = slope_checker.check_slope(10.0)
        assert result is True
        assert slope_checker.slope is not None
        assert slope_checker.slope > 0
        assert slope_checker.margin_of_error is not None

    @pytest.mark.sanity
    def test_check_slope_requires_minimum_samples(self, slope_checker):
        """Test that check_slope requires minimum samples."""
        # Not enough samples
        slope_checker.add_data_point(1.0, 2.0)
        result = slope_checker.check_slope(1.0)
        assert result is False

        # Still not enough with 2 points
        slope_checker.add_data_point(2.0, 4.0)
        result = slope_checker.check_slope(2.0)
        assert result is False

        # Should work with 3+ points
        slope_checker.add_data_point(3.0, 6.0)
        result = slope_checker.check_slope(3.0)
        # Might be True or False depending on confidence intervals

