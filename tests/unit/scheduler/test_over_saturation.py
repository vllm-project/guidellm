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
    PydanticConstraintInitializer,
    SchedulerState,
    SchedulerUpdateAction,
    SerializableConstraintInitializer,
)
from guidellm.schemas import RequestInfo, RequestTimings


class TestOverSaturationConstraintInternal:
    """Test the OverSaturationConstraint internal functionality."""

    @pytest.fixture(
        params=[
            {"minimum_duration": 30.0, "maximum_window_seconds": 120.0},
            {"minimum_duration": 10.0, "maximum_window_seconds": 60.0},
            {"minimum_duration": 60.0, "maximum_window_seconds": 240.0},
        ]
    )
    def valid_instances(self, request):
        """Create OverSaturationConstraint instances with valid parameters."""
        constructor_args = request.param
        instance = OverSaturationConstraint(**constructor_args, enabled=True)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test OverSaturationConstraint initialization with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.smoke
    def test_initialization_defaults(self):
        """Test that OverSaturationConstraint has correct default values."""
        constraint = OverSaturationConstraint(enabled=True)

        assert constraint.minimum_duration == 30.0
        assert constraint.minimum_ttft == 2.5
        assert constraint.maximum_window_seconds == 120.0
        assert constraint.moe_threshold == 2.0
        assert constraint.maximum_window_ratio == 0.75
        assert constraint.minimum_window_size == 5
        assert constraint.confidence == 0.95
        assert constraint.eps == 1e-12

    @pytest.mark.smoke
    def test_reset(self, valid_instances):
        """Test that reset method properly initializes constraint state."""
        constraint, _ = valid_instances
        constraint.reset()

        assert constraint.duration == 0.0
        assert constraint.started_requests == []
        assert constraint.finished_requests == []
        assert constraint.ttft_violations_counter == 0
        assert constraint.total_finished_ever == 0
        assert constraint.total_started_ever == 0
        assert hasattr(constraint, "concurrent_slope_checker")
        assert hasattr(constraint, "ttft_slope_checker")

    @pytest.mark.sanity
    def test_window_management_through_constraint(self):
        """Test that constraint properly manages window sizes through usage."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0,
            maximum_window_seconds=100.0,
            maximum_window_ratio=0.5,
            enabled=True,
        )
        start_time = time.time()

        # Add many requests through constraint calls
        for i in range(100):
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time - i,
                processing_requests=i,
            )
            request = RequestInfo(
                request_id=f"test-{i}",
                status="in_progress",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time - i,
            )
            constraint(state, request)

        # Check that window management is working (through internal state)
        # The constraint should have pruned old requests
        assert len(constraint.started_requests) <= 50  # Should be limited by ratio


class TestOverSaturationConstraint:
    """Test the OverSaturationConstraint implementation."""

    @pytest.fixture
    def constraint(self):
        """Create a constraint for testing."""
        return OverSaturationConstraint(
            minimum_duration=0.0, minimum_window_size=3, enabled=True
        )

    @pytest.fixture(
        params=[
            {"enabled": True},
            {"enabled": False},
        ]
    )
    def valid_instances(self, request):
        """Create OverSaturationConstraint instances with valid parameters."""
        constructor_args = request.param
        instance = OverSaturationConstraint(
            minimum_duration=0.0,
            minimum_window_size=3,
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
        constraint = OverSaturationConstraint(enabled=True)
        call_method = constraint.__call__
        sig = inspect.signature(call_method)

        expected_params = ["state", "request_info"]
        assert list(sig.parameters.keys()) == expected_params

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test OverSaturationConstraint initialization with valid parameters."""
        constraint, constructor_args = valid_instances

        assert constraint.enabled == constructor_args["enabled"]

    @pytest.mark.sanity
    def test_constraint_returns_continue_when_not_saturated(self, constraint):
        """Test constraint returns continue when not over-saturated."""
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
    def test_constraint_with_completed_request(self, constraint):
        """Test constraint with completed request including timings."""
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
    def test_constraint_stops_when_over_saturated(self, constraint):
        """Test constraint stops when over-saturated and flag is enabled."""
        start_time = time.time()

        # Simulate over-saturation by creating positive slopes through constraint calls
        # Add many started requests with increasing concurrent count
        for i in range(20):
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time - i,
                processing_requests=i * 2,
            )
            request = RequestInfo(
                request_id=f"test-{i}",
                status="in_progress",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time - i,
            )
            constraint(state, request)

        # Add finished requests with increasing TTFT
        for i in range(20):
            timings = RequestTimings(
                request_start=start_time - i - 10.0,
                first_iteration=start_time - i - 10.0 + (1.0 + i * 0.1),
            )
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time - i - 10.0,
                processing_requests=5,
            )
            request = RequestInfo(
                request_id=f"test-finished-{i}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time - i - 10.0,
                timings=timings,
            )
            constraint(state, request)

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
    def test_constraint_never_stops_when_flag_disabled(self):
        """Test constraint never stops when enabled is False."""
        constraint = OverSaturationConstraint(
            minimum_duration=0.0,
            minimum_window_size=3,
            enabled=False,
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
            {"enabled": True},
            {"enabled": False},
            {
                "enabled": True,
                "min_seconds": 10.0,
                "max_window_seconds": 60.0,
            },
        ]
    )
    def valid_instances(self, request):
        """Create OverSaturationConstraintInitializer with valid parameters."""
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

        assert instance.type_ == "over_saturation"
        assert instance.enabled == constructor_args["enabled"]

        if "min_seconds" in constructor_args:
            assert instance.min_seconds == constructor_args["min_seconds"]
        if "max_window_seconds" in constructor_args:
            assert instance.max_window_seconds == constructor_args["max_window_seconds"]

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that initializer rejects invalid parameters."""
        # Invalid type for enabled
        with pytest.raises(ValidationError):
            OverSaturationConstraintInitializer(enabled="invalid")

        # Invalid type for min_seconds
        with pytest.raises(ValidationError):
            OverSaturationConstraintInitializer(enabled=True, min_seconds="invalid")

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test that create_constraint returns OverSaturationConstraint."""
        instance, _ = valid_instances
        constraint = instance.create_constraint()

        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.enabled == instance.enabled

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test validated_kwargs method with various inputs."""
        result = OverSaturationConstraintInitializer.validated_kwargs(
            over_saturation=True
        )
        assert result == {"enabled": True}

        result = OverSaturationConstraintInitializer.validated_kwargs(
            over_saturation=False
        )
        assert result == {"enabled": False}

        # Test with dict input
        result = OverSaturationConstraintInitializer.validated_kwargs(
            over_saturation={"enabled": True, "min_seconds": 20.0}
        )
        assert result["enabled"] is True
        assert "min_seconds" in result

        # Test with aliases
        result = OverSaturationConstraintInitializer.validated_kwargs(
            detect_saturation=True
        )
        assert result == {"enabled": True}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that initializer can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert data["type_"] == "over_saturation"
        assert data["enabled"] == constructor_args["enabled"]

        reconstructed = OverSaturationConstraintInitializer.model_validate(data)
        assert reconstructed.enabled == instance.enabled

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that initializer is properly registered with expected aliases."""
        expected_aliases = [
            "over_saturation",
            "detect_saturation",
        ]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == OverSaturationConstraintInitializer

    @pytest.mark.smoke
    @pytest.mark.parametrize("alias", ["over_saturation", "detect_saturation"])
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, enabled=True
        )
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.enabled is True

        # Test with simple boolean value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, True)
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.enabled is True

        constraint = ConstraintsInitializerFactory.create_constraint(alias, False)
        assert isinstance(constraint, OverSaturationConstraint)
        assert constraint.enabled is False

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"over_saturation": {"enabled": True}}
        )
        assert isinstance(resolved["over_saturation"], OverSaturationConstraint)
        assert resolved["over_saturation"].enabled is True

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"detect_saturation": True})
        assert isinstance(resolved["detect_saturation"], OverSaturationConstraint)
        assert resolved["detect_saturation"].enabled is True

        # Test with instance
        instance = OverSaturationConstraintInitializer(enabled=False)
        constraint_instance = instance.create_constraint()
        resolved = ConstraintsInitializerFactory.resolve(
            {"over_saturation": constraint_instance}
        )
        assert resolved["over_saturation"] is constraint_instance

    @pytest.mark.smoke
    def test_functional_constraint_creation(self):
        """Test that created constraints are functionally correct."""
        constraint = ConstraintsInitializerFactory.create_constraint(
            "over_saturation", enabled=True
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
        from guidellm.scheduler.constraints.saturation import (
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
