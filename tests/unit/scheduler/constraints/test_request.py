import time

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    MaxDurationConstraint,
    MaxNumberConstraint,
    MinNumberConstraint,
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo


class TestMaxNumberConstraint:
    """Test the MaxNumberConstraint implementation."""

    @pytest.fixture(params=[{"max_num": 100}, {"max_num": 50.5}, {"max_num": 1}])
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxNumberConstraint(**constructor_args)

        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxNumberConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """Test MaxNumberConstraint satisfies the ConstraintInitializer protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxNumberConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxNumberConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxNumberConstraint()
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num=-1)
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num=0)
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions and progress"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        for num_requests in range(0, int(constructor_args["max_num"]) * 2 + 1, 1):
            state = SchedulerState(
                start_time=start_time,
                created_requests=num_requests,
                processed_requests=num_requests,
                errored_requests=0,
            )
            request_info = RequestInfo(
                request_id="test", status="completed", created_at=start_time
            )

            action = instance(state, request_info)
            assert isinstance(action, SchedulerUpdateAction)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxNumberConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxNumberConstraint.model_validate(data)
        assert reconstructed.max_num == instance.max_num

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_create_constraint_functionality(self, valid_instances):
        """Test the constraint initializer functionality."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxNumberConstraint)
        assert constraint.max_num == constructor_args["max_num"]

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MaxNumberConstraint.validated_kwargs class method."""
        result = MaxNumberConstraint.validated_kwargs(max_num=100)
        assert result == {"max_num": 100, "current_index": -1}

        result = MaxNumberConstraint.validated_kwargs(50.5)
        assert result == {"max_num": 50.5, "current_index": -1}

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MaxNumberConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MaxNumberConstraint)
        assert constraint is not instance  # Should return a copy
        assert constraint.max_num == instance.max_num
        assert instance.current_index == original_index + 1  # Original is incremented
        assert constraint.current_index == original_index + 1  # Copy has incremented

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MaxNumberConstraint is properly registered with expected aliases."""
        expected_aliases = ["max_number", "max_num", "max_requests", "max_req"]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MaxNumberConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias", ["max_number", "max_num", "max_requests", "max_req"]
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(alias, max_num=100)
        assert isinstance(constraint, MaxNumberConstraint)
        assert constraint.max_num == 100

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 50)
        assert isinstance(constraint, MaxNumberConstraint)
        assert constraint.max_num == 50

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_number": {"max_num": 200}}
        )
        assert isinstance(resolved["max_number"], MaxNumberConstraint)
        assert resolved["max_number"].max_num == 200

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"max_num": 150})
        assert isinstance(resolved["max_num"], MaxNumberConstraint)
        assert resolved["max_num"].max_num == 150

        # Test with instance
        instance = MaxNumberConstraint(max_num=75)
        resolved = ConstraintsInitializerFactory.resolve({"max_requests": instance})
        assert resolved["max_requests"] is instance


class TestMinNumberConstraint:
    """Test the MinNumberConstraint implementation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture(params=[{"min_num": 100}, {"min_num": 50.5}, {"min_num": 1}])
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MinNumberConstraint(**constructor_args)

        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MinNumberConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """Test MinNumberConstraint satisfies the ConstraintInitializer protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MinNumberConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MinNumberConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MinNumberConstraint()
        with pytest.raises(ValidationError):
            MinNumberConstraint(min_num=-1)
        with pytest.raises(ValidationError):
            MinNumberConstraint(min_num=0)
        with pytest.raises(ValidationError):
            MinNumberConstraint(min_num="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions and progress"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        for num_requests in range(0, int(constructor_args["min_num"]) * 2 + 1, 1):
            state = SchedulerState(
                start_time=start_time,
                created_requests=num_requests,
                processed_requests=num_requests,
                errored_requests=0,
            )
            request_info = RequestInfo(
                request_id="test", status="completed", created_at=start_time
            )

            action = instance(state, request_info)
            assert isinstance(action, SchedulerUpdateAction)

            processed_exceeded = num_requests >= constructor_args["min_num"]

            if not processed_exceeded:
                assert action.request_queuing == "continue"
                assert action.request_processing == "continue"
            else:
                assert action.request_queuing == "stop"
                assert action.request_processing == "stop_local"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MinNumberConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MinNumberConstraint.model_validate(data)
        assert reconstructed.min_num == instance.min_num

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_create_constraint_functionality(self, valid_instances):
        """Test the constraint initializer functionality."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MinNumberConstraint)
        assert constraint.min_num == constructor_args["min_num"]

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MinNumberConstraint.validated_kwargs class method."""
        result = MinNumberConstraint.validated_kwargs(min_num=100)
        assert result == {"min_num": 100, "current_index": -1}

        result = MinNumberConstraint.validated_kwargs(50.5)
        assert result == {"min_num": 50.5, "current_index": -1}

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MinNumberConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MinNumberConstraint)
        assert constraint is not instance  # Should return a copy
        assert constraint.min_num == instance.min_num
        assert instance.current_index == original_index + 1  # Original is incremented
        assert constraint.current_index == original_index + 1  # Copy has incremented

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MinNumberConstraint is properly registered with expected aliases."""
        expected_aliases = ["min_number", "min_num", "min_requests", "min_req"]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MinNumberConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias", ["min_number", "min_num", "min_requests", "min_req"]
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(alias, min_num=100)
        assert isinstance(constraint, MinNumberConstraint)
        assert constraint.min_num == 100

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 50)
        assert isinstance(constraint, MinNumberConstraint)
        assert constraint.min_num == 50

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"min_number": {"min_num": 200}}
        )
        assert isinstance(resolved["min_number"], MinNumberConstraint)
        assert resolved["min_number"].min_num == 200

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"min_num": 150})
        assert isinstance(resolved["min_num"], MinNumberConstraint)
        assert resolved["min_num"].min_num == 150

        # Test with instance
        instance = MinNumberConstraint(min_num=75)
        resolved = ConstraintsInitializerFactory.resolve({"min_requests": instance})
        assert resolved["min_requests"] is instance


class TestMaxDurationConstraint:
    """Test the MaxDurationConstraint implementation."""

    @pytest.fixture(
        params=[{"max_duration": 2.0}, {"max_duration": 1}, {"max_duration": 0.5}]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxDurationConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxDurationConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxDurationConstraint also satisfies
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxDurationConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxDurationConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxDurationConstraint()
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration=-1)
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration=0)
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions and progress through a time loop"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_duration = constructor_args["max_duration"]
        sleep_interval = max_duration * 0.05
        target_duration = max_duration * 1.5

        elapsed = 0.0
        step = 0

        while elapsed <= target_duration:
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=step + 1,
                processed_requests=step,
            )
            request = RequestInfo(
                request_id=f"test-{step}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)

            duration_exceeded = elapsed >= max_duration

            if not duration_exceeded:
                assert action.request_queuing == "continue"
                assert action.request_processing == "continue"
            else:
                assert action.request_queuing == "stop"
                assert action.request_processing == "stop_local"
            assert isinstance(action.metadata, dict)
            assert action.metadata["max_duration"] == max_duration
            assert action.metadata["elapsed_time"] == pytest.approx(elapsed, abs=0.01)
            assert action.metadata["duration_exceeded"] == duration_exceeded
            assert action.metadata["start_time"] == start_time

            assert isinstance(action.progress, SchedulerProgress)
            expected_remaining_fraction = max(0.0, 1.0 - elapsed / max_duration)
            expected_remaining_duration = max(0.0, max_duration - elapsed)
            assert action.progress.remaining_fraction == pytest.approx(
                expected_remaining_fraction, abs=0.1
            )
            assert action.progress.remaining_duration == pytest.approx(
                expected_remaining_duration, abs=0.1
            )
            time.sleep(sleep_interval)
            elapsed = time.time() - start_time
            step += 1

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxDurationConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxDurationConstraint.model_validate(data)
        assert reconstructed.max_duration == instance.max_duration

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_create_constraint_functionality(self, valid_instances):
        """Test the constraint initializer functionality."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxDurationConstraint)
        assert constraint.max_duration == constructor_args["max_duration"]

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MaxDurationConstraint.validated_kwargs class method."""
        result = MaxDurationConstraint.validated_kwargs(max_duration=60.0)
        assert result == {"max_duration": 60.0, "current_index": -1}

        result = MaxDurationConstraint.validated_kwargs(30)
        assert result == {"max_duration": 30, "current_index": -1}

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MaxDurationConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MaxDurationConstraint)
        assert constraint is not instance  # Should return a copy
        assert constraint.max_duration == instance.max_duration
        assert instance.current_index == original_index + 1  # Original is incremented
        assert constraint.current_index == original_index + 1  # Copy has incremented

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MaxDurationConstraint is properly registered with expected aliases."""
        expected_aliases = [
            "max_duration",
            "max_dur",
            "max_sec",
            "max_seconds",
            "max_min",
            "max_minutes",
        ]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MaxDurationConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias",
        ["max_duration", "max_dur", "max_sec", "max_seconds", "max_min", "max_minutes"],
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, max_duration=60.0
        )
        assert isinstance(constraint, MaxDurationConstraint)
        assert constraint.max_duration == 60.0

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 30.0)
        assert isinstance(constraint, MaxDurationConstraint)
        assert constraint.max_duration == 30.0

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_duration": {"max_duration": 120.0}}
        )
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert resolved["max_duration"].max_duration == 120.0

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"max_sec": 90.0})
        assert isinstance(resolved["max_sec"], MaxDurationConstraint)
        assert resolved["max_sec"].max_duration == 90.0

        # Test with instance
        instance = MaxDurationConstraint(max_duration=45.0)
        resolved = ConstraintsInitializerFactory.resolve({"max_minutes": instance})
        assert resolved["max_minutes"] is instance
