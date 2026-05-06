import random
import time

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo


class TestMaxErrorsConstraint:
    """Test the MaxErrorsConstraint implementation."""

    @pytest.fixture(params=[{"max_errors": 10}, {"max_errors": 5.5}, {"max_errors": 1}])
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxErrorsConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxErrorsConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxErrorsConstraint also satisfies
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxErrorsConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxErrorsConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorsConstraint()
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors=-1)
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors=0)
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        for num_errors in range(int(constructor_args["max_errors"] * 2)):
            created_requests = (num_errors + 1) * 2
            processed_requests = num_errors + 1
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=created_requests,
                processed_requests=processed_requests,
                errored_requests=num_errors,
            )
            request = RequestInfo(
                request_id=f"test-{num_errors}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )
            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)
            errors_exceeded = num_errors >= constructor_args["max_errors"]
            if not errors_exceeded:
                assert action.request_queuing == "continue"
                assert action.request_processing == "continue"
            else:
                assert action.request_queuing == "stop"
                assert action.request_processing == "stop_all"

            assert isinstance(action.metadata, dict)
            expected_metadata = {
                "max_errors": constructor_args["max_errors"],
                "errors_exceeded": errors_exceeded,
                "current_errors": num_errors,
            }
            # Note: metadata may have additional fields like stop_time
            for key, value in expected_metadata.items():
                assert action.metadata[key] == value

            assert isinstance(action.progress, SchedulerProgress)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxErrorsConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorsConstraint.model_validate(data)
        assert reconstructed.max_errors == instance.max_errors

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MaxErrorsConstraint.validated_kwargs class method."""
        result = MaxErrorsConstraint.validated_kwargs(max_errors=10)
        assert result == {"max_errors": 10, "current_index": -1}

        result = MaxErrorsConstraint.validated_kwargs(5.5)
        assert result == {"max_errors": 5.5, "current_index": -1}

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MaxErrorsConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MaxErrorsConstraint)
        assert constraint is not instance
        assert constraint.max_errors == instance.max_errors
        assert instance.current_index == original_index + 1
        assert constraint.current_index == original_index + 1

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MaxErrorsConstraint is properly registered with expected aliases."""
        expected_aliases = ["max_errors", "max_err", "max_error", "max_errs"]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MaxErrorsConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias", ["max_errors", "max_err", "max_error", "max_errs"]
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, max_errors=10
        )
        assert isinstance(constraint, MaxErrorsConstraint)
        assert constraint.max_errors == 10

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 5)
        assert isinstance(constraint, MaxErrorsConstraint)
        assert constraint.max_errors == 5

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_errors": {"max_errors": 15}}
        )
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)
        assert resolved["max_errors"].max_errors == 15

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"max_err": 8})
        assert isinstance(resolved["max_err"], MaxErrorsConstraint)
        assert resolved["max_err"].max_errors == 8

        # Test with instance
        instance = MaxErrorsConstraint(max_errors=3)
        resolved = ConstraintsInitializerFactory.resolve({"max_error": instance})
        assert resolved["max_error"] is instance


class TestMaxErrorRateConstraint:
    """Test the MaxErrorRateConstraint implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "window_size": 40},
            {"max_error_rate": 0.5, "window_size": 50},
            {"max_error_rate": 0.05, "window_size": 55},
        ]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxErrorRateConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxErrorRateConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxErrorRateConstraint also satisfies
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxErrorRateConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxErrorRateConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint()
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=0.5, window_size=0)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions with sliding window behavior"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_error_rate = constructor_args["max_error_rate"]
        window_size = constructor_args["window_size"]
        safety_factor = 1.5
        total_errors = 0
        error_window = []

        for request_num in range(window_size * 2):
            error_probability = max_error_rate * safety_factor

            if random.random() < error_probability:
                total_errors += 1
                status = "errored"
                error_window.append(1)
            else:
                status = "completed"
                error_window.append(0)
            error_window = (
                error_window[-window_size:]
                if len(error_window) > window_size
                else error_window
            )

            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=request_num + 1,
                processed_requests=request_num + 1,
            )
            request = RequestInfo(
                request_id=f"test-{request_num}",
                status=status,
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)
            error_count = sum(instance.error_window)
            processed_requests = state.processed_requests
            exceeded_min_processed = processed_requests >= window_size
            current_error_rate = (
                error_count / float(min(processed_requests, window_size))
                if processed_requests > 0
                else 0.0
            )
            exceeded_error_rate = current_error_rate >= max_error_rate
            should_stop = exceeded_min_processed and exceeded_error_rate
            expected_queuing = "stop" if should_stop else "continue"
            expected_processing = "stop_all" if should_stop else "continue"

            assert action.request_queuing == expected_queuing
            assert action.request_processing == expected_processing
            assert isinstance(action.metadata, dict)
            assert action.metadata["max_error_rate"] == max_error_rate
            assert action.metadata["window_size"] == window_size
            assert action.metadata["error_count"] == error_count
            assert action.metadata["current_error_rate"] == current_error_rate
            assert action.metadata["exceeded_error_rate"] == exceeded_error_rate

            assert isinstance(action.progress, SchedulerProgress)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxErrorRateConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorRateConstraint.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.window_size == instance.window_size

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MaxErrorRateConstraint.validated_kwargs class method."""
        result = MaxErrorRateConstraint.validated_kwargs(
            max_error_rate=0.1, window_size=50
        )
        assert result == {
            "max_error_rate": 0.1,
            "window_size": 50,
            "error_window": [],
            "current_index": -1,
        }

        result = MaxErrorRateConstraint.validated_kwargs(0.05)
        assert result == {
            "max_error_rate": 0.05,
            "window_size": 30,
            "error_window": [],
            "current_index": -1,
        }

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MaxErrorRateConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MaxErrorRateConstraint)
        assert constraint is not instance  # Should return a copy
        assert constraint.max_error_rate == instance.max_error_rate
        assert constraint.window_size == instance.window_size
        assert instance.current_index == original_index + 1  # Original is incremented
        assert constraint.current_index == original_index + 1  # Copy has incremented

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MaxErrorRateConstraint is properly registered with expected aliases."""
        expected_aliases = ["max_error_rate", "max_err_rate", "max_errors_rate"]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MaxErrorRateConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias", ["max_error_rate", "max_err_rate", "max_errors_rate"]
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, max_error_rate=0.1, window_size=50
        )
        assert isinstance(constraint, MaxErrorRateConstraint)
        assert constraint.max_error_rate == 0.1
        assert constraint.window_size == 50

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 0.05)
        assert isinstance(constraint, MaxErrorRateConstraint)
        assert constraint.max_error_rate == 0.05

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_error_rate": {"max_error_rate": 0.15, "window_size": 100}}
        )
        assert isinstance(resolved["max_error_rate"], MaxErrorRateConstraint)
        assert resolved["max_error_rate"].max_error_rate == 0.15
        assert resolved["max_error_rate"].window_size == 100

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"max_err_rate": 0.08})
        assert isinstance(resolved["max_err_rate"], MaxErrorRateConstraint)
        assert resolved["max_err_rate"].max_error_rate == 0.08

        # Test with instance
        instance = MaxErrorRateConstraint(max_error_rate=0.2, window_size=25)
        resolved = ConstraintsInitializerFactory.resolve({"max_errors_rate": instance})
        assert resolved["max_errors_rate"] is instance


class TestMaxGlobalErrorRateConstraint:
    """Test the MaxGlobalErrorRateConstraint implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "min_processed": 50},
            {"max_error_rate": 0.2, "min_processed": 100},
            {"max_error_rate": 0.05, "min_processed": 31},
        ]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxGlobalErrorRateConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxGlobalErrorRateConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxGlobalErrorRateConstraint also satisfies
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """
        Test that MaxGlobalErrorRateConstraint can be initialized
        with valid parameters.
        """
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxGlobalErrorRateConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint()
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=0.5, min_processed=0)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions based on global error rate"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_error_rate = constructor_args["max_error_rate"]
        min_processed = constructor_args["min_processed"]
        safety_factor = 1.5
        total_requests = min_processed * 2
        total_errors = 0

        for request_num in range(total_requests):
            error_probability = max_error_rate * safety_factor

            if random.random() < error_probability:
                total_errors += 1
                status = "errored"
            else:
                status = "completed"

            processed_requests = request_num + 1

            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=processed_requests + 10,
                processed_requests=processed_requests,
                errored_requests=total_errors,
            )
            request = RequestInfo(
                request_id=f"test-{request_num}",
                status=status,
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)

            exceeded_min_processed = processed_requests >= min_processed
            error_rate = (
                total_errors / float(processed_requests)
                if processed_requests > 0
                else 0.0
            )
            exceeded_error_rate = error_rate >= max_error_rate
            should_stop = exceeded_min_processed and exceeded_error_rate

            expected_queuing = "stop" if should_stop else "continue"
            expected_processing = "stop_all" if should_stop else "continue"

            assert action.request_queuing == expected_queuing
            assert action.request_processing == expected_processing

            assert isinstance(action.metadata, dict)
            expected_metadata = {
                "max_error_rate": max_error_rate,
                "min_processed": min_processed,
                "processed_requests": processed_requests,
                "errored_requests": total_errors,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            }
            # Note: metadata may have additional fields like stop_time and exceeded
            for key, value in expected_metadata.items():
                assert action.metadata[key] == value

            # Error constraints don't provide progress information
            assert isinstance(action.progress, SchedulerProgress)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxGlobalErrorRateConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxGlobalErrorRateConstraint.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.min_processed == instance.min_processed

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test MaxGlobalErrorRateConstraint.validated_kwargs class method."""
        result = MaxGlobalErrorRateConstraint.validated_kwargs(
            max_error_rate=0.1, min_processed=50
        )
        assert result == {
            "max_error_rate": 0.1,
            "min_processed": 50,
            "current_index": -1,
        }

        result = MaxGlobalErrorRateConstraint.validated_kwargs(0.05)
        assert result == {
            "max_error_rate": 0.05,
            "min_processed": 30,
            "current_index": -1,
        }

    @pytest.mark.smoke
    def test_create_constraint(self, valid_instances):
        """Test MaxGlobalErrorRateConstraint.create_constraint method."""
        instance, constructor_args = valid_instances
        original_index = instance.current_index
        constraint = instance.create_constraint()

        assert isinstance(constraint, MaxGlobalErrorRateConstraint)
        assert constraint is not instance  # Should return a copy
        assert constraint.max_error_rate == instance.max_error_rate
        assert constraint.min_processed == instance.min_processed
        assert instance.current_index == original_index + 1  # Original is incremented
        assert constraint.current_index == original_index + 1  # Copy has incremented

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test MaxGlobalErrorRateConstraint is properly registered with aliases."""
        expected_aliases = [
            "max_global_error_rate",
            "max_global_err_rate",
            "max_global_errors_rate",
        ]

        for alias in expected_aliases:
            assert ConstraintsInitializerFactory.is_registered(alias)
            registered_class = ConstraintsInitializerFactory.get_registered_object(
                alias
            )
            assert registered_class == MaxGlobalErrorRateConstraint

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "alias",
        ["max_global_error_rate", "max_global_err_rate", "max_global_errors_rate"],
    )
    def test_factory_creation_with_aliases(self, alias):
        """Test factory creation using different aliases."""
        # Test with dict configuration
        constraint = ConstraintsInitializerFactory.create_constraint(
            alias, max_error_rate=0.1, min_processed=50
        )
        assert isinstance(constraint, MaxGlobalErrorRateConstraint)
        assert constraint.max_error_rate == 0.1
        assert constraint.min_processed == 50

        # Test with simple value
        constraint = ConstraintsInitializerFactory.create_constraint(alias, 0.05)
        assert isinstance(constraint, MaxGlobalErrorRateConstraint)
        assert constraint.max_error_rate == 0.05

    @pytest.mark.smoke
    def test_factory_resolve_methods(self):
        """Test factory resolve methods with various input formats."""
        # Test with dict config
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_global_error_rate": {"max_error_rate": 0.12, "min_processed": 100}}
        )
        assert isinstance(
            resolved["max_global_error_rate"], MaxGlobalErrorRateConstraint
        )
        assert resolved["max_global_error_rate"].max_error_rate == 0.12
        assert resolved["max_global_error_rate"].min_processed == 100

        # Test with simple value
        resolved = ConstraintsInitializerFactory.resolve({"max_global_err_rate": 0.08})
        assert isinstance(resolved["max_global_err_rate"], MaxGlobalErrorRateConstraint)
        assert resolved["max_global_err_rate"].max_error_rate == 0.08

        # Test with instance
        instance = MaxGlobalErrorRateConstraint(max_error_rate=0.15, min_processed=75)
        resolved = ConstraintsInitializerFactory.resolve(
            {"max_global_errors_rate": instance}
        )
        assert resolved["max_global_errors_rate"] is instance
