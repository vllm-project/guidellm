import inspect
import random
import time
from abc import ABC
from typing import Protocol

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    MaxDurationConstraint,
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
    MaxNumberConstraint,
    PydanticConstraintInitializer,
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from guidellm.utils import InfoMixin, StandardBaseModel


class TestConstraint:
    """Test the Constraint protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that Constraint is a protocol and runtime checkable."""
        assert issubclass(Constraint, Protocol)
        assert hasattr(Constraint, "_is_protocol")
        assert Constraint._is_protocol is True
        assert hasattr(Constraint, "_is_runtime_protocol")
        assert Constraint._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that the Constraint protocol has the correct method signature."""
        call_method = Constraint.__call__
        sig = inspect.signature(call_method)

        expected_params = ["self", "state", "request"]
        assert list(sig.parameters.keys()) == expected_params

        params = sig.parameters
        assert "state" in params
        assert "request" in params

    @pytest.mark.smoke
    def test_runtime_is_constraint(self):
        """Test that Constraint can be checked at runtime using isinstance."""

        class ValidConstraint:
            def __call__(
                self,
                state: SchedulerState,
                request: ScheduledRequestInfo,
            ) -> SchedulerUpdateAction:
                return SchedulerUpdateAction()

        valid_instance = ValidConstraint()
        assert isinstance(valid_instance, Constraint)

        class InvalidConstraint:
            pass

        invalid_instance = InvalidConstraint()
        assert not isinstance(invalid_instance, Constraint)

    @pytest.mark.smoke
    def test_runtime_is_not_intializer(self):
        """
        Test that a class not implementing the ConstraintInitializer
        protocol is not recognized as such.
        """

        class ValidConstraint:
            def __call__(
                self,
                state: SchedulerState,
                request: ScheduledRequestInfo,
            ) -> SchedulerUpdateAction:
                return SchedulerUpdateAction()

        not_initializer_instance = ValidConstraint()
        assert not isinstance(not_initializer_instance, ConstraintInitializer)


class TestConstraintInitializer:
    """Test the ConstraintInitializer protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that ConstraintInitializer is a protocol and runtime checkable."""
        assert issubclass(ConstraintInitializer, Protocol)
        assert hasattr(ConstraintInitializer, "_is_protocol")
        assert ConstraintInitializer._is_protocol is True
        assert hasattr(ConstraintInitializer, "_is_runtime_protocol")
        assert ConstraintInitializer._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that ConstraintInitializer protocol has correct method signature."""
        create_constraint_method = ConstraintInitializer.create_constraint
        sig = inspect.signature(create_constraint_method)

        expected_params = ["self", "kwargs"]
        assert list(sig.parameters.keys()) == expected_params
        kwargs_param = sig.parameters["kwargs"]
        assert kwargs_param.kind == kwargs_param.VAR_KEYWORD

    @pytest.mark.smoke
    def test_runtime_is_initializer(self):
        """Test that ConstraintInitializer can be checked at runtime."""

        class ValidInitializer:
            def create_constraint(self, **kwargs) -> Constraint:
                class SimpleConstraint:
                    def __call__(
                        self,
                        state: SchedulerState,
                        request: ScheduledRequestInfo,
                    ) -> SchedulerUpdateAction:
                        return SchedulerUpdateAction()

                return SimpleConstraint()

        valid_instance = ValidInitializer()
        assert isinstance(valid_instance, ConstraintInitializer)

    @pytest.mark.smoke
    def test_runtime_is_not_constraint(self):
        """
        Test that a class not implementing the Constraint protocol
        is not recognized as such.
        """

        class ValidInitializer:
            def create_constraint(self, **kwargs) -> Constraint:
                class SimpleConstraint:
                    def __call__(
                        self,
                        state: SchedulerState,
                        request: ScheduledRequestInfo,
                    ) -> SchedulerUpdateAction:
                        return SchedulerUpdateAction()

                return SimpleConstraint()

        not_constraint_instance = ValidInitializer()
        assert not isinstance(not_constraint_instance, Constraint)


class TestSerializableConstraintInitializer:
    """Test the SerializableConstraintInitializer protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test SerializableConstraintInitializer is a protocol and checkable."""
        assert issubclass(SerializableConstraintInitializer, Protocol)
        assert hasattr(SerializableConstraintInitializer, "_is_protocol")
        assert SerializableConstraintInitializer._is_protocol is True
        assert hasattr(SerializableConstraintInitializer, "_is_runtime_protocol")
        assert SerializableConstraintInitializer._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signatures(self):
        """Test SerializableConstraintInitializer protocol has correct signatures."""
        methods = [
            "validated_kwargs",
            "model_validate",
            "model_dump",
            "create_constraint",
        ]

        for method_name in methods:
            assert hasattr(SerializableConstraintInitializer, method_name)

    @pytest.mark.smoke
    def test_runtime_is_serializable_initializer(self):
        """Test that SerializableConstraintInitializer can be checked at runtime."""

        class ValidSerializableInitializer:
            @classmethod
            def validated_kwargs(cls, *args, **kwargs):
                return kwargs

            @classmethod
            def model_validate(cls, **kwargs):
                return cls()

            def model_dump(self):
                return {}

            def create_constraint(self, **kwargs):
                class SimpleConstraint:
                    def __call__(self, state, request):
                        return SchedulerUpdateAction()

                return SimpleConstraint()

        valid_instance = ValidSerializableInitializer()
        assert isinstance(valid_instance, SerializableConstraintInitializer)


class TestPydanticConstraintInitializer:
    """Test the PydanticConstraintInitializer implementation."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test PydanticConstraintInitializer inheritance and abstract methods."""
        assert issubclass(PydanticConstraintInitializer, StandardBaseModel)
        assert issubclass(PydanticConstraintInitializer, ABC)
        assert issubclass(PydanticConstraintInitializer, InfoMixin)

    @pytest.mark.smoke
    def test_abstract_methods(self):
        """Test that PydanticConstraintInitializer has required abstract methods."""
        abstract_methods = PydanticConstraintInitializer.__abstractmethods__
        expected_methods = {"validated_kwargs", "create_constraint"}
        assert abstract_methods == expected_methods

    @pytest.mark.sanity
    def test_cannot_instantiate_directly(self):
        """Test that PydanticConstraintInitializer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PydanticConstraintInitializer(type_="test")


class TestUnserializableConstraintInitializer:
    """Test the UnserializableConstraintInitializer implementation."""

    @pytest.fixture(
        params=[
            {"orig_info": {}},
            {"orig_info": {"class": "SomeClass", "module": "some.module"}},
        ]
    )
    def valid_instances(self, request):
        """Fixture providing test data for UnserializableConstraintInitializer."""
        constructor_args = request.param
        instance = UnserializableConstraintInitializer(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test UnserializableConstraintInitializer inheritance."""
        assert issubclass(
            UnserializableConstraintInitializer, PydanticConstraintInitializer
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test UnserializableConstraintInitializer initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, UnserializableConstraintInitializer)
        assert instance.type_ == "unserializable"
        assert instance.orig_info == constructor_args["orig_info"]

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test validated_kwargs class method."""
        result = UnserializableConstraintInitializer.validated_kwargs(
            orig_info={"test": "data"}
        )
        assert result == {"orig_info": {"test": "data"}}

        result = UnserializableConstraintInitializer.validated_kwargs()
        assert result == {"orig_info": {}}

    @pytest.mark.sanity
    def test_create_constraint_raises(self, valid_instances):
        """Test that create_constraint raises RuntimeError."""
        instance, _ = valid_instances
        with pytest.raises(
            RuntimeError, match="Cannot create constraint from unserializable"
        ):
            instance.create_constraint()

    @pytest.mark.sanity
    def test_call_raises(self, valid_instances):
        """Test that calling constraint raises RuntimeError."""
        instance, _ = valid_instances
        state = SchedulerState()
        request = ScheduledRequestInfo()

        with pytest.raises(
            RuntimeError, match="Cannot invoke unserializable constraint"
        ):
            instance(state, request)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test UnserializableConstraintInitializer serialization/deserialization."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert data["type_"] == "unserializable"
        assert data["orig_info"] == constructor_args["orig_info"]

        reconstructed = UnserializableConstraintInitializer.model_validate(data)
        assert reconstructed.type_ == instance.type_
        assert reconstructed.orig_info == instance.orig_info


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
            request_info = ScheduledRequestInfo(
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
            request = ScheduledRequestInfo(
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
            assert isinstance(action.progress, dict)
            expected_remaining_fraction = max(0.0, 1.0 - elapsed / max_duration)
            expected_remaining_duration = max(0.0, max_duration - elapsed)
            assert action.progress["remaining_fraction"] == pytest.approx(
                expected_remaining_fraction, abs=0.1
            )
            assert action.progress["remaining_duration"] == pytest.approx(
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
            request = ScheduledRequestInfo(
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
            assert action.metadata == {
                "max_errors": constructor_args["max_errors"],
                "errors_exceeded": errors_exceeded,
                "current_errors": num_errors,
            }
            assert action.progress == {}

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
            request = ScheduledRequestInfo(
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
            assert action.progress == {}

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
            request = ScheduledRequestInfo(
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
            assert action.metadata == {
                "max_error_rate": max_error_rate,
                "min_processed": min_processed,
                "processed_requests": processed_requests,
                "errored_requests": total_errors,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            }

            # Error constraints don't provide progress information
            assert action.progress == {}

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


class TestConstraintsInitializerFactory:
    """Test the ConstraintsInitializerFactory implementation."""

    @pytest.mark.sanity
    def test_unregistered_key_fails(self):
        """Test that unregistered keys raise ValueError."""
        unregistered_key = "nonexistent_constraint"
        assert not ConstraintsInitializerFactory.is_registered(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create_constraint(unregistered_key)

    @pytest.mark.smoke
    def test_resolve_mixed_types(self):
        """Test resolve method with mixed constraint types."""
        max_num_constraint = MaxNumberConstraint(max_num=25)
        max_duration_initializer = MaxDurationConstraint(max_duration=120.0)

        mixed_spec = {
            "max_number": max_num_constraint,
            "max_duration": max_duration_initializer,
            "max_errors": {"max_errors": 15},
            "max_error_rate": 0.08,
        }

        resolved = ConstraintsInitializerFactory.resolve(mixed_spec)

        assert len(resolved) == 4
        assert all(isinstance(c, Constraint) for c in resolved.values())
        assert resolved["max_number"] is max_num_constraint
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)
        assert isinstance(resolved["max_error_rate"], MaxErrorRateConstraint)
        assert resolved["max_error_rate"].max_error_rate == 0.08

    @pytest.mark.sanity
    def test_resolve_with_invalid_key(self):
        """Test that resolve raises ValueError for unregistered keys."""
        invalid_spec = {
            "max_number": {"max_num": 100},
            "invalid_constraint": {"some_param": 42},
        }

        with pytest.raises(
            ValueError, match="Unknown constraint initializer key: invalid_constraint"
        ):
            ConstraintsInitializerFactory.resolve(invalid_spec)

    @pytest.mark.smoke
    def test_functional_constraint_creation(self):
        """Test that created constraints are functionally correct."""
        constraint = ConstraintsInitializerFactory.create_constraint(
            "max_number", max_num=10
        )
        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=5,
            processed_requests=5,
        )
        request = ScheduledRequestInfo(
            request_id="test-request",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"

        state_exceeded = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=15,
            processed_requests=15,
        )
        action_exceeded = constraint(state_exceeded, request)
        assert action_exceeded.request_queuing == "stop"
        assert action_exceeded.request_processing == "stop_local"
