import inspect
from abc import ABC
from typing import Protocol

import pytest

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    PydanticConstraintInitializer,
    SchedulerState,
    SchedulerUpdateAction,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.utils.mixins import InfoMixin


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
                request: RequestInfo,
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
                request: RequestInfo,
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
                        request: RequestInfo,
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
                        request: RequestInfo,
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
        state = SchedulerState(node_id=0, num_processes=1, start_time=0.0)
        request = RequestInfo(
            request_id="test_request",
            status="pending",
            scheduler_node_id=0,
            scheduler_process_id=1,
            scheduler_start_time=0.0,
        )

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
