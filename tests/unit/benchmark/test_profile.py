"""
Unit tests for the guidellm benchmark profile module.

This module contains comprehensive tests for all public classes and functions
in the guidellm.benchmark.profile module following the established template.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    Profile,
    ProfileType,
    SweepProfile,
    SynchronousProfile,
    ThroughputProfile,
)
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.utils import PydanticClassRegistryMixin


def async_timeout(delay: float):
    """Decorator adding asyncio timeout for async tests."""

    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


@pytest.mark.smoke
def test_profile_type():
    """Test that ProfileType is defined correctly as a Literal type."""
    assert ProfileType is not None
    # Test that it can be used in type annotations (basic usage test)
    profile_type: ProfileType = "synchronous"
    assert profile_type == "synchronous"


class TestProfile:
    """Test suite for abstract Profile."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Profile inheritance and type relationships."""
        assert issubclass(Profile, PydanticClassRegistryMixin)
        assert Profile.schema_discriminator == "type_"

    @pytest.mark.smoke
    def test_pydantic_schema_base_type(self):
        """Test that the pydantic schema base type is Profile."""
        assert Profile.__pydantic_schema_base_type__() is Profile

    @pytest.mark.sanity
    def test_cannot_instantiate_directly(self):
        """Test that the abstract Profile class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Profile"):
            Profile(type_="profile")

    @pytest.mark.smoke
    @patch.object(Profile, "get_registered_object")
    def test_create_factory_method(self, mock_get_registered):
        """Test the create factory method for Profile."""
        mock_profile_class = MagicMock()
        mock_profile_class.resolve_args.return_value = {"type_": "test_profile"}
        mock_get_registered.return_value = mock_profile_class

        Profile.create("test_profile", rate=None)

        mock_get_registered.assert_called_once_with("test_profile")
        mock_profile_class.resolve_args.assert_called_once_with(
            rate_type="test_profile", rate=None, random_seed=42
        )
        mock_profile_class.assert_called_once_with(type_="test_profile")

    @pytest.mark.sanity
    @patch.object(Profile, "get_registered_object", return_value=None)
    def test_create_factory_method_unregistered(self, mock_get_registered):
        """Test create factory method with an unregistered type."""
        with pytest.raises(AttributeError):  # None has no resolve_args method
            Profile.create("unregistered", rate=None)

    @pytest.mark.smoke
    def test_strategies_generator(self):
        """Test the strategies_generator method."""
        mock_profile = MagicMock(spec=Profile)
        mock_profile.next_strategy.side_effect = [
            MagicMock(spec=SchedulingStrategy),
            None,
        ]
        mock_profile.next_strategy_constraints.return_value = {"max_requests": 10}
        mock_profile.completed_strategies = []

        generator = Profile.strategies_generator(mock_profile)
        strategy, constraints = next(generator)

        assert strategy is not None
        assert constraints == {"max_requests": 10}
        mock_profile.next_strategy.assert_called_once_with(None, None)
        mock_profile.next_strategy_constraints.assert_called_once()

        with pytest.raises(StopIteration):
            generator.send(MagicMock())  # Send a mock benchmark result back

    @pytest.mark.sanity
    def test_next_strategy_constraints(self):
        """Test the next_strategy_constraints method."""
        mock_profile = MagicMock(spec=Profile)
        mock_profile.constraints = {"max_duration": 10}
        with patch.object(
            ConstraintsInitializerFactory, "resolve", return_value={"max_duration": 10}
        ) as mock_resolve:
            constraints = Profile.next_strategy_constraints(
                mock_profile, MagicMock(), None, None
            )
            assert constraints == {"max_duration": 10}
            mock_resolve.assert_called_once_with({"max_duration": 10})

    @pytest.mark.smoke
    def test_constraints_validator(self):
        """Test the constraints validator."""
        assert Profile._constraints_validator(None) is None
        assert Profile._constraints_validator({"max_requests": 10}) == {
            "max_requests": 10
        }

        # Test invalid constraints type
        with pytest.raises(ValueError, match="Constraints must be a dictionary"):
            Profile._constraints_validator("invalid_type")

    @pytest.mark.smoke
    def test_constraints_serializer(self):
        """Test the constraints serializer through model serialization."""
        # Test with None constraints
        profile = SynchronousProfile()
        data = profile.model_dump()
        assert data.get("constraints") is None

        # Test with dict constraint (what actually gets stored after validation)
        regular_constraint = {"workers": 5, "max_requests": 100}
        profile_regular = SynchronousProfile(constraints=regular_constraint)
        data = profile_regular.model_dump()
        assert data["constraints"] == regular_constraint

        # Test with constraint dict format that would come from deserialize
        constraint_dict = {"type_": "max_number", "max_num": 100, "current_index": -1}
        profile_with_constraint_dict = SynchronousProfile(
            constraints={"max_requests": constraint_dict}
        )
        data = profile_with_constraint_dict.model_dump()
        expected = constraint_dict
        assert data["constraints"]["max_requests"] == expected

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(2.0)
    async def test_async_timeout_decorator(self):
        """Test the async_timeout decorator."""
        await asyncio.sleep(0.01)
        assert True


class TestSynchronousProfile:
    """Test suite for SynchronousProfile."""

    @pytest.fixture(
        params=[
            {},
            {"constraints": {"max_requests": 100}},
        ],
        ids=["basic", "with_constraints"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for SynchronousProfile."""
        constructor_args = request.param
        instance = SynchronousProfile(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SynchronousProfile inheritance and type relationships."""
        assert issubclass(SynchronousProfile, Profile)
        # Check type_ value through instance instead of class
        instance = SynchronousProfile()
        assert instance.type_ == "synchronous"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SynchronousProfile initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SynchronousProfile)
        assert instance.constraints == constructor_args.get("constraints")

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test SynchronousProfile serialization and deserialization."""
        instance, _ = valid_instances
        dumped = instance.model_dump()
        validated = Profile.model_validate(dumped)
        assert isinstance(validated, SynchronousProfile)
        assert validated.type_ == "synchronous"

    @pytest.mark.smoke
    def test_resolve_args(self):
        """Test the resolve_args class method."""
        args = SynchronousProfile.resolve_args("synchronous", None, 42)
        assert args == {}

        args_with_kwargs = SynchronousProfile.resolve_args(
            "synchronous", None, 42, constraints={"max_requests": 100}
        )
        assert args_with_kwargs == {"constraints": {"max_requests": 100}}

    @pytest.mark.sanity
    def test_resolve_args_invalid_rate(self):
        """Test resolve_args raises error when rate is provided."""
        with pytest.raises(ValueError, match="does not accept a rate parameter"):
            SynchronousProfile.resolve_args("synchronous", 10.0, 42)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test SynchronousProfile initialization with invalid constraints."""
        # Test invalid constraints type
        with pytest.raises(ValidationError):
            SynchronousProfile(constraints="invalid_type")

    @pytest.mark.sanity
    def test_strategy_types(self, valid_instances):
        """Test the strategy_types property."""
        instance, _ = valid_instances
        assert instance.strategy_types == ["synchronous"]

    @pytest.mark.smoke
    def test_next_strategy(self, valid_instances):
        """Test the next_strategy method."""
        instance, _ = valid_instances
        # First call should return a strategy
        strategy = instance.next_strategy(None, None)
        assert isinstance(strategy, SynchronousStrategy)

        # Simulate the strategy being completed by adding to completed_strategies
        instance.completed_strategies.append(strategy)

        # Second call should return None
        assert instance.next_strategy(strategy, None) is None

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that SynchronousProfile is registered with the Profile factory."""
        instance = Profile.create("synchronous", rate=None)
        assert isinstance(instance, SynchronousProfile)


class TestConcurrentProfile:
    """Test suite for ConcurrentProfile."""

    @pytest.fixture(
        params=[
            {"streams": 4},
            {"streams": 2, "startup_duration": 1.0},  # Single stream instead of list
            {"streams": 1, "startup_duration": 0.0},
        ],
        ids=["single_stream", "with_startup", "minimal_startup"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for ConcurrentProfile."""
        constructor_args = request.param
        instance = ConcurrentProfile(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ConcurrentProfile inheritance and type relationships."""
        assert issubclass(ConcurrentProfile, Profile)
        # Check type_ value through instance instead of class
        instance = ConcurrentProfile(streams=1)
        assert instance.type_ == "concurrent"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ConcurrentProfile initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ConcurrentProfile)
        assert instance.streams == constructor_args["streams"]
        assert instance.startup_duration == constructor_args.get(
            "startup_duration", 0.0
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("streams", 0),
            ("streams", -1),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test ConcurrentProfile with invalid field values."""
        data = {"streams": 1, field: value}
        with pytest.raises(ValidationError):
            ConcurrentProfile(**data)

    @pytest.mark.smoke
    def test_resolve_args(self):
        """Test the resolve_args class method."""
        args = ConcurrentProfile.resolve_args("concurrent", 4, 42, startup_duration=1.0)
        assert args == {
            "streams": 4,
            "startup_duration": 1.0,
        }

    @pytest.mark.sanity
    def test_resolve_args_invalid_rate(self):
        """Test resolve_args when rate is None."""
        # Rate (streams) can be None since it gets set as the streams value
        args = ConcurrentProfile.resolve_args("concurrent", None, 42)
        assert args == {"streams": None}

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test ConcurrentProfile initialization without required streams field."""
        with pytest.raises(ValidationError):
            ConcurrentProfile()

    @pytest.mark.smoke
    def test_strategy_types(self, valid_instances):
        """Test the strategy_types property."""
        instance, _ = valid_instances
        assert instance.strategy_types == ["concurrent"]

    @pytest.mark.smoke
    def test_next_strategy(self, valid_instances):
        """Test the next_strategy method."""
        instance, constructor_args = valid_instances
        streams = (
            constructor_args["streams"]
            if isinstance(constructor_args["streams"], list)
            else [constructor_args["streams"]]
        )
        prev_strategy = None
        for i, stream_count in enumerate(streams):
            strategy = instance.next_strategy(prev_strategy, None)
            assert isinstance(strategy, ConcurrentStrategy)
            assert strategy.streams == stream_count
            assert len(instance.completed_strategies) == i

            # Simulate the strategy being completed
            instance.completed_strategies.append(strategy)
            prev_strategy = strategy

        assert instance.next_strategy(prev_strategy, None) is None
        assert len(instance.completed_strategies) == len(streams)

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that ConcurrentProfile is registered with the Profile factory."""
        instance = Profile.create("concurrent", rate=4)
        assert isinstance(instance, ConcurrentProfile)
        assert instance.streams == 4


class TestThroughputProfile:
    """Test suite for ThroughputProfile."""

    @pytest.fixture(
        params=[
            {},
            {"max_concurrency": 10},
            {"startup_duration": 2.0},
            {"max_concurrency": 5, "startup_duration": 1.0},
        ],
        ids=["basic", "with_concurrency", "with_startup", "full_config"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for ThroughputProfile."""
        constructor_args = request.param
        instance = ThroughputProfile(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ThroughputProfile inheritance and type relationships."""
        assert issubclass(ThroughputProfile, Profile)
        # Check type_ value through instance instead of class
        instance = ThroughputProfile()
        assert instance.type_ == "throughput"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ThroughputProfile initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ThroughputProfile)
        assert instance.max_concurrency == constructor_args.get("max_concurrency")
        assert instance.startup_duration == constructor_args.get(
            "startup_duration", 0.0
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_concurrency", 0),
            ("max_concurrency", -1),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test ThroughputProfile with invalid field values."""
        data = {field: value}
        with pytest.raises(ValidationError):
            ThroughputProfile(**data)

    @pytest.mark.smoke
    def test_resolve_args(self):
        """Test the resolve_args class method."""
        args = ThroughputProfile.resolve_args(
            "throughput", None, 42, max_concurrency=10, startup_duration=1.0
        )
        assert args == {
            "max_concurrency": 10,
            "startup_duration": 1.0,
        }

        # Test with rate mapping to max_concurrency
        args_with_rate = ThroughputProfile.resolve_args(
            "throughput", 5, 42, startup_duration=2.0
        )
        assert args_with_rate == {
            "max_concurrency": 5,
            "startup_duration": 2.0,
        }

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test ThroughputProfile can be initialized with no required fields."""
        # ThroughputProfile has all optional fields
        instance = ThroughputProfile()
        assert isinstance(instance, ThroughputProfile)
        assert instance.max_concurrency is None
        assert instance.startup_duration == 0.0

    @pytest.mark.smoke
    def test_strategy_types(self, valid_instances):
        """Test the strategy_types property."""
        instance, _ = valid_instances
        assert instance.strategy_types == ["throughput"]

    @pytest.mark.smoke
    def test_next_strategy(self, valid_instances):
        """Test the next_strategy method."""
        instance, _ = valid_instances
        strategy = instance.next_strategy(None, None)
        assert isinstance(strategy, ThroughputStrategy)

        # Simulate the strategy being completed
        instance.completed_strategies.append(strategy)

        assert instance.next_strategy(strategy, None) is None

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that ThroughputProfile is registered with the Profile factory."""
        instance = Profile.create("throughput", rate=None)
        assert isinstance(instance, ThroughputProfile)


class TestAsyncProfile:
    """Test suite for AsyncProfile."""

    @pytest.fixture(
        params=[
            {"strategy_type": "constant", "rate": 5.0},
            {"strategy_type": "poisson", "rate": 2.0, "random_seed": 123},
            {
                "strategy_type": "constant",
                "rate": 10.0,
                "max_concurrency": 8,
                "startup_duration": 1.0,
            },
        ],
        ids=["constant_single", "poisson_single", "full_config"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for AsyncProfile."""
        constructor_args = request.param
        instance = AsyncProfile(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test AsyncProfile inheritance and type relationships."""
        assert issubclass(AsyncProfile, Profile)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test AsyncProfile initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, AsyncProfile)
        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
            ("max_concurrency", 0),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test AsyncProfile with invalid field values."""
        data = {"strategy_type": "constant", "rate": 1.0, field: value}
        with pytest.raises(ValidationError):
            AsyncProfile(**data)

    @pytest.mark.smoke
    def test_resolve_args(self):
        """Test the resolve_args class method."""
        args = AsyncProfile.resolve_args("constant", 10.0, 123, max_concurrency=8)
        assert args == {
            "type_": "constant",  # rate_type is used for type_ when it's "constant"
            "strategy_type": "constant",
            "rate": 10.0,
            "random_seed": 123,
            "max_concurrency": 8,
        }

    @pytest.mark.sanity
    def test_resolve_args_invalid_rate(self):
        """Test resolve_args raises error when rate is None."""
        with pytest.raises(ValueError, match="requires a rate parameter"):
            AsyncProfile.resolve_args("constant", None, 42)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test AsyncProfile initialization without required fields."""
        with pytest.raises(ValidationError):
            AsyncProfile()  # Missing strategy_type and rate

    @pytest.mark.sanity
    def test_strategy_types(self, valid_instances):
        """Test the strategy_types property."""
        instance, constructor_args = valid_instances
        assert instance.strategy_types == [constructor_args["strategy_type"]]

    @pytest.mark.smoke
    def test_next_strategy(self, valid_instances):
        """Test the next_strategy method."""
        instance, constructor_args = valid_instances
        rates = (
            constructor_args["rate"]
            if isinstance(constructor_args["rate"], list)
            else [constructor_args["rate"]]
        )
        strategy_class = (
            AsyncConstantStrategy
            if constructor_args["strategy_type"] == "constant"
            else AsyncPoissonStrategy
        )
        prev_strategy = None
        for i, rate in enumerate(rates):
            strategy = instance.next_strategy(prev_strategy, None)
            assert isinstance(strategy, strategy_class)
            assert strategy.rate == rate
            assert len(instance.completed_strategies) == i

            # Simulate the strategy being completed
            instance.completed_strategies.append(strategy)
            prev_strategy = strategy

        assert instance.next_strategy(prev_strategy, None) is None
        assert len(instance.completed_strategies) == len(rates)

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that AsyncProfile is registered with the Profile factory."""
        for alias in ["async", "constant", "poisson"]:
            instance = Profile.create(alias, rate=5.0)
            assert isinstance(instance, AsyncProfile)
            assert instance.rate == 5.0

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test AsyncProfile serialization and deserialization."""
        instance, _ = valid_instances
        dumped = instance.model_dump()
        validated = Profile.model_validate(dumped)
        assert isinstance(validated, AsyncProfile)
        assert validated.type_ == "async"


class TestSweepProfile:
    """Test suite for SweepProfile."""

    @pytest.fixture(
        params=[
            {"sweep_size": 5},
            {"sweep_size": 3, "strategy_type": "poisson", "random_seed": 123},
            {"sweep_size": 4, "max_concurrency": 10, "startup_duration": 2.0},
        ],
        ids=["basic", "poisson", "full_config"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for SweepProfile."""
        constructor_args = request.param
        instance = SweepProfile(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SweepProfile inheritance and type relationships."""
        assert issubclass(SweepProfile, Profile)
        # Check type_ value through instance instead of class
        instance = SweepProfile(sweep_size=3)
        assert instance.type_ == "sweep"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SweepProfile initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SweepProfile)
        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_concurrency", 0),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test SweepProfile with invalid field values."""
        data = {"sweep_size": 5, field: value}
        with pytest.raises(ValidationError):
            SweepProfile(**data)

    @pytest.mark.smoke
    def test_resolve_args(self):
        """Test the resolve_args class method."""
        args = SweepProfile.resolve_args(
            "sweep", 5, 42, strategy_type="poisson", max_concurrency=10
        )
        assert args == {
            "sweep_size": 5,
            "strategy_type": "poisson",
            "random_seed": 42,
            "max_concurrency": 10,
        }

        # Test rate used as default sweep_size
        args_default_sweep = SweepProfile.resolve_args("constant", 3, 123)
        assert args_default_sweep == {
            "sweep_size": 3,
            "strategy_type": "constant",
            "random_seed": 123,
        }

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test SweepProfile initialization without required sweep_size field."""
        with pytest.raises(ValidationError):
            SweepProfile()  # Missing sweep_size

    @pytest.mark.smoke
    def test_strategy_types(self, valid_instances):
        """Test the strategy_types property."""
        instance, constructor_args = valid_instances
        expected_type = constructor_args.get("strategy_type", "constant")
        # SweepProfile returns complex strategy types list
        expected_types = ["synchronous", "throughput"]
        sweep_size = constructor_args.get("sweep_size", 5)
        expected_types += [expected_type] * (sweep_size - 2)  # 2 for sync + throughput
        assert instance.strategy_types == expected_types

    @pytest.mark.sanity
    def test_next_strategy_basic_flow(self, valid_instances):
        """Test that next_strategy returns a SynchronousStrategy first."""
        instance, _ = valid_instances
        # First call should return SynchronousStrategy
        strategy = instance.next_strategy(None, None)
        assert isinstance(strategy, SynchronousStrategy)

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that SweepProfile is registered with the Profile factory."""
        instance = Profile.create("sweep", rate=5)
        assert isinstance(instance, SweepProfile)
        assert instance.sweep_size == 5

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test SweepProfile serialization and deserialization."""
        instance, _ = valid_instances
        dumped = instance.model_dump()
        validated = Profile.model_validate(dumped)
        assert isinstance(validated, SweepProfile)
        assert validated.type_ == "sweep"
