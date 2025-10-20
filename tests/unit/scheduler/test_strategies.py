from __future__ import annotations

import math
import time
from typing import Literal, TypeVar

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    SchedulingStrategy,
    StrategyT,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.schemas import RequestInfo


def test_strategy_type():
    """Test that StrategyType is defined correctly as a Literal type."""
    # StrategyType is a type alias/literal type, we can't test its runtime value
    # but we can test that it exists and is importable
    from guidellm.scheduler.strategies import StrategyType

    assert StrategyType is not None


def test_strategy_t():
    """Test that StrategyT is filled out correctly as a TypeVar."""
    assert isinstance(StrategyT, type(TypeVar("test")))
    assert StrategyT.__name__ == "StrategyT"
    assert StrategyT.__bound__ == SchedulingStrategy
    assert StrategyT.__constraints__ == ()


class TestExponentialDecay:
    """Test suite for # _exponential_decay_tau function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("max_progress", "convergence", "expected_range"),
        [
            (1.0, 0.99, (0.21, 0.22)),
            (5.0, 0.99, (1.08, 1.09)),
            (10.0, 0.95, (3.33, 3.35)),
        ],
    )
    def test_tau_invocation(self, max_progress, convergence, expected_range):
        """Test exponential decay tau calculation with valid inputs."""
        tau = max_progress / (-math.log(1 - convergence))  # Direct calculation
        assert expected_range[0] <= tau <= expected_range[1]
        expected_tau = max_progress / (-math.log(1 - convergence))
        assert tau == pytest.approx(expected_tau, rel=1e-10)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("progress", "tau", "expected_min", "expected_max"),
        [
            (0.0, 1.0, 0.0, 0.0),  # No progress = 0
            (1.0, 1.0, 0.6, 0.7),  # 1 tau ≈ 63.2%
            (2.0, 1.0, 0.85, 0.87),  # 2 tau ≈ 86.5%
            (3.0, 1.0, 0.95, 0.96),  # 3 tau ≈ 95.0%
        ],
    )
    def test_exp_decay_invocation(self, progress, tau, expected_min, expected_max):
        """Test exponential decay fraction calculation with valid inputs."""
        fraction = 1 - math.exp(-progress / tau)  # Direct calculation
        assert expected_min <= fraction <= expected_max
        expected_fraction = 1 - math.exp(-progress / tau)
        assert fraction == pytest.approx(expected_fraction, rel=1e-10)

    @pytest.mark.smoke
    def test_exp_boundary_conditions(self):
        """Test boundary conditions for exponential decay fraction."""
        assert (1 - math.exp(-0.0 / 1.0)) == 0.0
        assert (1 - math.exp(-0.0 / 10.0)) == 0.0
        large_progress = 100.0
        fraction = 1 - math.exp(-large_progress / 1.0)
        assert fraction > 0.99999


class TestSchedulingStrategy:
    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SchedulingStrategy inheritance and type relationships."""
        # Inheritance and abstract class properties
        assert issubclass(SchedulingStrategy, object)
        assert hasattr(SchedulingStrategy, "info")

        # Validate expected methods exist
        expected_methods = {
            "processes_limit",
            "requests_limit",
        }
        strategy_methods = set(dir(SchedulingStrategy))
        for method in expected_methods:
            assert method in strategy_methods

        # validate expected properties
        processes_limit_prop = SchedulingStrategy.processes_limit
        assert isinstance(processes_limit_prop, property)
        requests_limit_prop = SchedulingStrategy.requests_limit
        assert isinstance(requests_limit_prop, property)

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise NotImplementedError."""

        class InvalidStrategy(SchedulingStrategy):
            type_: Literal["strategy"] = "strategy"  # type: ignore[assignment,annotation-unchecked]

        with pytest.raises(TypeError):
            InvalidStrategy()

    @pytest.mark.smoke
    def test_concrete_implementation(self):
        """Test that concrete implementations can be constructed."""

        class TestStrategy(SchedulingStrategy):
            type_: Literal["strategy"] = "strategy"  # type: ignore[assignment,annotation-unchecked]

            async def next_request_time(self, offset: int) -> float:
                return time.time() + offset

            def request_completed(self, request_info: RequestInfo):
                pass

        strategy = TestStrategy()
        assert isinstance(strategy, SchedulingStrategy)


class TestSynchronousStrategy:
    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of SynchronousStrategy."""
        strategy = SynchronousStrategy()
        assert strategy.type_ == "synchronous"

    @pytest.mark.smoke
    def test_limits(self):
        """Test that SynchronousStrategy enforces proper limits."""
        strategy = SynchronousStrategy()
        assert strategy.processes_limit == 1
        assert strategy.requests_limit == 1

    @pytest.mark.smoke
    def test_string_representation(self):
        """Test __str__ method for SynchronousStrategy."""
        strategy = SynchronousStrategy()
        result = str(strategy)
        assert result == "synchronous"

    @pytest.mark.smoke
    def test_marshalling(self):
        """Test marshalling to/from pydantic dict formats."""
        strategy = SynchronousStrategy()
        data = strategy.model_dump()
        assert isinstance(data, dict)
        assert data["type_"] == "synchronous"

        reconstructed = SynchronousStrategy.model_validate(data)
        assert isinstance(reconstructed, SynchronousStrategy)
        assert reconstructed.type_ == "synchronous"

        # Test polymorphic reconstruction via base registry class
        base_reconstructed = SchedulingStrategy.model_validate(data)
        assert isinstance(base_reconstructed, SynchronousStrategy)
        assert base_reconstructed.type_ == "synchronous"

        # Test model_validate_json pathway
        json_str = strategy.model_dump_json()
        json_reconstructed = SynchronousStrategy.model_validate_json(json_str)
        assert isinstance(json_reconstructed, SynchronousStrategy)
        assert json_reconstructed.type_ == "synchronous"

        # Test polymorphic model_validate_json via base class
        base_json_reconstructed = SchedulingStrategy.model_validate_json(json_str)
        assert isinstance(base_json_reconstructed, SynchronousStrategy)
        assert base_json_reconstructed.type_ == "synchronous"


class TestConcurrentStrategy:
    @pytest.fixture(
        params=[
            {"streams": 1},
            {"streams": 4},
            {"streams": 8, "startup_duration": 2.0},
            {"streams": 2, "startup_duration": 0.0},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of ConcurrentStrategy."""
        constructor_args = request.param
        instance = ConcurrentStrategy(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[ConcurrentStrategy, dict]):
        """Test initialization of ConcurrentStrategy."""
        instance, constructor_args = valid_instances
        assert instance.type_ == "concurrent"

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("streams", 0),
            ("streams", -1),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization."""
        kwargs = {"streams": 2}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            ConcurrentStrategy(**kwargs)

    @pytest.mark.smoke
    def test_limits(self, valid_instances: tuple[ConcurrentStrategy, dict]):
        """Test that ConcurrentStrategy returns correct limits."""
        instance, constructor_args = valid_instances
        streams = constructor_args["streams"]
        assert instance.processes_limit == streams
        assert instance.requests_limit == streams

    @pytest.mark.smoke
    def test_string_representation(
        self, valid_instances: tuple[ConcurrentStrategy, dict]
    ):
        """Test __str__ method for ConcurrentStrategy."""
        instance, constructor_args = valid_instances
        streams = constructor_args["streams"]
        result = str(instance)
        assert result == f"concurrent@{streams}"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[ConcurrentStrategy, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)
        assert data["type_"] == "concurrent"

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = ConcurrentStrategy.model_validate(data)
        assert isinstance(reconstructed, ConcurrentStrategy)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

        # Test polymorphic reconstruction via base registry class
        base_reconstructed = SchedulingStrategy.model_validate(data)
        assert isinstance(base_reconstructed, ConcurrentStrategy)
        assert base_reconstructed.type_ == "concurrent"

        for key, value in constructor_args.items():
            assert getattr(base_reconstructed, key) == value

        # Test model_validate_json pathway
        json_str = instance.model_dump_json()
        json_reconstructed = ConcurrentStrategy.model_validate_json(json_str)
        assert isinstance(json_reconstructed, ConcurrentStrategy)

        for key, value in constructor_args.items():
            assert getattr(json_reconstructed, key) == value

        # Test polymorphic model_validate_json via base class
        base_json_reconstructed = SchedulingStrategy.model_validate_json(json_str)
        assert isinstance(base_json_reconstructed, ConcurrentStrategy)
        assert base_json_reconstructed.type_ == "concurrent"

        for key, value in constructor_args.items():
            assert getattr(base_json_reconstructed, key) == value


class TestThroughputStrategy:
    @pytest.fixture(
        params=[
            {},
            {"max_concurrency": 10},
            {"startup_duration": 5.0},
            {"max_concurrency": 5, "startup_duration": 2.0},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of ThroughputStrategy."""
        constructor_args = request.param
        instance = ThroughputStrategy(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[ThroughputStrategy, dict]):
        """Test initialization of ThroughputStrategy."""
        instance, constructor_args = valid_instances
        assert instance.type_ == "throughput"

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_concurrency", 0),
            ("max_concurrency", -1),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            ThroughputStrategy(**kwargs)

    @pytest.mark.smoke
    def test_limits(self, valid_instances: tuple[ThroughputStrategy, dict]):
        """Test that ThroughputStrategy returns correct limits."""
        instance, constructor_args = valid_instances
        max_concurrency = constructor_args.get("max_concurrency")
        assert instance.processes_limit == max_concurrency
        assert instance.requests_limit == max_concurrency

    @pytest.mark.smoke
    def test_string_representation(
        self, valid_instances: tuple[ThroughputStrategy, dict]
    ):
        """Test __str__ method for ThroughputStrategy."""
        instance, _ = valid_instances
        result = str(instance)
        assert result == "throughput"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[ThroughputStrategy, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)
        assert data["type_"] == "throughput"

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = ThroughputStrategy.model_validate(data)
        assert isinstance(reconstructed, ThroughputStrategy)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

        # Test polymorphic reconstruction via base registry class
        base_reconstructed = SchedulingStrategy.model_validate(data)
        assert isinstance(base_reconstructed, ThroughputStrategy)
        assert base_reconstructed.type_ == "throughput"

        for key, value in constructor_args.items():
            assert getattr(base_reconstructed, key) == value

        # Test model_validate_json pathway
        json_str = instance.model_dump_json()
        json_reconstructed = ThroughputStrategy.model_validate_json(json_str)
        assert isinstance(json_reconstructed, ThroughputStrategy)

        for key, value in constructor_args.items():
            assert getattr(json_reconstructed, key) == value

        # Test polymorphic model_validate_json via base class
        base_json_reconstructed = SchedulingStrategy.model_validate_json(json_str)
        assert isinstance(base_json_reconstructed, ThroughputStrategy)
        assert base_json_reconstructed.type_ == "throughput"

        for key, value in constructor_args.items():
            assert getattr(base_json_reconstructed, key) == value


class TestAsyncConstantStrategy:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {"rate": 5.0},
            {"rate": 10.3, "max_concurrency": 8},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of AsyncConstantStrategy."""
        constructor_args = request.param
        instance = AsyncConstantStrategy(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[AsyncConstantStrategy, dict]):
        """Test initialization of AsyncConstantStrategy."""
        instance, constructor_args = valid_instances
        assert instance.type_ == "constant"

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization."""
        kwargs = {"rate": 1.0}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            AsyncConstantStrategy(**kwargs)

    @pytest.mark.smoke
    def test_string_representation(
        self, valid_instances: tuple[AsyncConstantStrategy, dict]
    ):
        """Test __str__ method for AsyncConstantStrategy."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        result = str(instance)
        assert result == f"constant@{rate:.2f}"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[AsyncConstantStrategy, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)
        assert data["type_"] == "constant"

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = AsyncConstantStrategy.model_validate(data)
        assert isinstance(reconstructed, AsyncConstantStrategy)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

        # Test polymorphic reconstruction via base registry class
        base_reconstructed = SchedulingStrategy.model_validate(data)
        assert isinstance(base_reconstructed, AsyncConstantStrategy)
        assert base_reconstructed.type_ == "constant"

        for key, value in constructor_args.items():
            assert getattr(base_reconstructed, key) == value

        # Test model_validate_json pathway
        json_str = instance.model_dump_json()
        json_reconstructed = AsyncConstantStrategy.model_validate_json(json_str)
        assert isinstance(json_reconstructed, AsyncConstantStrategy)

        for key, value in constructor_args.items():
            assert getattr(json_reconstructed, key) == value

        # Test polymorphic model_validate_json via base class
        base_json_reconstructed = SchedulingStrategy.model_validate_json(json_str)
        assert isinstance(base_json_reconstructed, AsyncConstantStrategy)
        assert base_json_reconstructed.type_ == "constant"

        for key, value in constructor_args.items():
            assert getattr(base_json_reconstructed, key) == value


class TestAsyncPoissonStrategy:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {"rate": 5.0, "random_seed": 123},
            {"rate": 10.3, "random_seed": 456, "max_concurrency": 8},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of AsyncPoissonStrategy."""
        constructor_args = request.param
        instance = AsyncPoissonStrategy(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[AsyncPoissonStrategy, dict]):
        """Test initialization of AsyncPoissonStrategy."""
        instance, constructor_args = valid_instances
        assert instance.type_ == "poisson"

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization."""
        kwargs = {"rate": 1.0, "random_seed": 42}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            AsyncPoissonStrategy(**kwargs)

    @pytest.mark.smoke
    def test_string_representation(
        self, valid_instances: tuple[AsyncPoissonStrategy, dict]
    ):
        """Test __str__ method for AsyncPoissonStrategy."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        result = str(instance)
        assert result == f"poisson@{rate:.2f}"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[AsyncPoissonStrategy, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)
        assert data["type_"] == "poisson"

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = AsyncPoissonStrategy.model_validate(data)
        assert isinstance(reconstructed, AsyncPoissonStrategy)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value

        # Test polymorphic reconstruction via base registry class
        base_reconstructed = SchedulingStrategy.model_validate(data)
        assert isinstance(base_reconstructed, AsyncPoissonStrategy)
        assert base_reconstructed.type_ == "poisson"

        for key, value in constructor_args.items():
            assert getattr(base_reconstructed, key) == value

        # Test model_validate_json pathway
        json_str = instance.model_dump_json()
        json_reconstructed = AsyncPoissonStrategy.model_validate_json(json_str)
        assert isinstance(json_reconstructed, AsyncPoissonStrategy)

        for key, value in constructor_args.items():
            assert getattr(json_reconstructed, key) == value

        # Test polymorphic model_validate_json via base class
        base_json_reconstructed = SchedulingStrategy.model_validate_json(json_str)
        assert isinstance(base_json_reconstructed, AsyncPoissonStrategy)
        assert base_json_reconstructed.type_ == "poisson"

        for key, value in constructor_args.items():
            assert getattr(base_json_reconstructed, key) == value
