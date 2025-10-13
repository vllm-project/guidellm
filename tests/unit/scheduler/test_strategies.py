from __future__ import annotations

import inspect
import math
import statistics
import time
from abc import ABC
from typing import Literal, TypeVar

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    ConstantRateRequestTimings,
    LastCompletionRequestTimings,
    NoDelayRequestTimings,
    PoissonRateRequestTimings,
    ScheduledRequestInfo,
    ScheduledRequestTimings,
    SchedulingStrategy,
    StrategyT,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.scheduler.strategies import (
    _exponential_decay_fraction,
    _exponential_decay_tau,
)


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
    """Test suite for _exponential_decay_tau function."""

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
        tau = _exponential_decay_tau(max_progress, convergence)
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
        fraction = _exponential_decay_fraction(progress, tau)
        assert expected_min <= fraction <= expected_max
        expected_fraction = 1 - math.exp(-progress / tau)
        assert fraction == pytest.approx(expected_fraction, rel=1e-10)

    @pytest.mark.smoke
    def test_exp_boundary_conditions(self):
        """Test boundary conditions for exponential decay fraction."""
        assert _exponential_decay_fraction(0.0, 1.0) == 0.0
        assert _exponential_decay_fraction(0.0, 10.0) == 0.0
        large_progress = 100.0
        fraction = _exponential_decay_fraction(large_progress, 1.0)
        assert fraction > 0.99999


class TestScheduledRequestTimings:
    @pytest.mark.smoke
    def test_signatures(self):
        """Test that ScheduledRequestTimings is an abstract base class."""
        assert issubclass(ScheduledRequestTimings, ABC)
        assert inspect.isabstract(ScheduledRequestTimings)

        abstract_methods = ScheduledRequestTimings.__abstractmethods__
        expected_methods = {"next_offset", "request_completed"}
        assert abstract_methods == expected_methods

        # Validate method signatures
        next_offset_method = ScheduledRequestTimings.next_offset
        assert callable(next_offset_method)
        request_completed_method = ScheduledRequestTimings.request_completed
        assert callable(request_completed_method)

        # Check signature parameters using inspect
        next_offset_sig = inspect.signature(next_offset_method)
        assert len(next_offset_sig.parameters) == 1
        assert str(next_offset_sig.return_annotation) == "float"
        request_completed_sig = inspect.signature(request_completed_method)
        assert len(request_completed_sig.parameters) == 2
        params = list(request_completed_sig.parameters.values())
        param_annotation = params[1].annotation
        assert param_annotation in {ScheduledRequestInfo, "ScheduledRequestInfo"}

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise TypeError."""

        class InvalidImplementation(ScheduledRequestTimings):
            pass  # Missing required abstract methods

        with pytest.raises(TypeError):
            InvalidImplementation()

    @pytest.mark.smoke
    def test_child_implementation(self):
        """Test that concrete implementations can be constructed."""

        class TestRequestTimings(ScheduledRequestTimings):
            offset: float = 0.0

            def next_offset(self) -> float:
                self.offset += 1.0
                return self.offset

            def request_completed(self, request_info: ScheduledRequestInfo):
                pass

        timing = TestRequestTimings()
        assert isinstance(timing, ScheduledRequestTimings)

        assert timing.next_offset() == 1.0
        assert timing.next_offset() == 2.0

        mock_request = ScheduledRequestInfo(
            request_id="test",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=time.time(),
        )
        timing.request_completed(mock_request)


class TestLastCompletionRequestTimings:
    @pytest.fixture(
        params=[
            {},
            {"offset": 10.0},
            {"startup_requests": 5, "startup_requests_delay": 0.5},
            {
                "offset": 0.0,
                "startup_requests": 0,
                "startup_requests_delay": 0.0,
            },
            {
                "offset": 2.5,
                "startup_requests": 3,
                "startup_requests_delay": 1.0,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of LastCompletionRequestTimings."""
        constructor_args = request.param
        instance = LastCompletionRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, LastCompletionRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("startup_requests", -1),
            ("startup_requests_delay", -0.5),
            ("offset", "invalid"),
            ("startup_requests", 1.5),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            LastCompletionRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_lifecycle(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test the complete lifecycle of next_offset and request_completed calls."""
        instance, constructor_args = valid_instances
        initial_offset = instance.offset
        startup_requests = constructor_args.get("startup_requests", 0)
        startup_delay = constructor_args.get("startup_requests_delay", 0.0)
        request_times = []

        for index in range(max(5, startup_requests + 2)):
            offset = instance.next_offset()
            assert isinstance(offset, int | float)

            if index < startup_requests:
                expected_offset = initial_offset + (index + 1) * startup_delay
                assert offset == pytest.approx(expected_offset, abs=1e-5)

            completion_time = time.time() + offset
            request_times.append(completion_time)

            mock_request: ScheduledRequestInfo = ScheduledRequestInfo(
                request_id=f"test-{index}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=time.time(),
            )
            mock_request.scheduler_timings.resolve_end = completion_time
            instance.request_completed(mock_request)

    @pytest.mark.smoke
    def test_marshalling(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = LastCompletionRequestTimings.model_validate(data)
        assert isinstance(reconstructed, LastCompletionRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestNoDelayRequestTimings:
    @pytest.fixture(
        params=[
            {},
            {"offset": 0.2},
            {"startup_duration": 0.3, "startup_target_requests": 5},
            {
                "offset": 0.15,
                "startup_duration": 0.2,
                "startup_target_requests": 20,
                "startup_convergence": 0.9,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of NoDelayRequestTimings."""
        constructor_args = request.param
        instance = NoDelayRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, NoDelayRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("offset", -1.0),
            ("startup_duration", -1.0),
            ("startup_target_requests", 0),
            ("startup_target_requests", -1),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            NoDelayRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_lifecycle(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test the complete lifecycle of timing methods."""
        instance, constructor_args = valid_instances
        startup_duration = constructor_args.get("startup_duration", 0.0)
        base_offset = constructor_args.get("offset", 0.0)
        start_time = time.time()
        min_time = base_offset + startup_duration + 0.2
        end_time = start_time + min_time
        last_offset = -1 * math.inf

        while (current_time := time.time()) < end_time:
            offset = instance.next_offset()

            if startup_duration > 0 and (current_time - start_time) <= startup_duration:
                assert offset < base_offset + startup_duration
                assert offset > last_offset
            elif startup_duration > 0:
                assert offset == base_offset + startup_duration
            else:
                assert offset == base_offset

            last_offset = offset
            time.sleep(0.025)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = NoDelayRequestTimings.model_validate(data)
        assert isinstance(reconstructed, NoDelayRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestConstantRateRequestTimings:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {"rate": 5.0, "offset": 2.0},
            {"rate": 10.5, "offset": 1.0},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of ConstantRateRequestTimings."""
        constructor_args = request.param
        instance = ConstantRateRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ConstantRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
            ("offset", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {"rate": 1.0}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            ConstantRateRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_constant_rate_behavior(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test that requests are scheduled at constant intervals."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        expected_interval = 1.0 / rate
        base_offset = constructor_args.get("offset", 0.0)
        num_requests = int(5 * rate)  # simulate 5 seconds

        for ind in range(num_requests):
            offset = instance.next_offset()
            assert offset >= base_offset
            assert offset == pytest.approx(
                base_offset + ind * expected_interval, rel=1e-2
            )

    @pytest.mark.smoke
    def test_marshalling(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = ConstantRateRequestTimings.model_validate(data)
        assert isinstance(reconstructed, ConstantRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestPoissonRateRequestTimings:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {
                "rate": 5.0,
                "random_seed": 123,
                "offset": 1.0,
            },
            {
                "rate": 0.5,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of PoissonRateRequestTimings."""
        constructor_args = request.param
        instance = PoissonRateRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[PoissonRateRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, PoissonRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
            ("offset", "invalid"),
            ("random_seed", "invalid"),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {"rate": 1.0}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            PoissonRateRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_lifecycle(self, valid_instances: tuple[PoissonRateRequestTimings, dict]):
        """Test that Poisson timing produces variable intervals."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        base_offset = constructor_args.get("offset", 0.0)
        num_requests = 200
        last_offset = 0.0
        intervals = []

        for index in range(num_requests):
            offset = instance.next_offset()

            if index == 0:
                assert offset == base_offset
            else:
                assert offset > last_offset

            intervals.append(offset - last_offset)
            last_offset = offset

        expected_mean_interval = 1.0 / rate
        actual_mean_interval = statistics.mean(intervals)
        tolerance = 0.2 * expected_mean_interval
        assert abs(actual_mean_interval - expected_mean_interval) < tolerance

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[PoissonRateRequestTimings, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = PoissonRateRequestTimings.model_validate(data)
        assert isinstance(reconstructed, PoissonRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


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
            "create_request_timings",
        }
        strategy_methods = set(dir(SchedulingStrategy))
        for method in expected_methods:
            assert method in strategy_methods

        # validate expected properties
        processes_limit_prop = SchedulingStrategy.processes_limit
        assert isinstance(processes_limit_prop, property)
        requests_limit_prop = SchedulingStrategy.requests_limit
        assert isinstance(requests_limit_prop, property)
        create_request_timings_method = SchedulingStrategy.create_request_timings
        assert callable(create_request_timings_method)

        # Validate method signature
        sig = inspect.signature(create_request_timings_method)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "local_rank",
            "local_world_size",
            "local_max_concurrency",
        ]
        assert params == expected_params

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise NotImplementedError."""

        class InvalidStrategy(SchedulingStrategy):
            type_: Literal["strategy"] = "strategy"  # type: ignore[assignment,annotation-unchecked]

        strategy = InvalidStrategy()
        with pytest.raises(NotImplementedError):
            strategy.create_request_timings(0, 1, 1)

    @pytest.mark.smoke
    def test_concrete_implementation(self):
        """Test that concrete implementations can be constructed."""

        class TestStrategy(SchedulingStrategy):
            type_: Literal["strategy"] = "strategy"  # type: ignore[assignment,annotation-unchecked]

            def create_request_timings(
                self,
                local_rank: int,
                local_world_size: int,
                local_max_concurrency: int,
            ):
                return LastCompletionRequestTimings()

        strategy = TestStrategy()
        assert isinstance(strategy, SchedulingStrategy)
        timing = strategy.create_request_timings(0, 1, 1)
        assert isinstance(timing, ScheduledRequestTimings)


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
    def test_create_timings_valid(self):
        """Test creating timings with valid parameters."""
        strategy = SynchronousStrategy()
        timing = strategy.create_request_timings(0, 1, 1)
        assert isinstance(timing, LastCompletionRequestTimings)

    @pytest.mark.sanity
    def test_create_timings_invalid(self):
        """Test that invalid parameters raise ValueError."""
        strategy = SynchronousStrategy()

        with pytest.raises(ValueError):
            strategy.create_request_timings(1, 1, 1)  # rank != 0

        with pytest.raises(ValueError):
            strategy.create_request_timings(0, 2, 1)  # world_size > 1

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
    def test_create_timings(self, valid_instances: tuple[ConcurrentStrategy, dict]):
        """Test creating timings."""
        instance, constructor_args = valid_instances
        streams = constructor_args["streams"]
        startup_duration = constructor_args.get("startup_duration", 0.0)

        # Test with different rank and world_size combinations
        for local_rank in range(min(streams, 2)):
            for local_world_size in range(1, min(streams + 1, 3)):
                if local_rank < local_world_size:
                    timing = instance.create_request_timings(
                        local_rank, local_world_size, streams
                    )
                    assert isinstance(timing, LastCompletionRequestTimings)

                    # Verify startup behavior
                    if startup_duration > 0:
                        # Check that timing has proper startup configuration
                        expected_delay_per_stream = startup_duration / streams
                        streams_per_worker = streams // local_world_size
                        expected_offset = (
                            local_rank * streams_per_worker * expected_delay_per_stream
                        )
                        assert timing.offset == pytest.approx(expected_offset, abs=1e-5)

    @pytest.mark.sanity
    def test_create_timings_invalid(
        self, valid_instances: tuple[ConcurrentStrategy, dict]
    ):
        """Test invalid inputs for create request timings."""
        instance, constructor_args = valid_instances
        streams = constructor_args["streams"]

        # Test various invalid configurations
        invalid_configs = [
            (streams, 1, 1),  # rank >= streams
            (0, streams + 1, 1),  # world_size > streams
        ]

        for local_rank, local_world_size, local_max_concurrency in invalid_configs:
            if local_rank >= streams or local_world_size > streams:
                with pytest.raises(ValueError):
                    instance.create_request_timings(
                        local_rank, local_world_size, local_max_concurrency
                    )

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
    def test_create_timings(self, valid_instances: tuple[ThroughputStrategy, dict]):
        """Test creating timings."""
        instance, constructor_args = valid_instances
        startup_duration = constructor_args.get("startup_duration", 0.0)

        # Test with different configurations
        for local_rank in range(3):
            for local_world_size in range(1, 4):
                for local_max_concurrency in range(1, 6):
                    timing = instance.create_request_timings(
                        local_rank, local_world_size, local_max_concurrency
                    )
                    assert isinstance(timing, NoDelayRequestTimings)

                    # Verify startup configuration
                    if startup_duration > 0:
                        assert timing.startup_duration == startup_duration
                        assert timing.startup_target_requests == local_max_concurrency
                        expected_offset = (
                            0.05 * startup_duration * (local_rank / local_world_size)
                        )
                        assert timing.offset == pytest.approx(expected_offset, abs=1e-5)
                    else:
                        assert timing.startup_duration == 0.0
                        assert timing.offset == 0.0

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
    def test_create_timings(self, valid_instances: tuple[AsyncConstantStrategy, dict]):
        """Test creating timings."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]

        # Test with different worker configurations
        for local_world_size in range(1, 5):
            timing = instance.create_request_timings(0, local_world_size, 1)
            assert isinstance(timing, ConstantRateRequestTimings)

            # Rate should be distributed across workers
            expected_worker_rate = rate / local_world_size
            assert timing.rate == pytest.approx(expected_worker_rate, abs=1e-5)

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
    def test_create_timings(self, valid_instances: tuple[AsyncPoissonStrategy, dict]):
        """Test creating timings."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        base_seed = constructor_args.get("random_seed", 42)

        # Test with different worker configurations
        for local_rank in range(3):
            for local_world_size in range(1, 4):
                timing = instance.create_request_timings(
                    local_rank, local_world_size, 1
                )
                assert isinstance(timing, PoissonRateRequestTimings)

                # Rate should be distributed across workers
                expected_worker_rate = rate / local_world_size
                assert timing.rate == pytest.approx(expected_worker_rate, abs=1e-5)

                # Each worker should have a unique seed
                expected_seed = base_seed + local_rank
                assert timing.random_seed == expected_seed

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
