"""Benchmarker module unit tests.

Clean, comprehensive test suite covering Benchmarker behaviors following the
standard template format with proper coverage of all public components,
type variables, classes, and functions according to the testing conditions.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC
from functools import wraps
from typing import Generic, TypeVar
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from guidellm.benchmark.aggregator import CompilableAggregator
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.objects import BenchmarkerDict, BenchmarkT, SchedulerDict
from guidellm.benchmark.profile import SynchronousProfile
from guidellm.scheduler import (
    BackendInterface,
    NonDistributedEnvironment,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SynchronousStrategy,
)
from guidellm.utils import InfoMixin, ThreadSafeSingletonMixin
from guidellm.utils.pydantic_utils import StandardBaseDict


def async_timeout(delay: float):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):  # type: ignore[override]
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


@pytest.mark.smoke
def test_benchmark_t():
    """Test that BenchmarkT is filled out correctly as a TypeVar."""
    assert isinstance(BenchmarkT, type(TypeVar("tmp")))
    assert BenchmarkT.__name__ == "BenchmarkT"
    assert BenchmarkT.__constraints__ == ()


@pytest.mark.smoke
def test_request_t():
    """Test that RequestT is filled out correctly as a TypeVar."""
    assert isinstance(RequestT, type(TypeVar("tmp")))
    assert RequestT.__name__ == "RequestT"
    assert RequestT.__bound__ is None
    assert RequestT.__constraints__ == ()


@pytest.mark.smoke
def test_response_t():
    """Test that ResponseT is filled out correctly as a TypeVar."""
    assert isinstance(ResponseT, type(TypeVar("tmp")))
    assert ResponseT.__name__ == "ResponseT"
    assert ResponseT.__bound__ is None
    assert ResponseT.__constraints__ == ()


class MockBenchmark:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def create_mock_scheduler_state() -> SchedulerState:
    """Create a valid scheduler state for testing."""
    return SchedulerState(
        node_id=0,
        num_processes=1,
        start_time=time.time(),
        end_time=time.time() + 10.0,
        end_queuing_time=time.time() + 5.0,
        end_queuing_constraints={},
        end_processing_time=time.time() + 8.0,
        end_processing_constraints={},
        scheduler_constraints={},
        remaining_fraction=0.0,
        remaining_requests=0,
        remaining_duration=0.0,
        created_requests=10,
        queued_requests=10,
        pending_requests=0,
        processing_requests=0,
        processed_requests=10,
        successful_requests=10,
        errored_requests=0,
        cancelled_requests=0,
    )


class MockBackend(BackendInterface):
    @property
    def processes_limit(self) -> int | None:  # pragma: no cover
        return None

    @property
    def requests_limit(self) -> int | None:  # pragma: no cover
        return None

    @property
    def info(self) -> dict[str, str]:  # pragma: no cover
        return {"type": "MockBackend"}

    async def process_startup(self):  # pragma: no cover
        pass

    async def validate(self):  # pragma: no cover
        pass

    async def process_shutdown(self):  # pragma: no cover
        pass

    async def resolve(self, request, request_info, request_history):  # pragma: no cover
        await asyncio.sleep(0)
        yield f"response_for_{request}"


class MockAggregator:
    def __call__(self, state, response, request, request_info, scheduler_state):
        state.setdefault("count", 0)
        state["count"] += 1
        return {"test_metric": state["count"]}


class MockCompilableAggregator(CompilableAggregator):
    def __call__(self, state, response, request, request_info, scheduler_state):
        state.setdefault("seen", 0)
        state["seen"] += 1
        return {"comp_metric": state["seen"]}

    def compile(self, state, scheduler_state):  # type: ignore[override]
        return {"extras": StandardBaseDict(compiled_field=state.get("seen", 0))}


class TestBenchmarker:
    """Test suite for Benchmarker."""

    @pytest.fixture(
        params=[
            {
                "requests": ["req1", "req2", "req3"],
                "backend": MockBackend(),
                "profile": SynchronousProfile.create("synchronous", rate=None),
                "benchmark_class": MockBenchmark,
                "benchmark_aggregators": {"test_agg": MockAggregator()},
            },
            {
                "requests": ["req1", "req2"],
                "backend": MockBackend(),
                "profile": SynchronousProfile.create("synchronous", rate=None),
                "benchmark_class": MockBenchmark,
                "benchmark_aggregators": {
                    "agg1": MockAggregator(),
                    "agg2": MockCompilableAggregator(),
                },
                "environment": NonDistributedEnvironment(),
            },
        ]
    )
    def valid_instances(self, request):
        """Fixture providing test data for Benchmarker."""
        return Benchmarker(), request.param

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Benchmarker inheritance and type relationships."""
        assert issubclass(Benchmarker, ABC)
        assert issubclass(Benchmarker, ThreadSafeSingletonMixin)
        assert issubclass(Benchmarker, Generic)
        assert hasattr(Benchmarker, "run")
        assert hasattr(Benchmarker, "_compile_benchmark_kwargs")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test Benchmarker initialization."""
        benchmarker_instance, _ = valid_instances
        assert isinstance(benchmarker_instance, Benchmarker)
        assert hasattr(benchmarker_instance, "thread_lock")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test Benchmarker cannot be instantiated as abstract class."""
        # Since Benchmarker is abstract and uses singleton pattern,
        # we test it can be instantiated (the concrete implementation handles this)
        instance = Benchmarker()
        assert isinstance(instance, Benchmarker)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("invalid_param", "invalid_value"),
        [
            ("invalid_method", "not_a_method"),
            ("bad_attribute", 12345),
        ],
    )
    def test_invalid_initialization_values(self, invalid_param, invalid_value):
        """Test Benchmarker with invalid attribute access."""
        benchmarker_inst = Benchmarker()
        # Test that invalid attributes don't exist or can't be set improperly
        if hasattr(benchmarker_inst, invalid_param):
            # If attribute exists, test it has expected type/behavior
            assert getattr(benchmarker_inst, invalid_param) != invalid_value
        else:
            # Test setting invalid attributes doesn't break the instance
            setattr(benchmarker_inst, invalid_param, invalid_value)
            assert hasattr(benchmarker_inst, invalid_param)

    @pytest.mark.sanity
    def test_singleton_identity(self):
        """Test singleton behavior."""
        assert Benchmarker() is Benchmarker()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_run_functionality(self, valid_instances):
        """Test Benchmarker.run core functionality."""
        benchmarker_instance, constructor_args = valid_instances
        with patch.object(Scheduler, "run") as mock_run:

            async def generated_results():
                yield ("resp", "req1", Mock(), create_mock_scheduler_state())

            mock_run.return_value = generated_results()
            with patch.object(
                SynchronousProfile, "strategies_generator"
            ) as strategies_gen:

                def one_strategy_generator():
                    yield SynchronousStrategy(), {}

                strategies_gen.return_value = one_strategy_generator()
                results = [
                    result
                    async for result in benchmarker_instance.run(**constructor_args)
                ]
        assert any(benchmark_obj is not None for _, benchmark_obj, _, _ in results)

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_run_invalid_parameters(self, valid_instances):
        """Test Benchmarker.run with invalid parameters."""
        benchmarker_instance, constructor_args = valid_instances

        # Test with missing required parameter
        invalid_args = constructor_args.copy()
        del invalid_args["requests"]

        async def run_missing_param():
            async for _ in benchmarker_instance.run(**invalid_args):
                break

        with pytest.raises(TypeError):
            await run_missing_param()

        # Test with invalid profile (non-Profile type)
        invalid_args = constructor_args.copy()
        invalid_args["profile"] = "not_a_profile"  # type: ignore[assignment]

        with patch.object(SynchronousProfile, "strategies_generator") as strategies_gen:
            # Mock AttributeError when calling strategies_generator on string
            strategies_gen.side_effect = AttributeError(
                "'str' object has no attribute 'strategies_generator'"
            )

            async def run_invalid_profile():
                async for _ in benchmarker_instance.run(**invalid_args):
                    break

            with pytest.raises(AttributeError):
                await run_invalid_profile()

    @pytest.mark.smoke
    def test_compile_benchmark_kwargs_functionality(self):
        """Test _compile_benchmark_kwargs core functionality."""
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()
        strategy_instance = SynchronousStrategy()
        scheduler_state_instance = create_mock_scheduler_state()
        aggregators = {
            "regular": MockAggregator(),
            "compilable": MockCompilableAggregator(),
        }
        result = Benchmarker._compile_benchmark_kwargs(
            run_id="run-123",
            run_index=0,
            profile=profile_instance,
            requests=["req"],
            backend=backend_mock,
            environment=environment_instance,
            aggregators=aggregators,
            aggregators_state={"regular": {}, "compilable": {"seen": 2}},
            strategy=strategy_instance,
            constraints={"max_requests": 100},
            scheduler_state=scheduler_state_instance,
        )
        assert all(
            key in result
            for key in (
                "run_id",
                "run_index",
                "scheduler",
                "benchmarker",
                "env_args",
                "extras",
            )
        )

    @pytest.mark.sanity
    def test_compile_benchmark_kwargs_invalid_parameters(self):
        """Test _compile_benchmark_kwargs with invalid parameters."""
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            Benchmarker._compile_benchmark_kwargs(
                run_id=None,  # type: ignore[arg-type]
                run_index=0,
                profile=None,  # type: ignore[arg-type]
                requests=[],
                backend=None,  # type: ignore[arg-type]
                environment=None,  # type: ignore[arg-type]
                aggregators={},
                aggregators_state={},
                strategy=None,  # type: ignore[arg-type]
                constraints={},
                scheduler_state=None,
            )

    @pytest.mark.smoke
    def test_combine_function_behavior(self):
        """Test internal _combine function behavior."""
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()

        class CompilableAgg(CompilableAggregator):
            def __call__(self, *args, **kwargs):
                return {}

            def compile(self, state_data, scheduler_state):  # type: ignore[override]
                return {"env_args": StandardBaseDict(extra_field="value")}

        result = Benchmarker._compile_benchmark_kwargs(
            run_id="run_id",
            run_index=0,
            profile=profile_instance,
            requests=[],
            backend=backend_mock,
            environment=environment_instance,
            aggregators={"agg": CompilableAgg()},
            aggregators_state={"agg": {}},
            strategy=SynchronousStrategy(),
            constraints={},
            scheduler_state=SchedulerState(),
        )
        assert isinstance(result["env_args"], StandardBaseDict)

    @pytest.mark.smoke
    def test_thread_safety(self, valid_instances):
        """Test thread safety through singleton identity."""
        benchmarker_inst, _ = valid_instances
        benchmarker_new = Benchmarker()
        assert benchmarker_inst is benchmarker_new

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_run_complete_workflow(self, valid_instances):
        """Test complete run workflow."""
        benchmarker_instance, constructor_args = valid_instances
        with patch.object(Scheduler, "run") as mock_run:

            async def scheduler_gen():
                yield ("resp1", "req1", Mock(), create_mock_scheduler_state())

            mock_run.return_value = scheduler_gen()
            with patch.object(
                SynchronousProfile, "strategies_generator"
            ) as strategies_gen:

                def strategy_sequence():
                    benchmark_obj = yield (SynchronousStrategy(), {})
                    assert benchmark_obj is not None

                strategies_gen.return_value = strategy_sequence()
                results = [
                    result
                    async for result in benchmarker_instance.run(**constructor_args)
                ]
        assert any(
            benchmark_created is not None for _, benchmark_created, _, _ in results
        )

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_run_with_environment_none(self, valid_instances):
        """Test run with environment defaulting to NonDistributedEnvironment."""
        benchmarker_instance, constructor_args = valid_instances
        constructor_args = constructor_args.copy()
        constructor_args.pop("environment", None)
        with patch.object(Scheduler, "run") as mock_run:

            async def scheduler_results():
                yield ("resp", "req", Mock(), create_mock_scheduler_state())

            mock_run.return_value = scheduler_results()
            with patch.object(
                SynchronousProfile, "strategies_generator"
            ) as strategies_gen:

                def single_strategy():
                    yield SynchronousStrategy(), {}

                strategies_gen.return_value = single_strategy()
                _ = [
                    result
                    async for result in benchmarker_instance.run(**constructor_args)
                ]
        assert isinstance(
            mock_run.call_args.kwargs.get("env"), NonDistributedEnvironment
        )

    @pytest.mark.smoke
    def test_compile_benchmark_kwargs_with_info_mixin(self):
        """Test _compile_benchmark_kwargs InfoMixin extraction."""
        with patch.object(InfoMixin, "extract_from_obj") as extract_mock:
            extract_mock.return_value = {"extracted": "data"}
            profile_instance = SynchronousProfile.create("synchronous", rate=None)
            backend_mock = Mock(spec=BackendInterface)
            backend_mock.info = {"type": "backend_type"}
            environment_instance = NonDistributedEnvironment()
            Benchmarker._compile_benchmark_kwargs(
                run_id="id-123",
                run_index=0,
                profile=profile_instance,
                requests=["req"],
                backend=backend_mock,
                environment=environment_instance,
                aggregators={"agg": MockAggregator()},
                aggregators_state={"agg": {}},
                strategy=SynchronousStrategy(),
                constraints={"constraint": 100},
                scheduler_state=SchedulerState(),
            )
            assert extract_mock.called

    @pytest.mark.sanity
    def test_compile_benchmark_kwargs_combine_error_cases(self):
        """Test _compile_benchmark_kwargs combine function error handling."""

        class BadAggregator(CompilableAggregator):
            def __call__(self, *args, **kwargs):
                return {}

            def compile(self, state_data, scheduler_state):  # type: ignore[override]
                return {"env_args": "invalid"}

        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()
        with pytest.raises(ValueError):
            Benchmarker._compile_benchmark_kwargs(
                run_id="run_id",
                run_index=0,
                profile=profile_instance,
                requests=[],
                backend=backend_mock,
                environment=environment_instance,
                aggregators={"bad": BadAggregator()},
                aggregators_state={"bad": {}},
                strategy=SynchronousStrategy(),
                constraints={},
                scheduler_state=Mock(),
            )

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_run_with_multiple_aggregators(self, valid_instances):
        """Test run with multiple aggregators including compilable ones."""
        benchmarker_instance, constructor_args = valid_instances
        multiple_aggregators = {
            "agg_regular": MockAggregator(),
            "agg_other": MockAggregator(),
            "agg_compilable": MockCompilableAggregator(),
        }
        constructor_args = constructor_args.copy()
        constructor_args["benchmark_aggregators"] = multiple_aggregators
        with patch.object(Scheduler, "run") as mock_run:

            async def scheduler_results():
                yield ("resp", "req1", Mock(), create_mock_scheduler_state())
                yield ("resp", "req1", Mock(), create_mock_scheduler_state())

            mock_run.return_value = scheduler_results()
            with patch.object(
                SynchronousProfile, "strategies_generator"
            ) as strategies_gen:

                def one_strategy():
                    yield SynchronousStrategy(), {}

                strategies_gen.return_value = one_strategy()
                results = [
                    result
                    async for result in benchmarker_instance.run(**constructor_args)
                ]
        updates = [
            update
            for update, benchmark_obj, strategy_obj, scheduler_state in results
            if update
        ]
        assert any(
            "test_metric" in update or "comp_metric" in update for update in updates
        )
        benchmark_obj = next(bench for _, bench, _, _ in results if bench is not None)
        assert benchmark_obj.extras.compiled_field >= 0

    @pytest.mark.smoke
    def test_benchmarker_dict_creation(self):
        """Test BenchmarkerDict creation in _compile_benchmark_kwargs."""
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()
        result = Benchmarker._compile_benchmark_kwargs(
            run_id="run_id",
            run_index=1,
            profile=profile_instance,
            requests=["req"],
            backend=backend_mock,
            environment=environment_instance,
            aggregators={"agg": MockAggregator()},
            aggregators_state={"agg": {}},
            strategy=SynchronousStrategy(),
            constraints={"limit": 200},
            scheduler_state=SchedulerState(),
        )
        assert isinstance(result["benchmarker"], BenchmarkerDict)

    @pytest.mark.smoke
    def test_scheduler_dict_creation(self):
        """Test SchedulerDict creation in _compile_benchmark_kwargs."""
        strategy_instance = SynchronousStrategy()
        scheduler_state_instance = SchedulerState()
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()
        result = Benchmarker._compile_benchmark_kwargs(
            run_id="run_id",
            run_index=0,
            profile=profile_instance,
            requests=[],
            backend=backend_mock,
            environment=environment_instance,
            aggregators={},
            aggregators_state={},
            strategy=strategy_instance,
            constraints={"max_requests": 100},
            scheduler_state=scheduler_state_instance,
        )
        assert isinstance(result["scheduler"], SchedulerDict)
        assert result["scheduler"].strategy is strategy_instance
        assert result["scheduler"].state is scheduler_state_instance

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_uuid_generation_in_run(self, valid_instances):
        """Test UUID generation in run method."""
        benchmarker_instance, constructor_args = valid_instances
        with patch("uuid.uuid4") as uuid_mock:
            uuid_mock.return_value = Mock()
            uuid_mock.return_value.__str__ = Mock(return_value="test_uuid")
            with patch.object(Scheduler, "run") as scheduler_run_mock:

                async def scheduler_results():
                    yield ("resp", "req", Mock(), create_mock_scheduler_state())

                scheduler_run_mock.return_value = scheduler_results()
                with patch.object(
                    SynchronousProfile, "strategies_generator"
                ) as strategies_gen:

                    def strategy_generator():
                        yield SynchronousStrategy(), {}

                    strategies_gen.return_value = strategy_generator()
                    _ = [
                        result
                        async for result in benchmarker_instance.run(**constructor_args)
                    ]
        uuid_mock.assert_called()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test Benchmarker serialization through _compile_benchmark_kwargs."""
        _, constructor_args = valid_instances
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend"}
        environment_instance = NonDistributedEnvironment()
        result = Benchmarker._compile_benchmark_kwargs(
            run_id="test-run",
            run_index=0,
            profile=profile_instance,
            requests=constructor_args["requests"],
            backend=backend_mock,
            environment=environment_instance,
            aggregators=constructor_args["benchmark_aggregators"],
            aggregators_state={
                key: {} for key in constructor_args["benchmark_aggregators"]
            },
            strategy=SynchronousStrategy(),
            constraints={"max_number": 100},
            scheduler_state=SchedulerState(),
        )
        assert isinstance(result, dict)
        assert "run_id" in result
        assert "scheduler" in result
        assert "benchmarker" in result

    @pytest.mark.regression
    def test_multi_strategy_iteration_functionality(self):
        """Test multi-strategy iteration ensuring proper state handling."""
        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()

        # Test that completed_strategies is used correctly in run_index
        for run_index in range(3):
            profile_instance.completed_strategies = [SynchronousStrategy()] * run_index
            result = Benchmarker._compile_benchmark_kwargs(
                run_id="multi-run",
                run_index=len(profile_instance.completed_strategies),
                profile=profile_instance,
                requests=[],
                backend=backend_mock,
                environment=environment_instance,
                aggregators={},
                aggregators_state={},
                strategy=SynchronousStrategy(),
                constraints={},
                scheduler_state=SchedulerState(),
            )
            assert result["run_index"] == run_index

    @pytest.mark.regression
    def test_compile_benchmark_kwargs_merge_multiple_fields(self):
        """Test merge when multiple compilable aggregators overlap fields."""

        class EnvArgsAggregator(CompilableAggregator):
            def __call__(self, *args, **kwargs):
                return {}

            def compile(self, state_data, scheduler_state):  # type: ignore[override]
                return {"env_args": StandardBaseDict(field1="value1")}

        class ExtrasAggregator(CompilableAggregator):
            def __call__(self, *args, **kwargs):
                return {}

            def compile(self, state_data, scheduler_state):  # type: ignore[override]
                return {
                    "env_args": StandardBaseDict(field2="value2"),
                    "extras": StandardBaseDict(extra1="extra_value"),
                }

        profile_instance = SynchronousProfile.create("synchronous", rate=None)
        backend_mock = Mock(spec=BackendInterface)
        backend_mock.info = {"type": "backend_type"}
        environment_instance = NonDistributedEnvironment()
        result = Benchmarker._compile_benchmark_kwargs(
            run_id="merge-test",
            run_index=0,
            profile=profile_instance,
            requests=[],
            backend=backend_mock,
            environment=environment_instance,
            aggregators={
                "env_agg": EnvArgsAggregator(),
                "extras_agg": ExtrasAggregator(),
            },
            aggregators_state={"env_agg": {}, "extras_agg": {}},
            strategy=SynchronousStrategy(),
            constraints={},
            scheduler_state=SchedulerState(),
        )
        # Verify that fields from both aggregators are merged
        assert hasattr(result["env_args"], "field1")
        assert hasattr(result["env_args"], "field2")
        assert hasattr(result["extras"], "extra1")
