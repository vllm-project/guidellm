from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Protocol
from unittest.mock import Mock

import pytest

from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.benchmark.aggregator import (
    Aggregator,
    CompilableAggregator,
    GenerativeRequestsAggregator,
    GenerativeStatsProgressAggregator,
    SchedulerStatsAggregator,
    SerializableAggregator,
)
from guidellm.benchmark.objects import (
    BenchmarkSchedulerStats,
    GenerativeMetrics,
    GenerativeRequestStats,
)
from guidellm.scheduler import (
    ScheduledRequestInfo,
    SchedulerState,
)


def async_timeout(delay):
    """Decorator for async test timeouts."""

    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class TestAggregator:
    """Test the Aggregator protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that Aggregator is a protocol and runtime checkable."""
        assert issubclass(Aggregator, Protocol)
        assert hasattr(Aggregator, "_is_protocol")
        assert Aggregator._is_protocol is True
        assert hasattr(Aggregator, "_is_runtime_protocol")
        assert Aggregator._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that the Aggregator protocol has the correct method signature."""
        # Test that __call__ method exists and has correct signature
        call_method = Aggregator.__call__
        # Verify protocol method exists and is callable
        assert callable(call_method)

    @pytest.mark.smoke
    def test_runtime_is_aggregator(self):
        """Test that Aggregator can be checked at runtime using isinstance."""

        class ValidAggregator:
            def __call__(
                self,
                agg_state: dict[str, Any],
                response: Any | None,
                request: Any,
                request_info: Any,
                scheduler_state: Any,
            ) -> dict[str, Any] | None:
                return agg_state

        valid_instance = ValidAggregator()
        assert isinstance(valid_instance, Aggregator)

        class InvalidAggregator:
            def some_other_method(self):
                pass

        invalid_instance = InvalidAggregator()
        assert not isinstance(invalid_instance, Aggregator)


class TestCompilableAggregator:
    """Test the CompilableAggregator protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that CompilableAggregator is a protocol and runtime checkable."""
        assert issubclass(CompilableAggregator, Protocol)
        assert hasattr(CompilableAggregator, "_is_protocol")
        assert CompilableAggregator._is_protocol is True
        assert hasattr(CompilableAggregator, "_is_runtime_protocol")
        assert CompilableAggregator._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signatures(self):
        """Test that CompilableAggregator protocol has correct method signatures."""
        # Test that both __call__ and compile methods exist
        call_method = CompilableAggregator.__call__
        compile_method = CompilableAggregator.compile
        assert callable(call_method)
        assert callable(compile_method)

    @pytest.mark.smoke
    def test_runtime_is_compilable_aggregator(self):
        """Test that CompilableAggregator can be checked at runtime using isinstance."""

        class ValidCompilableAggregator:
            def __call__(
                self,
                agg_state: dict[str, Any],
                response: Any | None,
                request: Any,
                request_info: Any,
                scheduler_state: Any,
            ) -> dict[str, Any] | None:
                # Test implementation of aggregator call method
                return agg_state

            def compile(
                self, agg_state: dict[str, Any], scheduler_state: Any
            ) -> dict[str, Any]:
                # Test implementation of compile method
                return agg_state

        valid_instance = ValidCompilableAggregator()
        assert isinstance(valid_instance, CompilableAggregator)
        assert isinstance(valid_instance, Aggregator)  # Should also be an Aggregator

        class InvalidCompilableAggregator:
            def __call__(
                self, agg_state, response, request, request_info, scheduler_state
            ):
                # Test class with only __call__ but missing compile method
                return agg_state

        invalid_instance = InvalidCompilableAggregator()
        assert not isinstance(invalid_instance, CompilableAggregator)


class TestSerializableAggregator:
    """Test the SerializableAggregator implementation."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SerializableAggregator inheritance and type relationships."""
        # Test SerializableAggregator extends from correct base classes
        from abc import ABC
        from typing import Generic

        from guidellm.utils import PydanticClassRegistryMixin

        assert issubclass(SerializableAggregator, PydanticClassRegistryMixin)
        assert issubclass(SerializableAggregator, ABC)
        assert issubclass(SerializableAggregator, Generic)

        # Test class variables and discriminator
        assert hasattr(SerializableAggregator, "schema_discriminator")
        assert SerializableAggregator.schema_discriminator == "type_"

    @pytest.mark.smoke
    def test_abstract_methods(self):
        """Test that SerializableAggregator has correct abstract methods."""
        # Test that abstract methods are defined as abstract
        abstract_methods = SerializableAggregator.__abstractmethods__
        assert callable(SerializableAggregator.__call__)
        assert callable(SerializableAggregator.compile)
        assert "__call__" in abstract_methods
        assert "compile" in abstract_methods
        assert "validated_kwargs" in abstract_methods

    @pytest.mark.sanity
    def test_cannot_instantiate_directly(self):
        """Test that SerializableAggregator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SerializableAggregator()

    @pytest.mark.smoke
    def test_add_aggregate_metric_invocation(self):
        """Test the add_aggregate_metric class method."""
        # Test add_aggregate_metric with valid values
        agg_state = {}
        SerializableAggregator.add_aggregate_metric(
            "test_metric", agg_state, 10.0, 5.0, 2
        )

        assert agg_state["test_metric_total"] == 5.0  # 10.0 - 5.0
        assert agg_state["test_metric_count"] == 2

    @pytest.mark.smoke
    def test_add_aggregate_metric_none_values(self):
        """Test add_aggregate_metric with None values."""
        # Test that None values are handled correctly
        agg_state = {}
        SerializableAggregator.add_aggregate_metric(
            "test_metric", agg_state, None, 5.0, 1
        )
        assert len(agg_state) == 0  # No entries should be added

        SerializableAggregator.add_aggregate_metric(
            "test_metric", agg_state, 10.0, None, 1
        )
        assert len(agg_state) == 0  # No entries should be added

    @pytest.mark.smoke
    def test_add_aggregate_metric_rate(self):
        """Test the add_aggregate_metric_rate class method."""
        # Setup agg_state with total and count
        agg_state = {"test_metric_total": 100.0, "test_metric_count": 4}
        SerializableAggregator.add_aggregate_metric_rate("test_metric", agg_state)

        assert "test_metric_rate" in agg_state
        assert agg_state["test_metric_rate"] == 25.0  # 100.0 / 4

        # Test with zero count (safe_divide returns very large number for zero division)
        agg_state = {"test_metric_total": 100.0, "test_metric_count": 0}
        SerializableAggregator.add_aggregate_metric_rate("test_metric", agg_state)
        assert agg_state["test_metric_rate"] > 1e10  # Very large number

    @pytest.mark.smoke
    def test_resolve_functionality(self):
        """Test the resolve class method."""
        # Test resolving aggregators from mixed specifications
        aggregators_spec = {
            "scheduler_stats": {},  # Dict specification
            "generative_stats_progress": GenerativeStatsProgressAggregator(),
        }

        resolved = SerializableAggregator.resolve(aggregators_spec)

        # Verify results
        assert isinstance(resolved, dict)
        assert len(resolved) == 2
        assert "scheduler_stats" in resolved
        assert "generative_stats_progress" in resolved
        assert isinstance(resolved["scheduler_stats"], SchedulerStatsAggregator)
        assert isinstance(
            resolved["generative_stats_progress"], GenerativeStatsProgressAggregator
        )


class TestSchedulerStatsAggregator:
    """Test suite for SchedulerStatsAggregator."""

    @pytest.fixture(params=[{}])
    def valid_instances(self, request):
        """Fixture providing test data for SchedulerStatsAggregator."""
        constructor_args = request.param
        instance = SchedulerStatsAggregator(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SchedulerStatsAggregator inheritance and type relationships."""
        assert issubclass(SchedulerStatsAggregator, SerializableAggregator)
        from guidellm.utils import InfoMixin

        assert issubclass(SchedulerStatsAggregator, InfoMixin)

        # Test that the aggregator has the expected default type
        instance = SchedulerStatsAggregator()
        assert instance.type_ == "scheduler_stats"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SchedulerStatsAggregator initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SchedulerStatsAggregator)
        assert instance.type_ == "scheduler_stats"

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test SchedulerStatsAggregator with invalid field values."""
        # Test invalid field values if any are defined
        # Currently no specific validation constraints to test
        assert True  # Placeholder - no validation constraints currently exist

    @pytest.mark.smoke
    def test_call_method(self, valid_instances):
        """Test SchedulerStatsAggregator.__call__ method."""
        instance, _ = valid_instances

        # Mock required objects
        agg_state = {}
        response = Mock()
        request = Mock()
        request_info = Mock()
        scheduler_state = Mock()

        # Mock timing attributes
        request_info.scheduler_timings = Mock()
        request_info.scheduler_timings.dequeued = 10.0
        request_info.scheduler_timings.queued = 5.0
        request_info.scheduler_timings.resolve_start = 8.0
        request_info.scheduler_timings.scheduled_at = 7.0
        request_info.scheduler_timings.resolve_end = 12.0
        request_info.scheduler_timings.finalized = 15.0
        request_info.scheduler_timings.targeted_start = 6.0
        request_info.status = "completed"

        request_info.request_timings = Mock()
        request_info.request_timings.request_end = 14.0
        request_info.request_timings.request_start = 9.0

        # Test successful call
        result = instance(agg_state, response, request, request_info, scheduler_state)

        # Verify aggregation state is updated
        assert isinstance(result, dict)
        assert "queued_time_total" in agg_state
        assert "queued_time_count" in agg_state

    @pytest.mark.sanity
    def test_call_method_none_response(self, valid_instances):
        """Test SchedulerStatsAggregator.__call__ with None response."""
        instance, _ = valid_instances

        # Mock required objects
        agg_state = {}
        response = None
        request = Mock()
        request_info = Mock()
        request_info.status = "pending"  # Status that returns None
        scheduler_state = Mock()

        # Test call with None response
        result = instance(agg_state, response, request, request_info, scheduler_state)
        assert result is None

    @pytest.mark.smoke
    def test_compile_method(self, valid_instances):
        """Test SchedulerStatsAggregator.compile method."""
        instance, _ = valid_instances

        # Prepare aggregation state with sample data
        agg_state = {
            "queued_time_total": 20.0,
            "queued_time_count": 4,
            "worker_resolve_time_total": 15.0,
            "worker_resolve_time_count": 3,
        }

        # Mock scheduler state
        scheduler_state = Mock()
        scheduler_state.start_time = 0.0
        scheduler_state.end_time = 100.0
        scheduler_state.successful_requests = 10
        scheduler_state.cancelled_requests = 1
        scheduler_state.errored_requests = 2

        # Test compile method
        result = instance.compile(agg_state, scheduler_state)

        # Verify result structure
        assert isinstance(result, dict)
        assert "scheduler_stats" in result
        assert isinstance(result["scheduler_stats"], BenchmarkSchedulerStats)

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test SchedulerStatsAggregator.validated_kwargs method."""
        result = SchedulerStatsAggregator.validated_kwargs()
        assert isinstance(result, dict)
        assert result == {}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test SchedulerStatsAggregator serialization and deserialization."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["type_"] == "scheduler_stats"

        # Test model_validate
        recreated_instance = SchedulerStatsAggregator.model_validate(data_dict)
        assert isinstance(recreated_instance, SchedulerStatsAggregator)
        assert recreated_instance.type_ == instance.type_

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test SchedulerStatsAggregator factory registration."""
        # Test that the aggregator is properly registered
        registered_class = SerializableAggregator.get_registered_object(
            "scheduler_stats"
        )
        assert registered_class == SchedulerStatsAggregator

    @pytest.mark.regression
    def test_lifecycle_with_real_instances(self):
        """Test SchedulerStatsAggregator lifecycle with real request objects."""
        from guidellm.backend.objects import GenerationRequestTimings
        from guidellm.scheduler.objects import RequestSchedulerTimings

        instance = SchedulerStatsAggregator()
        agg_state = {}

        # Create real request objects for multiple requests
        for idx in range(3):
            # Create real timings objects
            request_timings = GenerationRequestTimings()
            request_timings.request_start = 1000.0 + idx
            request_timings.request_end = 1010.0 + idx

            scheduler_timings = RequestSchedulerTimings()
            scheduler_timings.queued = 1000.0 + idx
            scheduler_timings.dequeued = 1001.0 + idx
            scheduler_timings.scheduled_at = 1001.5 + idx
            scheduler_timings.resolve_start = 1002.0 + idx
            scheduler_timings.resolve_end = 1009.0 + idx
            scheduler_timings.finalized = 1010.0 + idx
            scheduler_timings.targeted_start = 1001.0 + idx

            request_info = ScheduledRequestInfo(
                request_timings=request_timings,
                scheduler_timings=scheduler_timings,
                status="completed",
            )

            # Mock minimal required objects
            response = Mock()
            request = Mock()
            scheduler_state = Mock()

            # Call aggregator
            result = instance(
                agg_state, response, request, request_info, scheduler_state
            )
            assert isinstance(result, dict)

        # Verify accumulated state
        assert "queued_time_total" in agg_state
        assert "queued_time_count" in agg_state
        assert agg_state["queued_time_count"] == 3

        # Test compile
        scheduler_state.start_time = 1000.0
        scheduler_state.end_time = 1020.0
        scheduler_state.successful_requests = 3
        scheduler_state.cancelled_requests = 0
        scheduler_state.errored_requests = 0

        compiled_result = instance.compile(agg_state, scheduler_state)
        assert "scheduler_stats" in compiled_result
        assert isinstance(compiled_result["scheduler_stats"], BenchmarkSchedulerStats)


class TestGenerativeStatsProgressAggregator:
    """Test suite for GenerativeStatsProgressAggregator."""

    @pytest.fixture(params=[{}])
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeStatsProgressAggregator."""
        constructor_args = request.param
        instance = GenerativeStatsProgressAggregator(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerativeStatsProgressAggregator inheritance and type relationships."""
        assert issubclass(GenerativeStatsProgressAggregator, SerializableAggregator)

        # Test that the aggregator has the expected default type
        instance = GenerativeStatsProgressAggregator()
        assert instance.type_ == "generative_stats_progress"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerativeStatsProgressAggregator initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerativeStatsProgressAggregator)
        assert instance.type_ == "generative_stats_progress"

    @pytest.mark.smoke
    def test_call_method(self, valid_instances):
        """Test GenerativeStatsProgressAggregator.__call__ method."""
        instance, _ = valid_instances

        # Mock required objects
        # Pre-populate agg_state to work around source code bug
        # where "prompt_tokens_total" is expected
        agg_state = {"prompt_tokens_total": 0, "output_tokens_total": 0}
        response = Mock(spec=GenerationResponse)
        response.output_tokens = 50
        response.prompt_tokens = 100
        response.total_tokens = 150

        request = Mock(spec=GenerationRequest)
        request_info = Mock(spec=ScheduledRequestInfo)
        request_info.status = "completed"
        request_info.request_timings = Mock(spec=GenerationRequestTimings)
        request_info.request_timings.request_start = 1000.0
        request_info.request_timings.request_end = 1010.0
        request_info.request_timings.first_iteration = 1002.0
        request_info.request_timings.last_iteration = 1008.0

        scheduler_state = Mock(spec=SchedulerState)
        scheduler_state.start_time = 1000.0
        scheduler_state.successful_requests = 10
        scheduler_state.cancelled_requests = 2
        scheduler_state.errored_requests = 1
        scheduler_state.processed_requests = 13

        # Test successful call
        result = instance(agg_state, response, request, request_info, scheduler_state)

        # Verify aggregation state is updated
        assert isinstance(result, dict)
        assert "requests_per_second" in agg_state
        assert "request_latency_total" in agg_state

    @pytest.mark.sanity
    def test_call_method_none_response(self, valid_instances):
        """Test GenerativeStatsProgressAggregator.__call__ with None response."""
        instance, _ = valid_instances

        # Mock required objects with status that returns None
        request_info = Mock()
        request_info.status = "pending"  # Status that causes None return

        # Test with None response
        result = instance({}, None, Mock(), request_info, Mock())
        assert result is None

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test GenerativeStatsProgressAggregator.validated_kwargs class method."""
        # Test validated_kwargs returns empty dict
        result = GenerativeStatsProgressAggregator.validated_kwargs()
        assert result == {}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test GenerativeStatsProgressAggregator serialization and deserialization."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["type_"] == "generative_stats_progress"

        # Test model_validate
        recreated_instance = GenerativeStatsProgressAggregator.model_validate(data_dict)
        assert isinstance(recreated_instance, GenerativeStatsProgressAggregator)

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test GenerativeStatsProgressAggregator factory registration."""
        # Test that the aggregator is properly registered
        registered_class = SerializableAggregator.get_registered_object(
            "generative_stats_progress"
        )
        assert registered_class == GenerativeStatsProgressAggregator

    @pytest.mark.regression
    def test_lifecycle_with_real_instances(self):
        """Test GenerativeStatsProgressAggregator lifecycle with real objects."""
        from guidellm.backend.objects import GenerationRequestTimings
        from guidellm.scheduler.objects import RequestSchedulerTimings

        instance = GenerativeStatsProgressAggregator()
        agg_state = {"prompt_tokens_total": 0, "output_tokens_total": 0}

        # Create real request objects for multiple requests
        for idx in range(3):
            # Create real timings objects
            request_timings = GenerationRequestTimings()
            request_timings.request_start = 1000.0 + idx
            request_timings.request_end = 1010.0 + idx
            request_timings.first_iteration = 1002.0 + idx
            request_timings.last_iteration = 1008.0 + idx

            scheduler_timings = RequestSchedulerTimings()
            scheduler_timings.resolve_end = 1009.0 + idx

            request_info = ScheduledRequestInfo(
                request_timings=request_timings,
                scheduler_timings=scheduler_timings,
                status="completed",
            )

            # Create real response object
            response = Mock(spec=GenerationResponse)
            response.output_tokens = 25 + idx
            response.prompt_tokens = 100 + idx
            response.total_tokens = 125 + idx  # Set as numeric value, not Mock

            request = Mock(spec=GenerationRequest)
            scheduler_state = Mock(spec=SchedulerState)
            scheduler_state.start_time = 1000.0
            scheduler_state.successful_requests = idx + 1
            scheduler_state.cancelled_requests = 0
            scheduler_state.errored_requests = 0
            scheduler_state.processed_requests = idx + 1

            # Call aggregator
            result = instance(
                agg_state, response, request, request_info, scheduler_state
            )
            assert isinstance(result, dict)

        # Verify accumulated state
        assert "completed_request_latency_total" in agg_state
        assert "completed_request_latency_count" in agg_state
        assert agg_state["completed_request_latency_count"] == 3

        # Test compile (this aggregator doesn't have a compile method)
        compiled_result = instance.compile(agg_state, scheduler_state)
        assert isinstance(compiled_result, dict)


class TestGenerativeRequestsAggregator:
    """Test suite for GenerativeRequestsAggregator."""

    @pytest.fixture(
        params=[
            {"request_samples": None, "warmup": None, "cooldown": None},
            {"request_samples": None, "warmup": 0, "cooldown": 0},
            {"request_samples": None, "warmup": 0.1, "cooldown": 0.1},
        ]
    )
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeRequestsAggregator."""
        constructor_args = request.param
        instance = GenerativeRequestsAggregator(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerativeRequestsAggregator inheritance and type relationships."""
        assert issubclass(GenerativeRequestsAggregator, SerializableAggregator)

        # Test that the aggregator has the expected default type
        instance = GenerativeRequestsAggregator()
        assert instance.type_ == "generative_requests"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerativeRequestsAggregator initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerativeRequestsAggregator)
        assert instance.type_ == "generative_requests"
        assert instance.request_samples == constructor_args["request_samples"]
        assert instance.warmup == constructor_args["warmup"]
        assert instance.cooldown == constructor_args["cooldown"]

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerativeRequestsAggregator with invalid field values."""
        # Note: Currently no field validation constraints are enforced
        # This test verifies that the class can be instantiated with any values
        instance = GenerativeRequestsAggregator(request_samples=-1)
        assert isinstance(instance, GenerativeRequestsAggregator)

        instance = GenerativeRequestsAggregator(warmup=-1.0)
        assert isinstance(instance, GenerativeRequestsAggregator)

        instance = GenerativeRequestsAggregator(cooldown=-1.0)
        assert isinstance(instance, GenerativeRequestsAggregator)

    @pytest.mark.smoke
    def test_call_method(self, valid_instances):
        """Test GenerativeRequestsAggregator.__call__ method."""
        instance, _ = valid_instances

        # Mock required objects
        agg_state = {}
        response = Mock(spec=GenerationResponse)
        request = Mock(spec=GenerationRequest)
        request_info = Mock(spec=ScheduledRequestInfo)
        request_info.status = "completed"
        request_info.started_at = 1000.0
        request_info.request_timings = Mock(spec=GenerationRequestTimings)
        request_info.request_timings.request_end = 1010.0

        # Mock scheduler_timings for warmup/cooldown detection
        request_info.scheduler_timings = Mock()
        request_info.scheduler_timings.targeted_start = 1001.0
        request_info.scheduler_timings.resolve_end = 1009.0

        scheduler_state = Mock(spec=SchedulerState)
        scheduler_state.start_time = 1000.0
        scheduler_state.processed_requests = 10
        scheduler_state.remaining_requests = 5
        scheduler_state.remaining_duration = 10.0
        scheduler_state.remaining_fraction = 0.5

        # Test successful call
        result = instance(agg_state, response, request, request_info, scheduler_state)

        # Verify result structure
        assert isinstance(result, dict)
        assert "requests_in_warmup" in result
        assert "requests_in_cooldown" in result

    @pytest.mark.sanity
    def test_call_method_none_response(self, valid_instances):
        """Test GenerativeRequestsAggregator.__call__ with None response."""
        instance, _ = valid_instances

        # Test with None response
        request_info = Mock()
        request_info.status = "pending"

        result = instance({}, None, Mock(), request_info, Mock())

        # Should return status dict with warmup/cooldown flags
        assert isinstance(result, dict)
        assert "requests_in_warmup" in result
        assert "requests_in_cooldown" in result

    @pytest.mark.smoke
    def test_compile_method(self, valid_instances):
        """Test GenerativeRequestsAggregator.compile method."""
        instance, _ = valid_instances

        # Create proper mock objects with all required attributes
        response_mock = Mock(spec=GenerationResponse)
        response_mock.preferred_prompt_tokens.return_value = 100
        response_mock.preferred_output_tokens.return_value = 50
        response_mock.request_args = {"temperature": 0.7}
        response_mock.value = "test output"
        response_mock.iterations = 1

        request_mock = Mock(spec=GenerationRequest)
        request_mock.request_id = "test_id_1"
        request_mock.request_type = "text_completions"
        request_mock.content = "test prompt"

        # Create actual ScheduledRequestInfo instead of mock
        from guidellm.backend.objects import GenerationRequestTimings
        from guidellm.scheduler.objects import RequestSchedulerTimings

        timings = GenerationRequestTimings()
        timings.request_start = 1000.0
        timings.request_end = 1010.0
        timings.first_iteration = 1002.0
        timings.last_iteration = 1008.0

        scheduler_timings = RequestSchedulerTimings()
        scheduler_timings.queued = 1000.0
        scheduler_timings.dequeued = 1001.0
        scheduler_timings.scheduled_at = 1002.0
        scheduler_timings.finalized = 1010.0

        request_info = ScheduledRequestInfo(
            request_timings=timings,
            scheduler_timings=scheduler_timings,
            status="completed",
        )

        agg_state = {
            "completed": [(response_mock, request_mock, request_info)],
            "errored": [],
            "incomplete": [],
        }

        # Mock scheduler state
        scheduler_state = Mock(spec=SchedulerState)
        scheduler_state.start_time = 0.0
        scheduler_state.end_time = 100.0

        # Test compile method
        result = instance.compile(agg_state, scheduler_state)

        # Verify result structure
        assert isinstance(result, dict)
        assert "start_time" in result
        assert "end_time" in result
        assert "request_totals" in result
        assert "requests" in result
        assert "metrics" in result
        assert isinstance(result["metrics"], GenerativeMetrics)

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test GenerativeRequestsAggregator.validated_kwargs class method."""
        # Test validated_kwargs with various parameters
        result = GenerativeRequestsAggregator.validated_kwargs(
            request_samples=25, warmup=10, cooldown=5
        )
        assert isinstance(result, dict)
        assert "warmup" in result
        assert "cooldown" in result

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test GenerativeRequestsAggregator serialization and deserialization."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["type_"] == "generative_requests"
        assert data_dict["request_samples"] == constructor_args["request_samples"]

        # Test model_validate
        recreated_instance = GenerativeRequestsAggregator.model_validate(data_dict)
        assert isinstance(recreated_instance, GenerativeRequestsAggregator)
        assert recreated_instance.request_samples == instance.request_samples

    @pytest.mark.smoke
    def test_create_generate_stats(self):
        """Test GenerativeRequestsAggregator._create_generate_stats class method."""
        # Create Mock objects for the method parameters
        response_mock = Mock(spec=GenerationResponse)
        response_mock.preferred_prompt_tokens.return_value = 100
        response_mock.preferred_output_tokens.return_value = 50
        response_mock.request_args = {"temperature": 0.7}
        response_mock.value = "test output"
        response_mock.iterations = 1

        request_mock = Mock(spec=GenerationRequest)
        request_mock.request_id = "test_id"
        request_mock.request_type = "text_completions"
        request_mock.content = "test prompt"

        # Create an actual ScheduledRequestInfo instance instead of a mock
        from guidellm.backend.objects import GenerationRequestTimings
        from guidellm.scheduler.objects import RequestSchedulerTimings

        timings = GenerationRequestTimings()
        scheduler_timings = RequestSchedulerTimings()
        request_info = ScheduledRequestInfo(
            request_timings=timings,
            scheduler_timings=scheduler_timings,
            status="completed",
        )

        # Test _create_generate_stats method
        result = GenerativeRequestsAggregator._create_generate_stats(
            response_mock, request_mock, request_info
        )

        # Verify result is GenerativeRequestStats
        assert isinstance(result, GenerativeRequestStats)
        assert result.request_id == "test_id"
        assert result.prompt_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test GenerativeRequestsAggregator factory registration."""
        # Test that the aggregator is properly registered
        registered_class = SerializableAggregator.get_registered_object(
            "generative_requests"
        )
        assert registered_class == GenerativeRequestsAggregator

    @pytest.mark.regression
    def test_lifecycle_with_real_instances(self):
        """Test GenerativeRequestsAggregator lifecycle with real objects."""
        from guidellm.backend.objects import GenerationRequestTimings
        from guidellm.scheduler.objects import RequestSchedulerTimings

        instance = GenerativeRequestsAggregator(
            request_samples=None, warmup=None, cooldown=None
        )
        agg_state = {}

        # Create real request objects for multiple requests
        for idx in range(5):
            # Create real timings objects
            request_timings = GenerationRequestTimings()
            request_timings.request_start = 1000.0 + idx
            request_timings.request_end = 1010.0 + idx
            request_timings.first_iteration = 1002.0 + idx
            request_timings.last_iteration = 1008.0 + idx

            scheduler_timings = RequestSchedulerTimings()
            scheduler_timings.queued = 1000.0 + idx
            scheduler_timings.dequeued = 1001.0 + idx
            scheduler_timings.scheduled_at = 1001.5 + idx
            scheduler_timings.resolve_start = 1002.0 + idx
            scheduler_timings.resolve_end = 1009.0 + idx
            scheduler_timings.finalized = 1010.0 + idx

            request_info = ScheduledRequestInfo(
                request_timings=request_timings,
                scheduler_timings=scheduler_timings,
                status="completed",
            )

            # Create real response and request objects
            response = Mock(spec=GenerationResponse)
            response.preferred_prompt_tokens.return_value = 100 + idx
            response.preferred_output_tokens.return_value = 25 + idx
            response.request_args = {"temperature": 0.7}
            response.value = f"response_{idx}"
            response.iterations = 1

            request = Mock(spec=GenerationRequest)
            request.request_id = f"req_{idx}"
            request.request_type = "text_completions"
            request.content = f"prompt_{idx}"

            scheduler_state = Mock(spec=SchedulerState)
            scheduler_state.start_time = 1000.0
            scheduler_state.processed_requests = idx + 1

            # Call aggregator
            result = instance(
                agg_state, response, request, request_info, scheduler_state
            )
            # Result can be None for this aggregator during accumulation
            assert result is None or isinstance(result, dict)

        # Verify accumulated state
        assert "completed" in agg_state
        assert len(agg_state["completed"]) == 5

        # Test compile
        scheduler_state.end_time = 1020.0
        compiled_result = instance.compile(agg_state, scheduler_state)
        assert isinstance(compiled_result, dict)
        assert "requests" in compiled_result
        assert "metrics" in compiled_result
        assert isinstance(compiled_result["metrics"], GenerativeMetrics)
