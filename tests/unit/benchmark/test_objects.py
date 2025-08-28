"""
Unit tests for the guidellm benchmark objects module.

This module contains comprehensive tests for all public classes and functions
in the guidellm.benchmark.objects module following the established template.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import TypeVar
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from guidellm.backend import GenerationRequestTimings
from guidellm.benchmark.objects import (
    Benchmark,
    BenchmarkerDict,
    BenchmarkMetrics,
    BenchmarkMetricsT,
    BenchmarkRequestStats,
    BenchmarkRequestStatsT,
    BenchmarkSchedulerStats,
    BenchmarkT,
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
    GenerativeMetrics,
    GenerativeRequestStats,
    SchedulerDict,
)
from guidellm.benchmark.profile import SynchronousProfile
from guidellm.scheduler import (
    ScheduledRequestInfo,
    SchedulerState,
    SynchronousStrategy,
)
from guidellm.utils.pydantic_utils import (
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
)
from guidellm.utils.statistics import (
    DistributionSummary,
    Percentiles,
    StatusDistributionSummary,
)


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


def _dist(v: float = 1.0) -> DistributionSummary:
    return DistributionSummary(
        mean=v,
        median=v,
        mode=v,
        variance=0.0,
        std_dev=0.0,
        min=v,
        max=v,
        count=1,
        total_sum=v,
        percentiles=Percentiles(
            p001=v,
            p01=v,
            p05=v,
            p10=v,
            p25=v,
            p50=v,
            p75=v,
            p90=v,
            p95=v,
            p99=v,
            p999=v,
        ),
    )


def _status_dist() -> StatusDistributionSummary:
    return StatusDistributionSummary(
        successful=_dist(1),
        incomplete=_dist(2),
        errored=_dist(3),
        total=_dist(6),
    )


# Reusable baseline argument dictionaries / factories to cut duplication
BASE_SCHEDULER_STATS_ARGS = {
    "start_time": 1.0,
    "end_time": 2.0,
    "requests_made": StatusBreakdown(successful=1, incomplete=0, errored=0, total=1),
    "queued_time_avg": 0.1,
    "worker_resolve_start_delay_avg": 0.1,
    "worker_resolve_time_avg": 0.1,
    "worker_resolve_end_delay_avg": 0.1,
    "finalized_delay_avg": 0.1,
    "worker_targeted_start_delay_avg": 0.1,
    "request_start_delay_avg": 0.1,
    "request_time_avg": 0.1,
    "request_targeted_delay_avg": 0.1,
}


def _benchmark_base_args():
    return {
        "run_id": "r",
        "run_index": 0,
        "scheduler": SchedulerDict(
            strategy=SynchronousStrategy(), constraints={}, state=SchedulerState()
        ),
        "benchmarker": BenchmarkerDict(
            profile=SynchronousProfile.create("synchronous", rate=None),
            requests={},
            backend={},
            environment={},
            aggregators={},
        ),
        "env_args": StandardBaseDict(),
        "extras": StandardBaseDict(),
        "run_stats": BenchmarkSchedulerStats(**BASE_SCHEDULER_STATS_ARGS),
        "start_time": 0.0,
        "end_time": 1.0,
        "metrics": BenchmarkMetrics(
            requests_per_second=StatusDistributionSummary(),
            request_concurrency=StatusDistributionSummary(),
            request_latency=StatusDistributionSummary(),
        ),
        "request_totals": StatusBreakdown(
            successful=0, incomplete=0, errored=0, total=0
        ),
        "requests": StatusBreakdown(
            successful=[], incomplete=[], errored=[], total=None
        ),
    }


@pytest.mark.smoke
def test_benchmark_metrics_t():
    """Test that BenchmarkMetricsT is filled out correctly as a TypeVar."""
    assert isinstance(BenchmarkMetricsT, type(TypeVar("test")))
    assert BenchmarkMetricsT.__name__ == "BenchmarkMetricsT"
    assert BenchmarkMetricsT.__bound__ == BenchmarkMetrics
    assert BenchmarkMetricsT.__constraints__ == ()


@pytest.mark.smoke
def test_benchmark_request_stats_t():
    """Test that BenchmarkRequestStatsT is filled out correctly as a TypeVar."""
    assert isinstance(BenchmarkRequestStatsT, type(TypeVar("test")))
    assert BenchmarkRequestStatsT.__name__ == "BenchmarkRequestStatsT"
    assert BenchmarkRequestStatsT.__bound__ == BenchmarkRequestStats
    assert BenchmarkRequestStatsT.__constraints__ == ()


@pytest.mark.smoke
def test_benchmark_t():
    """Test that BenchmarkT is filled out correctly as a TypeVar."""
    assert isinstance(BenchmarkT, type(TypeVar("test")))
    assert BenchmarkT.__name__ == "BenchmarkT"
    assert BenchmarkT.__bound__ == Benchmark
    assert BenchmarkT.__constraints__ == ()


class TestBenchmarkSchedulerStats:
    """Test suite for BenchmarkSchedulerStats."""

    @pytest.fixture(
        params=[
            {
                "start_time": 1000.0,
                "end_time": 2000.0,
                "requests_made": StatusBreakdown(
                    successful=100, incomplete=5, errored=2, total=107
                ),
                "queued_time_avg": 0.5,
                "worker_resolve_start_delay_avg": 0.1,
                "worker_resolve_time_avg": 2.0,
                "worker_resolve_end_delay_avg": 0.05,
                "finalized_delay_avg": 0.02,
                "worker_targeted_start_delay_avg": 0.03,
                "request_start_delay_avg": 0.01,
                "request_time_avg": 1.5,
                "request_targeted_delay_avg": 0.04,
            },
            {
                "start_time": 5000.0,
                "end_time": 6000.0,
                "requests_made": StatusBreakdown(
                    successful=50, incomplete=0, errored=1, total=51
                ),
                "queued_time_avg": 0.2,
                "worker_resolve_start_delay_avg": 0.05,
                "worker_resolve_time_avg": 1.8,
                "worker_resolve_end_delay_avg": 0.03,
                "finalized_delay_avg": 0.01,
                "worker_targeted_start_delay_avg": 0.02,
                "request_start_delay_avg": 0.005,
                "request_time_avg": 1.2,
                "request_targeted_delay_avg": 0.025,
            },
        ],
        ids=["standard_stats", "minimal_errors"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for BenchmarkSchedulerStats."""
        constructor_args = request.param
        instance = BenchmarkSchedulerStats(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(BenchmarkSchedulerStats, StandardBaseDict)
        fields = set(BenchmarkSchedulerStats.model_fields.keys())
        expected = {
            "start_time",
            "end_time",
            "requests_made",
            "queued_time_avg",
            "worker_resolve_start_delay_avg",
            "worker_resolve_time_avg",
            "worker_resolve_end_delay_avg",
            "finalized_delay_avg",
            "worker_targeted_start_delay_avg",
            "request_start_delay_avg",
            "request_time_avg",
            "request_targeted_delay_avg",
        }
        assert expected.issubset(fields)
        assert BenchmarkSchedulerStats.model_fields[
            "queued_time_avg"
        ].description.startswith("Avg time")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        instance, data = valid_instances
        assert isinstance(instance, BenchmarkSchedulerStats)
        for k, v in data.items():
            assert getattr(instance, k) == v

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("start_time", "invalid"),
            ("end_time", None),
            ("requests_made", "not_breakdown"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        data = {
            "start_time": 1.0,
            "end_time": 2.0,
            "requests_made": StatusBreakdown(
                successful=1, incomplete=0, errored=0, total=1
            ),
            "queued_time_avg": 0.1,
            "worker_resolve_start_delay_avg": 0.1,
            "worker_resolve_time_avg": 0.1,
            "worker_resolve_end_delay_avg": 0.1,
            "finalized_delay_avg": 0.1,
            "worker_targeted_start_delay_avg": 0.1,
            "request_start_delay_avg": 0.1,
            "request_time_avg": 0.1,
            "request_targeted_delay_avg": 0.1,
        }
        data[field] = value
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            BenchmarkSchedulerStats(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            BenchmarkSchedulerStats()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        instance, data = valid_instances
        dumped = instance.model_dump()
        for k, v in data.items():
            if hasattr(v, "model_dump"):
                assert dumped[k] == v.model_dump()
            else:
                assert dumped[k] == v
        re = BenchmarkSchedulerStats.model_validate(dumped)
        assert re == instance


class TestSchedulerDict:
    """Test suite for SchedulerDict."""

    @pytest.fixture(
        params=[
            {
                "strategy": SynchronousStrategy(),
                "constraints": {"max_requests": {"value": 100}},
                "state": SchedulerState(node_id=0, num_processes=1),
            },
        ],
        ids=["basic_scheduler"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for SchedulerDict."""
        constructor_args = request.param
        instance = SchedulerDict(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(SchedulerDict, StandardBaseDict)
        assert {"strategy", "constraints", "state"}.issubset(
            SchedulerDict.model_fields.keys()
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        instance, data = valid_instances
        for k, v in data.items():
            assert getattr(instance, k) == v

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            SchedulerDict(strategy=1, constraints={}, state=SchedulerState())  # type: ignore
        with pytest.raises(ValidationError):
            SchedulerDict(
                strategy=SynchronousStrategy(), constraints=5, state=SchedulerState()
            )  # type: ignore

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            SchedulerDict()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        dumped = inst.model_dump()
        SchedulerDict.model_validate(dumped)


class TestBenchmarkerDict:
    """Test suite for BenchmarkerDict."""

    @pytest.fixture(
        params=[
            {
                "profile": SynchronousProfile.create("synchronous", rate=None),
                "requests": {"count": 100, "type": "text"},
                "backend": {"type": "openai", "model": "gpt-3.5"},
                "environment": {"nodes": 1, "processes": 4},
                "aggregators": {"stats": {"enabled": True}},
            },
        ],
        ids=["basic_benchmarker"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for BenchmarkerDict."""
        constructor_args = request.param
        instance = BenchmarkerDict(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(BenchmarkerDict, StandardBaseDict)
        assert set(BenchmarkerDict.model_fields.keys()) == {
            "profile",
            "requests",
            "backend",
            "environment",
            "aggregators",
        }

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        for k, v in data.items():
            assert getattr(inst, k) == v

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            BenchmarkerDict(
                profile=1, requests={}, backend={}, environment={}, aggregators={}
            )  # type: ignore
        with pytest.raises(ValidationError):
            BenchmarkerDict(
                profile=SynchronousProfile.create("synchronous", rate=None),
                requests=5,
                backend={},
                environment={},
                aggregators={},
            )  # type: ignore

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            BenchmarkerDict()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        BenchmarkerDict.model_validate(inst.model_dump())


class TestBenchmarkMetrics:
    """Test suite for BenchmarkMetrics."""

    @pytest.fixture(
        params=[
            {
                "requests_per_second": Mock(spec=StatusDistributionSummary),
                "request_concurrency": Mock(spec=StatusDistributionSummary),
                "request_latency": Mock(spec=StatusDistributionSummary),
            },
        ],
        ids=["basic_metrics"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for BenchmarkMetrics."""
        constructor_args = request.param
        instance = BenchmarkMetrics(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(BenchmarkMetrics, StandardBaseDict)
        assert set(BenchmarkMetrics.model_fields.keys()) == {
            "requests_per_second",
            "request_concurrency",
            "request_latency",
        }
        assert (
            "requests per second"
            in BenchmarkMetrics.model_fields["requests_per_second"].description
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        for k, v in data.items():
            assert getattr(inst, k) is v

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            BenchmarkMetrics(
                requests_per_second=1,
                request_concurrency=Mock(),
                request_latency=Mock(),
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            BenchmarkMetrics()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        BenchmarkMetrics.model_validate(inst.model_dump())


class TestBenchmarkRequestStats:
    """Test suite for BenchmarkRequestStats."""

    @pytest.fixture(
        params=[
            {
                "scheduler_info": ScheduledRequestInfo(),
            },
        ],
        ids=["basic_request_stats"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for BenchmarkRequestStats."""
        constructor_args = request.param
        instance = BenchmarkRequestStats(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(BenchmarkRequestStats, StandardBaseDict)
        assert "scheduler_info" in BenchmarkRequestStats.model_fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        assert inst.scheduler_info == data["scheduler_info"]

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            BenchmarkRequestStats(scheduler_info=1)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            BenchmarkRequestStats()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        BenchmarkRequestStats.model_validate(inst.model_dump())


class TestBenchmark:
    """Test suite for Benchmark."""

    @pytest.fixture(
        params=[
            {
                "run_id": "test-run-123",
                "run_index": 0,
                "scheduler": SchedulerDict(
                    strategy=SynchronousStrategy(),
                    constraints={},
                    state=SchedulerState(node_id=0, num_processes=1),
                ),
                "benchmarker": BenchmarkerDict(
                    profile=SynchronousProfile.create("synchronous", rate=None),
                    requests={},
                    backend={},
                    environment={},
                    aggregators={},
                ),
                "env_args": StandardBaseDict(),
                "extras": StandardBaseDict(),
                "run_stats": BenchmarkSchedulerStats(
                    start_time=1.0,
                    end_time=2.0,
                    requests_made=StatusBreakdown(
                        successful=1, incomplete=0, errored=0, total=1
                    ),
                    queued_time_avg=0.1,
                    worker_resolve_start_delay_avg=0.1,
                    worker_resolve_time_avg=0.1,
                    worker_resolve_end_delay_avg=0.1,
                    finalized_delay_avg=0.1,
                    worker_targeted_start_delay_avg=0.1,
                    request_start_delay_avg=0.1,
                    request_time_avg=0.1,
                    request_targeted_start_delay_avg=0.1,
                ),
                "start_time": 1000.0,
                "end_time": 2000.0,
                "metrics": BenchmarkMetrics(
                    requests_per_second=_status_dist(),
                    request_concurrency=_status_dist(),
                    request_latency=_status_dist(),
                ),
                "request_totals": StatusBreakdown(
                    successful=1, incomplete=0, errored=0, total=1
                ),
                "requests": StatusBreakdown(
                    successful=[
                        BenchmarkRequestStats(scheduler_info=ScheduledRequestInfo())
                    ],
                    incomplete=[],
                    errored=[],
                    total=None,
                ),
            },
        ],
        ids=["basic_benchmark"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for Benchmark."""
        constructor_args = request.param
        instance = Benchmark(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(Benchmark, StandardBaseDict)
        assert Benchmark.model_fields["type_"].default == "benchmark"
        assert "id_" in Benchmark.model_fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        for k, v in data.items():
            assert getattr(inst, k) == v
        assert isinstance(inst.id_, str)
        assert inst.id_

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            Benchmark(
                run_id=1,
                run_index=0,
                scheduler=SchedulerDict(
                    strategy=SynchronousStrategy(),
                    constraints={},
                    state=SchedulerState(),
                ),
                benchmarker=BenchmarkerDict(
                    profile=SynchronousProfile.create("synchronous", rate=None),
                    requests={},
                    backend={},
                    environment={},
                    aggregators={},
                ),
                env_args=StandardBaseDict(),
                extras=StandardBaseDict(),
                run_stats=BenchmarkSchedulerStats(
                    start_time=1,
                    end_time=2,
                    requests_made=StatusBreakdown(
                        successful=0, incomplete=0, errored=0, total=0
                    ),
                    queued_time_avg=0.1,
                    worker_resolve_start_delay_avg=0.1,
                    worker_resolve_time_avg=0.1,
                    worker_resolve_end_delay_avg=0.1,
                    finalized_delay_avg=0.1,
                    worker_targeted_start_delay_avg=0.1,
                    request_start_delay_avg=0.1,
                    request_time_avg=0.1,
                    request_targeted_delay_avg=0.1,
                ),
                start_time=0,
                end_time=1,
                metrics=BenchmarkMetrics(
                    requests_per_second=StatusDistributionSummary(),
                    request_concurrency=StatusDistributionSummary(),
                    request_latency=StatusDistributionSummary(),
                ),
                request_totals=StatusBreakdown(
                    successful=0, incomplete=0, errored=0, total=0
                ),
                requests=StatusBreakdown(
                    successful=[], incomplete=[], errored=[], total=None
                ),
            )  # type: ignore
        with pytest.raises(ValidationError):
            Benchmark(
                run_id="r",
                run_index="x",
                scheduler=SchedulerDict(
                    strategy=SynchronousStrategy(),
                    constraints={},
                    state=SchedulerState(),
                ),
                benchmarker=BenchmarkerDict(
                    profile=SynchronousProfile.create("synchronous", rate=None),
                    requests={},
                    backend={},
                    environment={},
                    aggregators={},
                ),
                env_args=StandardBaseDict(),
                extras=StandardBaseDict(),
                run_stats=BenchmarkSchedulerStats(
                    start_time=1,
                    end_time=2,
                    requests_made=StatusBreakdown(
                        successful=0, incomplete=0, errored=0, total=0
                    ),
                    queued_time_avg=0.1,
                    worker_resolve_start_delay_avg=0.1,
                    worker_resolve_time_avg=0.1,
                    worker_resolve_end_delay_avg=0.1,
                    finalized_delay_avg=0.1,
                    worker_targeted_start_delay_avg=0.1,
                    request_start_delay_avg=0.1,
                    request_time_avg=0.1,
                    request_targeted_start_delay_avg=0.1,
                ),
                start_time=0,
                end_time=1,
                metrics=BenchmarkMetrics(
                    requests_per_second=StatusDistributionSummary(),
                    request_concurrency=StatusDistributionSummary(),
                    request_latency=StatusDistributionSummary(),
                ),
                request_totals=StatusBreakdown(
                    successful=0, incomplete=0, errored=0, total=0
                ),
                requests=StatusBreakdown(
                    successful=[], incomplete=[], errored=[], total=None
                ),
            )  # type: ignore

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            Benchmark()

    @pytest.mark.smoke
    def test_duration_computed_field(self, valid_instances):
        inst, data = valid_instances
        assert inst.duration == data["end_time"] - data["start_time"]
        inst.start_time = 5
        inst.end_time = 3
        assert inst.duration == -2

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        dumped = inst.model_dump()
        assert "duration" in dumped
        Benchmark.model_validate(dumped)


class TestGenerativeRequestStats:
    """Test suite for GenerativeRequestStats."""

    @pytest.fixture(
        params=[
            {
                "scheduler_info": ScheduledRequestInfo(),
                "request_id": "test-request-123",
                "request_type": "text_completions",
                "prompt": "Test prompt",
                "request_args": {"max_tokens": 100},
                "output": "Test output",
                "iterations": 5,
                "prompt_tokens": 10,
                "output_tokens": 20,
            },
            {
                "scheduler_info": ScheduledRequestInfo(),
                "request_id": "test-request-456",
                "request_type": "chat_completions",
                "prompt": "Chat prompt",
                "request_args": {"temperature": 0.7},
                "output": None,
                "iterations": 0,
                "prompt_tokens": None,
                "output_tokens": None,
            },
        ],
        ids=["text_completion", "chat_completion_incomplete"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeRequestStats."""
        constructor_args = request.param

        # Mock the scheduler_info with request timings
        mock_timings = Mock(spec=GenerationRequestTimings)
        mock_timings.request_start = 1000.0
        mock_timings.request_end = 1005.0
        mock_timings.first_iteration = 1001.0
        mock_timings.last_iteration = 1004.0

        constructor_args["scheduler_info"].request_timings = mock_timings

        instance = GenerativeRequestStats(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(GenerativeRequestStats, BenchmarkRequestStats)
        assert (
            GenerativeRequestStats.model_fields["type_"].default
            == "generative_request_stats"
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        for k, v in data.items():
            assert getattr(inst, k) == v

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            GenerativeRequestStats(
                scheduler_info=ScheduledRequestInfo(),
                request_id="r",
                request_type="invalid_type",  # type: ignore
                prompt="p",
                request_args={},
                output="o",
                iterations=1,
                prompt_tokens=1,
                output_tokens=1,
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            GenerativeRequestStats()

    @pytest.mark.smoke
    def test_total_tokens_computed_field(self, valid_instances):
        inst, data = valid_instances
        if data["prompt_tokens"] is None:
            assert inst.total_tokens is None
        else:
            assert inst.total_tokens == data["prompt_tokens"] + data["output_tokens"]

    @pytest.mark.smoke
    def test_request_latency_computed_field(self, valid_instances):
        inst, _ = valid_instances
        assert inst.request_latency == 5.0
        inst.scheduler_info.request_timings.request_start = None
        assert inst.request_latency is None
        inst.scheduler_info.request_timings.request_start = 1000

    @pytest.mark.smoke
    def test_time_to_first_token_ms_computed_field(self, valid_instances):
        inst, _ = valid_instances
        assert inst.time_to_first_token_ms == 1000
        inst.scheduler_info.request_timings.first_iteration = None
        assert inst.time_to_first_token_ms is None
        inst.scheduler_info.request_timings.first_iteration = 1001

    @pytest.mark.smoke
    def test_time_per_output_token_ms_computed_field(self, valid_instances):
        inst, data = valid_instances
        if data["output_tokens"]:
            assert inst.time_per_output_token_ms == pytest.approx(
                1000 * (1004 - 1000) / data["output_tokens"]
            )  # ms per token
        inst.scheduler_info.request_timings.last_iteration = None
        assert inst.time_per_output_token_ms is None
        inst.scheduler_info.request_timings.last_iteration = 1004

    @pytest.mark.smoke
    def test_inter_token_latency_ms_computed_field(self, valid_instances):
        inst, data = valid_instances
        if data["output_tokens"] and data["output_tokens"] > 1:
            assert inst.inter_token_latency_ms == pytest.approx(
                1000 * (1004 - 1001) / (data["output_tokens"] - 1)
            )
        inst.scheduler_info.request_timings.first_iteration = None
        assert inst.inter_token_latency_ms is None
        inst.scheduler_info.request_timings.first_iteration = 1001

    @pytest.mark.smoke
    def test_tokens_per_second_computed_field(self, valid_instances):
        inst, data = valid_instances
        if data["prompt_tokens"] is None:
            assert inst.tokens_per_second is None
        else:
            assert inst.tokens_per_second == pytest.approx(
                (data["prompt_tokens"] + data["output_tokens"]) / 5.0
            )

    @pytest.mark.smoke
    def test_output_tokens_per_second_computed_field(self, valid_instances):
        inst, data = valid_instances
        if data["output_tokens"]:
            assert inst.output_tokens_per_second == pytest.approx(
                data["output_tokens"] / 5.0
            )
        else:
            assert inst.output_tokens_per_second is None

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        d = inst.model_dump()
        for f in [
            "total_tokens",
            "request_latency",
            "time_to_first_token_ms",
        ]:
            assert f in d
        GenerativeRequestStats.model_validate(d)


class TestGenerativeMetrics:
    """Test suite for GenerativeMetrics."""

    @pytest.fixture(
        params=[
            {
                "requests_per_second": Mock(spec=StatusDistributionSummary),
                "request_concurrency": Mock(spec=StatusDistributionSummary),
                "request_latency": Mock(spec=StatusDistributionSummary),
                "prompt_token_count": Mock(spec=StatusDistributionSummary),
                "output_token_count": Mock(spec=StatusDistributionSummary),
                "total_token_count": Mock(spec=StatusDistributionSummary),
                "time_to_first_token_ms": Mock(spec=StatusDistributionSummary),
                "time_per_output_token_ms": Mock(spec=StatusDistributionSummary),
                "inter_token_latency_ms": Mock(spec=StatusDistributionSummary),
                "output_tokens_per_second": Mock(spec=StatusDistributionSummary),
                "tokens_per_second": Mock(spec=StatusDistributionSummary),
            },
        ],
        ids=["complete_metrics"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeMetrics."""
        constructor_args = request.param
        instance = GenerativeMetrics(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(GenerativeMetrics, BenchmarkMetrics)
        for f in GenerativeMetrics.model_fields:
            assert (
                GenerativeMetrics.model_fields[f].annotation
                is StatusDistributionSummary
            )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        for k, v in data.items():
            assert getattr(inst, k) is v

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            GenerativeMetrics(
                requests_per_second=1,
                request_concurrency=Mock(),
                request_latency=Mock(),
                prompt_token_count=Mock(),
                output_token_count=Mock(),
                total_token_count=Mock(),
                time_to_first_token_ms=Mock(),
                time_per_output_token_ms=Mock(),
                inter_token_latency_ms=Mock(),
                output_tokens_per_second=Mock(),
                tokens_per_second=Mock(),
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            GenerativeMetrics()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        GenerativeMetrics.model_validate(inst.model_dump())


class TestGenerativeBenchmark:
    """Test suite for GenerativeBenchmark."""

    @pytest.fixture(
        params=[
            {
                "run_id": "test-run-gen",
                "run_index": 0,
                "scheduler": SchedulerDict(
                    strategy=SynchronousStrategy(),
                    constraints={},
                    state=SchedulerState(node_id=0, num_processes=1),
                ),
                "benchmarker": BenchmarkerDict(
                    profile=SynchronousProfile.create("synchronous", rate=None),
                    requests={},
                    backend={},
                    environment={},
                    aggregators={},
                ),
                "env_args": StandardBaseDict(),
                "extras": StandardBaseDict(),
                "run_stats": BenchmarkSchedulerStats(
                    start_time=1,
                    end_time=2,
                    requests_made=StatusBreakdown(
                        successful=1, incomplete=0, errored=0, total=1
                    ),
                    queued_time_avg=0.1,
                    worker_resolve_start_delay_avg=0.1,
                    worker_resolve_time_avg=0.1,
                    worker_resolve_end_delay_avg=0.1,
                    finalized_delay_avg=0.1,
                    worker_targeted_start_delay_avg=0.1,
                    request_start_delay_avg=0.1,
                    request_time_avg=0.1,
                    request_targeted_start_delay_avg=0.1,
                ),
                "start_time": 1000.0,
                "end_time": 2000.0,
                "metrics": GenerativeMetrics(
                    requests_per_second=_status_dist(),
                    request_concurrency=_status_dist(),
                    request_latency=_status_dist(),
                    prompt_token_count=_status_dist(),
                    output_token_count=_status_dist(),
                    total_token_count=_status_dist(),
                    time_to_first_token_ms=_status_dist(),
                    time_per_output_token_ms=_status_dist(),
                    inter_token_latency_ms=_status_dist(),
                    output_tokens_per_second=_status_dist(),
                    tokens_per_second=_status_dist(),
                ),
                "request_totals": StatusBreakdown(
                    successful=1, incomplete=0, errored=0, total=1
                ),
                "requests": StatusBreakdown(
                    successful=[
                        GenerativeRequestStats(
                            scheduler_info=ScheduledRequestInfo(
                                request_timings=GenerationRequestTimings(
                                    request_start=1,
                                    first_iteration=2,
                                    last_iteration=6,
                                    request_end=6,
                                )
                            ),
                            request_id="a",
                            request_type="text_completions",
                            prompt="p",
                            request_args={},
                            output="o",
                            iterations=1,
                            prompt_tokens=1,
                            output_tokens=2,
                        )
                    ],
                    incomplete=[],
                    errored=[],
                    total=None,
                ),
            },
        ],
        ids=["generative_benchmark"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeBenchmark."""
        constructor_args = request.param
        instance = GenerativeBenchmark(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(GenerativeBenchmark, Benchmark)
        assert (
            GenerativeBenchmark.model_fields["type_"].default == "generative_benchmark"
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        assert inst.metrics is data["metrics"]

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        with pytest.raises(ValidationError):
            GenerativeBenchmark()

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        d = inst.model_dump()
        assert d["type_"] == "generative_benchmark"
        GenerativeBenchmark.model_validate(d)


class TestGenerativeBenchmarksReport:
    """Test suite for GenerativeBenchmarksReport."""

    @pytest.fixture(
        params=[
            {"benchmarks": []},
            {
                "benchmarks": [
                    GenerativeBenchmark(
                        run_id="r1",
                        run_index=0,
                        scheduler=SchedulerDict(
                            strategy=SynchronousStrategy(),
                            constraints={},
                            state=SchedulerState(node_id=0, num_processes=1),
                        ),
                        benchmarker=BenchmarkerDict(
                            profile=SynchronousProfile.create("synchronous", rate=None),
                            requests={},
                            backend={},
                            environment={},
                            aggregators={},
                        ),
                        env_args=StandardBaseDict(),
                        extras=StandardBaseDict(),
                        run_stats=BenchmarkSchedulerStats(
                            start_time=1,
                            end_time=2,
                            requests_made=StatusBreakdown(
                                successful=1, incomplete=0, errored=0, total=1
                            ),
                            queued_time_avg=0.1,
                            worker_resolve_start_delay_avg=0.1,
                            worker_resolve_time_avg=0.1,
                            worker_resolve_end_delay_avg=0.1,
                            finalized_delay_avg=0.1,
                            worker_targeted_start_delay_avg=0.1,
                            request_start_delay_avg=0.1,
                            request_time_avg=0.1,
                            request_targeted_start_delay_avg=0.1,
                        ),
                        start_time=10,
                        end_time=20,
                        metrics=GenerativeMetrics(
                            requests_per_second=_status_dist(),
                            request_concurrency=_status_dist(),
                            request_latency=_status_dist(),
                            prompt_token_count=_status_dist(),
                            output_token_count=_status_dist(),
                            total_token_count=_status_dist(),
                            time_to_first_token_ms=_status_dist(),
                            time_per_output_token_ms=_status_dist(),
                            inter_token_latency_ms=_status_dist(),
                            output_tokens_per_second=_status_dist(),
                            tokens_per_second=_status_dist(),
                        ),
                        request_totals=StatusBreakdown(
                            successful=1, incomplete=0, errored=0, total=1
                        ),
                        requests=StatusBreakdown(
                            successful=[], incomplete=[], errored=[], total=None
                        ),
                    ),
                    GenerativeBenchmark(
                        run_id="r2",
                        run_index=1,
                        scheduler=SchedulerDict(
                            strategy=SynchronousStrategy(),
                            constraints={},
                            state=SchedulerState(node_id=0, num_processes=1),
                        ),
                        benchmarker=BenchmarkerDict(
                            profile=SynchronousProfile.create("synchronous", rate=None),
                            requests={},
                            backend={},
                            environment={},
                            aggregators={},
                        ),
                        env_args=StandardBaseDict(),
                        extras=StandardBaseDict(),
                        run_stats=BenchmarkSchedulerStats(
                            start_time=1,
                            end_time=3,
                            requests_made=StatusBreakdown(
                                successful=2, incomplete=0, errored=0, total=2
                            ),
                            queued_time_avg=0.1,
                            worker_resolve_start_delay_avg=0.1,
                            worker_resolve_time_avg=0.1,
                            worker_resolve_end_delay_avg=0.1,
                            finalized_delay_avg=0.1,
                            worker_targeted_start_delay_avg=0.1,
                            request_start_delay_avg=0.1,
                            request_time_avg=0.1,
                            request_targeted_start_delay_avg=0.1,
                        ),
                        start_time=30,
                        end_time=40,
                        metrics=GenerativeMetrics(
                            requests_per_second=_status_dist(),
                            request_concurrency=_status_dist(),
                            request_latency=_status_dist(),
                            prompt_token_count=_status_dist(),
                            output_token_count=_status_dist(),
                            total_token_count=_status_dist(),
                            time_to_first_token_ms=_status_dist(),
                            time_per_output_token_ms=_status_dist(),
                            inter_token_latency_ms=_status_dist(),
                            output_tokens_per_second=_status_dist(),
                            tokens_per_second=_status_dist(),
                        ),
                        request_totals=StatusBreakdown(
                            successful=2, incomplete=0, errored=0, total=2
                        ),
                        requests=StatusBreakdown(
                            successful=[], incomplete=[], errored=[], total=None
                        ),
                    ),
                ]
            },
        ],
        ids=["empty_report", "populated_report"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for GenerativeBenchmarksReport."""
        constructor_args = request.param
        instance = GenerativeBenchmarksReport(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(GenerativeBenchmarksReport, StandardBaseModel)
        assert GenerativeBenchmarksReport.DEFAULT_FILE == "benchmarks.json"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, data = valid_instances
        assert isinstance(inst.benchmarks, list)

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ValidationError):
            GenerativeBenchmarksReport(benchmarks=5)
        with pytest.raises(ValidationError):
            GenerativeBenchmarksReport(benchmarks=[1])

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        inst = GenerativeBenchmarksReport()
        assert inst.benchmarks == []

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("file_type", "expected_extension"),
        [
            ("json", ".json"),
            ("yaml", ".yaml"),
            (None, ".json"),  # auto-detect from filename
        ],
    )
    def test_save_file(self, valid_instances, tmp_path, file_type, expected_extension):
        inst, _ = valid_instances
        path = tmp_path / f"report.{file_type or 'json'}"
        saved = inst.save_file(path, file_type)
        assert saved.suffix == expected_extension
        assert saved.exists()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "file_type",
        ["json", "yaml"],
    )
    def test_load_file(self, valid_instances, tmp_path, file_type):
        inst, _ = valid_instances
        path = tmp_path / f"report.{file_type}"
        inst.save_file(path)
        loaded = GenerativeBenchmarksReport.load_file(path)
        assert isinstance(loaded, GenerativeBenchmarksReport)

    @pytest.mark.sanity
    def test_save_file_invalid_type(self, valid_instances, tmp_path):
        inst, _ = valid_instances
        with pytest.raises(ValueError):
            inst.save_file(tmp_path / "report.txt")

    @pytest.mark.sanity
    def test_load_file_invalid_type(self, tmp_path):
        p = tmp_path / "report.txt"
        p.write_text("{}")
        with pytest.raises(ValueError):
            GenerativeBenchmarksReport.load_file(p)

    @pytest.mark.smoke
    def test_default_file_behavior(self, valid_instances, tmp_path):
        inst, _ = valid_instances
        saved = inst.save_file(tmp_path, None)
        assert saved.name == GenerativeBenchmarksReport.DEFAULT_FILE
        loaded = GenerativeBenchmarksReport.load_file(tmp_path)
        assert isinstance(loaded, GenerativeBenchmarksReport)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        inst, _ = valid_instances
        GenerativeBenchmarksReport.model_validate(inst.model_dump())
