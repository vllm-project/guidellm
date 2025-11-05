"""Mock benchmark objects for unit testing."""

from guidellm.benchmark import (
    BenchmarkSchedulerStats,
    GenerativeBenchmark,
    GenerativeMetrics,
)
from guidellm.benchmark.profile import SynchronousProfile
from guidellm.benchmark.schemas import (
    BenchmarkerDict,
    GenerativeAudioMetricsSummary,
    GenerativeImageMetricsSummary,
    GenerativeMetricsSummary,
    GenerativeTextMetricsSummary,
    GenerativeVideoMetricsSummary,
    SchedulerDict,
)
from guidellm.scheduler import SchedulerState, SynchronousStrategy
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    UsageMetrics,
)
from guidellm.utils import (
    DistributionSummary,
    Percentiles,
    StandardBaseDict,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = ["mock_generative_benchmark"]


def _create_mock_percentiles() -> Percentiles:
    """Create mock percentiles for testing."""
    return Percentiles(
        p001=0.1,
        p01=1.0,
        p05=5.0,
        p10=10.0,
        p25=25.0,
        p50=50.0,
        p75=75.0,
        p90=90.0,
        p95=95.0,
        p99=99.0,
        p999=99.9,
    )


def _create_mock_distribution() -> DistributionSummary:
    """Create mock distribution summary for testing."""
    return DistributionSummary(
        mean=50.0,
        median=50.0,
        mode=50.0,
        variance=10.0,
        std_dev=3.16,
        min=10.0,
        max=100.0,
        count=100,
        total_sum=5000.0,
        percentiles=_create_mock_percentiles(),
    )


def _create_status_dist() -> StatusDistributionSummary:
    """Create mock status distribution summary for testing."""
    dist = _create_mock_distribution()
    return StatusDistributionSummary(
        successful=dist,
        incomplete=dist,
        errored=dist,
        total=dist,
    )


def _create_metrics_summary() -> GenerativeMetricsSummary:
    """Create mock generative metrics summary for testing."""
    return GenerativeMetricsSummary(
        input=_create_status_dist(),
        input_per_second=_create_status_dist(),
        input_concurrency=_create_status_dist(),
        output=_create_status_dist(),
        output_per_second=_create_status_dist(),
        output_concurrency=_create_status_dist(),
        total=_create_status_dist(),
        total_per_second=_create_status_dist(),
        total_concurrency=_create_status_dist(),
    )


def mock_generative_benchmark() -> GenerativeBenchmark:
    """Create a minimal mock GenerativeBenchmark for testing purposes."""
    return GenerativeBenchmark(
        run_id="test-run-gen",
        run_index=0,
        scheduler=SchedulerDict(
            strategy=SynchronousStrategy(),
            constraints={},
            state=SchedulerState(node_id=0, num_processes=1),
        ),
        benchmarker=BenchmarkerDict(
            profile=SynchronousProfile.create("synchronous", rate=None),
            requests={
                "data": "prompt_tokens=256,output_tokens=128",
            },
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
                successful=1,
                incomplete=0,
                errored=0,
                total=1,
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
        start_time=1000.0,
        end_time=2000.0,
        metrics=GenerativeMetrics(
            requests_per_second=_create_status_dist(),
            request_concurrency=_create_status_dist(),
            request_latency=_create_status_dist(),
            request_streaming_iterations_count=_create_status_dist(),
            prompt_token_count=_create_status_dist(),
            output_token_count=_create_status_dist(),
            total_token_count=_create_status_dist(),
            time_to_first_token_ms=_create_status_dist(),
            time_per_output_token_ms=_create_status_dist(),
            inter_token_latency_ms=_create_status_dist(),
            output_tokens_wo_first_per_iteration=_create_status_dist(),
            output_tokens_per_iteration=_create_status_dist(),
            output_tokens_per_second=_create_status_dist(),
            tokens_per_second=_create_status_dist(),
            text=GenerativeTextMetricsSummary(
                tokens=_create_metrics_summary(),
                characters=_create_metrics_summary(),
                words=_create_metrics_summary(),
            ),
            image=GenerativeImageMetricsSummary(
                tokens=_create_metrics_summary(),
                images=_create_metrics_summary(),
                pixels=_create_metrics_summary(),
                bytes=_create_metrics_summary(),
            ),
            video=GenerativeVideoMetricsSummary(
                tokens=_create_metrics_summary(),
                frames=_create_metrics_summary(),
                seconds=_create_metrics_summary(),
                bytes=_create_metrics_summary(),
            ),
            audio=GenerativeAudioMetricsSummary(
                tokens=_create_metrics_summary(),
                samples=_create_metrics_summary(),
                seconds=_create_metrics_summary(),
                bytes=_create_metrics_summary(),
            ),
        ),
        request_totals=StatusBreakdown(
            successful=1,
            incomplete=0,
            errored=0,
            total=1,
        ),
        requests=StatusBreakdown(
            successful=[
                GenerativeRequestStats(
                    scheduler_info=RequestInfo(
                        request_timings=RequestTimings(
                            request_start=1,
                            request_end=6,
                        )
                    ),
                    request_id="a",
                    request_type="text_completions",
                    prompt="p",
                    request_args="{}",
                    output="o",
                    iterations=1,
                    prompt_tokens=1,
                    output_tokens=2,
                    info=RequestInfo(
                        request_timings=RequestTimings(
                            request_start=1,
                            request_end=6,
                        )
                    ),
                    input_metrics=UsageMetrics(),
                    output_metrics=UsageMetrics(),
                )
            ],
            incomplete=[],
            errored=[],
            total=None,
        ),
    )
