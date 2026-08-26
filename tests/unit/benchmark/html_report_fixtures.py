"""
Shared builders for HTML report unit tests and the Node/jsdom fixture generator.

## WRITTEN BY AI ##
"""

from __future__ import annotations

from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeAudioMetricsSummary,
    GenerativeBenchmark,
    GenerativeBenchmarkMetadata,
    GenerativeBenchmarksReport,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeTextMetricsSummary,
    GenerativeVideoMetricsSummary,
    SchedulerMetrics,
)
from guidellm.benchmark.schemas.metrics import GenerativeToolCallMetricsSummary
from guidellm.scheduler import (
    AsyncConstantStrategy,
    ConcurrentStrategy,
    SchedulerState,
    ThroughputStrategy,
)
from guidellm.schemas import (
    DistributionSummary,
    GenerativeRequestStats,
    Percentiles,
    RequestInfo,
    RequestTimings,
    StatusBreakdown,
    StatusDistributionSummary,
    UsageMetrics,
)
from guidellm.schemas.benchmark import BenchmarkScenario


def distribution(
    mean: float,
    *,
    median: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    count: int = 10,
) -> DistributionSummary:
    """
    Build a populated :class:`DistributionSummary` for HTML report fixtures.

    ## WRITTEN BY AI ##
    """
    median_value = mean if median is None else median
    p95_value = median_value if p95 is None else p95
    p99_value = p95_value if p99 is None else p99
    return DistributionSummary(
        mean=mean,
        median=median_value,
        mode=median_value,
        variance=0.0,
        std_dev=0.0,
        min=median_value,
        max=p99_value,
        count=count,
        total_sum=mean * count,
        percentiles=Percentiles(
            p001=median_value,
            p01=median_value,
            p05=median_value,
            p10=median_value,
            p25=median_value,
            p50=median_value,
            p75=p95_value,
            p90=p95_value,
            p95=p95_value,
            p99=p99_value,
            p999=p99_value,
        ),
        pdf=None,
    )


def status_dist(
    mean: float,
    *,
    median: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    count: int = 10,
) -> StatusDistributionSummary:
    """
    Wrap a distribution as a successful/total status breakdown.

    ## WRITTEN BY AI ##
    """
    populated = distribution(
        mean,
        median=median,
        p95=p95,
        p99=p99,
        count=count,
    )
    empty = DistributionSummary.from_values([])
    return StatusDistributionSummary(
        successful=populated,
        incomplete=empty,
        errored=empty,
        total=populated,
    )


def metric_summary(mean: float = 0.0, *, count: int = 0) -> GenerativeMetricsSummary:
    """
    Build a modality metric summary with matching input/output/total sides.

    ## WRITTEN BY AI ##
    """
    dist = status_dist(
        mean,
        median=mean,
        p95=mean,
        p99=mean,
        count=count,
    )
    return GenerativeMetricsSummary(
        input=dist,
        input_per_second=None,
        input_concurrency=None,
        output=dist,
        output_per_second=None,
        output_concurrency=None,
        total=dist,
        total_per_second=None,
        total_concurrency=None,
    )


def empty_metrics_base() -> GenerativeMetrics:
    """
    Zeroed generative metrics suitable for overriding in fixtures.

    ## WRITTEN BY AI ##
    """
    zero_dist = status_dist(0.0, median=0.0, p95=0.0, p99=0.0, count=0)
    zero_summary = metric_summary(0.0, count=0)
    return GenerativeMetrics(
        request_totals=StatusBreakdown(successful=0, incomplete=0, errored=0, total=0),
        requests_per_second=zero_dist,
        request_concurrency=zero_dist,
        request_latency=zero_dist,
        request_streaming_iterations_count=zero_dist,
        prompt_token_count=zero_dist,
        output_token_count=zero_dist,
        total_token_count=zero_dist,
        time_to_first_token_ms=zero_dist,
        time_to_first_output_token_ms=zero_dist,
        time_per_output_token_ms=zero_dist,
        inter_token_latency_ms=zero_dist,
        time_to_last_round_trip_ms=zero_dist,
        avg_round_trip_time_ms=zero_dist,
        prompt_tokens_per_second=zero_dist,
        output_tokens_per_second=zero_dist,
        tokens_per_second=zero_dist,
        output_tokens_per_iteration=zero_dist,
        iter_tokens_per_iteration=zero_dist,
        text=GenerativeTextMetricsSummary(
            tokens=zero_summary,
            words=zero_summary,
            characters=zero_summary,
        ),
        image=GenerativeImageMetricsSummary(
            tokens=zero_summary,
            images=zero_summary,
            pixels=zero_summary,
            bytes=zero_summary,
        ),
        video=GenerativeVideoMetricsSummary(
            tokens=zero_summary,
            frames=zero_summary,
            seconds=zero_summary,
            bytes=zero_summary,
        ),
        audio=GenerativeAudioMetricsSummary(
            tokens=zero_summary,
            samples=zero_summary,
            seconds=zero_summary,
            bytes=zero_summary,
        ),
        tool_call=GenerativeToolCallMetricsSummary(
            tokens=zero_summary,
            mixed_tokens=zero_summary,
            count=zero_summary,
        ),
    )


def make_scheduler_metrics(
    measure_start: float,
    request_count: int,
) -> SchedulerMetrics:
    """
    Build scheduler metrics spanning a 10s measurement window.

    ## WRITTEN BY AI ##
    """
    measure_end = measure_start + 10.0
    return SchedulerMetrics(
        start_time=measure_start - 1.0,
        request_start_time=measure_start,
        measure_start_time=measure_start,
        measure_end_time=measure_end,
        request_end_time=measure_end,
        end_time=measure_end + 1.0,
        requests_made=StatusBreakdown(
            successful=request_count,
            incomplete=0,
            errored=0,
            total=request_count,
        ),
        queued_time_avg=0.0,
        resolve_start_delay_avg=0.0,
        resolve_targeted_start_delay_avg=0.0,
        request_start_delay_avg=0.0,
        request_targeted_start_delay_avg=0.0,
        request_time_avg=0.0,
        resolve_end_delay_avg=0.0,
        resolve_time_avg=0.0,
        finalized_delay_avg=0.0,
        processed_delay_avg=0.0,
    )


def make_request(
    turn_index: int,
    *,
    latency_s: float,
    ttft_ms: float,
    itl_ms: float,
    prompt_tokens: int,
    history_len: int,
    start_time: float,
    output_tokens: int = 4,
    agent_id: str = "default",
) -> GenerativeRequestStats:
    """
    Build a completed request with the given turn and timing fields.

    ## WRITTEN BY AI ##
    """
    first_token_time = start_time + (ttft_ms / 1000.0)
    last_token_time = first_token_time + ((output_tokens - 1) * itl_ms / 1000.0)
    timings = RequestTimings(
        request_start=start_time,
        request_end=start_time + latency_s,
        first_token_iteration=first_token_time,
        first_output_token_iteration=first_token_time,
        last_token_iteration=last_token_time,
        token_iterations=output_tokens,
        resolve_end=start_time + latency_s,
    )
    return GenerativeRequestStats(
        request_id=f"req-{turn_index}-{history_len}-{start_time}",
        info=RequestInfo(
            turn_index=turn_index,
            history_len=history_len,
            agent_id=agent_id,
            timings=timings,
            status="completed",
        ),
        input_metrics=UsageMetrics(text_tokens=prompt_tokens),
        output_metrics=UsageMetrics(text_tokens=output_tokens),
    )


def make_scenario(model: str = "test-model") -> BenchmarkScenario:
    """
    Minimal scenario used by HTML report fixtures.

    ## WRITTEN BY AI ##
    """
    return BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": model,
                },
                "profile": {"kind": "constant", "rate": 10.0},
                "tokenizer": {"kind": "huggingface_auto", "model": "tokenizer-model"},
                "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
            }
        }
    )


def report(
    *benchmarks: GenerativeBenchmark,
    scenario: BenchmarkScenario | None = None,
) -> GenerativeBenchmarksReport:
    """
    Wrap benchmarks in a :class:`GenerativeBenchmarksReport`.

    ## WRITTEN BY AI ##
    """
    return GenerativeBenchmarksReport(
        config=scenario or make_scenario(),
        metadata=GenerativeBenchmarkMetadata(guidellm_version="0.0.0-test"),
        benchmarks=list(benchmarks),
    )


def make_benchmark(
    *,
    strategy: AsyncConstantStrategy | ConcurrentStrategy | ThroughputStrategy,
    rps: float,
    tps: float,
    backend_model: str | None = None,
    requests: list[GenerativeRequestStats] | None = None,
    measure_start: float = 1_700_000_000.0,
) -> GenerativeBenchmark:
    """
    Build a generative benchmark with populated rate/latency metrics.

    ## WRITTEN BY AI ##
    """
    request_list = requests or []
    request_count = len(request_list) if request_list else 10
    metrics = empty_metrics_base()
    metrics.request_totals = StatusBreakdown(
        successful=request_count,
        incomplete=0,
        errored=0,
        total=request_count,
    )
    metrics.requests_per_second = status_dist(rps, count=request_count)
    metrics.request_concurrency = status_dist(
        float(strategy.requests_limit or 1),
        count=request_count,
    )
    metrics.request_latency = status_dist(
        0.25,
        median=0.22,
        p95=0.3,
        p99=0.35,
        count=request_count,
    )
    metrics.time_to_first_token_ms = status_dist(
        100.0,
        median=90.0,
        p95=150.0,
        p99=180.0,
        count=request_count,
    )
    metrics.time_to_first_output_token_ms = status_dist(
        100.0,
        median=90.0,
        p95=150.0,
        p99=180.0,
        count=request_count,
    )
    metrics.inter_token_latency_ms = status_dist(
        15.0,
        median=14.0,
        p95=18.0,
        p99=22.0,
        count=request_count,
    )
    metrics.time_per_output_token_ms = status_dist(
        20.0,
        median=18.0,
        p95=25.0,
        p99=30.0,
        count=request_count,
    )
    metrics.tokens_per_second = status_dist(tps, count=request_count)
    metrics.prompt_tokens_per_second = status_dist(10.0, count=request_count)
    metrics.output_tokens_per_second = status_dist(20.0, count=request_count)
    metrics.prompt_token_count = status_dist(
        128.0,
        median=128.0,
        p95=140.0,
        p99=150.0,
        count=request_count,
    )
    metrics.output_token_count = status_dist(
        64.0,
        median=64.0,
        p95=80.0,
        p99=90.0,
        count=request_count,
    )

    return GenerativeBenchmark(
        config=BenchmarkConfig(
            run_id="test-run",
            run_index=0,
            strategy=strategy,
            constraints={},
            profile={"kind": "test_profile"},
            requests={"kind": "test_requests"},
            backend={
                "kind": "openai_http",
                "target": "http://localhost:8000/v1",
                **({"model": backend_model} if backend_model is not None else {}),
            },
            environment={},
        ),
        scheduler_state=SchedulerState(
            start_time=measure_start - 1.0,
            end_time=measure_start + 11.0,
        ),
        scheduler_metrics=make_scheduler_metrics(measure_start, request_count),
        metrics=metrics,
        requests=StatusBreakdown(
            successful=request_list,
            incomplete=[],
            errored=[],
            total=None,
        ),
    )
