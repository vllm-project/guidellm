"""
Unit tests for the self-contained HTML benchmark report output.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from guidellm.benchmark.outputs.html import (
    GenerativeBenchmarkerHTML,
    HTMLBenchmarkOutputArgs,
    build_report_view,
    render_html_report,
)
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
from guidellm.scheduler import AsyncConstantStrategy, ConcurrentStrategy, SchedulerState
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
from guidellm.schemas.benchmark import BenchmarkOutputArgs, BenchmarkScenario


def _distribution(
    mean: float,
    *,
    median: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    count: int = 10,
) -> DistributionSummary:
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


def _status_dist(
    mean: float,
    *,
    median: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    count: int = 10,
) -> StatusDistributionSummary:
    populated = _distribution(
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


def _metric_summary(mean: float = 0.0, *, count: int = 0) -> GenerativeMetricsSummary:
    distribution = _status_dist(
        mean,
        median=mean,
        p95=mean,
        p99=mean,
        count=count,
    )
    return GenerativeMetricsSummary(
        input=distribution,
        input_per_second=None,
        input_concurrency=None,
        output=distribution,
        output_per_second=None,
        output_concurrency=None,
        total=distribution,
        total_per_second=None,
        total_concurrency=None,
    )


def _empty_metrics_base() -> GenerativeMetrics:
    zero_dist = _status_dist(0.0, median=0.0, p95=0.0, p99=0.0, count=0)
    zero_summary = _metric_summary(0.0, count=0)
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


def _make_scheduler_metrics(
    measure_start: float,
    request_count: int,
) -> SchedulerMetrics:
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


def _make_request(
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


def _scenario(model: str = "test-model") -> BenchmarkScenario:
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


def _report(
    *benchmarks: GenerativeBenchmark,
    scenario: BenchmarkScenario | None = None,
) -> GenerativeBenchmarksReport:
    return GenerativeBenchmarksReport(
        config=scenario or _scenario(),
        metadata=GenerativeBenchmarkMetadata(guidellm_version="0.0.0-test"),
        benchmarks=list(benchmarks),
    )


def _make_benchmark(
    *,
    strategy: AsyncConstantStrategy | ConcurrentStrategy,
    rps: float,
    tps: float,
    backend_model: str | None = None,
    requests: list[GenerativeRequestStats] | None = None,
    measure_start: float = 1_700_000_000.0,
) -> GenerativeBenchmark:
    request_list = requests or []
    request_count = len(request_list) if request_list else 10
    metrics = _empty_metrics_base()
    metrics.request_totals = StatusBreakdown(
        successful=request_count,
        incomplete=0,
        errored=0,
        total=request_count,
    )
    metrics.requests_per_second = _status_dist(rps, count=request_count)
    metrics.request_concurrency = _status_dist(
        float(strategy.requests_limit or 1),
        count=request_count,
    )
    metrics.request_latency = _status_dist(
        0.25,
        median=0.22,
        p95=0.3,
        p99=0.35,
        count=request_count,
    )
    metrics.time_to_first_token_ms = _status_dist(
        100.0,
        median=90.0,
        p95=150.0,
        p99=180.0,
        count=request_count,
    )
    metrics.time_to_first_output_token_ms = _status_dist(
        100.0,
        median=90.0,
        p95=150.0,
        p99=180.0,
        count=request_count,
    )
    metrics.inter_token_latency_ms = _status_dist(
        15.0,
        median=14.0,
        p95=18.0,
        p99=22.0,
        count=request_count,
    )
    metrics.time_per_output_token_ms = _status_dist(
        20.0,
        median=18.0,
        p95=25.0,
        p99=30.0,
        count=request_count,
    )
    metrics.tokens_per_second = _status_dist(tps, count=request_count)
    metrics.prompt_tokens_per_second = _status_dist(10.0, count=request_count)
    metrics.output_tokens_per_second = _status_dist(20.0, count=request_count)
    metrics.prompt_token_count = _status_dist(
        128.0,
        median=128.0,
        p95=140.0,
        p99=150.0,
        count=request_count,
    )
    metrics.output_token_count = _status_dist(
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
        scheduler_metrics=_make_scheduler_metrics(measure_start, request_count),
        metrics=metrics,
        requests=StatusBreakdown(
            successful=request_list,
            incomplete=[],
            errored=[],
            total=None,
        ),
    )


@pytest.mark.smoke
def test_from_args_creates_instance():
    """
    HTML args should construct a GenerativeBenchmarkerHTML instance.

    ## WRITTEN BY AI ##
    """
    args = HTMLBenchmarkOutputArgs(path=Path("report.html"))
    output = GenerativeBenchmarkerHTML.from_args(args)
    assert output.output_path == Path("report.html")


@pytest.mark.smoke
def test_from_args_rejects_wrong_type():
    """
    Non-HTML args should raise TypeError.

    ## WRITTEN BY AI ##
    """

    class DummyArgs(BenchmarkOutputArgs):
        kind: str = "dummy"

    with pytest.raises(TypeError, match="Expected HTMLBenchmarkOutputArgs"):
        GenerativeBenchmarkerHTML.from_args(DummyArgs(kind="dummy"))


@pytest.mark.sanity
def test_build_report_view_single_and_multi_run():
    """
    Compact view should expose p95/p99 values and peak-throughput KPIs.

    ## WRITTEN BY AI ##
    """
    single_report = _report(
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=2.0),
            rps=2.0,
            tps=40.0,
        )
    )
    single_view = build_report_view(single_report)
    assert single_view["header"]["multi_run"] is False
    assert single_view["header"]["has_multi_turn"] is False
    assert single_view["runs"][0]["ttft_p95_ms"] == 150.0
    assert single_view["runs"][0]["ttft_p99_ms"] == 180.0
    assert single_view["runs"][0]["label"] == "constant@2.00"
    assert single_view["kpis"]["tokens_per_second"] == 40.0

    multi_report = _report(
        _make_benchmark(
            strategy=ConcurrentStrategy(streams=2),
            rps=1.0,
            tps=20.0,
            measure_start=1_700_000_000.0,
        ),
        _make_benchmark(
            strategy=ConcurrentStrategy(streams=4),
            rps=2.0,
            tps=50.0,
            measure_start=1_700_000_010.0,
        ),
    )
    multi_view = build_report_view(multi_report)
    assert multi_view["header"]["multi_run"] is True
    assert multi_view["header"]["peak_index"] == 1
    assert multi_view["kpis"]["tokens_per_second"] == 50.0
    assert multi_view["kpis"]["strategy"] == "concurrent@4"
    assert multi_view["runs"][1]["label"] == "concurrent@4"
    assert multi_view["runs"][1]["concurrency"] == 4.0
    assert multi_view["runs"][1]["configured_concurrency"] == 4
    assert multi_view["runs"][0]["configured_concurrency"] == 2


@pytest.mark.sanity
def test_header_model_prefers_resolved_backend_info():
    """
    Header model should prefer the resolved per-run backend model.

    ## WRITTEN BY AI ##
    """
    report = _report(
        _make_benchmark(
            strategy=ConcurrentStrategy(streams=4),
            rps=2.0,
            tps=50.0,
            backend_model="Qwen/Qwen3-0.6B",
        ),
        scenario=_scenario(model=""),
    )
    view = build_report_view(report)
    assert view["header"]["model"] == "Qwen/Qwen3-0.6B"


@pytest.mark.sanity
def test_build_report_view_multi_turn():
    """
    Multi-turn requests should produce per-turn aggregates with latency percentiles.

    ## WRITTEN BY AI ##
    """
    requests = [
        _make_request(
            0,
            latency_s=0.2,
            ttft_ms=80.0,
            itl_ms=10.0,
            prompt_tokens=50,
            history_len=0,
            start_time=10.0,
        ),
        _make_request(
            0,
            latency_s=0.22,
            ttft_ms=85.0,
            itl_ms=11.0,
            prompt_tokens=55,
            history_len=0,
            start_time=11.0,
        ),
        _make_request(
            1,
            latency_s=0.4,
            ttft_ms=120.0,
            itl_ms=14.0,
            prompt_tokens=120,
            history_len=1,
            start_time=12.0,
        ),
        _make_request(
            1,
            latency_s=0.45,
            ttft_ms=130.0,
            itl_ms=15.0,
            prompt_tokens=130,
            history_len=1,
            start_time=13.0,
        ),
        _make_request(
            2,
            latency_s=0.7,
            ttft_ms=180.0,
            itl_ms=18.0,
            prompt_tokens=220,
            history_len=2,
            start_time=14.0,
        ),
    ]
    report = _report(
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=1.0),
            rps=1.5,
            tps=30.0,
            requests=requests,
        )
    )
    view = build_report_view(report)
    assert view["header"]["has_multi_turn"] is True
    assert view["by_turn"] is not None
    assert [row["turn_index"] for row in view["by_turn"]] == [0, 1, 2]
    assert view["by_turn"][2]["prompt_tokens_median"] == 220.0
    assert view["by_turn"][2]["request_latency_p95_ms"] is not None


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_writes_self_contained_html(tmp_path: Path):
    """
    Finalize should write a standalone HTML file with embedded metrics.

    ## WRITTEN BY AI ##
    """
    report = _report(
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=1.0),
            rps=1.0,
            tps=20.0,
        ),
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=2.0),
            rps=2.0,
            tps=45.0,
            measure_start=1_700_000_010.0,
        ),
    )
    output_file = tmp_path / "benchmarks.html"
    path = await GenerativeBenchmarkerHTML(output_path=output_file).finalize(report)
    assert path == output_file
    assert path.exists()
    content = path.read_text(encoding="utf-8")

    assert "GuideLLM" in content
    assert "window.GUIDELLM_REPORT" in content
    assert "ttft_p95_ms" in content
    assert "ttft_p99_ms" in content
    assert "total_tps" in content
    assert re.search(r'<link[^>]+href=["\']https?://', content) is None
    assert re.search(r'<script[^>]+src=["\']https?://', content) is None
    assert "vllm-project.github.io" not in content


@pytest.mark.sanity
def test_render_html_includes_embedded_assets():
    """
    Rendered HTML should inline CSS and JS without template placeholders.

    ## WRITTEN BY AI ##
    """
    report = _report(
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=1.5),
            rps=1.5,
            tps=30.0,
        )
    )
    html = render_html_report(build_report_view(report))
    assert "<style>" in html
    assert "function" in html
    assert "__GUIDELLM_REPORT_CSS__" not in html
    assert "__GUIDELLM_REPORT_JS__" not in html
    assert "__GUIDELLM_REPORT_JSON__" not in html
    assert 'class="help-q"' in html
    assert "Time to First Token (TTFT)" in html
    assert "var HELP" in html
    assert "help-tip-also" in html
    assert 'also: ["p95", "p99"]' in html
