"""
Unit tests for the self-contained HTML benchmark report output.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from guidellm.benchmark.outputs.html import (
    GenerativeBenchmarkerHTML,
    HTMLBenchmarkOutputArgs,
    build_report_view,
    render_html_report,
)
from guidellm.schemas.benchmark import BenchmarkOutputArgs, BenchmarkScenario


class MockPercentiles:
    def __init__(self, p95: float = 10.0, p99: float = 12.0, p50: float = 5.0):
        self.p95 = p95
        self.p99 = p99
        self.p50 = p50


class MockDistribution:
    def __init__(
        self,
        mean: float = 5.0,
        median: float = 4.0,
        p95: float = 10.0,
        p99: float = 12.0,
        count: int = 10,
    ):
        self.mean = mean
        self.median = median
        self.count = count
        self.percentiles = MockPercentiles(p95=p95, p99=p99, p50=median)


class MockStatusDistribution:
    def __init__(
        self,
        mean: float = 5.0,
        median: float = 4.0,
        p95: float = 10.0,
        p99: float = 12.0,
        count: int = 10,
    ):
        self.successful = MockDistribution(mean, median, p95, p99, count)
        self.incomplete = MockDistribution(0.0, 0.0, 0.0, 0.0, 0)
        self.errored = MockDistribution(0.0, 0.0, 0.0, 0.0, 0)
        self.total = MockDistribution(mean, median, p95, p99, count)


class MockRequestTotals:
    def __init__(
        self,
        successful: int = 10,
        incomplete: int = 0,
        errored: int = 0,
    ):
        self.successful = successful
        self.incomplete = incomplete
        self.errored = errored
        self.total = successful + incomplete + errored


class MockMetrics:
    def __init__(self, rps: float = 1.5, tps: float = 30.0):
        self.request_totals = MockRequestTotals()
        self.requests_per_second = MockStatusDistribution(mean=rps)
        self.request_concurrency = MockStatusDistribution(mean=4.0)
        self.request_latency = MockStatusDistribution(
            mean=0.25, median=0.22, p95=0.3, p99=0.35
        )
        self.time_to_first_token_ms = MockStatusDistribution(
            mean=100.0, median=90.0, p95=150.0, p99=180.0
        )
        self.time_to_first_output_token_ms = MockStatusDistribution(
            mean=100.0, median=90.0, p95=150.0, p99=180.0
        )
        self.inter_token_latency_ms = MockStatusDistribution(
            mean=15.0, median=14.0, p95=18.0, p99=22.0
        )
        self.time_per_output_token_ms = MockStatusDistribution(
            mean=20.0, median=18.0, p95=25.0, p99=30.0
        )
        self.tokens_per_second = MockStatusDistribution(mean=tps)
        self.prompt_tokens_per_second = MockStatusDistribution(mean=10.0)
        self.output_tokens_per_second = MockStatusDistribution(mean=20.0)
        self.prompt_token_count = MockStatusDistribution(
            mean=128.0, median=128.0, p95=140.0, p99=150.0
        )
        self.output_token_count = MockStatusDistribution(
            mean=64.0, median=64.0, p95=80.0, p99=90.0
        )
        self.avg_round_trip_time_ms = MockStatusDistribution(
            mean=0.0, median=0.0, p95=0.0, p99=0.0, count=0
        )
        self.time_to_last_round_trip_ms = MockStatusDistribution(
            mean=0.0, median=0.0, p95=0.0, p99=0.0, count=0
        )
        self.text = SimpleNamespace()
        self.image = SimpleNamespace()
        self.video = SimpleNamespace()
        self.audio = SimpleNamespace()
        self.tool_call = SimpleNamespace()


class MockBenchmark:
    def __init__(
        self,
        strategy: str = "constant",
        rps: float = 1.5,
        tps: float = 30.0,
        requests: list | None = None,
        start_time: float = 1_700_000_000.0,
    ):
        self.metrics = MockMetrics(rps=rps, tps=tps)
        self.config = SimpleNamespace(strategy=SimpleNamespace(type_=strategy))
        self.requests = SimpleNamespace(
            successful=requests or [],
            incomplete=[],
            errored=[],
        )
        self.start_time = start_time


def _scenario() -> BenchmarkScenario:
    return BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": "test-model",
                },
                "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
                "profile": {"kind": "constant", "rate": 10.0},
            },
        }
    )


def _mock_request(
    turn_index: int,
    *,
    latency_s: float,
    ttft_ms: float,
    itl_ms: float,
    prompt_tokens: int,
    history_len: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_latency=latency_s,
        time_to_first_token_ms=ttft_ms,
        inter_token_latency_ms=itl_ms,
        prompt_tokens=prompt_tokens,
        info=SimpleNamespace(
            turn_index=turn_index,
            history_len=history_len,
            agent_id="default",
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
    Compact view should expose P95/P99 and peak-throughput KPIs for multi-run.

    ## WRITTEN BY AI ##
    """
    single = SimpleNamespace(
        config=_scenario(),
        metadata=SimpleNamespace(guidellm_version="0.0.0-test"),
        benchmarks=[MockBenchmark(strategy="constant", rps=2.0, tps=40.0)],
    )
    single_view = build_report_view(single)
    assert single_view["header"]["multi_run"] is False
    assert single_view["header"]["has_multi_turn"] is False
    assert single_view["runs"][0]["ttft_p95_ms"] == 150.0
    assert single_view["runs"][0]["ttft_p99_ms"] == 180.0
    assert single_view["kpis"]["tokens_per_second"] == 40.0

    multi = SimpleNamespace(
        config=_scenario(),
        metadata=SimpleNamespace(guidellm_version="0.0.0-test"),
        benchmarks=[
            MockBenchmark(strategy="rate_1", rps=1.0, tps=20.0),
            MockBenchmark(strategy="rate_2", rps=2.0, tps=50.0),
        ],
    )
    multi_view = build_report_view(multi)
    assert multi_view["header"]["multi_run"] is True
    assert multi_view["header"]["peak_index"] == 1
    assert multi_view["kpis"]["tokens_per_second"] == 50.0
    assert multi_view["kpis"]["strategy"] == "concurrent@4"
    assert multi_view["runs"][1]["label"] == "concurrent@4"


@pytest.mark.sanity
def test_build_report_view_multi_turn():
    """
    Multi-turn requests should produce by_turn aggregates with latency percentiles.

    ## WRITTEN BY AI ##
    """
    requests = [
        _mock_request(
            0, latency_s=0.2, ttft_ms=80, itl_ms=10, prompt_tokens=50, history_len=0
        ),
        _mock_request(
            0, latency_s=0.22, ttft_ms=85, itl_ms=11, prompt_tokens=55, history_len=0
        ),
        _mock_request(
            1, latency_s=0.4, ttft_ms=120, itl_ms=14, prompt_tokens=120, history_len=1
        ),
        _mock_request(
            1, latency_s=0.45, ttft_ms=130, itl_ms=15, prompt_tokens=130, history_len=1
        ),
        _mock_request(
            2, latency_s=0.7, ttft_ms=180, itl_ms=18, prompt_tokens=220, history_len=2
        ),
    ]
    report = SimpleNamespace(
        config=_scenario(),
        metadata=SimpleNamespace(guidellm_version="0.0.0-test"),
        benchmarks=[MockBenchmark(requests=requests)],
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
    finalize() should write a standalone HTML file with embedded metrics.

    ## WRITTEN BY AI ##
    """
    report = SimpleNamespace(
        config=_scenario(),
        metadata=SimpleNamespace(guidellm_version="0.0.0-test"),
        benchmarks=[
            MockBenchmark(strategy="a", rps=1.0, tps=20.0),
            MockBenchmark(strategy="b", rps=2.0, tps=45.0),
        ],
    )
    output_file = tmp_path / "benchmarks.html"
    path = await GenerativeBenchmarkerHTML(output_path=output_file).finalize(
        report  # type: ignore[arg-type]
    )
    assert path == output_file
    assert path.exists()
    content = path.read_text(encoding="utf-8")

    assert "GuideLLM" in content
    assert "window.GUIDELLM_REPORT" in content
    assert "ttft_p95_ms" in content
    assert "ttft_p99_ms" in content
    assert "total_tps" in content
    # Header may include localhost target from the scenario; forbid remote asset loads.
    assert re.search(r'<link[^>]+href=["\']https?://', content) is None
    assert re.search(r'<script[^>]+src=["\']https?://', content) is None
    assert "vllm-project.github.io" not in content


@pytest.mark.sanity
def test_render_html_includes_embedded_assets():
    """
    Rendered HTML should inline CSS/JS and not leave template placeholders.

    ## WRITTEN BY AI ##
    """
    report = SimpleNamespace(
        config=_scenario(),
        metadata=SimpleNamespace(guidellm_version="0.0.0-test"),
        benchmarks=[MockBenchmark()],
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
