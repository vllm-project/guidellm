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
from guidellm.scheduler import (
    AsyncConstantStrategy,
    ConcurrentStrategy,
    ThroughputStrategy,
)
from guidellm.schemas import DistributionSummary
from guidellm.schemas.benchmark import BenchmarkOutputArgs
from tests.unit.benchmark.html_report_fixtures import (
    make_benchmark as _make_benchmark,
)
from tests.unit.benchmark.html_report_fixtures import (
    make_request as _make_request,
)
from tests.unit.benchmark.html_report_fixtures import (
    make_scenario as _scenario,
)
from tests.unit.benchmark.html_report_fixtures import (
    report as _report,
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
    assert single_view["runs"][0]["configured_request_rate"] == 2.0
    assert "kpis" not in single_view
    assert "modalities" not in single_view["runs"][0]
    assert single_view["runs"][0]["total_tps"] == 40.0

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
    assert "kpis" not in multi_view
    assert multi_view["runs"][1]["total_tps"] == 50.0
    assert multi_view["runs"][1]["label"] == "concurrent@4"
    assert multi_view["runs"][1]["concurrency"] == 4.0
    assert multi_view["runs"][1]["configured_concurrency"] == 4
    assert multi_view["runs"][0]["configured_concurrency"] == 2
    assert multi_view["runs"][0]["configured_request_rate"] is None
    assert multi_view["runs"][1]["configured_request_rate"] is None

    throughput_report = _report(
        _make_benchmark(
            strategy=ThroughputStrategy(max_concurrency=8),
            rps=3.0,
            tps=60.0,
        )
    )
    throughput_view = build_report_view(throughput_report)
    assert throughput_view["runs"][0]["strategy"] == "throughput"
    assert throughput_view["runs"][0]["configured_concurrency"] == 8
    assert throughput_view["runs"][0]["configured_request_rate"] is None

    unlimited_throughput = _report(
        _make_benchmark(
            strategy=ThroughputStrategy(),
            rps=4.0,
            tps=80.0,
        )
    )
    unlimited_view = build_report_view(unlimited_throughput)
    assert unlimited_view["runs"][0]["configured_concurrency"] is None
    assert unlimited_view["runs"][0]["configured_request_rate"] is None


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
    # Single-agent multi-turn: no agent filter note.
    assert view["turn_note"] is None
    assert all("use_history_axis" not in row for row in view["by_turn"])
    # Turn P95 matches DistributionSummary for the single turn-2 sample.
    turn2_latency_ms = float(requests[4].request_latency) * 1000.0
    expected = DistributionSummary.from_values([turn2_latency_ms]).percentiles.p95
    assert view["by_turn"][2]["request_latency_p95_ms"] == expected


@pytest.mark.sanity
def test_build_report_view_multi_turn_filters_to_dominant_agent():
    """
    Turn curves should keep the dominant agent and mention the filter in turn_note.

    ## WRITTEN BY AI ##
    """
    requests = [
        _make_request(
            0,
            agent_id="primary",
            latency_s=0.2,
            ttft_ms=80.0,
            itl_ms=10.0,
            prompt_tokens=50,
            history_len=0,
            start_time=10.0,
        ),
        _make_request(
            1,
            agent_id="primary",
            latency_s=0.3,
            ttft_ms=90.0,
            itl_ms=11.0,
            prompt_tokens=60,
            history_len=1,
            start_time=11.0,
        ),
        _make_request(
            2,
            agent_id="primary",
            latency_s=0.4,
            ttft_ms=100.0,
            itl_ms=12.0,
            prompt_tokens=70,
            history_len=2,
            start_time=12.0,
        ),
        _make_request(
            0,
            agent_id="other",
            latency_s=0.9,
            ttft_ms=200.0,
            itl_ms=20.0,
            prompt_tokens=40,
            history_len=0,
            start_time=13.0,
        ),
    ]
    report = _report(
        _make_benchmark(
            strategy=AsyncConstantStrategy(rate=1.0),
            rps=1.0,
            tps=20.0,
            requests=requests,
        )
    )
    view = build_report_view(report)
    assert view["by_turn"] is not None
    assert [row["turn_index"] for row in view["by_turn"]] == [0, 1, 2]
    assert view["turn_note"] is not None
    assert 'agent "primary"' in view["turn_note"]
    assert "3/4 requests" in view["turn_note"]


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
    assert "__GUIDELLM_STATIC_TABLE__" not in html
    assert "__GUIDELLM_KPI_RPS__" not in html
    assert "static-summary" in html
    assert "<noscript>" not in html
    assert "require JavaScript" in html
    assert 'id="kpi-rps">' in html
    assert "constant@1.50" in html
    # Peak-run defaults are embedded for no-JS viewers.
    assert re.search(r'id="kpi-rps">[^<—]+</div>', html)
    assert re.search(r'id="meta-model">[^<—]+</span>', html)
    assert 'class="help-q"' in html
    assert "Time to First Token (TTFT)" in html
    assert "var HELP" in html
    assert "help-tip-also" in html
    assert 'also: ["p95", "p99"]' in html
    assert 'role="tablist"' in html
    assert "Latency components" in html
    assert "prompt_vs_turn" not in html
    assert "function niceNum" not in html
    assert "function drawStackedBars" not in html
    assert "function drawGroupedBars" not in html
    assert "COLORS.itl" in html
    assert "card-spaced" in html
    assert "modality-sections" in html
    assert 'style="margin-top:16px"' not in html
