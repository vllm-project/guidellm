"""
HTML output formatter for benchmark results.

Builds a fully self-contained HTML report (inline CSS/JS, no CDN) with customer-
facing throughput and latency visualizations, multi-run comparison charts,
optional multi-turn breakdowns, and specialty detail tables.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, Literal

from loguru import logger
from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import (
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
)
from guidellm.benchmark.schemas.metrics import (
    GenerativeAudioMetricsSummary,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeTextMetricsSummary,
    GenerativeToolCallMetricsSummary,
    GenerativeVideoMetricsSummary,
)
from guidellm.scheduler import AsyncConstantStrategy, AsyncPoissonStrategy
from guidellm.schemas import (
    DistributionSummary,
    GenerativeRequestStats,
    StatusBreakdown,
    StatusDistributionSummary,
)
from guidellm.schemas.benchmark import BenchmarkOutputArgs
from guidellm.schemas.benchmark.outputs import HTMLBenchmarkOutputArgs

__all__ = [
    "GenerativeBenchmarkerHTML",
    "HTMLBenchmarkOutputArgs",
    "build_report_view",
    "render_html_report",
]

_StatusName = Literal["successful", "incomplete", "errored", "total"]


_MODALITY_GROUPS: dict[str, list[tuple[str, str]]] = {
    "text": [("tokens", "Tokens"), ("words", "Words"), ("characters", "Characters")],
    "image": [
        ("tokens", "Tokens"),
        ("images", "Images"),
        ("pixels", "Pixels"),
        ("bytes", "Bytes"),
    ],
    "video": [
        ("tokens", "Tokens"),
        ("frames", "Frames"),
        ("seconds", "Seconds"),
        ("bytes", "Bytes"),
    ],
    "audio": [
        ("tokens", "Tokens"),
        ("samples", "Samples"),
        ("seconds", "Seconds"),
        ("bytes", "Bytes"),
    ],
    "tool_call": [
        ("tokens", "Tokens"),
        ("mixed_tokens", "Mixed Tokens"),
        ("count", "Count"),
    ],
}


@GenerativeBenchmarkerOutput.register("html")
class GenerativeBenchmarkerHTML(GenerativeBenchmarkerOutput):
    """
    Self-contained HTML report formatter for generative benchmarks.

    Embeds compact chart/table JSON into a packaged HTML/CSS/JS template so the
    resulting file can be shared without network access or versioned UI assets.

    :cvar DEFAULT_FILE: Default filename when ``path`` is a directory
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.html"

    output_path: Path = Field(
        default_factory=Path.cwd,
        description=(
            "Directory or file path for saving the HTML report, "
            "defaults to current working directory"
        ),
    )

    @classmethod
    def from_args(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerHTML:
        """
        Create an HTML output formatter from output arguments.

        :param args: Output configuration with path
        :return: Configured HTML output formatter
        """
        if not isinstance(args, HTMLBenchmarkOutputArgs):
            raise TypeError(f"Expected HTMLBenchmarkOutputArgs, got {type(args)}")

        return cls(output_path=args.path)

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Generate and save the self-contained HTML benchmark report.

        :param report: Completed benchmark report containing all results
        :return: Path to the saved HTML report file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / self.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        view = build_report_view(report)
        html = render_html_report(view)
        output_path.write_text(html, encoding="utf-8")
        logger.debug("Saved HTML report to {}", output_path)
        return output_path


def render_html_report(view: dict[str, Any]) -> str:
    """
    Render the packaged HTML template with report view data.

    :param view: Compact report dictionary from :func:`build_report_view`
    :return: Complete HTML document string
    """
    pkg = files("guidellm.benchmark.outputs.html_report")
    template = (pkg / "template.html").read_text(encoding="utf-8")
    css = (pkg / "report.css").read_text(encoding="utf-8")
    js = (pkg / "report.js").read_text(encoding="utf-8")
    payload = json.dumps(view, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = (
        template.replace("__GUIDELLM_REPORT_CSS__", css)
        .replace("__GUIDELLM_REPORT_JS__", js)
        .replace("__GUIDELLM_REPORT_JSON__", payload)
        .replace(
            "__GUIDELLM_STATIC_TABLE__",
            _build_static_summary_table(view.get("runs") or []),
        )
    )
    for token, value in _static_text_defaults(view).items():
        html = html.replace(token, value)
    return html


def build_report_view(report: GenerativeBenchmarksReport) -> dict[str, Any]:
    """
    Build the compact JSON view embedded in the HTML report.

    :param report: Benchmark report to summarize
    :return: Dictionary consumed by the inlined report JavaScript
    """
    benchmarks = report.benchmarks
    runs = [
        _build_run_row(benchmark, index) for index, benchmark in enumerate(benchmarks)
    ]
    peak_index = _peak_throughput_index(runs)
    peak_run = runs[peak_index] if runs else None

    by_turn: list[dict[str, Any]] | None = None
    turn_note: str | None = None
    turn_agents: list[dict[str, Any]] | None = None
    turn_samples: list[dict[str, Any]] | None = None
    turn_peak_note: str | None = None
    if benchmarks:
        source = benchmarks[peak_index]
        turn_bundle = _build_metrics_by_turn(source)
        if turn_bundle is not None:
            by_turn = turn_bundle["rows"]
            turn_agents = turn_bundle["turn_agents"]
            turn_samples = turn_bundle["turn_samples"]
            note_parts: list[str] = []
            # Note the default filter only when sub-agents exist (control is shown).
            if turn_bundle["has_subagents"]:
                note_parts.append(turn_bundle["default_note"])
            if len(benchmarks) > 1:
                turn_peak_note = (
                    "Peak-throughput run "
                    f"(#{peak_index + 1}: "
                    f"{peak_run['label'] if peak_run else 'n/a'})."
                )
                note_parts.append(turn_peak_note)
            turn_note = " ".join(note_parts) if note_parts else None

    header = _build_header(report, runs, peak_index, has_multi_turn=bool(by_turn))
    details = _build_details(benchmarks, runs)

    return {
        "header": header,
        "runs": runs,
        "by_turn": by_turn,
        "turn_note": turn_note,
        "turn_agents": turn_agents,
        "turn_samples": turn_samples,
        "turn_peak_note": turn_peak_note,
        "details": details,
    }


def _nonempty_str(value: Any) -> str | None:
    """Return stripped text, or ``None`` when missing or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_report_model(report: GenerativeBenchmarksReport) -> str:
    """Resolve the display model name for the report header."""
    for benchmark in report.benchmarks:
        model = _nonempty_str(benchmark.config.backend.get("model"))
        if model:
            return model

    backend_config = report.config.spec.backend.model_dump()
    model = _nonempty_str(backend_config.get("model"))
    if model:
        return model

    tokenizer_model = _nonempty_str(report.config.spec.tokenizer.model)
    if tokenizer_model:
        return tokenizer_model

    return "N/A"


def _build_header(
    report: GenerativeBenchmarksReport,
    runs: Sequence[dict[str, Any]],
    peak_index: int,
    *,
    has_multi_turn: bool,
) -> dict[str, Any]:
    backend_config = report.config.spec.backend.model_dump()
    target = _nonempty_str(backend_config.get("target")) or "N/A"
    profile_kind = report.config.spec.profile.kind or "N/A"

    timestamps = [float(benchmark.start_time) for benchmark in report.benchmarks]
    latest_timestamp = max(timestamps) if timestamps else None
    timestamp_iso = (
        datetime.fromtimestamp(latest_timestamp, tz=timezone.utc).isoformat()
        if latest_timestamp is not None
        else None
    )

    return {
        "model": _resolve_report_model(report),
        "target": target,
        "profile": profile_kind,
        "timestamp": timestamp_iso,
        "guidellm_version": report.metadata.guidellm_version or "N/A",
        "peak_index": peak_index,
        "multi_run": len(runs) > 1,
        "has_multi_turn": has_multi_turn,
        "run_count": len(runs),
    }


def _html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_num(value: Any) -> str:
    """Format a number like the report JS ``fmt`` helper."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number):
        return "—"
    abs_n = abs(number)
    if abs_n >= 100:  # noqa: PLR2004 — match report.js fmt thresholds
        return f"{number:.0f}"
    if abs_n >= 10:  # noqa: PLR2004 — match report.js fmt thresholds
        return f"{number:.1f}"
    return f"{number:.2f}"


def _fmt_pct(value: Any) -> str:
    """Format a ratio like the report JS ``pct`` helper."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number):
        return "—"
    return f"{number * 100:.2f}%"


def _static_text_defaults(view: dict[str, Any]) -> dict[str, str]:
    """
    Peak-run / header defaults embedded in the HTML for no-JS viewers.

    JavaScript overwrites these on init when available.
    """
    header = view.get("header") or {}
    runs = view.get("runs") or []
    peak_index = int(header.get("peak_index") or 0)
    run = (
        runs[peak_index]
        if runs and 0 <= peak_index < len(runs)
        else (runs[0] if runs else {})
    )

    return {
        "__GUIDELLM_META_MODEL__": _html_escape(header.get("model") or "N/A"),
        "__GUIDELLM_META_TARGET__": _html_escape(header.get("target") or "N/A"),
        "__GUIDELLM_META_PROFILE__": _html_escape(header.get("profile") or "N/A"),
        "__GUIDELLM_META_TIME__": _html_escape(header.get("timestamp") or "N/A"),
        "__GUIDELLM_META_VERSION__": _html_escape(
            header.get("guidellm_version") or "N/A"
        ),
        "__GUIDELLM_KPI_RPS__": _html_escape(_fmt_num(run.get("request_rate"))),
        "__GUIDELLM_KPI_OUT_TPS__": _html_escape(_fmt_num(run.get("output_tps"))),
        "__GUIDELLM_KPI_TOTAL_TPS__": _html_escape(_fmt_num(run.get("total_tps"))),
        "__GUIDELLM_KPI_LAT_P95__": _html_escape(
            _fmt_num(run.get("request_latency_p95_ms"))
        ),
        "__GUIDELLM_KPI_LAT_P99__": _html_escape(
            _fmt_num(run.get("request_latency_p99_ms"))
        ),
        "__GUIDELLM_KPI_TTFT_P95__": _html_escape(_fmt_num(run.get("ttft_p95_ms"))),
        "__GUIDELLM_KPI_TTFT_P99__": _html_escape(_fmt_num(run.get("ttft_p99_ms"))),
        "__GUIDELLM_KPI_ITL_P95__": _html_escape(_fmt_num(run.get("itl_p95_ms"))),
        "__GUIDELLM_KPI_ITL_P99__": _html_escape(_fmt_num(run.get("itl_p99_ms"))),
        "__GUIDELLM_KPI_ERROR__": _html_escape(_fmt_pct(run.get("error_rate"))),
    }


def _build_static_summary_table(runs: Sequence[dict[str, Any]]) -> str:
    """
    Build a static HTML comparison table for viewers without JavaScript.

    :param runs: Compact per-benchmark rows from the report view
    :return: Escaped HTML table markup (or an empty-state paragraph)
    """
    if not runs:
        return "<p>No benchmark runs in this report.</p>"

    def cell(value: Any) -> str:
        return f"<td>{_html_escape(value)}</td>"

    rows_html: list[str] = []
    for run in runs:
        label = run.get("label") or run.get("strategy") or "—"
        rows_html.append(
            "<tr>"
            + cell(label)
            + cell(_fmt_num(run.get("request_rate")))
            + cell(_fmt_num(run.get("total_tps")))
            + cell(_fmt_pct(run.get("error_rate")))
            + "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>Strategy</th><th>Req/s</th><th>Total tok/s</th><th>Error rate</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def _build_run_row(benchmark: GenerativeBenchmark, index: int) -> dict[str, Any]:
    metrics = benchmark.metrics
    strategy = benchmark.config.strategy
    label = str(strategy)
    strategy_name = strategy.type_

    request_totals = metrics.request_totals
    successful = _status_total(request_totals, "successful")
    incomplete = _status_total(request_totals, "incomplete")
    errored = _status_total(request_totals, "errored")
    total = _status_total(request_totals, "total") or (
        successful + incomplete + errored
    )
    error_rate = (errored / total) if total else 0.0

    # Request latency is stored in seconds; convert to ms for chart consistency.
    req_lat_p95 = _percentile(metrics.request_latency, "p95")
    req_lat_p99 = _percentile(metrics.request_latency, "p99")
    concurrency = _mean(metrics.request_concurrency, status="total")
    # Intended parallel-request cap from the strategy (e.g. streams); None = unlimited.
    configured_concurrency = strategy.requests_limit
    # Rate-based strategies expose target RPS as `.rate`; others have no intended rate.
    configured_request_rate = (
        strategy.rate
        if isinstance(strategy, (AsyncConstantStrategy, AsyncPoissonStrategy))
        else None
    )

    return {
        "index": index,
        "strategy": strategy_name,
        "label": label,
        "request_rate": _mean(metrics.requests_per_second, status="total"),
        "concurrency": concurrency,
        "configured_concurrency": configured_concurrency,
        "configured_request_rate": configured_request_rate,
        "total_tps": _mean(metrics.tokens_per_second, status="total"),
        "input_tps": _mean(metrics.prompt_tokens_per_second, status="total"),
        "output_tps": _mean(metrics.output_tokens_per_second, status="total"),
        "request_latency_p95_ms": _ms(req_lat_p95),
        "request_latency_p99_ms": _ms(req_lat_p99),
        "ttft_p95_ms": _percentile(metrics.time_to_first_token_ms, "p95"),
        "ttft_p99_ms": _percentile(metrics.time_to_first_token_ms, "p99"),
        "ttfot_p95_ms": _percentile(metrics.time_to_first_output_token_ms, "p95"),
        "ttfot_p99_ms": _percentile(metrics.time_to_first_output_token_ms, "p99"),
        "itl_p95_ms": _percentile(metrics.inter_token_latency_ms, "p95"),
        "itl_p99_ms": _percentile(metrics.inter_token_latency_ms, "p99"),
        "tpot_p95_ms": _percentile(metrics.time_per_output_token_ms, "p95"),
        "tpot_p99_ms": _percentile(metrics.time_per_output_token_ms, "p99"),
        "prompt_tokens_median": _median(metrics.prompt_token_count),
        "prompt_tokens_p95": _percentile(metrics.prompt_token_count, "p95"),
        "output_tokens_median": _median(metrics.output_token_count),
        "output_tokens_p95": _percentile(metrics.output_token_count, "p95"),
        "rtt_avg_p95_ms": _percentile(metrics.avg_round_trip_time_ms, "p95"),
        "rtt_avg_p99_ms": _percentile(metrics.avg_round_trip_time_ms, "p99"),
        "rtt_last_p95_ms": _percentile(metrics.time_to_last_round_trip_ms, "p95"),
        "rtt_last_p99_ms": _percentile(metrics.time_to_last_round_trip_ms, "p99"),
        "successful": successful,
        "incomplete": incomplete,
        "errored": errored,
        "total": total,
        "error_rate": error_rate,
    }


def _build_details(
    benchmarks: Sequence[GenerativeBenchmark],
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    show_ttfot = any(
        _ttfot_differs(run.get("ttft_p95_ms"), run.get("ttfot_p95_ms"))
        or _ttfot_differs(run.get("ttft_p99_ms"), run.get("ttfot_p99_ms"))
        for run in runs
    )
    show_rtt = any(
        (run.get("rtt_avg_p95_ms") or 0) > 0 or (run.get("rtt_last_p95_ms") or 0) > 0
        for run in runs
    )

    modality_sections: dict[str, dict[str, Any]] = {}
    for modality, groups in _MODALITY_GROUPS.items():
        metrics_out: list[dict[str, Any]] = []
        for attr, label in groups:
            metric_rows: list[dict[str, Any]] = []
            for index, benchmark in enumerate(benchmarks):
                run = runs[index] if index < len(runs) else {}
                mod = _extract_modality_stats(benchmark.metrics).get(modality) or {}
                stats = mod.get(attr) or {}
                input_stats = stats.get("input") or {}
                output_stats = stats.get("output") or {}
                input_mean = input_stats.get("mean")
                input_p95 = input_stats.get("p95")
                output_mean = output_stats.get("mean")
                output_p95 = output_stats.get("p95")
                if all(
                    value in (None, 0)
                    for value in (input_mean, input_p95, output_mean, output_p95)
                ):
                    continue
                metric_rows.append(
                    {
                        "strategy": run.get("label") or run.get("strategy"),
                        "input_mean": input_mean,
                        "input_p95": input_p95,
                        "output_mean": output_mean,
                        "output_p95": output_p95,
                    }
                )
            if metric_rows:
                metrics_out.append(
                    {
                        "key": attr,
                        "label": label,
                        "rows": metric_rows,
                    }
                )
        if metrics_out:
            modality_sections[modality] = {
                "label": modality.replace("_", " ").title(),
                "metrics": metrics_out,
            }

    return {
        "show_ttfot": show_ttfot,
        "show_rtt": show_rtt,
        "modality_sections": modality_sections,
        "extra_latency_rows": [
            {
                "strategy": run.get("label") or run["strategy"],
                "tpot_p95_ms": run.get("tpot_p95_ms"),
                "tpot_p99_ms": run.get("tpot_p99_ms"),
                "ttfot_p95_ms": run.get("ttfot_p95_ms"),
                "ttfot_p99_ms": run.get("ttfot_p99_ms"),
            }
            for run in runs
        ],
        "request_size_rows": [
            {
                "strategy": run.get("label") or run["strategy"],
                "prompt_tokens_median": run.get("prompt_tokens_median"),
                "prompt_tokens_p95": run.get("prompt_tokens_p95"),
                "output_tokens_median": run.get("output_tokens_median"),
                "output_tokens_p95": run.get("output_tokens_p95"),
            }
            for run in runs
        ],
        "rtt_rows": [
            {
                "strategy": run.get("label") or run["strategy"],
                "avg_p95_ms": run.get("rtt_avg_p95_ms"),
                "avg_p99_ms": run.get("rtt_avg_p99_ms"),
                "last_p95_ms": run.get("rtt_last_p95_ms"),
                "last_p99_ms": run.get("rtt_last_p99_ms"),
            }
            for run in runs
        ]
        if show_rtt
        else [],
        "benchmark_count": len(benchmarks),
    }


# Main conversation chain uses null / "default"; anything else is a sub-agent.
_MAIN_AGENT_IDS: frozenset[str | None] = frozenset({None, "default"})


def _agent_display_label(agent_id: str | None) -> str:
    """Human-readable agent label; null and ``default`` both show as default."""
    if agent_id is None or agent_id == "default":
        return "default"
    return str(agent_id)


def _is_subagent(agent_id: str | None) -> bool:
    """Return True when ``agent_id`` is not the main default chain."""
    return agent_id not in _MAIN_AGENT_IDS


def _turn_sample_from_request(req: GenerativeRequestStats) -> dict[str, Any]:
    """Compact per-request fields for client-side turn aggregation."""
    latency = req.request_latency
    ttft = req.time_to_first_token_ms
    itl = req.inter_token_latency_ms
    prompt_tokens = req.prompt_tokens
    return {
        "turn_index": int(req.info.turn_index),
        "agent_id": req.info.agent_id,
        "latency_ms": float(latency) * 1000.0 if latency is not None else None,
        "ttft_ms": float(ttft) if ttft is not None else None,
        "itl_ms": float(itl) if itl is not None else None,
        "prompt_tokens": float(prompt_tokens) if prompt_tokens is not None else None,
        "history_len": float(req.info.history_len),
    }


def _aggregate_turn_rows(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group compact turn samples by ``turn_index`` into report rows.

    Percentiles use the same empirical CDF path as run-level
    ``DistributionSummary`` metrics (via :func:`_safe_percentile`).

    :param samples: Filtered successful multi-turn sample dicts
    :return: Sorted per-turn aggregate rows for charts and tables
    """
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[int(sample["turn_index"])].append(sample)

    rows: list[dict[str, Any]] = []
    for turn_index in sorted(buckets):
        group = buckets[turn_index]
        latencies_ms = [
            float(s["latency_ms"]) for s in group if s.get("latency_ms") is not None
        ]
        ttfts = [float(s["ttft_ms"]) for s in group if s.get("ttft_ms") is not None]
        itls = [float(s["itl_ms"]) for s in group if s.get("itl_ms") is not None]
        prompt_tokens = [
            float(s["prompt_tokens"])
            for s in group
            if s.get("prompt_tokens") is not None
        ]
        history_clean = [float(s["history_len"]) for s in group]

        rows.append(
            {
                "turn_index": turn_index,
                "count": len(group),
                "history_len_median": _safe_median(history_clean),
                "prompt_tokens_median": _safe_median(prompt_tokens),
                "prompt_tokens_p95": _safe_percentile(prompt_tokens, 0.95),
                "request_latency_p95_ms": _safe_percentile(latencies_ms, 0.95),
                "request_latency_p99_ms": _safe_percentile(latencies_ms, 0.99),
                "ttft_p95_ms": _safe_percentile(ttfts, 0.95),
                "ttft_p99_ms": _safe_percentile(ttfts, 0.99),
                "itl_p95_ms": _safe_percentile(itls, 0.95),
                "itl_p99_ms": _safe_percentile(itls, 0.99),
            }
        )
    return rows


def _build_metrics_by_turn(
    benchmark: GenerativeBenchmark,
) -> dict[str, Any] | None:
    """
    Build multi-turn samples, agent catalog, and default per-turn aggregates.

    When parent and sub-agent requests both exist, default ``rows`` use parent
    agents only so branched traffic does not mix into first paint. Compact
    ``turn_samples`` let the report JS switch among all / parents / subagents.

    :param benchmark: Benchmark whose requests may include multi-turn metadata
    :return: Bundle with ``rows``, ``turn_agents``, ``turn_samples``, and agent
        filter metadata, or ``None`` when there is only a single turn (or no data)
    """
    successful = benchmark.requests.successful or []
    if not successful:
        return None

    turn_samples = [_turn_sample_from_request(req) for req in successful]
    if len({sample["turn_index"] for sample in turn_samples}) <= 1:
        return None

    by_agent_count: dict[str | None, int] = defaultdict(int)
    for sample in turn_samples:
        by_agent_count[sample["agent_id"]] += 1

    def _agent_sort_key(key: str | None) -> tuple[int, str]:
        # Higher count first; then lexicographic label for stability.
        return (-by_agent_count[key], _agent_display_label(key))

    turn_agents = [
        {
            "id": agent_id,
            "label": _agent_display_label(agent_id),
            "count": by_agent_count[agent_id],
            "is_subagent": _is_subagent(agent_id),
        }
        for agent_id in sorted(by_agent_count.keys(), key=_agent_sort_key)
    ]

    has_subagents = any(agent["is_subagent"] for agent in turn_agents)
    has_parents = any(not agent["is_subagent"] for agent in turn_agents)
    total_successful = len(successful)

    # Prefer parents on first paint when both categories exist; otherwise all.
    if has_subagents and has_parents:
        default_samples = [
            sample for sample in turn_samples if not _is_subagent(sample["agent_id"])
        ]
        default_mode = "parents"
        default_note = (
            "Turn curves use parent agents only "
            f"({len(default_samples)}/{total_successful} requests)."
        )
    else:
        default_samples = turn_samples
        default_mode = "all"
        default_note = (
            "Turn curves use all requests "
            f"({total_successful}/{total_successful} requests)."
        )

    rows = _aggregate_turn_rows(default_samples)

    return {
        "rows": rows,
        "turn_agents": turn_agents,
        "turn_samples": turn_samples,
        "has_subagents": has_subagents,
        "has_parents": has_parents,
        "default_mode": default_mode,
        "default_note": default_note,
        "agent_count": len(by_agent_count),
        "agent_request_count": len(default_samples),
        "total_successful": total_successful,
    }


def _side_stats(metric_obj: Any, side: str) -> dict[str, Any] | None:
    if metric_obj is None:
        return None
    dist = metric_obj.input if side == "input" else metric_obj.output
    successful = _select_distribution(dist, "successful")
    if successful is None:
        return None
    mean_val = successful.mean
    p95_val = _percentile_from_dist(successful, "p95")
    if mean_val is None and p95_val is None:
        return None
    count = successful.count
    if count <= 0 and not mean_val:
        return None
    return {"mean": mean_val, "p95": p95_val, "count": count}


def _extract_modality_stats(metrics: GenerativeMetrics) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality, groups in _MODALITY_GROUPS.items():
        modality_obj = _select_modality(metrics, modality)
        if modality_obj is None:
            continue
        modality_data: dict[str, Any] = {}
        for attr, _label in groups:
            metric_obj = _select_modality_metric(modality_obj, attr)
            if metric_obj is None:
                continue
            entry: dict[str, Any] = {}
            for side in ("input", "output"):
                side_entry = _side_stats(metric_obj, side)
                if side_entry is not None:
                    entry[side] = side_entry
            if entry:
                modality_data[attr] = entry
        if modality_data:
            result[modality] = modality_data
    return result


def _peak_throughput_index(runs: Sequence[dict[str, Any]]) -> int:
    if not runs:
        return 0
    best_index = 0
    best_tps = float("-inf")
    for index, run in enumerate(runs):
        tps = run.get("total_tps")
        value = float(tps) if tps is not None else float("-inf")
        if value > best_tps:
            best_tps = value
            best_index = index
    return best_index


def _select_modality(
    metrics: GenerativeMetrics,
    modality: str,
) -> (
    GenerativeTextMetricsSummary
    | GenerativeImageMetricsSummary
    | GenerativeVideoMetricsSummary
    | GenerativeAudioMetricsSummary
    | GenerativeToolCallMetricsSummary
    | None
):
    match modality:
        case "text":
            return metrics.text
        case "image":
            return metrics.image
        case "video":
            return metrics.video
        case "audio":
            return metrics.audio
        case "tool_call":
            return metrics.tool_call
        case _:
            return None


def _select_modality_metric(
    modality: (
        GenerativeTextMetricsSummary
        | GenerativeImageMetricsSummary
        | GenerativeVideoMetricsSummary
        | GenerativeAudioMetricsSummary
        | GenerativeToolCallMetricsSummary
    ),
    attr: str,
) -> GenerativeMetricsSummary | None:
    if isinstance(modality, GenerativeTextMetricsSummary):
        return _select_text_metric(modality, attr)

    if isinstance(modality, GenerativeImageMetricsSummary):
        return _select_image_metric(modality, attr)

    if isinstance(modality, GenerativeVideoMetricsSummary):
        return _select_video_metric(modality, attr)

    if isinstance(modality, GenerativeAudioMetricsSummary):
        return _select_audio_metric(modality, attr)

    return _select_tool_call_metric(modality, attr)


def _select_text_metric(
    modality: GenerativeTextMetricsSummary,
    attr: str,
) -> GenerativeMetricsSummary | None:
    match attr:
        case "tokens":
            return modality.tokens
        case "words":
            return modality.words
        case "characters":
            return modality.characters
        case _:
            return None


def _select_image_metric(
    modality: GenerativeImageMetricsSummary,
    attr: str,
) -> GenerativeMetricsSummary | None:
    match attr:
        case "tokens":
            return modality.tokens
        case "images":
            return modality.images
        case "pixels":
            return modality.pixels
        case "bytes":
            return modality.bytes
        case _:
            return None


def _select_video_metric(
    modality: GenerativeVideoMetricsSummary,
    attr: str,
) -> GenerativeMetricsSummary | None:
    match attr:
        case "tokens":
            return modality.tokens
        case "frames":
            return modality.frames
        case "seconds":
            return modality.seconds
        case "bytes":
            return modality.bytes
        case _:
            return None


def _select_audio_metric(
    modality: GenerativeAudioMetricsSummary,
    attr: str,
) -> GenerativeMetricsSummary | None:
    match attr:
        case "tokens":
            return modality.tokens
        case "samples":
            return modality.samples
        case "seconds":
            return modality.seconds
        case "bytes":
            return modality.bytes
        case _:
            return None


def _select_tool_call_metric(
    modality: GenerativeToolCallMetricsSummary,
    attr: str,
) -> GenerativeMetricsSummary | None:
    match attr:
        case "tokens":
            return modality.tokens
        case "mixed_tokens":
            return modality.mixed_tokens
        case "count":
            return modality.count
        case _:
            return None


def _select_distribution(
    metric: StatusDistributionSummary | DistributionSummary | None,
    status: _StatusName,
) -> DistributionSummary | None:
    if metric is None:
        return None
    if isinstance(metric, DistributionSummary):
        return metric
    if status == "successful":
        return metric.successful
    if status == "incomplete":
        return metric.incomplete
    if status == "errored":
        return metric.errored
    return metric.total


def _mean(
    metric: StatusDistributionSummary | DistributionSummary | None,
    *,
    status: _StatusName = "successful",
) -> float | None:
    distribution = _select_distribution(metric, status)
    value = distribution.mean if distribution is not None else None
    return float(value) if value is not None else None


def _median(
    metric: StatusDistributionSummary | DistributionSummary | None,
    *,
    status: _StatusName = "successful",
) -> float | None:
    distribution = _select_distribution(metric, status)
    value = distribution.median if distribution is not None else None
    return float(value) if value is not None else None


def _percentile(
    metric: StatusDistributionSummary | DistributionSummary | None,
    name: Literal["p50", "p90", "p95", "p99"],
    *,
    status: _StatusName = "successful",
) -> float | None:
    distribution = _select_distribution(metric, status)
    return _percentile_from_dist(distribution, name)


def _percentile_from_dist(
    distribution: DistributionSummary | None,
    name: Literal["p50", "p90", "p95", "p99"],
) -> float | None:
    if distribution is None:
        return None
    percentiles = distribution.percentiles
    if name == "p99":
        value = percentiles.p99
    elif name == "p95":
        value = percentiles.p95
    elif name == "p90":
        value = percentiles.p90
    else:
        value = percentiles.p50
    return float(value) if value is not None else None


def _status_total(
    totals: StatusBreakdown[int, int, int, int] | None,
    status: _StatusName,
) -> float:
    if totals is None:
        return 0.0
    if status == "successful":
        value = totals.successful
    elif status == "incomplete":
        value = totals.incomplete
    elif status == "errored":
        value = totals.errored
    else:
        value = totals.total
    return float(value) if value is not None else 0.0


def _ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return float(seconds) * 1000.0


def _ttfot_differs(ttft: float | None, ttfot: float | None) -> bool:
    if ttft is None or ttfot is None:
        return False
    if ttft == 0 and ttfot == 0:
        return False
    return not math.isclose(float(ttft), float(ttfot), rel_tol=0.01, abs_tol=0.5)


def _safe_median(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return float(statistics.median(clean))


def _safe_percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    # Match run-level DistributionSummary percentiles (empirical CDF, no interpolation).
    name: Literal["p50", "p90", "p95", "p99"]
    if quantile == 0.99:  # noqa: PLR2004
        name = "p99"
    elif quantile == 0.95:  # noqa: PLR2004
        name = "p95"
    elif quantile == 0.90:  # noqa: PLR2004
        name = "p90"
    else:
        name = "p50"
    return _percentile_from_dist(DistributionSummary.from_values(clean), name)
