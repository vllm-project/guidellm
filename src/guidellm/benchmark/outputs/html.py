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
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, Literal

from loguru import logger
from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.schemas.benchmark import BenchmarkOutputArgs
from guidellm.schemas.benchmark.outputs import HTMLBenchmarkOutputArgs

__all__ = [
    "GenerativeBenchmarkerHTML",
    "HTMLBenchmarkOutputArgs",
    "build_report_view",
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
    return (
        template.replace("__GUIDELLM_REPORT_CSS__", css)
        .replace("__GUIDELLM_REPORT_JS__", js)
        .replace("__GUIDELLM_REPORT_JSON__", payload)
    )


def build_report_view(report: Any) -> dict[str, Any]:
    """
    Build the compact JSON view embedded in the HTML report.

    :param report: Benchmark report to summarize (duck-typed for test doubles)
    :return: Dictionary consumed by the inlined report JavaScript
    """
    benchmarks = list(getattr(report, "benchmarks", []) or [])
    runs = [
        _build_run_row(benchmark, index) for index, benchmark in enumerate(benchmarks)
    ]
    peak_index = _peak_throughput_index(runs)
    peak_run = runs[peak_index] if runs else None

    by_turn: list[dict[str, Any]] | None = None
    turn_note: str | None = None
    if benchmarks:
        source = benchmarks[peak_index]
        by_turn = _build_metrics_by_turn(source)
        if by_turn and len(benchmarks) > 1:
            turn_note = (
                "Turn curves use the peak-throughput run "
                f"(#{peak_index + 1}: {peak_run['strategy'] if peak_run else 'n/a'})."
            )

    header = _build_header(report, runs, peak_index, has_multi_turn=bool(by_turn))
    kpis = _build_kpis(peak_run) if peak_run else {}
    details = _build_details(benchmarks, runs)

    return {
        "header": header,
        "kpis": kpis,
        "runs": runs,
        "by_turn": by_turn,
        "turn_note": turn_note,
        "details": details,
    }


def _build_header(
    report: Any,
    runs: Sequence[dict[str, Any]],
    peak_index: int,
    *,
    has_multi_turn: bool,
) -> dict[str, Any]:
    config = getattr(report, "config", None)
    spec = getattr(config, "spec", config)
    backend = getattr(spec, "backend", None) if spec is not None else None
    profile = getattr(spec, "profile", None) if spec is not None else None
    metadata = getattr(report, "metadata", None)

    timestamp = None
    for benchmark in getattr(report, "benchmarks", []) or []:
        start = getattr(benchmark, "start_time", None)
        if start is not None:
            timestamp = max(timestamp or 0.0, float(start))
    timestamp_iso = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp
        else None
    )

    return {
        "model": getattr(backend, "model", None) or "N/A",
        "target": getattr(backend, "target", None) or "N/A",
        "profile": getattr(profile, "kind", None) or "N/A",
        "timestamp": timestamp_iso,
        "guidellm_version": getattr(metadata, "guidellm_version", None) or "N/A",
        "peak_index": peak_index,
        "multi_run": len(runs) > 1,
        "has_multi_turn": has_multi_turn,
        "run_count": len(runs),
    }


def _build_kpis(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "requests_per_second": run.get("request_rate"),
        "output_tokens_per_second": run.get("output_tps"),
        "tokens_per_second": run.get("total_tps"),
        "request_latency_p95_ms": run.get("request_latency_p95_ms"),
        "request_latency_p99_ms": run.get("request_latency_p99_ms"),
        "ttft_p95_ms": run.get("ttft_p95_ms"),
        "ttft_p99_ms": run.get("ttft_p99_ms"),
        "itl_p95_ms": run.get("itl_p95_ms"),
        "itl_p99_ms": run.get("itl_p99_ms"),
        "error_rate": run.get("error_rate"),
        "strategy": run.get("label") or run.get("strategy"),
    }


def _build_run_row(benchmark: Any, index: int) -> dict[str, Any]:
    metrics = getattr(benchmark, "metrics", None)
    config = getattr(benchmark, "config", None)
    strategy_obj = getattr(config, "strategy", None) if config is not None else None
    if strategy_obj is not None:
        label = str(strategy_obj)
        strategy_name = getattr(strategy_obj, "type_", None) or label
    else:
        label = f"run_{index}"
        strategy_name = label

    request_totals = getattr(metrics, "request_totals", None)
    successful = _status_total(request_totals, "successful")
    incomplete = _status_total(request_totals, "incomplete")
    errored = _status_total(request_totals, "errored")
    total = _status_total(request_totals, "total") or (
        successful + incomplete + errored
    )
    error_rate = (errored / total) if total else 0.0

    # Request latency is stored in seconds; convert to ms for chart consistency.
    req_lat_p95 = _percentile(metrics, "request_latency", "p95")
    req_lat_p99 = _percentile(metrics, "request_latency", "p99")
    concurrency = _mean(metrics, "request_concurrency", status="total")
    # Intended parallel-request cap from the strategy (e.g. streams); None = unlimited.
    configured_concurrency = (
        getattr(strategy_obj, "requests_limit", None)
        if strategy_obj is not None
        else None
    )

    return {
        "index": index,
        "strategy": strategy_name,
        "label": label,
        "request_rate": _mean(metrics, "requests_per_second", status="total"),
        "concurrency": concurrency,
        "configured_concurrency": configured_concurrency,
        "total_tps": _mean(metrics, "tokens_per_second", status="total"),
        "input_tps": _mean(metrics, "prompt_tokens_per_second", status="total"),
        "output_tps": _mean(metrics, "output_tokens_per_second", status="total"),
        "request_latency_p95_ms": _ms(req_lat_p95),
        "request_latency_p99_ms": _ms(req_lat_p99),
        "ttft_p95_ms": _percentile(metrics, "time_to_first_token_ms", "p95"),
        "ttft_p99_ms": _percentile(metrics, "time_to_first_token_ms", "p99"),
        "ttfot_p95_ms": _percentile(metrics, "time_to_first_output_token_ms", "p95"),
        "ttfot_p99_ms": _percentile(metrics, "time_to_first_output_token_ms", "p99"),
        "itl_p95_ms": _percentile(metrics, "inter_token_latency_ms", "p95"),
        "itl_p99_ms": _percentile(metrics, "inter_token_latency_ms", "p99"),
        "tpot_p95_ms": _percentile(metrics, "time_per_output_token_ms", "p95"),
        "tpot_p99_ms": _percentile(metrics, "time_per_output_token_ms", "p99"),
        "prompt_tokens_median": _median(metrics, "prompt_token_count"),
        "prompt_tokens_p95": _percentile(metrics, "prompt_token_count", "p95"),
        "output_tokens_median": _median(metrics, "output_token_count"),
        "output_tokens_p95": _percentile(metrics, "output_token_count", "p95"),
        "rtt_avg_p95_ms": _percentile(metrics, "avg_round_trip_time_ms", "p95"),
        "rtt_avg_p99_ms": _percentile(metrics, "avg_round_trip_time_ms", "p99"),
        "rtt_last_p95_ms": _percentile(metrics, "time_to_last_round_trip_ms", "p95"),
        "rtt_last_p99_ms": _percentile(metrics, "time_to_last_round_trip_ms", "p99"),
        "successful": successful,
        "incomplete": incomplete,
        "errored": errored,
        "total": total,
        "error_rate": error_rate,
        "modalities": _extract_modality_stats(metrics),
    }


def _build_details(
    benchmarks: Sequence[Any],
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
            for run in runs:
                mod = (run.get("modalities") or {}).get(modality) or {}
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


def _build_metrics_by_turn(benchmark: Any) -> list[dict[str, Any]] | None:
    """
    Aggregate successful request metrics by ``info.turn_index``.

    :param benchmark: Benchmark whose requests may include multi-turn metadata
    :return: Per-turn rows, or ``None`` when there is only a single turn (or no data)
    """
    requests_obj = getattr(benchmark, "requests", None)
    successful = getattr(requests_obj, "successful", None) or []
    if not successful:
        return None

    # Prefer the dominant agent_id so branched/tool graphs do not mix series.
    by_agent: dict[str | None, list[Any]] = defaultdict(list)
    for req in successful:
        info = getattr(req, "info", None)
        agent = getattr(info, "agent_id", None)
        by_agent[agent].append(req)
    preferred_agent = max(by_agent.keys(), key=lambda key: len(by_agent[key]))
    selected = by_agent[preferred_agent]

    buckets: dict[int, list[Any]] = defaultdict(list)
    history_lens: set[int] = set()
    for req in selected:
        info = getattr(req, "info", None)
        turn_index = getattr(info, "turn_index", None)
        if turn_index is None:
            continue
        buckets[int(turn_index)].append(req)
        history_len = getattr(info, "history_len", None)
        if history_len is not None:
            history_lens.add(int(history_len))

    if len(buckets) <= 1:
        return None

    rows: list[dict[str, Any]] = []
    for turn_index in sorted(buckets):
        reqs = buckets[turn_index]
        latencies_ms = [
            float(v) * 1000.0
            for v in _request_values(reqs, "request_latency")
            if v is not None
        ]
        ttfts = _request_values(reqs, "time_to_first_token_ms")
        itls = _request_values(reqs, "inter_token_latency_ms")
        prompt_tokens = _request_values(reqs, "prompt_tokens")
        history = [
            getattr(getattr(req, "info", None), "history_len", None) for req in reqs
        ]
        history_clean = [float(h) for h in history if h is not None]

        rows.append(
            {
                "turn_index": turn_index,
                "count": len(reqs),
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

    # Attach a flag when history_len is a useful alternate x-axis.
    if history_lens and history_lens != {row["turn_index"] for row in rows}:
        for row in rows:
            row["use_history_axis"] = True
    return rows


def _side_stats(metric_obj: Any, side: str) -> dict[str, Any] | None:
    dist = getattr(metric_obj, side, None)
    successful = _select_distribution(dist, "successful")
    if successful is None:
        return None
    mean_val = getattr(successful, "mean", None)
    p95_val = _percentile_from_dist(successful, "p95")
    if mean_val is None and p95_val is None:
        return None
    count = getattr(successful, "count", 0) or 0
    if count <= 0 and not mean_val:
        return None
    return {"mean": mean_val, "p95": p95_val, "count": count}


def _extract_modality_stats(metrics: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if metrics is None:
        return result

    for modality, groups in _MODALITY_GROUPS.items():
        modality_obj = getattr(metrics, modality, None)
        if modality_obj is None:
            continue
        modality_data: dict[str, Any] = {}
        for attr, _label in groups:
            metric_obj = getattr(modality_obj, attr, None)
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


def _select_distribution(metric: Any, status: _StatusName) -> Any:
    if metric is None:
        return None
    # StatusDistributionSummary exposes successful/incomplete/errored/total.
    if hasattr(metric, status):
        return getattr(metric, status)
    # Already a DistributionSummary-like object.
    if hasattr(metric, "mean") or hasattr(metric, "percentiles"):
        return metric
    return None


def _mean(
    metrics: Any,
    field_name: str,
    *,
    status: _StatusName = "successful",
) -> float | None:
    if metrics is None:
        return None
    metric = getattr(metrics, field_name, None)
    distribution = _select_distribution(metric, status)
    value = getattr(distribution, "mean", None) if distribution is not None else None
    return float(value) if value is not None else None


def _median(
    metrics: Any,
    field_name: str,
    *,
    status: _StatusName = "successful",
) -> float | None:
    if metrics is None:
        return None
    metric = getattr(metrics, field_name, None)
    distribution = _select_distribution(metric, status)
    value = getattr(distribution, "median", None) if distribution is not None else None
    return float(value) if value is not None else None


def _percentile(
    metrics: Any,
    field_name: str,
    name: Literal["p50", "p90", "p95", "p99"],
    *,
    status: _StatusName = "successful",
) -> float | None:
    if metrics is None:
        return None
    metric = getattr(metrics, field_name, None)
    distribution = _select_distribution(metric, status)
    return _percentile_from_dist(distribution, name)


def _percentile_from_dist(
    distribution: Any,
    name: Literal["p50", "p90", "p95", "p99"],
) -> float | None:
    if distribution is None:
        return None
    percentiles = getattr(distribution, "percentiles", None)
    if percentiles is None:
        return None
    value = getattr(percentiles, name, None)
    return float(value) if value is not None else None


def _status_total(totals: Any, status: _StatusName) -> float:
    if totals is None:
        return 0.0
    value = getattr(totals, status, None)
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


def _request_values(reqs: Iterable[Any], attr: str) -> list[float]:
    values: list[float] = []
    for req in reqs:
        value = getattr(req, attr, None)
        if value is None:
            continue
        values.append(float(value))
    return values


def _safe_median(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return float(statistics.median(clean))


def _safe_percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = sorted(
        float(v) for v in values if v is not None and not math.isnan(float(v))
    )
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    # Nearest-rank style percentile for small request samples.
    rank = min(len(clean) - 1, max(0, math.ceil(quantile * len(clean)) - 1))
    return clean[rank]
