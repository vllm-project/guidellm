"""
Plot output formatter for benchmark results.

This module provides the GenerativeBenchmarkerPlot class which exports benchmark
reports to a static PNG image format containing comprehensive performance
visualization charts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import (
    BenchmarkOutputArgs,
    GenerativeBenchmarksReport,
)
from guidellm.benchmark.schemas.output import (
    ALLOWED_PLOT_SUFFIXES,
    PlotBenchmarkOutputArgs,
)
from guidellm.extras import plot

__all__ = [
    "GenerativeBenchmarkerPlot",
]


_StatusName = Literal["successful", "incomplete", "errored", "total"]
_PlotFunction = Callable[["plot.Axes", Sequence["_BenchmarkPoint"]], None]

_BLUE = "#0077b6"
_RED = "#d00000"
_PURPLE = "#7b2cbf"
_GREEN = "#2d6a4f"
_ORANGE = "#f77f00"
_PINK = "#c9184a"
_GRAY = "#6c757d"


@dataclass(frozen=True)
class _BenchmarkPoint:
    request_rate: float
    concurrency: float
    total_tps: float
    input_tps: float
    output_tps: float
    ttft_median_ms: float
    ttft_p95_ms: float
    ttfo_median_ms: float
    ttfo_p95_ms: float
    e2e_median_ms: float
    e2e_p95_ms: float
    tpot_median_ms: float
    tpot_p95_ms: float
    itl_median_ms: float
    itl_p95_ms: float
    successful_requests: float
    incomplete_requests: float
    errored_requests: float
    total_requests: float


def _select_distribution(metric: Any, status: _StatusName) -> Any:
    if metric is None:
        return None
    if status == "successful":
        return metric.successful
    if status == "incomplete":
        return metric.incomplete
    if status == "errored":
        return metric.errored
    return metric.total


def _get_val(
    metric: Any,
    is_median: bool = False,
    default: float = 0.0,
    status: _StatusName = "successful",
) -> float:
    """
    Get the mean or median value of a distribution summary.

    :param metric: Metric object containing the distribution summary
    :param is_median: Whether to return the median value instead of the mean
    :param default: Default value to return if the metric or value is missing
    :param status: Request status distribution to read
    :return: The extracted metric value as a float
    """
    distribution = _select_distribution(metric, status)
    if distribution is None:
        return default

    value = distribution.median if is_median else distribution.mean
    return float(value) if value is not None else default


def _get_percentile(
    metric: Any,
    p_name: Literal["p50", "p90", "p95"] = "p95",
    default: float = 0.0,
    status: _StatusName = "successful",
) -> float:
    """
    Get a specific percentile value of a distribution summary.

    :param metric: Metric object containing the distribution summary
    :param p_name: Name of the percentile to get
    :param default: Default value to return if the metric or percentile is missing
    :param status: Request status distribution to read
    :return: The extracted percentile value as a float
    """
    distribution = _select_distribution(metric, status)
    if distribution is None or distribution.percentiles is None:
        return default

    percentiles = distribution.percentiles
    if p_name == "p95":
        value = percentiles.p95
    elif p_name == "p90":
        value = percentiles.p90
    else:
        value = percentiles.p50
    return float(value) if value is not None else default


def _get_status_total(totals: Any, status: _StatusName) -> float:
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


def _build_points(benchmarks: Sequence[Any]) -> list[_BenchmarkPoint]:
    points = []
    for benchmark in benchmarks:
        metrics = benchmark.metrics
        request_totals = metrics.request_totals
        points.append(
            _BenchmarkPoint(
                request_rate=_get_val(
                    metrics.requests_per_second,
                    status="total",
                ),
                concurrency=_get_val(
                    metrics.request_concurrency,
                    status="total",
                ),
                total_tps=_get_val(metrics.tokens_per_second, status="total"),
                input_tps=_get_val(metrics.prompt_tokens_per_second, status="total"),
                output_tps=_get_val(metrics.output_tokens_per_second, status="total"),
                ttft_median_ms=_get_val(
                    metrics.time_to_first_token_ms,
                    is_median=True,
                ),
                ttft_p95_ms=_get_percentile(metrics.time_to_first_token_ms),
                ttfo_median_ms=_get_val(
                    metrics.time_to_first_output_token_ms,
                    is_median=True,
                ),
                ttfo_p95_ms=_get_percentile(metrics.time_to_first_output_token_ms),
                e2e_median_ms=_get_val(
                    metrics.request_latency,
                    is_median=True,
                )
                * 1000.0,
                e2e_p95_ms=_get_percentile(metrics.request_latency) * 1000.0,
                tpot_median_ms=_get_val(
                    metrics.time_per_output_token_ms,
                    is_median=True,
                ),
                tpot_p95_ms=_get_percentile(metrics.time_per_output_token_ms),
                itl_median_ms=_get_val(
                    metrics.inter_token_latency_ms,
                    is_median=True,
                ),
                itl_p95_ms=_get_percentile(metrics.inter_token_latency_ms),
                successful_requests=_get_status_total(request_totals, "successful"),
                incomplete_requests=_get_status_total(request_totals, "incomplete"),
                errored_requests=_get_status_total(request_totals, "errored"),
                total_requests=_get_status_total(request_totals, "total"),
            )
        )

    return points


def _sort_by_request_rate(points: Sequence[_BenchmarkPoint]) -> list[_BenchmarkPoint]:
    return sorted(points, key=lambda point: point.request_rate)


def _sort_by_concurrency(points: Sequence[_BenchmarkPoint]) -> list[_BenchmarkPoint]:
    return sorted(points, key=lambda point: point.concurrency)


def _sort_by_throughput(points: Sequence[_BenchmarkPoint]) -> list[_BenchmarkPoint]:
    return sorted(points, key=lambda point: point.total_tps)


def _series(
    points: Sequence[_BenchmarkPoint],
    extractor: Callable[[_BenchmarkPoint], float],
) -> list[float]:
    return [extractor(point) for point in points]


def _filter_series(
    x_values: Sequence[float],
    y_values: Sequence[float],
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[list[float], list[float]]:
    filtered_x = []
    filtered_y = []

    for x_value, y_value in zip(x_values, y_values, strict=True):
        if log_x and x_value <= 0.0:
            continue
        if log_y and y_value <= 0.0:
            continue
        filtered_x.append(x_value)
        filtered_y.append(y_value)

    return filtered_x, filtered_y


def _plot_line(
    ax: plot.Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    log_x: bool = False,
    log_y: bool = False,
    **kwargs: Any,
) -> None:
    x_data, y_data = _filter_series(x_values, y_values, log_x, log_y)
    if x_data and y_data:
        ax.plot(x_data, y_data, **kwargs)


def _set_log_scale(
    ax: plot.Axes, axis: Literal["x", "y"], values: Sequence[float]
) -> None:
    if not any(value > 0.0 for value in values):
        return
    if axis == "x":
        ax.set_xscale("log")
    else:
        ax.set_yscale("log")


def _set_title(ax: plot.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, color="black", weight="bold")


def _show_legend(ax: plot.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(facecolor="white", edgecolor="#cccccc")


def _label_for_concurrency(point: _BenchmarkPoint, index: int) -> str:
    if point.concurrency > 0.0:
        return f"{point.concurrency:g}"
    return str(index + 1)


def _style_axis(ax: plot.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#cccccc", linestyle=":", alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#999999")


def _set_plot_theme(dpi: int) -> None:
    plot.use("Agg")
    plot.plt.rcParams["figure.dpi"] = dpi
    plot.plt.rcParams["font.family"] = "sans-serif"
    plot.plt.rcParams["text.color"] = "black"
    plot.plt.rcParams["axes.labelcolor"] = "black"
    plot.plt.rcParams["xtick.color"] = "black"
    plot.plt.rcParams["ytick.color"] = "black"


def plot_latency_vs_request_rate(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot TTFT and end-to-end latency against observed request rate.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_request_rate(points)
    request_rate = _series(sorted_points, lambda point: point.request_rate)

    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.ttft_median_ms),
        log_x=True,
        log_y=True,
        marker="o",
        label="TTFT Median",
        color=_BLUE,
        linewidth=2,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.ttft_p95_ms),
        log_x=True,
        log_y=True,
        marker="^",
        linestyle="--",
        label="TTFT p95",
        color=_BLUE,
        alpha=0.7,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.e2e_median_ms),
        log_x=True,
        log_y=True,
        marker="s",
        label="E2E Median",
        color=_RED,
        linewidth=2,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.e2e_p95_ms),
        log_x=True,
        log_y=True,
        marker="d",
        linestyle="--",
        label="E2E p95",
        color=_RED,
        alpha=0.7,
    )
    _set_log_scale(ax, "x", request_rate)
    _set_log_scale(ax, "y", _series(sorted_points, lambda point: point.e2e_p95_ms))
    _set_title(ax, "1. Latency (TTFT & E2E) vs. Request Rate\n(lower is better)")
    ax.set_xlabel("Request Rate (RPS, Log Scale)")
    ax.set_ylabel("Latency (ms, Log Scale)")
    _show_legend(ax)


def plot_generation_latency_vs_request_rate(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot per-token generation latency against observed request rate.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_request_rate(points)
    request_rate = _series(sorted_points, lambda point: point.request_rate)

    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.tpot_median_ms),
        log_x=True,
        log_y=True,
        marker="o",
        label="TPOT Median",
        color=_PURPLE,
        linewidth=2,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.tpot_p95_ms),
        log_x=True,
        log_y=True,
        marker="^",
        linestyle="--",
        label="TPOT p95",
        color=_PURPLE,
        alpha=0.7,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.itl_median_ms),
        log_x=True,
        log_y=True,
        marker="s",
        label="ITL Median",
        color=_GREEN,
        linewidth=2,
    )
    _plot_line(
        ax,
        request_rate,
        _series(sorted_points, lambda point: point.itl_p95_ms),
        log_x=True,
        log_y=True,
        marker="d",
        linestyle="--",
        label="ITL p95",
        color=_GREEN,
        alpha=0.7,
    )
    _set_log_scale(ax, "x", request_rate)
    _set_log_scale(ax, "y", _series(sorted_points, lambda point: point.tpot_p95_ms))
    _set_title(
        ax,
        "2. Generation Speed (TPOT & ITL) vs. Request Rate\n(lower is better)",
    )
    ax.set_xlabel("Request Rate (RPS, Log Scale)")
    ax.set_ylabel("Latency (ms/token, Log Scale)")
    _show_legend(ax)


def plot_token_throughput_vs_concurrency(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot total, input, and output token throughput against request concurrency.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_concurrency(points)
    concurrency = _series(sorted_points, lambda point: point.concurrency)

    _plot_line(
        ax,
        concurrency,
        _series(sorted_points, lambda point: point.total_tps),
        log_x=True,
        log_y=True,
        marker="o",
        label="Total Tokens/s",
        color=_ORANGE,
        linewidth=2.5,
    )
    _plot_line(
        ax,
        concurrency,
        _series(sorted_points, lambda point: point.input_tps),
        log_x=True,
        log_y=True,
        marker="s",
        label="Input Tokens/s",
        color=_BLUE,
        alpha=0.8,
    )
    _plot_line(
        ax,
        concurrency,
        _series(sorted_points, lambda point: point.output_tps),
        log_x=True,
        log_y=True,
        marker="d",
        label="Output Tokens/s",
        color=_PINK,
        alpha=0.8,
    )
    _set_log_scale(ax, "x", concurrency)
    _set_log_scale(ax, "y", _series(sorted_points, lambda point: point.total_tps))
    _set_title(ax, "3. Token Throughput vs. Concurrency\n(higher is better)")
    ax.set_xlabel("Concurrency (Active Requests, Log Scale)")
    ax.set_ylabel("Throughput (Tokens/sec, Log Scale)")
    _show_legend(ax)


def plot_latency_vs_throughput(
    ax: plot.Axes, points: Sequence[_BenchmarkPoint]
) -> None:
    """
    Plot first-token latency against token throughput to expose the saturation knee.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_throughput(points)
    throughput = _series(sorted_points, lambda point: point.total_tps)

    _plot_line(
        ax,
        throughput,
        _series(sorted_points, lambda point: point.ttft_p95_ms),
        log_y=True,
        marker="o",
        color=_RED,
        linewidth=2,
        label="p95 TTFT",
    )
    _plot_line(
        ax,
        throughput,
        _series(sorted_points, lambda point: point.ttft_median_ms),
        log_y=True,
        marker="s",
        color=_BLUE,
        linewidth=2,
        label="Median TTFT",
    )
    _set_log_scale(ax, "y", _series(sorted_points, lambda point: point.ttft_p95_ms))
    _set_title(
        ax,
        "4. Knee Plot: Latency vs. Throughput\n"
        "(lower latency + higher throughput is better)",
    )
    ax.set_xlabel("Throughput (Total Tokens/sec)")
    ax.set_ylabel("Latency (ms)")
    _show_legend(ax)


def plot_request_status_counts(
    ax: plot.Axes, points: Sequence[_BenchmarkPoint]
) -> None:
    """
    Plot request status counts by concurrency level.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_concurrency(points)
    x_positions = list(range(len(sorted_points)))
    successful = _series(sorted_points, lambda point: point.successful_requests)
    incomplete = _series(sorted_points, lambda point: point.incomplete_requests)
    errored = _series(sorted_points, lambda point: point.errored_requests)

    ax.bar(x_positions, successful, color=_BLUE, label="Successful")
    ax.bar(
        x_positions, incomplete, bottom=successful, color=_ORANGE, label="Incomplete"
    )
    errored_bottom = [
        successful_count + incomplete_count
        for successful_count, incomplete_count in zip(
            successful, incomplete, strict=True
        )
    ]
    ax.bar(x_positions, errored, bottom=errored_bottom, color=_RED, label="Errored")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            _label_for_concurrency(point, index)
            for index, point in enumerate(sorted_points)
        ],
        rotation=35,
        ha="right",
    )
    _set_title(
        ax, "5. Request Status Counts vs. Concurrency\n(more complete is better)"
    )
    ax.set_xlabel("Concurrency (Active Requests)")
    ax.set_ylabel("Requests")
    _show_legend(ax)


def plot_latency_breakdown_at_peak_throughput(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot latency component medians and p95s for the highest-throughput benchmark.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    peak_point = max(points, key=lambda point: point.total_tps)
    labels = ["TTFT", "TTFOT", "ITL", "TPOT", "E2E"]
    medians = [
        peak_point.ttft_median_ms,
        peak_point.ttfo_median_ms,
        peak_point.itl_median_ms,
        peak_point.tpot_median_ms,
        peak_point.e2e_median_ms,
    ]
    p95s = [
        peak_point.ttft_p95_ms,
        peak_point.ttfo_p95_ms,
        peak_point.itl_p95_ms,
        peak_point.tpot_p95_ms,
        peak_point.e2e_p95_ms,
    ]
    x_positions = list(range(len(labels)))
    bar_width = 0.38

    ax.bar(
        [position - bar_width / 2 for position in x_positions],
        medians,
        width=bar_width,
        color=_BLUE,
        label="Median",
    )
    ax.bar(
        [position + bar_width / 2 for position in x_positions],
        p95s,
        width=bar_width,
        color=_RED,
        alpha=0.8,
        label="p95",
    )
    _set_log_scale(ax, "y", p95s)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    _set_title(
        ax,
        "6. Latency Breakdown at Peak Throughput\n(lower is better)",
    )
    ax.set_xlabel("Latency Component")
    ax.set_ylabel("Latency (ms, Log Scale)")
    _show_legend(ax)


def plot_throughput_efficiency_vs_concurrency(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot total token throughput per concurrent request.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_concurrency(points)
    concurrency = _series(sorted_points, lambda point: point.concurrency)
    efficiency = [
        point.total_tps / point.concurrency if point.concurrency > 0.0 else 0.0
        for point in sorted_points
    ]

    _plot_line(
        ax,
        concurrency,
        efficiency,
        log_x=True,
        marker="o",
        color=_GREEN,
        linewidth=2,
        label="Tokens/s per Active Request",
    )
    _set_log_scale(ax, "x", concurrency)
    _set_title(
        ax,
        "7. Throughput Efficiency vs. Concurrency\n(higher is better)",
    )
    ax.set_xlabel("Concurrency (Active Requests, Log Scale)")
    ax.set_ylabel("Total Tokens/sec per Active Request")
    _show_legend(ax)


def plot_token_throughput_mix_vs_request_rate(
    ax: plot.Axes,
    points: Sequence[_BenchmarkPoint],
) -> None:
    """
    Plot input and output token throughput mix against request rate.

    :param ax: Matplotlib axis to draw on
    :param points: Benchmark points extracted from the report
    """
    sorted_points = _sort_by_request_rate(points)
    request_rate = _series(sorted_points, lambda point: point.request_rate)
    input_tps = _series(sorted_points, lambda point: point.input_tps)
    output_tps = _series(sorted_points, lambda point: point.output_tps)
    x_data, input_data = _filter_series(request_rate, input_tps, log_x=True)
    _x_data, output_data = _filter_series(request_rate, output_tps, log_x=True)

    if x_data and input_data and output_data:
        ax.stackplot(
            x_data,
            input_data,
            output_data,
            labels=["Input Tokens/s", "Output Tokens/s"],
            colors=[_BLUE, _PINK],
            alpha=0.75,
        )
        _plot_line(
            ax,
            request_rate,
            _series(sorted_points, lambda point: point.total_tps),
            log_x=True,
            color=_ORANGE,
            linewidth=2,
            label="Total Tokens/s",
        )

    _set_log_scale(ax, "x", request_rate)
    _set_title(
        ax,
        "8. Token Throughput Mix vs. Request Rate\n(higher is better)",
    )
    ax.set_xlabel("Request Rate (RPS, Log Scale)")
    ax.set_ylabel("Throughput (Tokens/sec)")
    _show_legend(ax)


_PLOT_FUNCTIONS: tuple[_PlotFunction, ...] = (
    plot_latency_vs_request_rate,
    plot_generation_latency_vs_request_rate,
    plot_token_throughput_vs_concurrency,
    plot_latency_vs_throughput,
    plot_request_status_counts,
    plot_latency_breakdown_at_peak_throughput,
    plot_throughput_efficiency_vs_concurrency,
    plot_token_throughput_mix_vs_request_rate,
)


@GenerativeBenchmarkerOutput.register("plot")
class GenerativeBenchmarkerPlot(GenerativeBenchmarkerOutput):
    """
    Plot output formatter for benchmark results.

    Generates a high-quality dashboard visualization of LLM benchmark results,
    enforcing light theme and saving to a PNG image file.
    """

    output_path: Path = Field(
        default_factory=Path.cwd,
        description=(
            "Path where the PNG plot file will be saved, defaults to current directory"
        ),
    )
    dpi: int = Field(
        default=100,
        description="Resolution of the output image in Dots Per Inch.",
    )

    @classmethod
    def from_args(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerPlot:
        """
        Create a plot output formatter from output arguments.

        :param args: Output configuration with path and dpi
        :return: Configured plot output formatter
        """
        if not isinstance(args, PlotBenchmarkOutputArgs):
            raise ValueError(f"Expected PlotBenchmarkOutputArgs, got {type(args)}")

        return cls(output_path=args.path, dpi=args.dpi)

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as a static PNG plot visualization.

        :param report: The completed benchmark report
        :return: Path to the saved PNG plot file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / "benchmarks.png"
        elif output_path.suffix.lower() not in ALLOWED_PLOT_SUFFIXES:
            output_path = output_path.with_suffix(".png")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not report.benchmarks:
            fig, ax = plot.plt.subplots(figsize=(10, 8), facecolor="white")
            ax.text(0.5, 0.5, "No benchmark data available", fontsize=14, ha="center")
            ax.set_facecolor("white")
            plot.plt.savefig(
                output_path, facecolor="white", bbox_inches="tight", dpi=self.dpi
            )
            plot.plt.close(fig)
            return output_path

        points = _build_points(report.benchmarks)

        _set_plot_theme(self.dpi)

        fig, axs = plot.plt.subplots(4, 2, figsize=(16, 22), facecolor="white")
        fig.suptitle(
            "GuideLLM Benchmark Performance Visualization",
            fontsize=18,
            color="black",
            weight="bold",
            y=0.985,
        )

        for ax in axs.flat:
            _style_axis(ax)

        for ax, plot_function in zip(axs.flat, _PLOT_FUNCTIONS, strict=True):
            plot_function(ax, points)

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        plot.plt.tight_layout(rect=(0.04, 0.03, 0.96, 0.965), h_pad=3.0, w_pad=2.5)
        fig.savefig(output_path, facecolor="white", bbox_inches="tight", dpi=self.dpi)
        plot.plt.close(fig)

        return output_path
