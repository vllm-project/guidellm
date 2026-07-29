#!/usr/bin/env python3
"""Compare groups of GuideLLM benchmark JSON reports by filename.

Discovers report files, groups them by stripping a trailing ``_<run>`` index
from the filename (e.g. ``benchmarks_dag_1.json`` → group ``dag``), aggregates
replicate runs, and writes a multi-group CSV plus a matplotlib dashboard.

Usage::

    # Default: scan guidellm_output for benchmarks_*_<n>.json
    uv run python scripts/compare_benchmark_groups.py \\
      --baseline main \\
      --output comparison_groups

    # Explicit directory / pattern
    uv run python scripts/compare_benchmark_groups.py \\
      --dir /Users/joconnel/Documents/projects/ai/guidellm_output \\
      --pattern 'benchmarks_*_*.json' \\
      --baseline main \\
      --output comparison_groups

    # Or pass explicit file paths/globs
    uv run python scripts/compare_benchmark_groups.py \\
      /Users/joconnel/Documents/projects/ai/guidellm_output/benchmarks_*.json
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from guidellm.benchmark import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.schemas import DistributionSummary, StatusDistributionSummary

DEFAULT_DIR = Path("/Users/joconnel/Documents/projects/ai/guidellm_output")
DEFAULT_PATTERN = "benchmarks_*_*.json"
DEFAULT_STRIP_PREFIX = "benchmarks_"
SKIPPED_FILE_PREVIEW = 5
StatusName = Literal["successful", "incomplete", "errored", "total"]

# Trailing _<digits> before the extension identifies a replicate run index.
_RUN_SUFFIX_RE = re.compile(r"^(?P<label>.+)_(?P<run>\d+)$")

# Metric display names in CSV row order (matches comparison.csv layout).
METRIC_ORDER: list[str] = [
    "Duration (s)",
    "Successful Requests",
    "Errored Requests",
    "Total Requests",
    "Request Latency Median (s)",
    "Request Latency Mean (s)",
    "Request Latency p95 (s)",
    "TTFT Median (ms)",
    "TTFT Mean (ms)",
    "TTFT p95 (ms)",
    "ITL Median (ms)",
    "ITL Mean (ms)",
    "ITL p95 (ms)",
    "TPOT Median (ms)",
    "TPOT Mean (ms)",
    "TPOT p95 (ms)",
    "Requests/sec Mean",
    "Request Concurrency Median",
    "Request Concurrency Mean",
    "Input Tokens/sec Mean",
    "Output Tokens/sec Mean",
    "Total Tokens/sec Mean",
    "Input Tokens/req Median",
    "Input Tokens/req p95",
    "Output Tokens/req Median",
    "Output Tokens/req p95",
]

# Key metrics used on the plot dashboard (subset of METRIC_ORDER).
PLOT_THROUGHPUT_METRICS = ("Requests/sec Mean", "Output Tokens/sec Mean")
PLOT_LATENCY_METRICS = (
    "Request Latency Mean (s)",
    "TTFT Mean (ms)",
    "ITL Mean (ms)",
)
PLOT_HEATMAP_METRICS = (
    "Requests/sec Mean",
    "Output Tokens/sec Mean",
    "Request Latency Mean (s)",
    "TTFT Mean (ms)",
    "ITL Mean (ms)",
    "TPOT Mean (ms)",
)


@dataclass
class GroupedFiles:
    """Files belonging to one filename-derived group, sorted by run index."""

    name: str
    files: list[tuple[int, Path]] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        """Return file paths ordered by run index."""
        return [path for _, path in sorted(self.files, key=lambda item: item[0])]


# group -> strategy -> metric -> list of per-run values (aligned to sorted files)
MetricStore = dict[str, dict[str, dict[str, list[float | None]]]]
# group -> run_index -> strategy -> metric -> value
RawRuns = dict[str, list[dict[str, dict[str, float]]]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare groups of GuideLLM benchmark JSON reports grouped by filename."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Optional explicit report paths or globs. When provided, overrides "
            "--dir/--pattern."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory to scan when no files are given (default: {DEFAULT_DIR}).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob within --dir (default: {DEFAULT_PATTERN!r}).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Group name used for Diff%% (default: 'main' if present, else first).",
    )
    parser.add_argument(
        "--strip-prefix",
        default=DEFAULT_STRIP_PREFIX,
        help=(
            "Prefix stripped from the grouped stem for display labels "
            f"(default: {DEFAULT_STRIP_PREFIX!r})."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output basename or path stem for .csv/.png "
            "(default: <dir>/comparison_groups)."
        ),
    )
    parser.add_argument(
        "--status",
        choices=["successful", "incomplete", "errored", "total"],
        default="successful",
        help="Status slice for distribution metrics (default: successful).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for the PNG dashboard (default: 120).",
    )
    return parser.parse_args(argv)


def _expand_glob(pattern: str) -> list[Path]:
    """Expand a file glob using ``Path.glob`` relative to the pattern's parent."""
    path = Path(pattern)
    parent = path.parent if str(path.parent) not in {"", "."} else Path.cwd()
    return sorted(parent.glob(path.name))


def discover_files(args: argparse.Namespace) -> list[Path]:
    """Resolve input files from explicit paths/globs or --dir/--pattern."""
    if not args.files:
        directory: Path = args.dir
        if not directory.is_dir():
            raise SystemExit(f"Input directory does not exist: {directory}")
        return sorted(directory.glob(args.pattern))

    resolved: list[Path] = []
    for item in args.files:
        matches = _expand_glob(item)
        if matches:
            resolved.extend(matches)
            continue
        path = Path(item)
        if path.is_file():
            resolved.append(path)
        else:
            print(f"warning: no files matched {item!r}", file=sys.stderr)

    # Preserve order but drop duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        resolved_path = path.resolve()
        if resolved_path not in seen:
            seen.add(resolved_path)
            unique.append(path)
    return unique


def group_label_from_path(path: Path, strip_prefix: str) -> tuple[str, int] | None:
    """Derive (group_name, run_index) from a report filename.

    :param path: Report file path
    :param strip_prefix: Optional prefix removed from the grouped stem
    :return: ``(group, run)`` or ``None`` if the stem has no trailing ``_N``
    """
    match = _RUN_SUFFIX_RE.match(path.stem)
    if match is None:
        return None
    label = match.group("label")
    run = int(match.group("run"))
    if strip_prefix and label.startswith(strip_prefix):
        label = label[len(strip_prefix) :]
    if not label:
        return None
    return label, run


def group_files(paths: Iterable[Path], strip_prefix: str) -> dict[str, GroupedFiles]:
    """Group report paths by filename-derived labels."""
    groups: dict[str, GroupedFiles] = {}
    skipped: list[Path] = []
    for path in paths:
        parsed = group_label_from_path(path, strip_prefix)
        if parsed is None:
            skipped.append(path)
            continue
        name, run = parsed
        if name not in groups:
            groups[name] = GroupedFiles(name=name)
        groups[name].files.append((run, path))

    if skipped:
        preview = skipped[:SKIPPED_FILE_PREVIEW]
        names = ", ".join(path.name for path in preview)
        extra = len(skipped) - len(preview)
        more = f" (+{extra} more)" if extra > 0 else ""
        print(
            f"warning: skipped {len(skipped)} file(s) without _<run> suffix: "
            f"{names}{more}",
            file=sys.stderr,
        )
    if not groups:
        raise SystemExit(
            "No report files matched the grouping pattern '*_<digits>.json'"
        )
    return groups


def _status_dist(
    summary: StatusDistributionSummary, status: StatusName
) -> DistributionSummary:
    """Return the distribution for the requested status slice."""
    if status == "successful":
        return summary.successful
    if status == "incomplete":
        return summary.incomplete
    if status == "errored":
        return summary.errored
    return summary.total


def extract_metrics(
    benchmark: GenerativeBenchmark, status: StatusName
) -> dict[str, float]:
    """Extract comparison metrics from a single benchmark."""
    metrics = benchmark.metrics
    totals = metrics.request_totals
    latency = _status_dist(metrics.request_latency, status)
    ttft = _status_dist(metrics.time_to_first_token_ms, status)
    itl = _status_dist(metrics.inter_token_latency_ms, status)
    tpot = _status_dist(metrics.time_per_output_token_ms, status)
    rps = _status_dist(metrics.requests_per_second, status)
    concurrency = _status_dist(metrics.request_concurrency, status)
    prompt_tps = _status_dist(metrics.prompt_tokens_per_second, status)
    output_tps = _status_dist(metrics.output_tokens_per_second, status)
    total_tps = _status_dist(metrics.tokens_per_second, status)
    prompt_tokens = _status_dist(metrics.prompt_token_count, status)
    output_tokens = _status_dist(metrics.output_token_count, status)

    return {
        "Duration (s)": float(benchmark.duration),
        "Successful Requests": float(totals.successful),
        "Errored Requests": float(totals.errored),
        "Total Requests": float(totals.total),
        "Request Latency Median (s)": float(latency.median),
        "Request Latency Mean (s)": float(latency.mean),
        "Request Latency p95 (s)": float(latency.percentiles.p95),
        "TTFT Median (ms)": float(ttft.median),
        "TTFT Mean (ms)": float(ttft.mean),
        "TTFT p95 (ms)": float(ttft.percentiles.p95),
        "ITL Median (ms)": float(itl.median),
        "ITL Mean (ms)": float(itl.mean),
        "ITL p95 (ms)": float(itl.percentiles.p95),
        "TPOT Median (ms)": float(tpot.median),
        "TPOT Mean (ms)": float(tpot.mean),
        "TPOT p95 (ms)": float(tpot.percentiles.p95),
        "Requests/sec Mean": float(rps.mean),
        "Request Concurrency Median": float(concurrency.median),
        "Request Concurrency Mean": float(concurrency.mean),
        "Input Tokens/sec Mean": float(prompt_tps.mean),
        "Output Tokens/sec Mean": float(output_tps.mean),
        "Total Tokens/sec Mean": float(total_tps.mean),
        "Input Tokens/req Median": float(prompt_tokens.median),
        "Input Tokens/req p95": float(prompt_tokens.percentiles.p95),
        "Output Tokens/req Median": float(output_tokens.median),
        "Output Tokens/req p95": float(output_tokens.percentiles.p95),
    }


def sort_strategies(strategies: Iterable[str]) -> list[str]:
    """Sort strategy labels by concurrency/rate ascending.

    Labels like ``concurrent@10`` sort by the numeric suffix so that 5 comes
    before 10. Labels without a numeric suffix sort first by name.
    """

    def key(label: str) -> tuple[int, float, str]:
        if "@" not in label:
            return (0, 0.0, label)
        prefix, suffix = label.split("@", 1)
        try:
            return (1, float(suffix), prefix)
        except ValueError:
            # e.g. throughput@unlimited — after numbered strategies
            return (2, 0.0, label)

    return sorted(strategies, key=key)


def _load_raw_runs(
    groups: dict[str, GroupedFiles], status: StatusName
) -> tuple[RawRuns, dict[str, list[Path]], list[str]]:
    """Load reports into per-run metric maps and collect strategy labels."""
    raw: RawRuns = {}
    group_paths: dict[str, list[Path]] = {}
    all_strategies: set[str] = set()

    for group_name, grouped in sorted(groups.items()):
        paths = grouped.paths
        group_paths[group_name] = paths
        raw[group_name] = []
        for path in paths:
            report = GenerativeBenchmarksReport.load_file(path)
            run_data: dict[str, dict[str, float]] = {}
            for benchmark in report.benchmarks:
                strategy = str(benchmark.config.strategy)
                all_strategies.add(strategy)
                run_data[strategy] = extract_metrics(benchmark, status)
            raw[group_name].append(run_data)

    return raw, group_paths, sort_strategies(all_strategies)


def _fill_store_from_raw(raw: RawRuns, strategies: Sequence[str]) -> MetricStore:
    """Align per-run values into the metric store and emit coverage warnings."""
    store: MetricStore = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for group_name, runs in raw.items():
        for strategy in strategies:
            present_count = 0
            for run_data in runs:
                extracted = run_data.get(strategy)
                if extracted is None:
                    for metric_name in METRIC_ORDER:
                        store[group_name][strategy][metric_name].append(None)
                    continue
                present_count += 1
                for metric_name in METRIC_ORDER:
                    store[group_name][strategy][metric_name].append(
                        extracted[metric_name]
                    )

            if present_count == 0:
                print(
                    f"warning: group {group_name!r} lacks strategy {strategy!r}",
                    file=sys.stderr,
                )
            elif present_count < len(runs):
                print(
                    f"warning: group {group_name!r} has incomplete coverage for "
                    f"strategy {strategy!r}",
                    file=sys.stderr,
                )

    return store


def load_metric_store(
    groups: dict[str, GroupedFiles], status: StatusName
) -> tuple[MetricStore, dict[str, list[Path]], list[str]]:
    """Load all reports and build the nested metric store.

    Per-run values are aligned by file order within each group: if a replicate
    lacks a strategy, that slot is ``None``.

    :return: ``(store, group_paths, strategies)`` where strategies are sorted
        uniquely across all groups
    """
    raw, group_paths, strategies = _load_raw_runs(groups, status)
    store = _fill_store_from_raw(raw, strategies)
    return store, group_paths, strategies


def _finite_values(values: Sequence[float | None]) -> list[float]:
    """Filter out None entries from a run-value list."""
    return [value for value in values if value is not None]


def mean_of(values: Sequence[float | None]) -> float | None:
    """Return the arithmetic mean, or None if there are no values."""
    finite = _finite_values(values)
    if not finite:
        return None
    return statistics.fmean(finite)


def stddev_of(values: Sequence[float | None]) -> float | None:
    """Return sample stddev, 0.0 for a single value, or None if empty."""
    finite = _finite_values(values)
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    return statistics.stdev(finite)


def diff_pct(group_mean: float | None, baseline_mean: float | None) -> str:
    """Format Diff% of group_mean relative to baseline_mean."""
    if group_mean is None or baseline_mean is None:
        return "N/A"
    if baseline_mean == 0:
        return "N/A"
    pct = (group_mean - baseline_mean) / baseline_mean * 100.0
    return f"{pct:+.2f}%"


def resolve_baseline(groups: Sequence[str], requested: str | None) -> str:
    """Pick the baseline group name."""
    if requested is not None:
        if requested not in groups:
            available = ", ".join(groups)
            raise SystemExit(
                f"Baseline group {requested!r} not found. Available: {available}"
            )
        return requested
    if "main" in groups:
        return "main"
    return groups[0]


def resolve_output_stem(args: argparse.Namespace, input_dir: Path | None) -> Path:
    """Resolve the output path stem (without suffix)."""
    if args.output is not None:
        output = Path(args.output)
        if output.suffix.lower() in {".csv", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            return output.with_suffix("")
        return output
    base = input_dir if input_dir is not None else Path.cwd()
    return base / "comparison_groups"


def format_number(value: float | None) -> str:
    """Format a numeric cell for CSV output."""
    if value is None:
        return ""
    return f"{value:.4f}"


def _csv_headers(
    group_order: Sequence[str], run_counts: dict[str, int], baseline: str
) -> list[str]:
    """Build CSV column headers for all groups and Diff% columns."""
    headers = ["Strategy", "Metric"]
    for group in group_order:
        for idx in range(1, run_counts[group] + 1):
            headers.append(f"{group} Run{idx}")
        headers.append(f"{group} Mean")
        headers.append(f"{group} StdDev")
    for group in group_order:
        if group != baseline:
            headers.append(f"{group} Diff%")
    return headers


def _csv_row(
    strategy: str,
    metric: str,
    store: MetricStore,
    group_order: Sequence[str],
    run_counts: dict[str, int],
    baseline: str,
) -> list[str]:
    """Build one CSV data row for a strategy/metric pair."""
    row = [strategy, metric]
    means: dict[str, float | None] = {}
    for group in group_order:
        values = store.get(group, {}).get(strategy, {}).get(metric, [])
        padded = list(values) + [None] * (run_counts[group] - len(values))
        padded = padded[: run_counts[group]]
        for value in padded:
            row.append(format_number(value))
        mean_val = mean_of(padded)
        means[group] = mean_val
        row.append(format_number(mean_val))
        row.append(format_number(stddev_of(padded)))
    baseline_mean = means.get(baseline)
    for group in group_order:
        if group != baseline:
            row.append(diff_pct(means.get(group), baseline_mean))
    return row


def write_csv(
    path: Path,
    store: MetricStore,
    group_order: Sequence[str],
    group_paths: dict[str, list[Path]],
    strategies: Sequence[str],
    baseline: str,
) -> None:
    """Write the multi-group comparison CSV."""
    run_counts = {group: len(group_paths[group]) for group in group_order}
    headers = _csv_headers(group_order, run_counts, baseline)
    rows = [
        _csv_row(strategy, metric, store, group_order, run_counts, baseline)
        for strategy in strategies
        for metric in METRIC_ORDER
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _grouped_bar(
    ax: Any,
    strategies: Sequence[str],
    store: MetricStore,
    group_order: Sequence[str],
    metric: str,
    title: str,
) -> None:
    """Draw a grouped bar chart with stddev error bars for one metric."""
    n_groups = len(group_order)
    n_strats = len(strategies)
    if n_strats == 0 or n_groups == 0:
        ax.set_axis_off()
        ax.set_title(f"{title} (no data)")
        return

    x = np.arange(n_strats, dtype=float)
    width = min(0.8 / n_groups, 0.25)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * width

    for offset, group in zip(offsets, group_order, strict=True):
        means: list[float] = []
        errs: list[float] = []
        for strategy in strategies:
            values = store.get(group, {}).get(strategy, {}).get(metric, [])
            mean_val = mean_of(values)
            std_val = stddev_of(values)
            means.append(0.0 if mean_val is None else mean_val)
            errs.append(0.0 if std_val is None else std_val)
        ax.bar(x + offset, means, width=width, yerr=errs, label=group, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(list(strategies), rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def _heatmap(
    ax: Any,
    strategies: Sequence[str],
    metrics: Sequence[str],
    store: MetricStore,
    group: str,
    baseline: str,
    title: str,
) -> Any | None:
    """Draw a Diff% heatmap for one non-baseline group.

    :return: The image artist for colorbar attachment, or ``None`` if empty
    """
    if not strategies or not metrics:
        ax.set_axis_off()
        ax.set_title(f"{title} (no data)")
        return None

    data = np.full((len(strategies), len(metrics)), np.nan, dtype=float)
    for row_idx, strategy in enumerate(strategies):
        for col_idx, metric in enumerate(metrics):
            group_vals = store.get(group, {}).get(strategy, {}).get(metric, [])
            base_vals = store.get(baseline, {}).get(strategy, {}).get(metric, [])
            group_mean = mean_of(group_vals)
            base_mean = mean_of(base_vals)
            if group_mean is None or base_mean is None or base_mean == 0:
                continue
            data[row_idx, col_idx] = (group_mean - base_mean) / base_mean * 100.0

    finite = data[np.isfinite(data)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(list(strategies), fontsize=8)
    ax.set_title(title)

    for row_idx in range(len(strategies)):
        for col_idx in range(len(metrics)):
            value = data[row_idx, col_idx]
            text = "N/A" if math.isnan(value) else f"{value:+.1f}%"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=7)

    return im


def write_plot(
    path: Path,
    store: MetricStore,
    group_order: Sequence[str],
    strategies: Sequence[str],
    baseline: str,
    dpi: int,
) -> None:
    """Write the matplotlib comparison dashboard PNG."""
    bar_metrics = list(PLOT_THROUGHPUT_METRICS) + list(PLOT_LATENCY_METRICS)
    compare_groups = [group for group in group_order if group != baseline]
    n_bar_rows = (len(bar_metrics) + 1) // 2
    n_heat_rows = len(compare_groups)
    n_rows = max(n_bar_rows + n_heat_rows, 1)

    fig = plt.figure(figsize=(14, 3.6 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.25)

    for idx, metric in enumerate(bar_metrics):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        _grouped_bar(ax, strategies, store, group_order, metric, title=metric)

    # Odd leftover cell on the bar rows: turn off if unused.
    if len(bar_metrics) % 2 == 1:
        ax = fig.add_subplot(gs[n_bar_rows - 1, 1])
        ax.set_axis_off()

    for h_idx, group in enumerate(compare_groups):
        row = n_bar_rows + h_idx
        ax = fig.add_subplot(gs[row, :])
        im = _heatmap(
            ax,
            strategies,
            PLOT_HEATMAP_METRICS,
            store,
            group,
            baseline,
            title=f"{group} Diff% vs {baseline}",
        )
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Benchmark group comparison (baseline: {baseline})",
        fontsize=14,
        y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison CLI."""
    # Matplotlib may warn about cache dir in restricted environments.
    warnings.filterwarnings(
        "ignore",
        message=".*Matplotlib created a temporary cache directory.*",
    )

    args = parse_args(argv)
    paths = discover_files(args)
    if not paths:
        raise SystemExit("No input report files found.")

    groups = group_files(paths, args.strip_prefix)
    group_order = sorted(groups)
    baseline = resolve_baseline(group_order, args.baseline)

    input_dir: Path | None = (
        (paths[0].parent if paths else None) if args.files else args.dir
    )
    output_stem = resolve_output_stem(args, input_dir)
    csv_path = output_stem.with_suffix(".csv")
    png_path = output_stem.with_suffix(".png")

    print(f"Input files: {len(paths)}")
    if not args.files:
        print(f"Directory:  {args.dir}")
        print(f"Pattern:    {args.pattern}")
    for name in group_order:
        run_files = groups[name].paths
        print(f"  group {name!r}: {len(run_files)} run(s)")
        for path in run_files:
            print(f"    - {path.name}")
    print(f"Baseline:   {baseline}")

    store, group_paths, strategies = load_metric_store(groups, args.status)
    if not strategies:
        raise SystemExit("No benchmarks found inside the report files.")

    write_csv(csv_path, store, group_order, group_paths, strategies, baseline)
    write_plot(png_path, store, group_order, strategies, baseline, args.dpi)

    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
