"""
Collect metrics from Prometheus text-format endpoints, such as vLLM's /metrics.

The endpoint is scraped when a benchmark starts, at a fixed interval while it
runs, and once more after it finishes. Gauges, counters and histograms are then
summarized over the benchmark's measurement window.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx2

from guidellm.collectors.collector import Collector, PeriodicCollector
from guidellm.logger import logger
from guidellm.schemas import DistributionSummary
from guidellm.schemas.benchmark import (
    CollectorArgs,
    PrometheusCollectorArgs,
    PrometheusCounterSeries,
    PrometheusGaugeSeries,
    PrometheusHistogramSeries,
    PrometheusSummary,
)

__all__ = [
    "PrometheusCollector",
    "PrometheusSnapshot",
    "histogram_quantile",
    "parse_prometheus_text",
    "summarize_snapshots",
]

LabelSet = tuple[tuple[str, str], ...]

_QUANTILES: dict[str, float] = {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}
# Request the classic text format; OpenMetrics renames counter samples.
_ACCEPT_HEADER = "text/plain; version=0.0.4"


@dataclass
class _Histogram:
    buckets: dict[float, float] = field(default_factory=dict)
    sum: float = 0.0
    count: float = 0.0


@dataclass
class PrometheusSnapshot:
    """
    Values of the configured metric families from a single scrape.

    :param timestamp: Time of the scrape in seconds since epoch
    :param gauges: Gauge values by (family, labels)
    :param counters: Counter values by (family, labels)
    :param histograms: Histogram buckets, sum and count by (family, labels)
    """

    timestamp: float
    gauges: dict[tuple[str, LabelSet], float] = field(default_factory=dict)
    counters: dict[tuple[str, LabelSet], float] = field(default_factory=dict)
    histograms: dict[tuple[str, LabelSet], _Histogram] = field(default_factory=dict)


def _parse_labels(text: str) -> LabelSet:
    labels: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        equals = text.index("=", index)
        name = text[index:equals].strip().lstrip(",").strip()
        value_chars: list[str] = []
        index = equals + 2  # skip ="
        while text[index] != '"':
            if text[index] == "\\":
                index += 1
                value_chars.append("\n" if text[index] == "n" else text[index])
            else:
                value_chars.append(text[index])
            index += 1
        labels.append((name, "".join(value_chars)))
        index += 1
        while index < len(text) and text[index] in ", ":
            index += 1

    return tuple(sorted(labels))


def _split_sample(line: str) -> tuple[str, LabelSet, float]:
    # Drop an OpenMetrics exemplar, whose own braces would confuse the split.
    line = line.split(" # {", 1)[0]
    if "{" in line:
        name, rest = line.split("{", 1)
        label_text, value_text = rest.rsplit("}", 1)
        labels = _parse_labels(label_text)
    else:
        name, value_text = line.split(None, 1)
        labels = ()

    return name.strip(), labels, float(value_text.split()[0])


def _family_samples(family: str, metric_type: str) -> dict[str, str]:
    if metric_type == "histogram":
        return {
            f"{family}_bucket": "bucket",
            f"{family}_sum": "sum",
            f"{family}_count": "count",
        }
    if metric_type == "counter" and not family.endswith("_total"):
        return {family: "value", f"{family}_total": "value"}

    return {family: "value"}


def _record_sample(
    snapshot: PrometheusSnapshot,
    family: str,
    metric_type: str,
    part: str,
    labels: LabelSet,
    value: float,
) -> None:
    if metric_type == "gauge":
        snapshot.gauges[(family, labels)] = value
        return
    if metric_type == "counter":
        snapshot.counters[(family, labels)] = value
        return

    bucket_bound = dict(labels).get("le")
    series_labels = tuple(item for item in labels if item[0] != "le")
    histogram = snapshot.histograms.setdefault((family, series_labels), _Histogram())
    if part == "bucket" and bucket_bound is not None:
        histogram.buckets[float(bucket_bound)] = value
    elif part == "sum":
        histogram.sum = value
    elif part == "count":
        histogram.count = value


def parse_prometheus_text(
    text: str, families: list[str], timestamp: float
) -> PrometheusSnapshot:
    """
    Parse the configured metric families from Prometheus text exposition format.

    Only lines that start with a configured family name are parsed, so large
    responses with many unrelated metrics stay cheap. A counter family matches
    both ``name`` and ``name_total`` as declared by its ``# TYPE`` line. Families
    without a ``# TYPE`` line, types other than gauge, counter and histogram,
    malformed lines, and non-finite values are skipped.

    :param text: Response body in Prometheus text exposition format
    :param families: Metric family names to collect, without counter suffixes
    :param timestamp: Time of the scrape in seconds since epoch
    :return: Snapshot of the configured families found in the text
    """
    snapshot = PrometheusSnapshot(timestamp=timestamp)
    prefixes = tuple(families)
    families_by_type_name = {name: name for name in families}
    families_by_type_name.update({f"{name}_total": name for name in families})
    samples: dict[str, tuple[str, str, str]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# TYPE "):
            parts = line.split()
            if len(parts) < 4 or parts[2] not in families_by_type_name:  # noqa: PLR2004
                continue
            family, metric_type = families_by_type_name[parts[2]], parts[3]
            if metric_type in ("gauge", "counter", "histogram"):
                for sample_name, part in _family_samples(family, metric_type).items():
                    samples[sample_name] = (family, metric_type, part)
            continue
        if not line.startswith(prefixes):
            continue

        try:
            name, labels, value = _split_sample(line)
        except (ValueError, IndexError):
            continue
        if name in samples and math.isfinite(value):
            _record_sample(snapshot, *samples[name], labels, value)

    return snapshot


def histogram_quantile(quantile: float, buckets: dict[float, float]) -> float | None:
    """
    Estimate a quantile from cumulative histogram bucket counts.

    Follows Prometheus ``histogram_quantile``: the quantile is interpolated
    linearly within the bucket that contains it, and falls back to the highest
    finite bound when it lands in the ``+Inf`` bucket.

    :param quantile: Quantile to estimate, between 0 and 1
    :param buckets: Cumulative counts by upper bound, including ``+Inf``
    :return: The estimated value, or None when the histogram is empty
    """
    bounds = sorted(buckets)
    if not bounds or not math.isinf(bounds[-1]):
        return None
    counts: list[float] = []
    for bound in bounds:
        counts.append(max(buckets[bound], counts[-1] if counts else 0.0))
    total = counts[-1]
    if total <= 0:
        return None

    rank = quantile * total
    index = bisect_left(counts, rank)
    if math.isinf(bounds[index]):
        return bounds[index - 1] if index > 0 else None

    lower_bound = bounds[index - 1] if index > 0 else min(0.0, bounds[0])
    lower_count = counts[index - 1] if index > 0 else 0.0
    in_bucket = counts[index] - lower_count
    if in_bucket <= 0:
        return bounds[index]

    return (
        lower_bound + (bounds[index] - lower_bound) * (rank - lower_count) / in_bucket
    )


def _reset_corrected(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    corrected: list[tuple[float, float]] = []
    offset = 0.0
    previous: float | None = None
    for timestamp, value in points:
        if previous is not None and value < previous:
            # A drop means the server restarted and its counter began again at 0.
            offset += previous
        corrected.append((timestamp, value + offset))
        previous = value

    return corrected


def _value_at(points: list[tuple[float, float]], timestamp: float) -> float:
    if timestamp <= points[0][0]:
        return points[0][1]
    if timestamp >= points[-1][0]:
        return points[-1][1]
    index = bisect_left([point[0] for point in points], timestamp)
    (t0, v0), (t1, v1) = points[index - 1], points[index]

    return v0 + (v1 - v0) * (timestamp - t0) / (t1 - t0)


def _increase(
    points: list[tuple[float, float]], start_time: float, end_time: float
) -> float:
    corrected = _reset_corrected(points)

    return _value_at(corrected, end_time) - _value_at(corrected, start_time)


def _summarize_gauges(
    snapshots: list[PrometheusSnapshot], start_time: float, end_time: float
) -> dict[str, list[PrometheusGaugeSeries]]:
    series: dict[tuple[str, LabelSet], list[tuple[float, float]]] = defaultdict(list)
    for snapshot in snapshots:
        if start_time <= snapshot.timestamp <= end_time:
            for key, value in snapshot.gauges.items():
                series[key].append((snapshot.timestamp, value))

    gauges: dict[str, list[PrometheusGaugeSeries]] = defaultdict(list)
    for (family, labels), samples in series.items():
        gauges[family].append(
            PrometheusGaugeSeries(
                labels=dict(labels),
                summary=DistributionSummary.from_values(
                    [value for _, value in samples]
                ),
                samples=samples,
            )
        )

    return dict(gauges)


def _summarize_counters(
    snapshots: list[PrometheusSnapshot], start_time: float, end_time: float
) -> dict[str, list[PrometheusCounterSeries]]:
    series: dict[tuple[str, LabelSet], list[tuple[float, float]]] = defaultdict(list)
    previous_timestamp: float | None = None
    for snapshot in snapshots:
        for key, value in snapshot.counters.items():
            if key not in series and previous_timestamp is not None:
                # A series first exposed mid-run was at zero before it appeared.
                series[key].append((previous_timestamp, 0.0))
            series[key].append((snapshot.timestamp, value))
        previous_timestamp = snapshot.timestamp

    duration = end_time - start_time
    counters: dict[str, list[PrometheusCounterSeries]] = defaultdict(list)
    for (family, labels), points in series.items():
        increase = _increase(points, start_time, end_time)
        counters[family].append(
            PrometheusCounterSeries(
                labels=dict(labels),
                increase=increase,
                rate=increase / duration if duration > 0 else 0.0,
            )
        )

    return dict(counters)


def _summarize_histograms(
    snapshots: list[PrometheusSnapshot], start_time: float, end_time: float
) -> dict[str, list[PrometheusHistogramSeries]]:
    series: dict[tuple[str, LabelSet], list[tuple[float, _Histogram]]] = defaultdict(
        list
    )
    previous_timestamp: float | None = None
    for snapshot in snapshots:
        for key, histogram in snapshot.histograms.items():
            if key not in series and previous_timestamp is not None:
                empty = _Histogram(buckets=dict.fromkeys(histogram.buckets, 0.0))
                series[key].append((previous_timestamp, empty))
            series[key].append((snapshot.timestamp, histogram))
        previous_timestamp = snapshot.timestamp

    histograms: dict[str, list[PrometheusHistogramSeries]] = defaultdict(list)
    for (family, labels), points in series.items():
        bounds = set().union(*(histogram.buckets for _, histogram in points))
        buckets = {
            bound: _increase(
                [(t, h.buckets[bound]) for t, h in points if bound in h.buckets],
                start_time,
                end_time,
            )
            for bound in bounds
        }
        count = _increase([(t, h.count) for t, h in points], start_time, end_time)
        total = _increase([(t, h.sum) for t, h in points], start_time, end_time)
        quantiles = {
            name: value
            for name, quantile in _QUANTILES.items()
            if (value := histogram_quantile(quantile, buckets)) is not None
        }
        histograms[family].append(
            PrometheusHistogramSeries(
                labels=dict(labels),
                count=count,
                sum=total,
                mean=total / count if count > 0 else None,
                quantiles=quantiles,
            )
        )

    return dict(histograms)


def summarize_snapshots(
    source: str,
    snapshots: list[PrometheusSnapshot],
    scrape_errors: int,
    start_time: float,
    end_time: float,
) -> PrometheusSummary:
    """
    Summarize scrape snapshots over a measurement window.

    Gauges report the values scraped within the window. Counters and histograms
    report their increase across the window, interpolated linearly between the
    scrapes around each edge and corrected for counter resets.

    :param source: The scraped endpoint
    :param snapshots: Snapshots in scrape order
    :param scrape_errors: Number of failed scrapes
    :param start_time: Start of the measurement window in seconds since epoch
    :param end_time: End of the measurement window in seconds since epoch
    :return: Summary of the metrics within the window
    """
    return PrometheusSummary(
        source=source,
        scrapes=len(snapshots),
        scrape_errors=scrape_errors,
        gauges=_summarize_gauges(snapshots, start_time, end_time),
        counters=_summarize_counters(snapshots, start_time, end_time),
        histograms=_summarize_histograms(snapshots, start_time, end_time),
    )


@Collector.register("prometheus")
class PrometheusCollector(PeriodicCollector):
    """
    Scrape a Prometheus text-format endpoint, such as vLLM's ``/metrics``.

    Scrape failures are counted and logged, and never interrupt the benchmark.

    Example:
        ::

            collector = PrometheusCollector(
                url="http://localhost:8000/metrics",
                interval=1.0,
                timeout=5.0,
                metrics=["vllm:num_requests_running"],
            )
            await collector.start()
            ...
            await collector.stop()
            result = collector.result(start_time, end_time)
    """

    def __init__(
        self,
        url: str,
        interval: float,
        timeout: float,
        metrics: list[str],
        transport: httpx2.AsyncBaseTransport | None = None,
        kind: str = "prometheus",
    ):
        """
        :param url: URL of the Prometheus metrics endpoint
        :param interval: Seconds between scrapes while a benchmark runs
        :param timeout: Seconds to wait for a single scrape
        :param metrics: Metric family names to collect
        :param transport: Optional httpx2 transport, used to substitute the server
        :param kind: The kind this collector was configured with
        """
        super().__init__(kind, interval)
        self.url = url
        self.timeout = timeout
        self.metrics = metrics
        self._transport = transport
        self._client: httpx2.AsyncClient | None = None
        self._snapshots: list[PrometheusSnapshot] = []
        self._scrape_errors = 0

    @classmethod
    def from_args(cls, args: CollectorArgs) -> PrometheusCollector:
        """
        Create a collector from Prometheus collector arguments.

        :param args: Prometheus collector arguments
        :return: The configured collector
        :raises TypeError: If the arguments are not Prometheus arguments
        """
        if not isinstance(args, PrometheusCollectorArgs):
            raise TypeError(
                f"Expected PrometheusCollectorArgs, got {type(args).__name__}"
            )

        return cls(
            url=args.url,
            interval=args.interval,
            timeout=args.timeout,
            metrics=args.metrics,
            kind=args.kind,
        )

    async def reset(self) -> None:
        """Discard the previous benchmark's scrapes."""
        self._snapshots = []
        self._scrape_errors = 0

    async def sample(self) -> None:
        """Scrape the endpoint once, counting a failure instead of raising."""
        if self._client is None:
            self._client = httpx2.AsyncClient(
                timeout=self.timeout, transport=self._transport
            )
        requested_at = time.time()
        try:
            response = await self._client.get(
                self.url, headers={"Accept": _ACCEPT_HEADER}
            )
            response.raise_for_status()
        except (httpx2.HTTPError, httpx2.InvalidURL) as err:
            self._scrape_errors += 1
            if self._scrape_errors == 1:
                logger.warning("Failed to scrape metrics from {}: {}", self.url, err)
            return
        # The midpoint of the request is the best estimate of when it was sampled.
        timestamp = (requested_at + time.time()) / 2
        self._snapshots.append(
            parse_prometheus_text(response.text, self.metrics, timestamp)
        )

    def result(self, start_time: float, end_time: float) -> dict[str, Any]:
        """
        Summarize the latest benchmark's scrapes over its measurement window.

        See :func:`summarize_snapshots` for how each metric type is summarized.

        :param start_time: Start of the measurement window in seconds since epoch
        :param end_time: End of the measurement window in seconds since epoch
        :return: The :class:`PrometheusSummary` as a JSON-serializable dict
        """
        if end_time - start_time < 2 * self.interval:
            logger.warning(
                "Measurement window of {:.2f}s is shorter than twice the {}s "
                "scrape interval of {}, so its metrics are approximate",
                end_time - start_time,
                self.interval,
                self.url,
            )

        return summarize_snapshots(
            self.url, self._snapshots, self._scrape_errors, start_time, end_time
        ).model_dump(mode="json")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
