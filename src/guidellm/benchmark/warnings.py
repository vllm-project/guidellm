"""Post-benchmark checks defined by a path into the compiled metric schemas.

The analyzer runs from ``GenerativeBenchmark.compile``. A rule names a dotted
path on the scheduler metrics or the generative metrics. No metric names are
hard-coded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from guidellm.benchmark.schemas.metrics import GenerativeMetrics, SchedulerMetrics
from guidellm.benchmark.schemas.warnings import BenchmarkWarning
from guidellm.logger import logger
from guidellm.schemas.benchmark.warnings import (
    MetricRef,
    WarningCondition,
    WarningRuleArgs,
    WarningStatistic,
)

__all__ = ["BenchmarkWarningAnalyzer"]


class BenchmarkWarningAnalyzer:
    """Evaluate configured metric rules against one compiled benchmark."""

    def __init__(self, rules: list[WarningRuleArgs]):
        """
        :param rules: Rules to evaluate, including the default checks
        """
        self.rules = rules

    def analyze(
        self,
        *,
        scheduler_metrics: SchedulerMetrics,
        metrics: GenerativeMetrics,
    ) -> list[BenchmarkWarning]:
        """
        Return warnings whose observed value exceeded the rule threshold.

        A rule with ``relative_to`` compares the ratio of two metrics. A rule
        without it compares the metric itself. A path that names a missing
        field is logged and reported here, because the interactive progress
        display clears stderr when it finishes. A field that is present but
        null is skipped.

        :param scheduler_metrics: Compiled scheduler timings
        :param metrics: Compiled request metrics
        :return: Warnings that fired, in rule order
        """
        warnings: list[BenchmarkWarning] = []
        for rule in self.rules:
            _evaluate_rule(
                rule,
                warnings,
                scheduler_metrics=scheduler_metrics,
                metrics=metrics,
            )
        return warnings


def _evaluate_rule(
    rule: WarningRuleArgs,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> None:
    """
    Evaluate one rule and append any warning it produces.

    :param rule: Metric, optional baseline metric, and threshold
    :param warnings: Warnings collected for this benchmark
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    """
    if not rule.enabled:
        return
    if rule.when is not None and not _when_matches(
        rule,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    ):
        return

    observed = _resolve_metric(
        rule,
        rule.metric,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if observed is None or observed.count == 0:
        return

    warning = (
        _absolute_warning(rule, observed)
        if rule.relative_to is None
        else _ratio_warning(
            rule,
            observed,
            warnings,
            scheduler_metrics=scheduler_metrics,
            metrics=metrics,
        )
    )
    if warning is not None:
        warnings.append(warning)


def _when_matches(
    rule: WarningRuleArgs,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> bool:
    """
    Apply the rule's ``when`` condition.

    :param rule: Rule that names a path and a required value
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: Whether the path equals the required value. A missing path is
        logged and does not match.
    """
    if rule.when is None:
        return True

    return _condition_matches(
        rule,
        rule.when,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )


def _absolute_warning(
    rule: WarningRuleArgs,
    observed: _MetricValue,
) -> BenchmarkWarning | None:
    """
    Warn when a single metric exceeds its threshold.

    :param rule: Rule without a baseline metric
    :param observed: Resolved primary metric
    :return: The warning, or None when the value is within the threshold
    """
    if observed.value <= rule.threshold:
        return None
    return BenchmarkWarning(
        code=rule.code,
        message=(
            _boolean_message(rule.metric, observed.value)
            if observed.boolean
            else _absolute_message(
                rule.metric, observed.value, observed.count, rule.threshold
            )
        ),
        observed=observed.value,
        threshold=rule.threshold,
        unit="boolean" if observed.boolean else "seconds",
        sample_count=observed.count,
        note=rule.note,
    )


def _ratio_warning(
    rule: WarningRuleArgs,
    observed: _MetricValue,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> BenchmarkWarning | None:
    """
    Warn when metric / relative_to exceeds the threshold.

    :param rule: Rule with a baseline metric
    :param observed: Resolved primary metric
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: The warning, or None when the baseline is missing or the ratio
        is within the threshold
    """
    if rule.relative_to is None:
        return None

    baseline = _resolve_metric(
        rule,
        rule.relative_to,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if baseline is None or baseline.count == 0 or baseline.value <= 0:
        return None

    ratio = observed.value / baseline.value
    if ratio <= rule.threshold:
        return None
    return BenchmarkWarning(
        code=rule.code,
        message=_ratio_message(
            rule.metric,
            observed.value,
            rule.relative_to,
            baseline.value,
            ratio,
            rule.threshold,
        ),
        observed=ratio,
        threshold=rule.threshold,
        unit="ratio",
        sample_count=observed.count,
        note=rule.note,
    )


def _log_unknown_metric(
    rule: WarningRuleArgs,
    path: str,
    warnings: list[BenchmarkWarning],
) -> None:
    """
    Record that a rule named a path the compiled schemas do not have.

    The message is logged and stored on the benchmark. The interactive progress
    display redirects stderr and clears it when the run finishes, so a log line
    alone does not remain on screen. The rule's note is left off, because it
    describes the condition the metric was meant to check.

    :param rule: Rule whose metric path failed to resolve
    :param path: Dotted path that was not found, or did not end on a metric
    :param warnings: Warnings collected for this benchmark
    """
    message = (
        f"Metric '{path}' in rule '{rule.code}' was not found on the compiled "
        "scheduler or generative metrics."
    )
    logger.warning(message)
    warnings.append(
        BenchmarkWarning(
            code="unknown_metric",
            message=message,
            observed=0.0,
            threshold=rule.threshold,
            unit="seconds",
            sample_count=0,
        )
    )


class _MetricValue:
    """A resolved metric statistic and how many samples it summarizes."""

    def __init__(self, value: float, count: int, *, boolean: bool = False):
        self.value = value
        self.count = count
        self.boolean = boolean


def _resolve_metric(
    rule: WarningRuleArgs,
    ref: MetricRef,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> _MetricValue | None:
    """
    Read one metric path from the compiled scheduler or generative metrics.

    The first path segment selects whichever schema defines that field.
    Later segments walk into status breakdowns and nested summaries.
    ``scale`` is applied to the statistic. A missing path is logged here.

    :param rule: Rule that named the path, used when the path is missing
    :param ref: Dotted path and statistic
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: The statistic and sample count, or None when the field is missing
        or present but null
    """
    root = _metric_root(
        rule,
        ref.name,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if root is None:
        return None
    node = _walk_path(rule, root, ref.name, warnings)
    if node is None:
        return None
    resolved = _read_statistic(node, ref.statistic)
    if resolved is None:
        _log_unknown_metric(rule, ref.name, warnings)
        return None
    resolved.value *= ref.scale
    return resolved


def _metric_root(
    rule: WarningRuleArgs,
    path: str,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> BaseModel | None:
    """
    Select the schema that defines the first segment of a metric path.

    :param rule: Rule that named the path, logged when neither schema has it
    :param path: Dotted field path
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: The schema to walk, or None when the first segment is unknown
    """
    first = path.split(".", 1)[0]
    if first in type(scheduler_metrics).model_fields:
        return scheduler_metrics
    if first in type(metrics).model_fields:
        return metrics
    _log_unknown_metric(rule, path, warnings)
    return None


def _walk_path(
    rule: WarningRuleArgs,
    root: BaseModel,
    path: str,
    warnings: list[BenchmarkWarning],
) -> Any:
    """
    Walk a dotted path through one metric schema.

    :param rule: Rule that named the path, logged when a segment is missing
    :param root: Compiled metric model selected by the first path segment
    :param path: Dotted field path
    :return: The value at the path, or None when a present field is null or a
        segment is not defined
    """
    current: Any = root
    for key in path.split("."):
        if not _has_segment(current, key):
            _log_unknown_metric(rule, path, warnings)
            return None
        current = _read_segment(current, key)
        if current is None:
            return None
    return current


def _has_segment(current: Any, key: str) -> bool:
    """
    :param current: Model or mapping reached so far
    :param key: Next path segment
    :return: Whether ``key`` is defined on ``current``
    """
    if key == "":
        return False
    if isinstance(current, BaseModel):
        return key in type(current).model_fields
    if isinstance(current, dict):
        return key in current
    return False


def _read_segment(current: Any, key: str) -> Any:
    """
    Read one field from a model or one key from a dumped mapping.

    :param current: Model or mapping that defines ``key``
    :param key: Next path segment
    :return: The child value, or None when that value is null
    """
    if isinstance(current, BaseModel):
        return current.model_dump(include={key})[key]
    return current[key]


def _condition_matches(
    rule: WarningRuleArgs,
    condition: WarningCondition,
    warnings: list[BenchmarkWarning],
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> bool:
    """
    Compare one path with the value a rule requires.

    :param rule: Rule that named the path, logged when the path is missing
    :param condition: Path and the value it must equal
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: Whether the path equals the value. A missing path does not match.
    """
    root = _metric_root(
        rule,
        condition.name,
        warnings,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if root is None:
        return False
    node = _walk_path(rule, root, condition.name, warnings)
    if node is None:
        return False
    return _values_equal(node, condition.equals)


def _values_equal(left: Any, right: str | float | int | bool) -> bool:
    """
    Compare a compiled value with a configured condition value.

    Booleans are compared only with booleans. ``True`` is not equal to ``1``.

    :param left: Value read from a metric schema
    :param right: Value configured on the rule
    :return: Whether the two values are the same
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _read_statistic(node: Any, statistic: WarningStatistic) -> _MetricValue | None:
    """
    Read a summary from a distribution, or the number a path already ended on.

    :param node: Value at the end of a metric path
    :param statistic: Summary to read when ``node`` is a distribution
    :return: The value and sample count, or None when the node has no samples
    """
    if node is None:
        return None
    if isinstance(node, bool):
        return _MetricValue(float(node), 1, boolean=True)
    if isinstance(node, (int, float)):
        return _MetricValue(float(node), 1)
    if not isinstance(node, dict) or "count" not in node:
        return None
    return _distribution_statistic(node, statistic)


def _distribution_statistic(
    node: dict[str, Any],
    statistic: WarningStatistic,
) -> _MetricValue | None:
    """
    Read one statistic from a dumped distribution summary.

    :param node: Serialized distribution, including ``count``
    :param statistic: ``mean``, ``max``, ``p95``, or ``sum``
    :return: The value and sample count, or None when the statistic is absent
    """
    count = int(node["count"])
    if count == 0:
        return _MetricValue(0.0, 0)

    value = _distribution_value(node, statistic)
    if value is None:
        return None
    return _MetricValue(value, count)


def _distribution_value(
    node: dict[str, Any],
    statistic: WarningStatistic,
) -> float | None:
    """
    :param node: Serialized distribution with at least one sample
    :param statistic: ``mean``, ``max``, ``p95``, or ``sum``
    :return: The selected value, or None when that summary is missing
    """
    if statistic == "mean":
        return float(node["mean"])
    if statistic == "max":
        return float(node["max"])
    if statistic == "sum":
        return float(node["total_sum"])
    percentiles = node.get("percentiles")
    if not isinstance(percentiles, dict) or "p95" not in percentiles:
        return None
    return float(percentiles["p95"])


def _ratio_message(
    metric: MetricRef,
    metric_value: float,
    relative_to: MetricRef,
    baseline_value: float,
    ratio: float,
    threshold: float,
) -> str:
    """
    :param metric: Numerator metric
    :param metric_value: Numerator in seconds
    :param relative_to: Denominator metric
    :param baseline_value: Denominator in seconds
    :param ratio: metric_value / baseline_value
    :param threshold: Configured ratio
    :return: User-facing sentence for a ratio rule
    """
    return (
        f"{_metric_label(metric)} was {_format_ratio(ratio)} of "
        f"{_metric_label(relative_to)} "
        f"({_format_seconds(metric_value)} / {_format_seconds(baseline_value)}; "
        f"threshold {_format_ratio(threshold)})."
    )


def _boolean_message(metric: MetricRef, observed: float) -> str:
    """
    :param metric: Boolean metric that exceeded the threshold
    :param observed: ``1.0`` when the metric is true, otherwise ``0.0``
    :return: User-facing sentence for a boolean rule
    """
    value = "true" if observed else "false"
    return f"{_metric_label(metric)} was {value}."


def _absolute_message(
    metric: MetricRef,
    observed: float,
    count: int,
    threshold: float,
) -> str:
    """
    :param metric: Metric that exceeded the threshold
    :param observed: Observed value in seconds
    :param count: Number of samples in the metric
    :param threshold: Configured threshold in seconds
    :return: User-facing sentence for a single-metric rule
    """
    sample = "sample" if count == 1 else "samples"
    return (
        f"{_metric_label(metric)} was {_format_seconds(observed)} "
        f"over {count} {sample} (threshold {_format_seconds(threshold)})."
    )


def _metric_label(metric: MetricRef) -> str:
    """
    :param metric: Metric reference
    :return: Statistic and metric name, with underscores written as spaces
    """
    return f"{metric.statistic} {metric.name.replace('_', ' ')}"


def _decimal_places(value: float, minimum: int, limit: int) -> int:
    """
    :param value: Number to display
    :param minimum: Fewest digits after the decimal point
    :param limit: Most digits after the decimal point
    :return: ``minimum``, or more when that would round a non-zero value to zero
    """
    decimals = minimum
    while decimals < limit and value != 0 and round(value, decimals) == 0:
        decimals += 1
    return decimals


def _format_ratio(ratio: float) -> str:
    """
    :param ratio: Fraction, such as 0.02 for 2%
    :return: Percent with enough digits that a non-zero ratio does not print as 0%
    """
    percent = ratio * 100
    decimals = _decimal_places(percent, minimum=0, limit=4)
    return f"{percent:.{decimals}f}%"


def _format_seconds(value: float) -> str:
    """
    :param value: Duration in seconds
    :return: Duration with a unit suffix. Three decimal places, or more when
        those places would round the value to zero.
    """
    decimals = _decimal_places(value, minimum=3, limit=9)
    return f"{value:.{decimals}f}s"
