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
from guidellm.schemas.benchmark.warnings import (
    BenchmarkWarningsArgs,
    MetricRef,
    WarningCondition,
    WarningRuleArgs,
    WarningStatistic,
)

__all__ = ["BenchmarkWarningAnalyzer"]


class BenchmarkWarningAnalyzer:
    """Evaluate configured metric rules against one compiled benchmark."""

    def __init__(self, args: BenchmarkWarningsArgs):
        """
        :param args: Rules to evaluate, including the default checks
        """
        self.args = args

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
        field produces an ``unknown_metric`` warning. A field that is present
        but null is skipped.

        :param scheduler_metrics: Compiled scheduler timings
        :param metrics: Compiled request metrics
        :return: Warnings that fired, in rule order
        """
        warnings: list[BenchmarkWarning] = []
        for rule in self.args.rules:
            warning = _evaluate_rule(
                rule,
                scheduler_metrics=scheduler_metrics,
                metrics=metrics,
            )
            if warning is not None:
                warnings.append(warning)
        return warnings


_CONTINUE = object()


def _gate_condition(
    rule: WarningRuleArgs,
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> BenchmarkWarning | None | object:
    """
    Apply the rule's ``when`` condition.

    :param rule: Rule that may name a path and a required value
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: ``_CONTINUE`` when the rule should be evaluated, a warning when
        the condition path is missing, or None when the value does not match
    """
    if rule.when is None:
        return _CONTINUE

    matched = _condition_matches(
        rule.when,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if isinstance(matched, _UnresolvedMetric):
        return _unknown_metric_warning(rule, matched.path)
    if not matched:
        return None
    return _CONTINUE


def _evaluate_rule(
    rule: WarningRuleArgs,
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> BenchmarkWarning | None:
    """
    Evaluate one rule.

    :param rule: Metric, optional baseline metric, and threshold
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: The warning, or None when the rule is disabled, a metric does not
        apply, or the value is within the threshold
    """
    if not rule.enabled:
        return None

    gated = _gate_condition(
        rule,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if gated is None or isinstance(gated, BenchmarkWarning):
        return gated

    observed = _resolve_metric(
        rule.metric,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if isinstance(observed, _UnresolvedMetric):
        return _unknown_metric_warning(rule, observed.path)
    if observed is None or observed.count == 0:
        return None

    if rule.relative_to is None:
        return _absolute_warning(rule, observed)

    return _ratio_warning(
        rule,
        observed,
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
        rule.relative_to,
        scheduler_metrics=scheduler_metrics,
        metrics=metrics,
    )
    if isinstance(baseline, _UnresolvedMetric):
        return _unknown_metric_warning(rule, baseline.path)
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


def _unknown_metric_warning(rule: WarningRuleArgs, path: str) -> BenchmarkWarning:
    """
    Warn that a rule named a path the compiled schemas do not have.

    :param rule: Rule whose metric path failed to resolve
    :param path: Dotted path that was not found, or did not end on a metric
    :return: A warning that names the path and the rule
    """
    return BenchmarkWarning(
        code="unknown_metric",
        message=(
            f"Metric '{path}' in rule '{rule.code}' was not found on the "
            "compiled scheduler or generative metrics."
        ),
        observed=0.0,
        threshold=rule.threshold,
        unit="seconds",
        sample_count=0,
        note=rule.note,
    )


class _UnresolvedMetric:
    """A metric path that is not on the schemas, or does not end on a metric."""

    def __init__(self, path: str):
        self.path = path


class _MetricValue:
    """A resolved metric statistic and how many samples it summarizes."""

    def __init__(self, value: float, count: int, *, boolean: bool = False):
        self.value = value
        self.count = count
        self.boolean = boolean


class _Absent:
    """A path segment that is not defined on the object being walked."""


def _resolve_metric(
    ref: MetricRef,
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> _MetricValue | _UnresolvedMetric | None:
    """
    Read one metric path from the compiled scheduler or generative metrics.

    The first path segment selects whichever schema defines that field.
    Later segments walk into status breakdowns and nested summaries.
    ``scale`` is applied to the statistic.

    :param ref: Dotted path and statistic
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: The statistic and sample count, an unresolved path, or None when
        the field is present but null
    """
    node = _walk_path(scheduler_metrics, ref.name)
    if isinstance(node, _Absent):
        node = _walk_path(metrics, ref.name)
    if isinstance(node, _Absent):
        return _UnresolvedMetric(ref.name)
    if node is None:
        return None
    resolved = _read_statistic(node, ref.statistic)
    if resolved is None:
        return _UnresolvedMetric(ref.name)
    resolved.value *= ref.scale
    return resolved


def _walk_path(root: BaseModel, path: str) -> Any:
    """
    Walk a dotted path through a metric schema.

    :param root: Compiled metric model
    :param path: Dotted field path
    :return: The value at the path, None when a present field is null, or
        ``_Absent`` when a segment is not defined
    """
    current: Any = root
    for key in path.split("."):
        if key == "":
            return _Absent()
        current = _step(current, key)
        if isinstance(current, _Absent) or current is None:
            return current
    return current


def _step(current: Any, key: str) -> Any:
    """
    Read one field from a model or one key from a dumped mapping.

    :param current: Model or mapping reached so far
    :param key: Next path segment
    :return: The child value, None when that value is null, or ``_Absent``
        when the segment is not on this object
    """
    if isinstance(current, BaseModel):
        if key not in type(current).model_fields:
            return _Absent()
        return current.model_dump(include={key})[key]
    if isinstance(current, dict):
        if key not in current:
            return _Absent()
        return current[key]
    return _Absent()


def _condition_matches(
    condition: WarningCondition,
    *,
    scheduler_metrics: SchedulerMetrics,
    metrics: GenerativeMetrics,
) -> bool | _UnresolvedMetric:
    """
    Compare one path with the value a rule requires.

    :param condition: Path and the value it must equal
    :param scheduler_metrics: Compiled scheduler timings
    :param metrics: Compiled request metrics
    :return: Whether the path equals the value, or an unresolved path
    """
    node = _walk_path(scheduler_metrics, condition.name)
    if isinstance(node, _Absent):
        node = _walk_path(metrics, condition.name)
    if isinstance(node, _Absent):
        return _UnresolvedMetric(condition.name)
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
