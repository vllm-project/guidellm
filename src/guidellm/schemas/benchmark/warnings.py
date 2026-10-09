"""Configuration for post-benchmark warning checks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.base.base import StandardBaseModel

__all__ = [
    "MetricRef",
    "WarningCondition",
    "WarningRuleArgs",
]

WarningStatistic = Literal["mean", "max", "p95", "sum"]

_PREFETCH_GUIDE = (
    "https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/"
    "troubleshooting.md#requests-load-after-they-are-due"
)


class MetricRef(StandardBaseModel):
    """A dotted path into the compiled metric schemas, and which summary to read.

    The path is resolved against scheduler metrics and then generative metrics.
    A status breakdown needs the status in the path, such as
    ``request_latency.total``. ``statistic`` selects ``mean``, ``max``,
    ``p95``, or ``sum`` when the path ends on a distribution. A path that
    already ends on a number is used as that number.
    """

    name: str = Field(
        description=(
            "Dotted path to a metric on the compiled scheduler or generative "
            "metrics, such as generation_delay or request_latency.total"
        ),
    )
    statistic: WarningStatistic = Field(
        default="mean",
        description=(
            "Summary compared or combined. mean is the typical sample and does "
            "not grow with the length of the run; max catches a single sample; "
            "p95 warns from the tail; sum is the total."
        ),
    )
    scale: float = Field(
        default=1.0,
        description=(
            "Multiply the statistic by this before comparing. Use 0.001 when "
            "the metric is in milliseconds and the other metric is in seconds."
        ),
    )


class WarningCondition(StandardBaseModel):
    """Require a compiled metric path to equal a value before the rule runs.

    The path is the same dotted path as a metric reference. The comparison is
    the value itself, not a distribution statistic, so it can match text such
    as a strategy type.
    """

    name: str = Field(description="Dotted path that must equal the configured value")
    equals: str | float | int | bool = Field(
        description="Value the path must equal for the rule to run",
    )


class WarningRuleArgs(StandardBaseModel):
    """Compare a metric, or its ratio to a second metric, with a threshold.

    When ``relative_to`` is set, the observed value is ``metric / relative_to``
    and ``threshold`` is a fraction of that second metric. Otherwise the
    observed value is the metric itself and ``threshold`` is in that metric's
    units, seconds for the built-in metrics.
    """

    enabled: bool = Field(
        default=True,
        description="Whether this rule is reported when its threshold is exceeded",
    )
    code: str = Field(
        description=(
            "Free-form tag stored on the warning. Any string is accepted. "
            "Short snake_case tags are the convention, and the tag does not "
            "have to match the metric name."
        ),
    )
    metric: MetricRef = Field(description="Metric the rule reads")
    relative_to: MetricRef | None = Field(
        default=None,
        description=(
            "When set, the rule compares metric / relative_to instead of the "
            "metric alone"
        ),
    )
    threshold: float = Field(
        ge=0,
        description=(
            "Warn when the observed value is greater than this. A ratio when "
            "relative_to is set, otherwise the metric's own units."
        ),
    )
    note: str = Field(
        default="",
        description=(
            "Extra context printed with the warning, such as what to change "
            "or a link to follow"
        ),
    )
    when: WarningCondition | None = Field(
        default=None,
        description=(
            "When set, the rule runs only if this path equals the given value"
        ),
    )


def default_warning_rules() -> list[WarningRuleArgs]:
    """
    :return: Generation delay as a fraction of request latency and of time to
        first token, late first turns measured in seconds, and an incomplete
        dataset for trace replay
    """
    return [
        WarningRuleArgs(
            code="generation_delay_ttft",
            metric=MetricRef(name="generation_delay", statistic="mean"),
            relative_to=MetricRef(
                name="time_to_first_token_ms.total",
                statistic="mean",
                scale=0.001,
            ),
            threshold=0.01,
            note=(
                "This means that the benchmark was likely bottlenecked by the "
                f"data generation. See {_PREFETCH_GUIDE}"
            ),
        ),
        WarningRuleArgs(
            code="root_late_p95",
            metric=MetricRef(name="root_dispatch_delay", statistic="p95"),
            threshold=0.75,
            note=(
                "This means that some conversations were loaded late. See "
                f"{_PREFETCH_GUIDE}"
            ),
        ),
        WarningRuleArgs(
            code="root_late_mean",
            metric=MetricRef(name="root_dispatch_delay", statistic="mean"),
            threshold=0.25,
            note=(
                f"This means that conversations were loaded late. See {_PREFETCH_GUIDE}"
            ),
        ),
        WarningRuleArgs(
            code="dataset_incomplete",
            metric=MetricRef(name="dataset_incomplete"),
            threshold=0,
            when=WarningCondition(name="strategy_type", equals="trace"),
            note=(
                "The trace dataset was not fully loaded. This could result in "
                f"late arrivals of trace conversations. See {_PREFETCH_GUIDE}"
            ),
        ),
    ]
