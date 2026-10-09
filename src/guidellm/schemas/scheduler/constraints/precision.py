"""
Target margin of error constraint argument schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.scheduler.constraints.args import ConstraintArgs

__all__ = [
    "TargetMoeConstraintArgs",
    "TargetMoeMetric",
    "TargetMoeStatistic",
]

TargetMoeMetric = Literal["time_to_first_token_ms", "request_latency"]
TargetMoeStatistic = Literal["mean", "p25", "p50", "p75", "p90", "p95", "p99", "p999"]


@ConstraintArgs.register("target_moe")
class TargetMoeConstraintArgs(ConstraintArgs):
    """
    Arguments for the target margin of error constraint.

    Stops a benchmark once one request-level statistic has been measured to the
    requested relative precision, so a run lasts as long as that precision needs
    rather than a fixed duration or request count.

    :cvar kind: Always "target_moe"
    """

    kind: Literal["target_moe"] = Field(
        default="target_moe",
        description="Constraint type discriminator",
    )
    metric: TargetMoeMetric = Field(
        default="time_to_first_token_ms",
        description=(
            "Request-level metric to measure, named as in the benchmark report. "
            "Only unweighted request-level metrics are supported, matching the "
            "metrics the report attaches confidence intervals to."
        ),
    )
    statistic: TargetMoeStatistic = Field(
        default="mean",
        description=(
            "Statistic of the metric whose margin of error is targeted, either "
            "the mean or a percentile key as used in the report (e.g. p95)."
        ),
    )
    moe: float = Field(
        gt=0,
        lt=1,
        description=(
            "Target relative margin of error: the larger distance from the point "
            "estimate to either confidence bound, as a fraction of the estimate. "
            "For example, 0.05 stops once the statistic is known to within 5%."
        ),
    )
    confidence: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Two-sided confidence level of the interval",
    )
    min_samples: int = Field(
        default=30,
        ge=2,
        description=(
            "Minimum successful requests before the margin of error is checked. "
            "Guards against stopping on an early, underestimated spread."
        ),
    )
    check_interval: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of new successful requests between margin of error checks. "
            "Checking less often than every request limits optional stopping."
        ),
    )

    @property
    def constraint_key(self) -> str:
        return "target_moe"
