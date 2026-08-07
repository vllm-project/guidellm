"""
Error-based constraint argument schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.scheduler.constraints.args import (
    ConstraintArgs,
    ErrorRateOrList,
    PositiveNumOrList,
)
from guidellm.settings import settings

__all__ = [
    "MaxErrorRateConstraintArgs",
    "MaxErrorsConstraintArgs",
    "MaxGlobalErrorRateConstraintArgs",
]


@ConstraintArgs.register("max_errors")
class MaxErrorsConstraintArgs(ConstraintArgs):
    """
    Arguments for maximum error count constraint.

    Stops execution when total errors reach the threshold.

    :cvar kind: Always "max_errors"
    """

    kind: Literal["max_errors"] = Field(
        default="max_errors",
        description="Constraint type discriminator",
    )
    count: PositiveNumOrList = Field(
        description="Maximum number of errors before stopping execution",
    )


@ConstraintArgs.register("max_error_rate")
class MaxErrorRateConstraintArgs(ConstraintArgs):
    """
    Arguments for maximum error rate constraint (sliding window).

    Stops execution when the windowed error rate exceeds the threshold.

    :cvar kind: Always "max_error_rate"
    """

    kind: Literal["max_error_rate"] = Field(
        default="max_error_rate",
        description="Constraint type discriminator",
    )
    rate: ErrorRateOrList = Field(
        description="Maximum error rate (0.0 to 1.0) before stopping execution",
    )
    window: int | float = Field(
        default_factory=lambda: settings.constraint_error_window_size,
        gt=0,
        description="Size of sliding window for calculating error rate",
    )


@ConstraintArgs.register("max_global_error_rate")
class MaxGlobalErrorRateConstraintArgs(ConstraintArgs):
    """
    Arguments for maximum global error rate constraint.

    Stops execution when the overall error rate across all requests exceeds
    the threshold. Only applies after min_processed requests are completed.

    :cvar kind: Always "max_global_error_rate"
    """

    kind: Literal["max_global_error_rate"] = Field(
        default="max_global_error_rate",
        description="Constraint type discriminator",
    )
    rate: ErrorRateOrList = Field(
        description="Maximum global error rate (0.0 to 1.0) before stopping",
    )
    minimum: int | float | None = Field(
        default_factory=lambda: settings.constraint_error_min_processed,
        gt=0,
        description="Minimum requests processed before applying error rate constraint",
    )
