"""
Request-based constraint argument schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.scheduler.constraints.args import (
    ConstraintArgs,
    PositiveNumOrList,
)

__all__ = [
    "MaxDurationConstraintArgs",
    "MaxRequestsConstraintArgs",
]


@ConstraintArgs.register("max_duration")
class MaxDurationConstraintArgs(ConstraintArgs):
    """
    Arguments for maximum duration constraint.

    Limits benchmark execution time per strategy.

    :cvar kind: Always "max_duration"
    """

    kind: Literal["max_duration"] = Field(
        default="max_duration",
        description="Constraint type discriminator",
    )
    seconds: PositiveNumOrList = Field(
        description="Maximum duration in seconds before stopping execution",
    )


@ConstraintArgs.register("max_requests")
class MaxRequestsConstraintArgs(ConstraintArgs):
    """
    Arguments for maximum request count constraint.

    Limits the number of requests processed per strategy.

    :cvar kind: Always "max_requests"
    """

    kind: Literal["max_requests"] = Field(
        default="max_requests",
        description="Constraint type discriminator",
    )
    count: PositiveNumOrList = Field(
        description="Maximum number of requests before stopping execution",
    )
