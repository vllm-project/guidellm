"""
Over-saturation detection constraint argument schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.scheduler.constraints.args import ConstraintArgs

__all__ = [
    "OverSaturationConstraintArgs",
]


@ConstraintArgs.register("over_saturation")
class OverSaturationConstraintArgs(ConstraintArgs):
    """
    Arguments for over-saturation detection constraint.

    Detects when a model becomes over-saturated using statistical slope analysis
    of concurrent requests and time-to-first-token metrics.

    :cvar kind: Always "over_saturation"
    """

    kind: Literal["over_saturation"] = Field(
        default="over_saturation",
        description="Constraint type discriminator",
    )
    mode: Literal["enforce", "monitor"] = Field(
        default="enforce",
        description=(
            "Whether to stop the benchmark if over-saturation is detected. "
            "Set to `enforce` to stop the benchmark if over-saturation is "
            "detected, and `monitor` to only report over-saturation."
        ),
    )
    min_seconds: int | float = Field(
        default=30.0,
        ge=0,
        description="Minimum seconds before checking for over-saturation",
    )
    max_window_seconds: int | float = Field(
        default=120.0,
        ge=0,
        description="Maximum over-saturation checking window size in seconds",
    )
    moe_threshold: float = Field(
        default=2.0,
        ge=0,
        description="Margin of error threshold for slope detection",
    )
    minimum_ttft: float = Field(
        default=2.5,
        ge=0,
        description="Minimum TTFT threshold for violation counting",
    )
    maximum_window_ratio: float = Field(
        default=0.75,
        ge=0,
        le=1.0,
        description="Maximum window size as ratio of total requests",
    )
    minimum_window_size: int = Field(
        default=5,
        ge=0,
        description="Minimum data points required for slope estimation",
    )
    confidence: float = Field(
        default=0.95,
        ge=0,
        le=1.0,
        description="Statistical confidence level for t-distribution",
    )

    @property
    def constraint_key(self) -> str:
        return "over_saturation"
