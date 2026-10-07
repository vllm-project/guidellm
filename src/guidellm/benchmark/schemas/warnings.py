"""Serializable warnings produced after a benchmark completes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.base.base import StandardBaseModel

__all__ = ["BenchmarkWarning"]

WarningUnit = Literal["ratio", "seconds", "boolean"]


class BenchmarkWarning(StandardBaseModel):
    """One post-benchmark warning, with the measurement that triggered it.

    ``message`` is the user-facing sentence printed on the console. The
    remaining fields are the JSON record of why it fired.
    """

    code: str = Field(description="Stable identifier for this warning")
    message: str = Field(description="User-facing description of what went wrong")
    observed: float = Field(
        description="Observed value compared with the threshold",
    )
    threshold: float = Field(
        description="Configured value above which this warning fires",
    )
    unit: WarningUnit = Field(
        description=(
            "ratio when the rule divides two metrics, boolean when the metric "
            "is true or false, otherwise seconds"
        ),
    )
    sample_count: int = Field(
        description="Number of samples in the rule's primary metric",
    )
    note: str = Field(
        default="",
        description=(
            "Extra context from the rule, such as what to change or a link to follow"
        ),
    )
