"""
Scheduler-related argument schemas.
"""

from guidellm.schemas.scheduler.constraints import (
    ConstraintArgs,
    ErrorRate,
    ErrorRateOrList,
    MaxDurationConstraintArgs,
    MaxErrorRateConstraintArgs,
    MaxErrorsConstraintArgs,
    MaxGlobalErrorRateConstraintArgs,
    MaxRequestsConstraintArgs,
    OverSaturationConstraintArgs,
    PositiveNum,
    PositiveNumOrList,
)

__all__ = [
    "ConstraintArgs",
    "ErrorRate",
    "ErrorRateOrList",
    "MaxDurationConstraintArgs",
    "MaxErrorRateConstraintArgs",
    "MaxErrorsConstraintArgs",
    "MaxGlobalErrorRateConstraintArgs",
    "MaxRequestsConstraintArgs",
    "OverSaturationConstraintArgs",
    "PositiveNum",
    "PositiveNumOrList",
]
