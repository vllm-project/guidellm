"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts. Constraints evaluate
scheduler state and individual requests to determine whether processing should
continue or stop based on predefined limits. The constraint system enables
sophisticated benchmark stopping criteria through composable constraint types.
"""

from guidellm.schemas.scheduler.constraints import (
    ConstraintArgs,
    MaxDurationConstraintArgs,
    MaxErrorRateConstraintArgs,
    MaxErrorsConstraintArgs,
    MaxGlobalErrorRateConstraintArgs,
    MaxRequestsConstraintArgs,
    OverSaturationConstraintArgs,
)

from .constraint import (
    Constraint,
    ConstraintInitializer,
    PydanticConstraintInitializer,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from .error import (
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
)
from .factory import ConstraintsInitializerFactory
from .request import (
    MaxDurationConstraint,
    MaxNumberConstraint,
    RequestsExhaustedConstraint,
)
from .saturation import (
    OverSaturationConstraint,
    OverSaturationConstraintInitializer,
)

__all__ = [
    "Constraint",
    "ConstraintArgs",
    "ConstraintInitializer",
    "ConstraintsInitializerFactory",
    "MaxDurationConstraint",
    "MaxDurationConstraintArgs",
    "MaxErrorRateConstraint",
    "MaxErrorRateConstraintArgs",
    "MaxErrorsConstraint",
    "MaxErrorsConstraintArgs",
    "MaxGlobalErrorRateConstraint",
    "MaxGlobalErrorRateConstraintArgs",
    "MaxNumberConstraint",
    "MaxRequestsConstraintArgs",
    "OverSaturationConstraint",
    "OverSaturationConstraintArgs",
    "OverSaturationConstraintInitializer",
    "PydanticConstraintInitializer",
    "RequestsExhaustedConstraint",
    "SerializableConstraintInitializer",
    "UnserializableConstraintInitializer",
]
