"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts. Constraints evaluate
scheduler state and individual requests to determine whether processing should
continue or stop based on predefined limits. The constraint system enables
sophisticated benchmark stopping criteria through composable constraint types.
"""

from .args import ConstraintArgs
from .constraint import (
    Constraint,
    ConstraintInitializer,
    PydanticConstraintInitializer,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from .error import (
    MaxErrorRateConstraint,
    MaxErrorRateConstraintArgs,
    MaxErrorsConstraint,
    MaxErrorsConstraintArgs,
    MaxGlobalErrorRateConstraint,
    MaxGlobalErrorRateConstraintArgs,
)
from .factory import ConstraintsInitializerFactory, constraint_args_to_initializer
from .request import (
    MaxDurationConstraint,
    MaxDurationConstraintArgs,
    MaxNumberConstraint,
    MaxRequestsConstraintArgs,
    RequestsExhaustedConstraint,
)
from .saturation import (
    OverSaturationConstraint,
    OverSaturationConstraintArgs,
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
    "constraint_args_to_initializer",
]
