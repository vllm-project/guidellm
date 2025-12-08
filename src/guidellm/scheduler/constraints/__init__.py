"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts. Constraints evaluate
scheduler state and individual requests to determine whether processing should
continue or stop based on predefined limits. The constraint system enables
sophisticated benchmark stopping criteria through composable constraint types.
"""

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
    "ConstraintInitializer",
    "ConstraintsInitializerFactory",
    "MaxDurationConstraint",
    "MaxErrorRateConstraint",
    "MaxErrorsConstraint",
    "MaxGlobalErrorRateConstraint",
    "MaxNumberConstraint",
    "OverSaturationConstraint",
    "OverSaturationConstraintInitializer",
    "PydanticConstraintInitializer",
    "RequestsExhaustedConstraint",
    "SerializableConstraintInitializer",
    "UnserializableConstraintInitializer",
]
