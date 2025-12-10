"""
Scheduler subsystem for orchestrating benchmark workloads and managing worker processes.

This module provides the core scheduling infrastructure for guidellm, including
strategies for controlling request timing patterns (synchronous, asynchronous,
constant rate, Poisson), constraints for limiting benchmark execution (duration,
error rates, request counts), and distributed execution through worker processes.
The scheduler coordinates between backend interfaces, manages benchmark state
transitions, and handles multi-turn request sequences with customizable timing
strategies and resource constraints.
"""

from .constraints import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    MaxDurationConstraint,
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
    MaxNumberConstraint,
    OverSaturationConstraint,
    OverSaturationConstraintInitializer,
    PydanticConstraintInitializer,
    SerializableConstraintInitializer,
    UnserializableConstraintInitializer,
)
from .environments import Environment, NonDistributedEnvironment
from .scheduler import Scheduler
from .schemas import (
    BackendInterface,
    BackendT,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    SchedulerMessagingPydanticRegistry,
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from .strategies import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    SchedulingStrategy,
    StrategyT,
    StrategyType,
    SynchronousStrategy,
    ThroughputStrategy,
)
from .worker import WorkerProcess
from .worker_group import WorkerProcessGroup

__all__ = [
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
    "BackendInterface",
    "BackendT",
    "ConcurrentStrategy",
    "Constraint",
    "ConstraintInitializer",
    "ConstraintsInitializerFactory",
    "Environment",
    "MaxDurationConstraint",
    "MaxErrorRateConstraint",
    "MaxErrorsConstraint",
    "MaxGlobalErrorRateConstraint",
    "MaxNumberConstraint",
    "MultiTurnRequestT",
    "NonDistributedEnvironment",
    "OverSaturationConstraint",
    "OverSaturationConstraintInitializer",
    "PydanticConstraintInitializer",
    "RequestT",
    "ResponseT",
    "Scheduler",
    "SchedulerMessagingPydanticRegistry",
    "SchedulerProgress",
    "SchedulerState",
    "SchedulerUpdateAction",
    "SchedulingStrategy",
    "SerializableConstraintInitializer",
    "StrategyT",
    "StrategyType",
    "SynchronousStrategy",
    "ThroughputStrategy",
    "UnserializableConstraintInitializer",
    "WorkerProcess",
    "WorkerProcessGroup",
]
