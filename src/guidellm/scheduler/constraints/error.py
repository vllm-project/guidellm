"""
Error-based constraint implementations.

Provides constraint types for limiting benchmark execution based on error rates
and error counts. These constraints monitor request error status to determine
when to stop benchmark execution due to excessive errors.
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

from pydantic import Field

from guidellm.scheduler.constraints.args import (
    ConstraintArgs,
    ErrorRateOrList,
    PositiveNumOrList,
)
from guidellm.scheduler.schemas import (
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo
from guidellm.settings import settings

from .constraint import Constraint, PydanticConstraintInitializer
from .factory import ConstraintsInitializerFactory

__all__ = [
    "MaxErrorRateConstraint",
    "MaxErrorRateConstraintArgs",
    "MaxErrorsConstraint",
    "MaxErrorsConstraintArgs",
    "MaxGlobalErrorRateConstraint",
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
    max_errors: PositiveNumOrList = Field(
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
    max_error_rate: ErrorRateOrList = Field(
        description="Maximum error rate (0.0 to 1.0) before stopping execution",
    )
    window_size: int | float = Field(
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
    max_error_rate: ErrorRateOrList = Field(
        description="Maximum global error rate (0.0 to 1.0) before stopping",
    )
    min_processed: int | float | None = Field(
        default_factory=lambda: settings.constraint_error_min_processed,
        gt=0,
        description="Minimum requests processed before applying error rate constraint",
    )


@ConstraintsInitializerFactory.register("max_errors")
class MaxErrorsConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on absolute error count.

    Stops both request queuing and all request processing when the total number
    of errored requests reaches the maximum threshold. Uses global error tracking
    across all requests for immediate constraint evaluation.
    """

    type_: Literal["max_errors"] = "max_errors"  # type: ignore[assignment]
    args: MaxErrorsConstraintArgs = Field(
        description="Configuration arguments for max errors constraint",
    )
    current_index: int = Field(default=-1, description="Current index in error list")

    @classmethod
    def validated_kwargs(
        cls, max_errors: int | float | list[int | float] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxErrorsConstraint creation.

        :param max_errors: Maximum number of errors to allow
        :param kwargs: Additional keyword arguments (optional type_, current_index)
        :return: Validated dictionary with args and runtime state fields
        """
        return {
            "args": MaxErrorsConstraintArgs(max_errors=max_errors),  # type: ignore[arg-type]
            "current_index": kwargs.get("current_index", -1),
        }

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return cast("Constraint", self.model_copy())

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current error count.

        :param state: Current scheduler state with error counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        _ = request_info  # Unused parameters
        current_index = max(0, self.current_index)
        max_errors = (
            self.args.max_errors
            if isinstance(self.args.max_errors, int | float)
            else self.args.max_errors[min(current_index, len(self.args.max_errors) - 1)]
        )
        errors_exceeded = state.errored_requests >= max_errors
        stop_time = (
            None if not errors_exceeded else request_info.completed_at or time.time()
        )

        return SchedulerUpdateAction(
            request_queuing="stop" if errors_exceeded else "continue",
            request_processing="stop_all" if errors_exceeded else "continue",
            metadata={
                "max_errors": max_errors,
                "errors_exceeded": errors_exceeded,
                "current_errors": state.errored_requests,
                "stop_time": stop_time,
            },
            progress=SchedulerProgress(stop_time=stop_time),
        )


@ConstraintsInitializerFactory.register("max_error_rate")
class MaxErrorRateConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on sliding window error rate.

    Tracks error status of recent requests in a sliding window and stops all
    processing when the error rate exceeds the threshold. Only applies the
    constraint after processing enough requests to fill the minimum window size
    for statistical significance.
    """

    type_: Literal["max_error_rate"] = "max_error_rate"  # type: ignore[assignment]
    args: MaxErrorRateConstraintArgs = Field(
        description="Configuration arguments for max error rate constraint",
    )
    error_window: list[bool] = Field(
        default_factory=list,
        description="Sliding window tracking error status of recent requests",
    )
    current_index: int = Field(
        default=-1, description="Current index in the error window"
    )

    @classmethod
    def validated_kwargs(
        cls,
        max_error_rate: int | float | list[int | float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxErrorRateConstraint creation.

        :param max_error_rate: Maximum error rate to allow
        :param kwargs: Additional keyword arguments (optional window_size,
            type_, current_index)
        :return: Validated dictionary with args and runtime state fields
        """
        return {
            "args": MaxErrorRateConstraintArgs(
                max_error_rate=max_error_rate,  # type: ignore[arg-type]
                window_size=kwargs.get(
                    "window_size", settings.constraint_error_window_size
                ),
            ),
            "error_window": kwargs.get("error_window", []),
            "current_index": kwargs.get("current_index", -1),
        }

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a new instance of MaxErrorRateConstraint (due to stateful window).

        :param kwargs: Additional keyword arguments (unused)
        :return: New instance of the constraint
        """
        self.current_index += 1

        return cast("Constraint", self.model_copy())

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against sliding window error rate.

        :param state: Current scheduler state with request counts
        :param request_info: Individual request with completion status
        :return: Action indicating whether to continue or stop operations
        """
        current_index = max(0, self.current_index)
        max_error_rate = (
            self.args.max_error_rate
            if isinstance(self.args.max_error_rate, int | float)
            else self.args.max_error_rate[
                min(current_index, len(self.args.max_error_rate) - 1)
            ]
        )

        if request_info.status in ["completed", "errored", "cancelled"]:
            self.error_window.append(request_info.status == "errored")
            if len(self.error_window) > self.args.window_size:
                self.error_window.pop(0)

        error_count = sum(self.error_window)
        window_requests = len(self.error_window)
        error_rate = (
            error_count / float(window_requests) if window_requests > 0 else 0.0
        )
        exceeded_min_processed = state.processed_requests >= self.args.window_size
        exceeded_error_rate = error_rate >= max_error_rate
        exceeded = exceeded_min_processed and exceeded_error_rate
        stop_time = None if not exceeded else request_info.completed_at or time.time()

        return SchedulerUpdateAction(
            request_queuing="stop" if exceeded else "continue",
            request_processing="stop_all" if exceeded else "continue",
            metadata={
                "max_error_rate": max_error_rate,
                "window_size": self.args.window_size,
                "error_count": error_count,
                "processed_count": state.processed_requests,
                "current_window_size": len(self.error_window),
                "current_error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
                "exceeded": exceeded,
                "stop_time": stop_time,
            },
        )


@ConstraintsInitializerFactory.register("max_global_error_rate")
class MaxGlobalErrorRateConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on global error rate.

    Calculates error rate across all processed requests and stops all processing
    when the rate exceeds the threshold. Only applies the constraint after
    processing the minimum number of requests to ensure statistical significance
    for global error rate calculations.
    """

    type_: Literal["max_global_error_rate"] = "max_global_error_rate"  # type: ignore[assignment]
    args: MaxGlobalErrorRateConstraintArgs = Field(
        description="Configuration arguments for max global error rate constraint",
    )
    current_index: int = Field(
        default=-1, description="Current index for list-based max_error_rate values"
    )

    @classmethod
    def validated_kwargs(
        cls,
        max_error_rate: int | float | list[int | float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxGlobalErrorRateConstraint creation.

        :param max_error_rate: Maximum error rate to allow
        :param kwargs: Additional keyword arguments (optional min_processed,
            type_, current_index)
        :return: Validated dictionary with args and runtime state fields
        """
        return {
            "args": MaxGlobalErrorRateConstraintArgs(
                max_error_rate=max_error_rate,  # type: ignore[arg-type]
                min_processed=kwargs.get(
                    "min_processed", settings.constraint_error_min_processed
                ),
            ),
            "current_index": kwargs.get("current_index", -1),
        }

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return cast("Constraint", self.model_copy())

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against global error rate.

        :param state: Current scheduler state with global request and error counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        _ = request_info  # Unused parameters
        current_index = max(0, self.current_index)
        max_error_rate = (
            self.args.max_error_rate
            if isinstance(self.args.max_error_rate, int | float)
            else self.args.max_error_rate[
                min(current_index, len(self.args.max_error_rate) - 1)
            ]
        )

        exceeded_min_processed = (
            self.args.min_processed is None
            or state.processed_requests >= self.args.min_processed
        )
        error_rate = (
            state.errored_requests / float(state.processed_requests)
            if state.processed_requests > 0
            else 0.0
        )
        exceeded_error_rate = error_rate >= max_error_rate
        exceeded = exceeded_min_processed and exceeded_error_rate
        stop_time = None if not exceeded else request_info.completed_at or time.time()

        return SchedulerUpdateAction(
            request_queuing="stop" if exceeded else "continue",
            request_processing="stop_all" if exceeded else "continue",
            metadata={
                "max_error_rate": max_error_rate,
                "min_processed": self.args.min_processed,
                "processed_requests": state.processed_requests,
                "errored_requests": state.errored_requests,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
                "exceeded": exceeded,
                "stop_time": stop_time,
            },
            progress=SchedulerProgress(stop_time=stop_time),
        )
