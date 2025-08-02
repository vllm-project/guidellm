"""
Constraint system for controlling scheduler behavior and request processing.

This module provides a flexible constraint system for managing scheduler behavior
in the GuideLLM toolkit. Constraints can limit request queuing and processing for
scheduler runs based on various criteria such as time limits, error rates, and counts.

Classes:
    CallableConstraint: Type alias for constraint function signature.
    ConstraintsResolveArgs: Dictionary for passing arguments to constraint factories.
    ConstraintsFactory: Factory class for creating and resolving constraint instances.

Functions:
    max_number_constraint: Creates a constraint that limits the total number of
        requests.
    max_duration_constraint: Creates a constraint that limits execution duration.
    max_errors_constraint: Creates a constraint that limits the total number of
        errors.
    max_error_rate_constraint: Creates a constraint that limits error rate within a
        sliding window.
    max_global_error_rate_constraint: Creates a constraint that limits global error
        rate.
"""

import time
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from guidellm.config import settings
from guidellm.scheduler.objects import (
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.utils import RegistryMixin

__all__ = [
    "CallableConstraint",
    "ConstraintsFactory",
    "ConstraintsResolveArgs",
    "max_duration_constraint",
    "max_error_rate_constraint",
    "max_errors_constraint",
    "max_global_error_rate_constraint",
    "max_number_constraint",
]

CallableConstraint = Callable[
    [SchedulerState, ScheduledRequestInfo], SchedulerUpdateAction
]


class ConstraintsResolveArgs(dict[str, Any]):
    """
    Dictionary container for passing arguments to constraint factory functions.

    This class extends dict to provide a type-safe way to pass arguments when
    creating constraints through the factory system. It allows constraints to
    accept complex parameter configurations beyond simple values.

    Example:
    ::
        args = ConstraintsResolveArgs({
            "max_error_rate": 0.1,
            "window_size": 100
        })
        constraint = factory.create_constraint("max_error_rate", args)
    """


class ConstraintsFactory(RegistryMixin[Callable[..., CallableConstraint]]):
    """
    Factory class for creating and resolving constraint instances.

    This factory provides a centralized mechanism for creating constraint functions
    from configuration parameters. It uses the class registry system to enable
    dynamic discovery of constraint implementations and supports both simple value
    and complex parameter configurations.

    The factory can resolve constraints from dictionaries containing mixed types
    (CallableConstraint instances or parameter values) and convert them into
    a uniform dictionary of callable constraints.

    Example:
    ::
        factory = ConstraintsFactory()
        constraints = {
            "max_number": 1000,
            "max_duration": 300.0,
            "max_error_rate": ConstraintsResolveArgs({
                "max_error_rate": 0.05,
                "window_size": 50
            })
        }
        resolved = factory.resolve_constraints(constraints)
    """

    @classmethod
    def resolve_constraints(
        cls,
        constraints: dict[str, Union[Any, CallableConstraint]],
    ) -> dict[str, CallableConstraint]:
        """
        Resolve a dictionary of constraint specifications into callable constraints.

        Takes a dictionary where values can be either already-instantiated
        CallableConstraint functions or parameter values/configurations that need
        to be converted into constraint functions using the registered factories.

        Example:
        ::
            constraints = {
                "max_number": 500,
                "max_duration": 120.0,
                "custom_constraint": lambda state, info: SchedulerUpdateAction()
            }
            resolved = ConstraintsFactory.resolve_constraints(constraints)

        :param constraints: Dictionary mapping constraint names to either callable
            constraints or parameter values for creating constraints.
        :return: Dictionary mapping constraint names to callable constraint functions.
        :raises ValueError: If an unknown constraint key is encountered.
        """
        resolved_constraints: dict[str, CallableConstraint] = {}

        for key, value in constraints.items():
            if value is None:
                continue

            resolved_constraints[key] = (
                value
                if isinstance(value, CallableConstraint)
                else cls.create_constraint(key, value)
            )

        return resolved_constraints

    @classmethod
    def create_constraint(cls, key: str, value: Any) -> CallableConstraint:
        """
        Create a constraint function from a registered factory using the given
        parameters.

        Uses the class registry to find the appropriate constraint factory function
        for the given key and creates a constraint instance with the provided value
        or parameter configuration.

        Example:
        ::
            # Create with simple parameter
            constraint = ConstraintsFactory.create_constraint("max_number", 100)

            # Create with complex parameters
            args = ConstraintsResolveArgs({"max_error_rate": 0.1, "window_size": 50})
            constraint = ConstraintsFactory.create_constraint("max_error_rate", args)

        :param key: The name of the constraint type to create.
        :param value: The parameter value or ConstraintsResolveArgs configuration
            to pass to the constraint factory.
        :return: A callable constraint function.
        :raises ValueError: If the constraint key is not registered in the factory.
        """
        constraint = cls.get_registered_object(key)

        return (
            callable(value)
            if not isinstance(value, ConstraintsResolveArgs)
            else constraint(**value)
        )


@ConstraintsFactory.register("max_number")
def max_number_constraint(max_num: Union[int, float]) -> CallableConstraint:
    """
    Create a constraint that limits the total number of requests.

    This constraint monitors both created and processed request counts and stops
    queuing new requests when the created count reaches the limit, and stops
    processing when the processed count reaches the limit.

    Example:
    ::
        # Limit to 1000 requests
        constraint = max_number_constraint(1000)

        # Use in scheduler configuration
        constraints = {"max_number": 500}

    :param max_num: Maximum number of requests allowed. Must be positive.
    :return: A callable constraint function that enforces the request limit.
    :raises ValueError: If max_num is not a number or is not positive.
    """
    if not isinstance(max_num, (int, float)):
        raise ValueError(
            f"max_number constraint must be an int or float, got {type(max_num)}"
        )

    if max_num <= 0:
        raise ValueError(f"max_number constraint must be positive, got {max_num}")

    def _constraint(
        state: SchedulerState, _: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        create_exceeded = state.created_requests >= max_num
        processed_exceeded = state.processed_requests >= max_num

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "max_number": max_num,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
            },
        )

    return _constraint


@ConstraintsFactory.register("max_duration")
def max_duration_constraint(max_duration: Union[int, float]) -> CallableConstraint:
    """
    Create a constraint that limits the total execution duration.

    This constraint monitors the elapsed time since the scheduler started and stops
    both queuing and processing when the duration limit is reached. The duration
    is measured in seconds.

    Example:
    ::
        # Limit to 5 minutes (300 seconds)
        constraint = max_duration_constraint(300)

        # Use in scheduler configuration
        constraints = {"max_duration": 120.0}

    :param max_duration: Maximum execution duration in seconds. Must be positive.
    :return: A callable constraint function that enforces the duration limit.
    :raises ValueError: If max_duration is not a number or is not positive.
    """
    if not isinstance(max_duration, (int, float)):
        raise ValueError(
            f"max_duration constraint must be an int or float, got {type(max_duration)}"
        )

    if max_duration <= 0:
        raise ValueError(
            f"max_duration constraint must be positive, got {max_duration}"
        )

    def _constraint(
        state: SchedulerState, _: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        current_time = time.time()
        elapsed = current_time - state.start_time
        duration_exceeded = elapsed >= max_duration

        return SchedulerUpdateAction(
            request_queuing="stop" if duration_exceeded else "continue",
            request_processing="stop_local" if duration_exceeded else "continue",
            metadata={
                "max_duration": max_duration,
                "elapsed_time": elapsed,
                "duration_exceeded": duration_exceeded,
                "start_time": state.start_time,
                "current_time": current_time,
            },
        )

    return _constraint


@ConstraintsFactory.register("max_errors")
def max_errors_constraint(max_errors: Union[int, float]) -> CallableConstraint:
    """
    Create a constraint that limits the total number of errors.

    This constraint monitors the cumulative error count and stops both queuing
    and processing when the error limit is reached. This provides a circuit
    breaker mechanism to prevent continued execution when error rates are high.

    Example:
    ::
        # Allow up to 10 errors before stopping
        constraint = max_errors_constraint(10)

        # Use in scheduler configuration
        constraints = {"max_errors": 5}

    :param max_errors: Maximum number of errors allowed. Must be non-negative.
    :return: A callable constraint function that enforces the error limit.
    :raises ValueError: If max_errors is not a number or is negative.
    """
    if not isinstance(max_errors, (int, float)):
        raise ValueError(
            f"max_errors constraint must be an int or float, got {type(max_errors)}"
        )

    if max_errors < 0:
        raise ValueError(
            f"max_errors constraint must be non-negative, got {max_errors}"
        )

    def _constraint(
        state: SchedulerState, _: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        errors_exceeded = state.errored_requests >= max_errors

        return SchedulerUpdateAction(
            request_queuing="stop" if errors_exceeded else "continue",
            request_processing="stop_all" if errors_exceeded else "continue",
            metadata={
                "max_errors": max_errors,
                "errors_exceeded": errors_exceeded,
                "current_errors": state.errored_requests,
            },
        )

    return _constraint


@ConstraintsFactory.register("max_error_rate")
def max_error_rate_constraint(
    max_error_rate: Union[int, float],
    window_size: Optional[Union[int, float]] = None,
) -> CallableConstraint:
    """
    Create a constraint that limits error rate within a sliding window.

    This constraint monitors the error rate over a sliding window of recent requests
    and stops processing when the error rate exceeds the threshold within the window.
    This provides more responsive error handling than global error rate constraints.

    Example:
    ::
        # Stop if error rate exceeds 10% in last 50 requests
        constraint = max_error_rate_constraint(0.1, 50)

        # Use in scheduler configuration with default window
        constraints = {"max_error_rate": 0.05}

        # Use with ConstraintsResolveArgs for complex parameters
        args = ConstraintsResolveArgs({
            "max_error_rate": 0.15,
            "window_size": 100
        })
        constraints = {"max_error_rate": args}

    :param max_error_rate: Maximum error rate allowed (0.0 to 1.0). Must be between
        0 and 1 inclusive.
    :param window_size: Size of the sliding window for calculating error rate.
        Defaults to settings.constraint_max_error_rate_min_processed if None.
        Must be positive.
    :return: A callable constraint function that enforces the error rate limit.
    :raises ValueError: If parameters are invalid or out of range.
    """
    if not isinstance(max_error_rate, (int, float)):
        raise ValueError(
            f"max_error_rate constraint must be int | float, got {type(max_error_rate)}"
        )

    if max_error_rate < 0 or max_error_rate > 1:
        raise ValueError(
            f"max_error_rate constraint must be between 0 and 1, got {max_error_rate}"
        )

    if window_size is None:
        window_size = settings.constraint_max_error_rate_min_processed

    if not isinstance(window_size, (int, float)):
        raise ValueError(
            f"window_size must be an int or float, got {type(window_size)}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    window_errors: list[bool] = []

    def _constraint(
        _: SchedulerState, request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        nonlocal window_errors

        if request_info.status in ["completed", "errored", "cancelled"]:
            window_errors.append(request_info.status == "errored")
            if len(window_errors) > window_size:
                window_errors.pop(0)

            error_count = sum(window_errors)
            processed_count = len(window_errors)
            error_rate = error_count / float(processed_count)

            if processed_count >= window_size and error_rate >= max_error_rate:
                return SchedulerUpdateAction(
                    request_queuing="stop",
                    request_processing="stop_all",
                    metadata={
                        "max_error_rate": max_error_rate,
                        "window_size": window_size,
                        "error_count": error_count,
                        "processed_count": processed_count,
                        "current_error_rate": error_rate,
                        "current_window_size": len(window_errors),
                    },
                )

        return SchedulerUpdateAction(
            request_queuing="continue",
            request_processing="continue",
            metadata={
                "max_error_rate": max_error_rate,
                "window_size": window_size,
            },
        )

    return _constraint


@ConstraintsFactory.register("max_global_error_rate")
def max_global_error_rate_constraint(
    max_error_rate: Union[int, float],
    min_processed: Optional[Union[int, float]] = None,
) -> CallableConstraint:
    """
    Create a constraint that limits the global error rate across all requests.

    This constraint monitors the overall error rate across all processed requests
    and stops processing when the error rate exceeds the threshold after a minimum
    number of requests have been processed. This provides a global view of system
    health but is less responsive than sliding window constraints.

    Example:
    ::
        # Stop if global error rate exceeds 5% after 100 requests
        constraint = max_global_error_rate_constraint(0.05, 100)

        # Use in scheduler configuration with default minimum
        constraints = {"max_global_error_rate": 0.1}

        # Use with ConstraintsResolveArgs for complex parameters
        args = ConstraintsResolveArgs({
            "max_error_rate": 0.08,
            "min_processed": 200
        })
        constraints = {"max_global_error_rate": args}

    :param max_error_rate: Maximum global error rate allowed (0.0 to 1.0).
        Must be between 0 and 1 inclusive.
    :param min_processed: Minimum number of requests that must be processed before
        the constraint becomes active. Defaults to
        settings.constraint_max_error_rate_min_processed if None. Must be positive.
    :return: A callable constraint function that enforces the global error rate limit.
    :raises ValueError: If parameters are invalid or out of range.
    """
    if not isinstance(max_error_rate, (int, float)):
        raise ValueError(
            f"max_error_rate constraint must be int | float, got {type(max_error_rate)}"
        )

    if max_error_rate < 0 or max_error_rate > 1:
        raise ValueError(
            f"max_error_rate constraint must be between 0 and 1, got {max_error_rate}"
        )

    if min_processed is None:
        min_processed = settings.constraint_max_error_rate_min_processed

    if not isinstance(min_processed, (int, float)):
        raise ValueError(
            f"min_processed must be an int or float, got {type(min_processed)}"
        )
    if min_processed <= 0:
        raise ValueError(f"min_processed must be positive, got {min_processed}")

    def _constraint(
        state: SchedulerState, _: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        exceeded_min_processed = state.processed_requests >= min_processed
        error_rate = (
            state.errored_requests / float(state.processed_requests)
            if state.processed_requests > 0
            else 0.0
        )
        exceeded_error_rate = error_rate >= max_error_rate

        return SchedulerUpdateAction(
            request_queuing=(
                "stop" if exceeded_min_processed and exceeded_error_rate else "continue"
            ),
            request_processing=(
                "stop_all"
                if exceeded_min_processed and exceeded_error_rate
                else "continue"
            ),
            metadata={
                "max_error_rate": max_error_rate,
                "min_processed": min_processed,
                "processed_requests": state.processed_requests,
                "errored_requests": state.errored_requests,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            },
        )

    return _constraint
