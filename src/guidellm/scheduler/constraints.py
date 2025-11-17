"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts. Constraints evaluate
scheduler state and individual requests to determine whether processing should
continue or stop based on predefined limits. The constraint system enables
sophisticated benchmark stopping criteria through composable constraint types.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import Field, field_validator

from guidellm.scheduler.schemas import (
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.settings import settings
from guidellm.utils import InfoMixin, RegistryMixin

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
    "OverSaturationDetector",
    "PydanticConstraintInitializer",
    "RequestsExhaustedConstraint",
    "SerializableConstraintInitializer",
    "UnserializableConstraintInitializer",
]


@runtime_checkable
class Constraint(Protocol):
    """Protocol for constraint evaluation functions that control scheduler behavior."""

    def __call__(
        self, state: SchedulerState, request: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against scheduler state and request information.

        :param state: Current scheduler state with metrics and timing information
        :param request: Individual request information and metadata
        :return: Action indicating whether to continue or stop scheduler operations
        """


@runtime_checkable
class ConstraintInitializer(Protocol):
    """Protocol for constraint initializer factory functions that create constraints."""

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance from configuration parameters.

        :param kwargs: Configuration parameters for constraint creation
        :return: Configured constraint evaluation function
        """


@runtime_checkable
class SerializableConstraintInitializer(Protocol):
    """Protocol for serializable constraint initializers supporting persistence."""

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        :param args: Positional arguments for constraint configuration
        :param kwargs: Keyword arguments for constraint configuration
        :return: Validated parameter dictionary for constraint creation
        """

    @classmethod
    def model_validate(cls, **kwargs) -> ConstraintInitializer:
        """
        Create validated constraint initializer from configuration.

        :param kwargs: Configuration dictionary for initializer creation
        :return: Validated constraint initializer instance
        """

    def model_dump(self) -> dict[str, Any]:
        """
        Serialize constraint initializer to dictionary format.

        :return: Dictionary representation of constraint initializer
        """

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create constraint instance from this initializer.

        :param kwargs: Additional configuration parameters
        :return: Configured constraint evaluation function
        """


class ConstraintsInitializerFactory(RegistryMixin[ConstraintInitializer]):
    """
    Registry factory for creating and managing constraint initializers.

    Provides centralized access to registered constraint types with support for
    creating constraints from configuration dictionaries, simple values, or
    pre-configured instances. Handles constraint resolution and type validation
    for the scheduler constraint system.

    Example:
    ::
        from guidellm.scheduler import ConstraintsInitializerFactory

        # Register new constraint type
        @ConstraintsInitializerFactory.register("new_constraint")
        class NewConstraint:
            def create_constraint(self, **kwargs) -> Constraint:
                return lambda state, request: SchedulerUpdateAction()

        # Create and use constraint
        constraint = ConstraintsInitializerFactory.create_constraint("new_constraint")
    """

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> ConstraintInitializer:
        """
        Create a constraint initializer for the specified key.

        :param key: Registered constraint initializer key
        :param args: Positional arguments for initializer creation
        :param kwargs: Keyword arguments for initializer creation
        :return: Configured constraint initializer instance
        :raises ValueError: If the key is not registered in the factory
        """
        if cls.registry is None or key not in cls.registry:
            raise ValueError(f"Unknown constraint initializer key: {key}")

        initializer_class = cls.registry[key]

        return (
            initializer_class(*args, **kwargs)  # type: ignore[operator]
            if not isinstance(initializer_class, type)
            or not issubclass(initializer_class, SerializableConstraintInitializer)
            else initializer_class(
                **initializer_class.validated_kwargs(*args, **kwargs)  # type: ignore[misc]
            )
        )

    @classmethod
    def serialize(cls, initializer: ConstraintInitializer) -> dict[str, Any]:
        """
        Serialize constraint initializer to dictionary format.

        :param initializer: Constraint initializer to serialize
        :return: Dictionary representation or unserializable placeholder
        """
        if isinstance(initializer, SerializableConstraintInitializer):
            return initializer.model_dump()
        else:
            unserializable = UnserializableConstraintInitializer(
                orig_info=InfoMixin.extract_from_obj(initializer)
            )
            return unserializable.model_dump()

    @classmethod
    def deserialize(
        cls, initializer_dict: dict[str, Any]
    ) -> SerializableConstraintInitializer | UnserializableConstraintInitializer:
        """
        Deserialize constraint initializer from dictionary format.

        :param initializer_dict: Dictionary representation of constraint initializer
        :return: Reconstructed constraint initializer instance
        :raises ValueError: If constraint type is unknown or cannot be deserialized
        """
        if initializer_dict.get("type_") == "unserializable":
            return UnserializableConstraintInitializer.model_validate(initializer_dict)

        if (
            cls.registry is not None
            and initializer_dict.get("type_")
            and initializer_dict["type_"] in cls.registry
        ):
            initializer_class = cls.registry[initializer_dict["type_"]]
            if hasattr(initializer_class, "model_validate"):
                return initializer_class.model_validate(initializer_dict)  # type: ignore[return-value]
            else:
                return initializer_class(**initializer_dict)  # type: ignore[return-value,operator]

        raise ValueError(
            f"Cannot deserialize unknown constraint initializer: "
            f"{initializer_dict.get('type_', 'unknown')}"
        )

    @classmethod
    def create_constraint(cls, key: str, *args, **kwargs) -> Constraint:
        """
        Create a constraint instance for the specified key.

        :param key: Registered constraint initializer key
        :param args: Positional arguments for constraint creation
        :param kwargs: Keyword arguments for constraint creation
        :return: Configured constraint function ready for evaluation
        :raises ValueError: If the key is not registered in the factory
        """
        return cls.create(key, *args, **kwargs).create_constraint()

    @classmethod
    def resolve(
        cls,
        initializers: dict[
            str,
            Any | dict[str, Any] | Constraint | ConstraintInitializer,
        ],
    ) -> dict[str, Constraint]:
        """
        Resolve mixed constraint specifications to callable constraints.

        :param initializers: Dictionary mapping constraint keys to specifications
        :return: Dictionary mapping constraint keys to callable functions
        :raises ValueError: If any key is not registered in the factory
        """
        constraints = {}

        for key, val in initializers.items():
            if isinstance(val, Constraint):
                constraints[key] = val
            elif isinstance(val, ConstraintInitializer):
                constraints[key] = val.create_constraint()
            elif isinstance(val, dict):
                constraints[key] = cls.create_constraint(key, **val)
            else:
                constraints[key] = cls.create_constraint(key, val)

        return constraints

    @classmethod
    def resolve_constraints(
        cls,
        constraints: dict[str, Any | dict[str, Any] | Constraint],
    ) -> dict[str, Constraint]:
        """
        Resolve constraints from mixed constraint specifications.

        :param constraints: Dictionary mapping constraint keys to specifications
        :return: Dictionary mapping constraint keys to callable functions
        :raises ValueError: If any constraint key is not registered
        """
        resolved_constraints = {}

        for key, val in constraints.items():
            if isinstance(val, Constraint):
                resolved_constraints[key] = val
            elif isinstance(val, dict):
                resolved_constraints[key] = cls.create_constraint(key, **val)
            else:
                resolved_constraints[key] = cls.create_constraint(key, val)

        return resolved_constraints


class PydanticConstraintInitializer(StandardBaseModel, ABC, InfoMixin):
    """
    Abstract base for Pydantic-based constraint initializers.

    Provides standardized serialization, validation, and metadata handling for
    constraint initializers using Pydantic models. Subclasses implement specific
    constraint creation logic while inheriting validation and persistence support.
    """

    type_: str = Field(description="Type identifier for the constraint initializer")

    @property
    def info(self) -> dict[str, Any]:
        """
        Extract serializable information from this constraint initializer.

        :return: Dictionary containing constraint configuration and metadata
        """
        return self.model_dump()

    @classmethod
    @abstractmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        Must be implemented by subclasses to handle their specific parameter patterns
        and validation requirements.

        :param args: Positional arguments passed to the constraint
        :param kwargs: Keyword arguments passed to the constraint
        :return: Validated dictionary of parameters for constraint creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @abstractmethod
    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance.

        Must be implemented by subclasses to return their specific constraint type
        with appropriate configuration and validation.

        :param kwargs: Additional keyword arguments (usually unused)
        :return: Configured constraint instance
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...


class UnserializableConstraintInitializer(PydanticConstraintInitializer):
    """
    Placeholder for constraints that cannot be serialized or executed.

    Represents constraint initializers that failed serialization or contain
    non-serializable components. Cannot be executed and raises errors when
    invoked to prevent runtime failures from invalid constraint state.
    """

    type_: Literal["unserializable"] = "unserializable"  # type: ignore[assignment]
    orig_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Original constraint information before serialization failure",
    )

    @classmethod
    def validated_kwargs(
        cls, orig_info: dict[str, Any] | None = None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate arguments for unserializable constraint creation.

        :param orig_info: Original constraint information before serialization failure
        :param kwargs: Additional arguments (ignored)
        :return: Validated parameters for unserializable constraint creation
        """
        return {"orig_info": orig_info or {}}

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Raise error for unserializable constraint creation attempt.

        :param kwargs: Additional keyword arguments (unused)
        :raises RuntimeError: Always raised since unserializable constraints
            cannot be executed
        """
        raise RuntimeError(
            "Cannot create constraint from unserializable constraint instance. "
            "This constraint cannot be serialized and therefore cannot be executed."
        )

    def __call__(
        self, state: SchedulerState, request: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Raise error since unserializable constraints cannot be invoked.

        :param state: Current scheduler state (unused)
        :param request: Individual request information (unused)
        :raises RuntimeError: Always raised for unserializable constraints
        """
        _ = (state, request)  # Unused parameters
        raise RuntimeError(
            "Cannot invoke unserializable constraint instance. "
            "This constraint was not properly serialized and cannot be executed."
        )


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
    ["max_number", "max_num", "max_requests", "max_req"]
)
class MaxNumberConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on maximum request counts.

    Stops request queuing when created requests reach the limit and stops local
    request processing when processed requests reach the limit. Provides progress
    tracking based on remaining requests and completion fraction.
    """

    type_: Literal["max_number"] = "max_number"  # type: ignore[assignment]
    max_num: int | float | list[int | float] = Field(
        description="Maximum number of requests allowed before triggering constraint",
    )
    current_index: int = Field(
        default=-1, description="Current index for list-based max_num values"
    )

    @classmethod
    def validated_kwargs(
        cls, max_num: int | float | list[int | float], **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxNumberConstraint creation.

        :param max_num: Maximum number of requests to allow
        :param kwargs: Supports max_num, max_number, max_requests, max_req,
            and optional type_
        :return: Validated dictionary with max_num and type_ fields
        """
        aliases = ["max_number", "max_num", "max_requests", "max_req"]
        for alias in aliases:
            if max_num is None:
                max_num = kwargs.get(alias)

        return {"max_num": max_num, "current_index": kwargs.get("current_index", -1)}

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
        Evaluate constraint against current scheduler state and request count.

        :param state: Current scheduler state with request counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        _ = request_info  # Unused parameters
        current_index = max(0, self.current_index)
        max_num = (
            self.max_num
            if isinstance(self.max_num, int | float)
            else self.max_num[min(current_index, len(self.max_num) - 1)]
        )

        create_exceeded = state.created_requests >= max_num
        processed_exceeded = state.processed_requests >= max_num
        remaining_requests = min(max(0, max_num - state.processed_requests), max_num)
        stop_time = (
            None if remaining_requests > 0 else request_info.completed_at or time.time()
        )

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "max_number": max_num,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
                "remaining_requests": remaining_requests,
                "stop_time": stop_time,
            },
            progress=SchedulerProgress(
                remaining_requests=remaining_requests,
                total_requests=max_num,
                stop_time=stop_time,
            ),
        )

    @field_validator("max_num")
    @classmethod
    def _validate_max_num(
        cls, value: int | float | list[int | float]
    ) -> int | float | list[int | float]:
        if not isinstance(value, list):
            value = [value]
        for val in value:
            if not val:
                raise ValueError(
                    f"max_num must be set and truthful, received {value} ({val} failed)"
                )
            if not isinstance(val, int | float) or val <= 0:
                raise ValueError(
                    f"max_num must be a positive num, received {value} ({val} failed)"
                )

        return value[0] if isinstance(value, list) and len(value) == 1 else value


@ConstraintsInitializerFactory.register(
    ["max_duration", "max_dur", "max_sec", "max_seconds", "max_min", "max_minutes"]
)
class MaxDurationConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on maximum time duration.

    Stops both request queuing and processing when the elapsed time since scheduler
    start exceeds the maximum duration. Provides progress tracking based on
    remaining time and completion fraction.
    """

    type_: Literal["max_duration"] = "max_duration"  # type: ignore[assignment]
    max_duration: int | float | list[int | float] = Field(
        description="Maximum duration in seconds before triggering constraint"
    )
    current_index: int = Field(default=-1, description="Current index in duration list")

    @classmethod
    def validated_kwargs(
        cls, max_duration: int | float | list[int | float] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxDurationConstraint creation.

        :param max_duration: Maximum duration in seconds
        :param kwargs: Supports max_duration, max_dur, max_sec, max_seconds,
            max_min, max_minutes, and optional type_
        :return: Validated dictionary with max_duration and type_ fields
        """
        seconds_aliases = ["max_dur", "max_sec", "max_seconds"]
        for alias in seconds_aliases:
            if max_duration is None:
                max_duration = kwargs.get(alias)
        minutes_aliases = ["max_min", "max_minutes"]
        for alias in minutes_aliases:
            minutes = kwargs.get(alias)
            if minutes is not None and max_duration is None:
                max_duration = minutes * 60

        return {
            "max_duration": max_duration,
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
        Evaluate constraint against current scheduler state and elapsed time.

        :param state: Current scheduler state with start time
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        _ = request_info  # Unused parameters
        current_index = max(0, self.current_index)
        max_duration = (
            self.max_duration
            if isinstance(self.max_duration, int | float)
            else self.max_duration[min(current_index, len(self.max_duration) - 1)]
        )

        current_time = time.time()
        elapsed = current_time - state.start_time
        duration_exceeded = elapsed >= max_duration
        remaining_duration = min(max(0.0, max_duration - elapsed), max_duration)
        stop_time = None if not duration_exceeded else state.start_time + max_duration

        return SchedulerUpdateAction(
            request_queuing="stop" if duration_exceeded else "continue",
            request_processing="stop_local" if duration_exceeded else "continue",
            metadata={
                "max_duration": max_duration,
                "elapsed_time": elapsed,
                "duration_exceeded": duration_exceeded,
                "start_time": state.start_time,
                "current_time": current_time,
                "stop_time": stop_time,
            },
            progress=SchedulerProgress(
                remaining_duration=remaining_duration,
                total_duration=max_duration,
                stop_time=stop_time,
            ),
        )

    @field_validator("max_duration")
    @classmethod
    def _validate_max_duration(
        cls, value: int | float | list[int | float]
    ) -> int | float | list[int | float]:
        if not isinstance(value, list):
            value = [value]
        for val in value:
            if not val:
                raise ValueError(
                    "max_duration must be set and truthful, "
                    f"received {value} ({val} failed)"
                )
            if not isinstance(val, int | float) or val <= 0:
                raise ValueError(
                    "max_duration must be a positive num,"
                    f"received {value} ({val} failed)"
                )

        return value[0] if isinstance(value, list) and len(value) == 1 else value


@ConstraintsInitializerFactory.register(
    ["max_errors", "max_err", "max_error", "max_errs"]
)
class MaxErrorsConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on absolute error count.

    Stops both request queuing and all request processing when the total number
    of errored requests reaches the maximum threshold. Uses global error tracking
    across all requests for immediate constraint evaluation.
    """

    type_: Literal["max_errors"] = "max_errors"  # type: ignore[assignment]
    max_errors: int | float | list[int | float] = Field(
        description="Maximum number of errors allowed before triggering constraint",
    )
    current_index: int = Field(default=-1, description="Current index in error list")

    @classmethod
    def validated_kwargs(
        cls, max_errors: int | float | list[int | float] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxErrorsConstraint creation.

        :param max_errors: Maximum number of errors to allow
        :param kwargs: Supports max_errors, max_err, max_error, max_errs,
            and optional type_
        :return: Validated dictionary with max_errors and type_ fields
        """
        aliases = ["max_errors", "max_err", "max_error", "max_errs"]
        for alias in aliases:
            if max_errors is None:
                max_errors = kwargs.get(alias)

        return {
            "max_errors": max_errors,
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
            self.max_errors
            if isinstance(self.max_errors, int | float)
            else self.max_errors[min(current_index, len(self.max_errors) - 1)]
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

    @field_validator("max_errors")
    @classmethod
    def _validate_max_errors(
        cls, value: int | float | list[int | float]
    ) -> int | float | list[int | float]:
        if not isinstance(value, list):
            value = [value]
        for val in value:
            if not val:
                raise ValueError(
                    "max_errors must be set and truthful, "
                    f"received {value} ({val} failed)"
                )
            if not isinstance(val, int | float) or val <= 0:
                raise ValueError(
                    f"max_errors must be a positive num,received {value} ({val} failed)"
                )

        return value[0] if isinstance(value, list) and len(value) == 1 else value


@ConstraintsInitializerFactory.register(
    ["max_error_rate", "max_err_rate", "max_errors_rate"]
)
class MaxErrorRateConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on sliding window error rate.

    Tracks error status of recent requests in a sliding window and stops all
    processing when the error rate exceeds the threshold. Only applies the
    constraint after processing enough requests to fill the minimum window size
    for statistical significance.
    """

    type_: Literal["max_error_rate"] = "max_error_rate"  # type: ignore[assignment]
    max_error_rate: int | float | list[int | float] = Field(
        description="Maximum error rate allowed (0.0, 1.0)"
    )
    window_size: int | float = Field(
        default=30,
        gt=0,
        description="Size of sliding window for calculating error rate",
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
        cls, max_error_rate: int | float | list[int | float], **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxErrorRateConstraint creation.

        :param max_error_rate: Maximum error rate to allow
        :param kwargs: Supports max_error_rate, max_err_rate, max_errors_rate,
            optional window_size, and optional type_
        :return: Validated dictionary with max_error_rate, window_size,
            and type_ fields
        """
        aliases = ["max_error_rate", "max_err_rate", "max_errors_rate"]
        for alias in aliases:
            if max_error_rate is None:
                max_error_rate = kwargs.get(alias)

        return {
            "max_error_rate": max_error_rate,
            "window_size": kwargs.get(
                "window_size", settings.constraint_error_window_size
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
            self.max_error_rate
            if isinstance(self.max_error_rate, int | float)
            else self.max_error_rate[min(current_index, len(self.max_error_rate) - 1)]
        )

        if request_info.status in ["completed", "errored", "cancelled"]:
            self.error_window.append(request_info.status == "errored")
            if len(self.error_window) > self.window_size:
                self.error_window.pop(0)

        error_count = sum(self.error_window)
        window_requests = len(self.error_window)
        error_rate = (
            error_count / float(window_requests) if window_requests > 0 else 0.0
        )
        exceeded_min_processed = state.processed_requests >= self.window_size
        exceeded_error_rate = error_rate >= max_error_rate
        exceeded = exceeded_min_processed and exceeded_error_rate
        stop_time = None if not exceeded else request_info.completed_at or time.time()

        return SchedulerUpdateAction(
            request_queuing="stop" if exceeded else "continue",
            request_processing="stop_all" if exceeded else "continue",
            metadata={
                "max_error_rate": max_error_rate,
                "window_size": self.window_size,
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

    @field_validator("max_error_rate")
    @classmethod
    def _validate_max_error_rate(
        cls, value: int | float | list[int | float]
    ) -> int | float | list[int | float]:
        if not isinstance(value, list):
            value = [value]
        for val in value:
            if not val:
                raise ValueError(
                    "max_error_rate must be set and truthful, "
                    f"received {value} ({val} failed)"
                )
            if not isinstance(val, int | float) or val <= 0 or val >= 1:
                raise ValueError(
                    "max_error_rate must be a number between 0 and 1,"
                    f"received {value} ({val} failed)"
                )

        return value[0] if isinstance(value, list) and len(value) == 1 else value


@ConstraintsInitializerFactory.register(
    ["max_global_error_rate", "max_global_err_rate", "max_global_errors_rate"]
)
class MaxGlobalErrorRateConstraint(PydanticConstraintInitializer):
    """
    Constraint that limits execution based on global error rate.

    Calculates error rate across all processed requests and stops all processing
    when the rate exceeds the threshold. Only applies the constraint after
    processing the minimum number of requests to ensure statistical significance
    for global error rate calculations.
    """

    type_: Literal["max_global_error_rate"] = "max_global_error_rate"  # type: ignore[assignment]
    max_error_rate: int | float = Field(
        description="Maximum error rate allowed (0.0 to 1.0)"
    )
    min_processed: int | float | None = Field(
        default=30,
        gt=0,
        description="Minimum requests processed before applying error rate constraint",
    )
    current_index: int = Field(
        default=-1, description="Current index for list-based max_error_rate values"
    )

    @classmethod
    def validated_kwargs(
        cls, max_error_rate: int | float | list[int | float], **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for MaxGlobalErrorRateConstraint creation.

        :param max_error_rate: Maximum error rate to allow
        :param kwargs: Supports max_global_error_rate, max_global_err_rate,
            max_global_errors_rate, optional min_processed, and optional type_
        :return: Validated dictionary with max_error_rate, min_processed,
            and type_ fields
        """
        for alias in [
            "max_global_error_rate",
            "max_global_err_rate",
            "max_global_errors_rate",
        ]:
            if max_error_rate is None:
                max_error_rate = kwargs.get(alias)

        return {
            "max_error_rate": max_error_rate,
            "min_processed": kwargs.get(
                "min_processed", settings.constraint_error_min_processed
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
            self.max_error_rate
            if isinstance(self.max_error_rate, int | float)
            else self.max_error_rate[min(current_index, len(self.max_error_rate) - 1)]
        )

        exceeded_min_processed = (
            self.min_processed is None or state.processed_requests >= self.min_processed
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
                "min_processed": self.min_processed,
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

    @field_validator("max_error_rate")
    @classmethod
    def _validate_max_error_rate(
        cls, value: int | float | list[int | float]
    ) -> int | float | list[int | float]:
        if not isinstance(value, list):
            value = [value]
        for val in value:
            if not val:
                raise ValueError(
                    "max_error_rate must be set and truthful, "
                    f"received {value} ({val} failed)"
                )
            if not isinstance(val, int | float) or val <= 0 or val >= 1:
                raise ValueError(
                    "max_error_rate must be a number between 0 and 1,"
                    f"received {value} ({val} failed)"
                )

        return value[0] if isinstance(value, list) and len(value) == 1 else value


class RequestsExhaustedConstraint(StandardBaseModel, InfoMixin):
    type_: Literal["requests_exhausted"] = "requests_exhausted"  # type: ignore[assignment]
    num_requests: int

    @property
    def info(self) -> dict[str, Any]:
        """
        Extract serializable information from this constraint initializer.

        :return: Dictionary containing constraint configuration and metadata
        """
        return self.model_dump()

    def __call__(
        self, state: SchedulerState, request: RequestInfo
    ) -> SchedulerUpdateAction:
        _ = request  # Unused parameter
        create_exceeded = state.created_requests >= self.num_requests
        processed_exceeded = state.processed_requests >= self.num_requests
        remaining_requests = max(0, self.num_requests - state.processed_requests)
        stop_time = (
            None if remaining_requests > 0 else request.completed_at or time.time()
        )

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "num_requests": self.num_requests,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
                "remaining_requests": remaining_requests,
                "stop_time": stop_time,
            },
            progress=SchedulerProgress(
                remaining_requests=remaining_requests,
                total_requests=self.num_requests,
                stop_time=stop_time,
            ),
        )


# Over-saturation detection classes
class OverSaturationDetectorBase(ABC):
    @abstractmethod
    def add_finished(self, request: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def add_started(self, request: dict[str, Any]) -> None:
        pass

    def update_duration(self, duration: float) -> None:
        self.duration = duration

    @abstractmethod
    def check_alert(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


def approx_t_ppf(p, df):
    """
    Approximates the percent point function (PPF) for the t-distribution.
    This provides a close but not exact value compared to scipy.stats.t.ppf,
    but is much faster.

    Reference:
        Milton Abramowitz and Irene A. Stegun (Eds.). (1965).
        Handbook of Mathematical Functions: with Formulas, Graphs,
        and Mathematical Tables. Dover Publications.

        An electronic version of this book is available at:
        https://personal.math.ubc.ca/~cbm/aands/.

    Args:
        p (float): The probability (e.g., 0.975 for a 95% CI).
        df (float): The degrees of freedom.
    """
    dof = df
    if dof <= 0:
        return float("nan")

    # 1. Approximate the PPF of the Normal distribution (z-score)
    # Uses Abramowitz & Stegun formula 26.2.23.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    numerical_stability_threshold = 0.5
    if p < numerical_stability_threshold:
        t = math.sqrt(-2.0 * math.log(p))
        z = -(
            t
            - ((c[2] * t + c[1]) * t + c[0])
            / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
        )
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        z = t - ((c[2] * t + c[1]) * t + c[0]) / (
            ((d[2] * t + d[1]) * t + d[0]) * t + 1.0
        )

    # 2. Convert the z-score to a t-score
    # Uses the Cornish-Fisher expansion (first few terms).
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z

    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z4 + 16.0 * z3 + 3.0 * z2) / 96.0

    # Adjust z using the degrees of freedom (dof)
    return z + g1 / dof + g2 / (dof * dof)


class SlopeChecker:
    def __init__(
        self, moe_threshold: float = 1.0, confidence: float = 0.95, eps: float = 1e-12
    ) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.moe_threshold = moe_threshold
        self.eps = eps
        self.confidence = confidence
        self.slope: float | None = None
        self.margin_of_error: float | None = None

    def add_data_point(self, x_new: float, y_new: float) -> None:
        """
        Integrates a new data point into the accumulated statistics.
        This operation is O(1).

        Args:
            x_new (float): The new x-coordinate.
            y_new (float): The new y-coordinate.
        """
        self.n += 1
        self.sum_x += x_new
        self.sum_y += y_new
        self.sum_xy += x_new * y_new
        self.sum_x2 += x_new**2
        self.sum_y2 += y_new**2

    def remove_data_point(self, x_old: float, y_old: float) -> None:
        """
        Remove a data point from the accumulated statistics.
        This operation is O(1).

        Args:
            x_old (float): The x-coordinate to remove.
            y_old (float): The y-coordinate to remove.
        """
        self.n -= 1
        self.sum_x -= x_old
        self.sum_y -= y_old
        self.sum_xy -= x_old * y_old
        self.sum_x2 -= x_old**2
        self.sum_y2 -= y_old**2

    def check_slope(self, effective_n: float) -> bool:
        minimal_n_for_slope_estimation = 3
        if effective_n < minimal_n_for_slope_estimation:
            return False

        # Calculate sums of squares and cross-products
        # These formulas are numerically stable for online calculation.
        centered_sum_xx = self.sum_x2 - (self.sum_x**2) / self.n
        centered_sum_xy = self.sum_xy - (self.sum_x * self.sum_y) / self.n
        centered_sum_yy = self.sum_y2 - (self.sum_y**2) / self.n

        # Safeguard against division by zero for SS_xx
        centered_sum_xx_safe = max(centered_sum_xx, self.eps)

        slope = centered_sum_xy / centered_sum_xx_safe

        # Calculate Residual Sum of Squares (RSS)
        # This is a direct calculation using the sums of squares.
        residual_sum_of_squares = centered_sum_yy - (
            centered_sum_xy**2 / centered_sum_xx_safe
        )

        # Ensure RSS is non-negative due to potential floating point inaccuracies
        residual_sum_of_squares = max(residual_sum_of_squares, 0.0)

        # Degrees of freedom for standard error (n - 2 for simple linear regression)
        dof = effective_n - 2

        residual_variance = residual_sum_of_squares / dof
        standard_error = (residual_variance / centered_sum_xx_safe) ** 0.5

        # t-critical value
        alpha = 1 - self.confidence
        t_crit = approx_t_ppf(1 - alpha / 2, df=dof)

        # Margin Of Error
        margin_of_error = t_crit * standard_error / max(slope, self.eps)

        self.slope = slope
        self.margin_of_error = margin_of_error
        return (slope > 0) and (margin_of_error < self.moe_threshold)


class OverSaturationDetector(OverSaturationDetectorBase):
    def __init__(
        self,
        minimum_duration: float = 30.0,
        minimum_ttft: float = 2.5,
        maximum_window_seconds: float = 120.0,
        moe_threshold: float = 2.0,
        maximum_window_ratio: float = 0.75,
        minimum_window_size: int = 5,
        confidence: float = 0.95,
        eps: float = 1e-12,
    ) -> None:
        self.minimum_duration = minimum_duration
        self.minimum_ttft = minimum_ttft
        self.maximum_window_seconds = maximum_window_seconds
        self.maximum_window_ratio = maximum_window_ratio
        self.minimum_window_size = minimum_window_size
        self.moe_threshold = moe_threshold
        self.confidence = confidence
        self.eps = eps
        self.reset()

    def add_finished(self, request: dict[str, Any]) -> None:
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft is not None:
            self.total_finished_ever += 1
            self.finished_requests.append(request)
            if ttft > self.minimum_ttft:
                self.ttft_violations_counter += 1
            self.ttft_slope_checker.add_data_point(duration, ttft)

    def remove_finished(self, request: dict[str, Any]) -> None:
        del self.finished_requests[0]
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft > self.minimum_ttft:
            self.ttft_violations_counter -= 1
        self.ttft_slope_checker.remove_data_point(duration, ttft)

    def add_started(self, request: dict[str, Any]) -> None:
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        if concurrent is not None:
            self.total_started_ever += 1
            self.started_requests.append(request)
            self.concurrent_slope_checker.add_data_point(duration, concurrent)

    def remove_started(self, request: dict[str, Any]) -> None:
        del self.started_requests[0]
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        self.concurrent_slope_checker.remove_data_point(duration, concurrent)

    def update_duration(self, duration: float) -> None:
        self.duration = duration

        maximum_finished_window_size = int(
            self.total_finished_ever * self.maximum_window_ratio
        )
        while len(self.finished_requests) > maximum_finished_window_size:
            self.remove_finished(self.finished_requests[0])

        while (len(self.finished_requests) > 0) and (
            (
                time_since_earliest_request := duration
                - self.finished_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self.remove_finished(self.finished_requests[0])

        maximum_started_window_size = int(
            self.total_started_ever * self.maximum_window_ratio
        )
        while len(self.started_requests) > maximum_started_window_size:
            self.remove_started(self.started_requests[0])

        while (len(self.started_requests) > 0) and (
            (
                time_since_earliest_request := duration  # noqa: F841
                - self.started_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self.remove_started(self.started_requests[0])

    def check_alert(self) -> bool:
        # Use duration as the maximum n value since requests from the
        # same second are highly correlated, this is simple and good enough
        # given that the MOE has a custom threshold anyway.
        concurrent_n = min(self.duration, self.concurrent_slope_checker.n)
        ttft_n = min(self.duration, self.ttft_slope_checker.n)

        if (
            (self.duration < self.minimum_duration)
            or (self.ttft_slope_checker.n > self.ttft_violations_counter * 2)
            or (self.duration < self.minimum_ttft)
            or (concurrent_n < self.minimum_window_size)
        ):
            return False

        is_concurrent_slope_positive = self.concurrent_slope_checker.check_slope(
            concurrent_n
        )

        if ttft_n < self.minimum_window_size:
            return is_concurrent_slope_positive

        is_ttft_slope_positive = self.ttft_slope_checker.check_slope(ttft_n)

        return is_concurrent_slope_positive and is_ttft_slope_positive

    def reset(self) -> None:
        self.duration = 0.0
        self.started_requests: list[dict[str, Any]] = []
        self.finished_requests: list[dict[str, Any]] = []
        self.ttft_violations_counter = 0
        self.total_finished_ever = 0
        self.total_started_ever = 0
        self.concurrent_slope_checker = SlopeChecker(
            moe_threshold=self.moe_threshold, confidence=self.confidence, eps=self.eps
        )
        self.ttft_slope_checker = SlopeChecker(
            moe_threshold=self.moe_threshold, confidence=self.confidence, eps=self.eps
        )


class OverSaturationConstraint:  # type: ignore[misc]
    """
    Constraint that limits execution based on over-saturation detection.

    Stops request queuing when over-saturation is detected (i.e response-rate
    doesn't keep up with the request-rate).
    """

    def __init__(
        self,
        over_saturation_detector: OverSaturationDetector,
        stop_over_saturated: bool,
    ) -> None:
        self.over_saturation_detector = over_saturation_detector
        self.stop_over_saturated = stop_over_saturated

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state.

        :param state: Current scheduler state.
        :param request_info: Individual request information.
        :return: Action indicating whether to continue or stop operations.
        """
        duration = time.time() - state.start_time

        if request_info.status == "in_progress":
            concurrent_requests = state.processing_requests
            self.over_saturation_detector.add_started(
                {"concurrent_requests": concurrent_requests, "duration": duration}
            )
        elif (
            request_info.status == "completed"
            and request_info.timings
            and request_info.timings.first_iteration
        ):
            ttft = (
                request_info.timings.first_iteration
                - request_info.timings.request_start
            )
            self.over_saturation_detector.add_finished(
                {"ttft": ttft, "duration": duration}
            )

        self.over_saturation_detector.update_duration(duration)
        is_over_saturated = self.over_saturation_detector.check_alert()

        ttft_slope = self.over_saturation_detector.ttft_slope_checker.slope
        ttft_slope_moe = (
            self.over_saturation_detector.ttft_slope_checker.margin_of_error
        )
        ttft_n = self.over_saturation_detector.ttft_slope_checker.n
        ttft_violations = self.over_saturation_detector.ttft_violations_counter
        concurrent_slope = self.over_saturation_detector.concurrent_slope_checker.slope
        concurrent_slope_moe = (
            self.over_saturation_detector.concurrent_slope_checker.margin_of_error
        )
        concurrent_n = self.over_saturation_detector.concurrent_slope_checker.n

        should_stop = is_over_saturated and self.stop_over_saturated
        return SchedulerUpdateAction(
            request_queuing="stop" if should_stop else "continue",
            request_processing="stop_all" if should_stop else "continue",
            metadata={
                "ttft_slope": ttft_slope,
                "ttft_slope_moe": ttft_slope_moe,
                "ttft_n": ttft_n,
                "ttft_violations": ttft_violations,
                "concurrent_slope": concurrent_slope,
                "concurrent_slope_moe": concurrent_slope_moe,
                "concurrent_n": concurrent_n,
                "is_over_saturated": is_over_saturated,
            },
        )


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
    ["stop_over_saturated", "stop_over_sat", "stop_osd"]
)
class OverSaturationConstraintInitializer(PydanticConstraintInitializer):
    """Factory for creating OverSaturationConstraint instances from configuration."""

    type_: Literal["stop_over_saturated"] = "stop_over_saturated"  # type: ignore[assignment]
    stop_over_saturated: bool = Field(
        description="Whether to stop the benchmark if the model is over-saturated",
    )
    min_seconds: int | float = Field(
        default_factory=lambda: settings.constraint_over_saturation_min_seconds,
        ge=0,
        description="Minimum seconds before checking for over-saturation",
    )
    max_window_seconds: int | float = Field(
        default_factory=lambda: settings.constraint_over_saturation_max_window_seconds,
        ge=0,
        description="Maximum over-saturation checking window size in seconds",
    )

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a OverSaturationConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured OverSaturationConstraint instance.
        """
        over_saturation_detector = OverSaturationDetector(
            minimum_duration=self.min_seconds,
            maximum_window_seconds=self.max_window_seconds,
        )
        return OverSaturationConstraint(
            over_saturation_detector=over_saturation_detector,
            stop_over_saturated=self.stop_over_saturated,
        )

    @classmethod
    def validated_kwargs(cls, stop_over_saturated: bool, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for OverSaturationConstraint creation.

        :param stop_over_saturated: Whether to stop the benchmark if over-saturated
        :param kwargs: Supports stop_over_saturated, stop_over_sat, stop_osd
        :return: Validated dictionary with stop_over_saturated field
        """
        aliases = ["stop_over_saturated", "stop_over_sat", "stop_osd"]
        for alias in aliases:
            stop_over_saturated = stop_over_saturated or kwargs.get(alias)

        return {"stop_over_saturated": stop_over_saturated}
