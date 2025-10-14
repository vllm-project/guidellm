"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts. Constraints evaluate
scheduler state and individual requests to determine whether processing should
continue or stop based on predefined limits. The constraint system enables
sophisticated benchmark stopping criteria through composable constraint types.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator

from guidellm.scheduler.schemas import (
    SchedulerState,
    SchedulerUpdateAction,
    SchedulerUpdateActionProgress,
)
from guidellm.schemas import RequestInfo
from guidellm.settings import settings
from guidellm.utils import InfoMixin, RegistryMixin, StandardBaseModel

__all__ = [
    "Constraint",
    "ConstraintInitializer",
    "ConstraintsInitializerFactory",
    "MaxDurationConstraint",
    "MaxErrorRateConstraint",
    "MaxErrorsConstraint",
    "MaxGlobalErrorRateConstraint",
    "MaxNumberConstraint",
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
        cls,
        orig_info: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG003
    ) -> dict[str, Any]:
        """
        Validate arguments for unserializable constraint creation.

        :param orig_info: Original constraint information before serialization failure
        :param kwargs: Additional arguments (ignored)
        :return: Validated parameters for unserializable constraint creation
        """
        return {"orig_info": orig_info or {}}

    def create_constraint(
        self,
        **kwargs,  # noqa: ARG002
    ) -> Constraint:
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
        self,
        state: SchedulerState,  # noqa: ARG002
        request: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        """
        Raise error since unserializable constraints cannot be invoked.

        :param state: Current scheduler state (unused)
        :param request: Individual request information (unused)
        :raises RuntimeError: Always raised for unserializable constraints
        """
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

    def create_constraint(self, **kwargs) -> Constraint:  # noqa: ARG002
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return self.model_copy()  # type: ignore[return-value]

    def __call__(
        self,
        state: SchedulerState,
        request_info: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state and request count.

        :param state: Current scheduler state with request counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        current_index = max(0, self.current_index)
        max_num = (
            self.max_num
            if isinstance(self.max_num, int | float)
            else self.max_num[min(current_index, len(self.max_num) - 1)]
        )

        create_exceeded = state.created_requests >= max_num
        processed_exceeded = state.processed_requests >= max_num
        remaining_requests = min(max(0, max_num - state.processed_requests), max_num)
        remaining_fraction = remaining_requests / float(max_num)

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "max_number": max_num,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
                "remaining_fraction": remaining_fraction,
                "remaining_requests": remaining_requests,
            },
            progress=SchedulerUpdateActionProgress(
                remaining_fraction=remaining_fraction,
                remaining_requests=remaining_requests,
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


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
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

    def create_constraint(self, **kwargs) -> Constraint:  # noqa: ARG002
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return self.model_copy()  # type: ignore[return-value]

    def __call__(
        self,
        state: SchedulerState,
        request_info: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state and elapsed time.

        :param state: Current scheduler state with start time
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
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
        remaining_fraction = remaining_duration / float(max_duration)

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
            progress=SchedulerUpdateActionProgress(
                remaining_fraction=remaining_fraction,
                remaining_duration=remaining_duration,
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


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
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

    def create_constraint(self, **kwargs) -> Constraint:  # noqa: ARG002
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return self.model_copy()  # type: ignore[return-value]

    def __call__(
        self,
        state: SchedulerState,
        request_info: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current error count.

        :param state: Current scheduler state with error counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
        current_index = max(0, self.current_index)
        max_errors = (
            self.max_errors
            if isinstance(self.max_errors, int | float)
            else self.max_errors[min(current_index, len(self.max_errors) - 1)]
        )
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


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
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

    def create_constraint(self, **kwargs) -> Constraint:  # noqa: ARG002
        """
        Create a new instance of MaxErrorRateConstraint (due to stateful window).

        :param kwargs: Additional keyword arguments (unused)
        :return: New instance of the constraint
        """
        self.current_index += 1

        return self.model_copy()  # type: ignore[return-value]

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
                "window_size": self.window_size,
                "error_count": error_count,
                "processed_count": state.processed_requests,
                "current_window_size": len(self.error_window),
                "current_error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
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


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
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

    def create_constraint(self, **kwargs) -> Constraint:  # noqa: ARG002
        """
        Return self as the constraint instance.

        :param kwargs: Additional keyword arguments (unused)
        :return: Self instance as the constraint
        """
        self.current_index += 1

        return self.model_copy()  # type: ignore[return-value]

    def __call__(
        self,
        state: SchedulerState,
        request_info: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against global error rate.

        :param state: Current scheduler state with global request and error counts
        :param request_info: Individual request information (unused)
        :return: Action indicating whether to continue or stop operations
        """
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
        should_stop = exceeded_min_processed and exceeded_error_rate

        return SchedulerUpdateAction(
            request_queuing="stop" if should_stop else "continue",
            request_processing="stop_all" if should_stop else "continue",
            metadata={
                "max_error_rate": max_error_rate,
                "min_processed": self.min_processed,
                "processed_requests": state.processed_requests,
                "errored_requests": state.errored_requests,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
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
        self,
        state: SchedulerState,
        request_info: RequestInfo,  # noqa: ARG002
    ) -> SchedulerUpdateAction:
        create_exceeded = state.created_requests >= self.num_requests
        processed_exceeded = state.processed_requests >= self.num_requests
        remaining_fraction = min(
            max(0.0, 1.0 - state.processed_requests / float(self.num_requests)), 1.0
        )
        remaining_requests = max(0, self.num_requests - state.processed_requests)

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "num_requests": self.num_requests,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
                "remaining_fraction": remaining_fraction,
                "remaining_requests": remaining_requests,
            },
            progress=SchedulerUpdateActionProgress(
                remaining_fraction=remaining_fraction,
                remaining_requests=remaining_requests,
            ),
        )
