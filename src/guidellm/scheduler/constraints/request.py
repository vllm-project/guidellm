"""
Request-based constraint implementations.

Provides constraint types for limiting benchmark execution based on request counts
and time duration. These constraints monitor request creation, processing, and
elapsed time to determine when to stop benchmark execution.
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

from pydantic import Field, field_validator

from guidellm.scheduler.constraints.constraint import (
    Constraint,
    PydanticConstraintInitializer,
)
from guidellm.scheduler.constraints.factory import ConstraintsInitializerFactory
from guidellm.scheduler.schemas import (
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.utils import InfoMixin

__all__ = [
    "MaxDurationConstraint",
    "MaxNumberConstraint",
    "RequestsExhaustedConstraint",
]


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

        start_time = state.start_requests_time or state.start_time
        current_time = time.time()
        elapsed = current_time - start_time
        duration_exceeded = elapsed >= max_duration
        remaining_duration = min(max(0.0, max_duration - elapsed), max_duration)
        stop_time = None if not duration_exceeded else start_time + max_duration

        return SchedulerUpdateAction(
            request_queuing="stop" if duration_exceeded else "continue",
            request_processing="stop_local" if duration_exceeded else "continue",
            metadata={
                "max_duration": max_duration,
                "elapsed_time": elapsed,
                "duration_exceeded": duration_exceeded,
                "start_time": start_time,
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
