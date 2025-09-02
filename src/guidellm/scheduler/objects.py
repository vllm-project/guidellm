"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations. Central to
the scheduler architecture, enabling request lifecycle tracking, backend
coordination, and state management across distributed worker processes.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
)

from pydantic import Field, computed_field
from typing_extensions import TypeAliasType, TypedDict

from guidellm.utils import (
    PydanticClassRegistryMixin,
    RegistryMixin,
    StandardBaseModel,
)
from guidellm.utils.registry import RegistryObjT

__all__ = [
    "BackendInterface",
    "BackendT",
    "MeasuredRequestTimings",
    "MultiTurnRequestT",
    "RequestSchedulerTimings",
    "RequestT",
    "ResponseT",
    "ScheduledRequestInfo",
    "SchedulerMessagingPydanticRegistry",
    "SchedulerState",
    "SchedulerUpdateAction",
    "SchedulerUpdateActionProgress",
]

RequestT = TypeVar("RequestT")
"""Generic request object type for scheduler processing."""

ResponseT = TypeVar("ResponseT")
"""Generic response object type returned by backend processing."""

MultiTurnRequestT = TypeAliasType(
    "MultiTurnRequestT",
    Union[
        list[Union[RequestT, tuple[RequestT, float]]],
        tuple[Union[RequestT, tuple[RequestT, float]]],
    ],
    type_params=(RequestT,),
)
"""Multi-turn request structure supporting conversation history with optional delays."""


class SchedulerMessagingPydanticRegistry(RegistryMixin[RegistryObjT]):
    """
    Registry for enabling a generic interface to define the pydantic class types used
    for inter-process messaging within the scheduler.
    """


@SchedulerMessagingPydanticRegistry.register()
class RequestSchedulerTimings(StandardBaseModel):
    """
    Scheduler-level timing measurements for request lifecycle tracking.
    All timestamps are expected to be in Unix time (seconds since epoch).
    """

    targeted_start: float | None = Field(
        default=None,
        description="When the request was initially targeted for execution",
    )
    queued: float | None = Field(
        default=None,
        description="When the request was placed into the processing queue",
    )
    dequeued: float | None = Field(
        default=None,
        description="When the request was removed from the queue for processing",
    )
    scheduled_at: float | None = Field(
        default=None, description="When the request was scheduled for processing"
    )
    resolve_start: float | None = Field(
        default=None, description="When backend resolution of the request began"
    )
    resolve_end: float | None = Field(
        default=None, description="When backend resolution of the request completed"
    )
    finalized: float | None = Field(
        default=None,
        description="When the request was processed/acknowledged by the scheduler",
    )


@SchedulerMessagingPydanticRegistry.register()
class MeasuredRequestTimings(PydanticClassRegistryMixin["MeasuredRequestTimings"]):
    """
    Base timing measurements for backend request processing.
    All timestamps are expected to be in Unix time (seconds since epoch).
    """

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[MeasuredRequestTimings]:
        if cls.__name__ == "MeasuredRequestTimings":
            return cls

        return MeasuredRequestTimings

    schema_discriminator: ClassVar[str] = "timings_type"

    timings_type: Literal["measured_request_timings"] = Field(
        description="Type identifier for the timing measurement",
    )
    request_start: float | None = Field(
        default=None, description="When the backend began processing the request"
    )
    request_end: float | None = Field(
        default=None, description="When the backend completed processing the request"
    )


@SchedulerMessagingPydanticRegistry.register()
class ScheduledRequestInfo(StandardBaseModel):
    """
    Complete request information including status, timings, and metadata.

    Central data structure for tracking request lifecycle from creation through
    completion, containing scheduling metadata, timing measurements, and processing
    status. Used by scheduler components to coordinate request processing across
    distributed worker processes.

    Example:
    ::
        from guidellm.scheduler.objects import ScheduledRequestInfo

        # Create request info with automatic ID generation
        request_info = ScheduledRequestInfo()
        request_info.status = "in_progress"
        request_info.scheduler_timings.queued = time.time()

        # Check processing completion
        if request_info.completed_at:
            duration = request_info.completed_at - request_info.started_at
    """

    request_id: str = Field(
        description="Unique identifier for the request",
        default_factory=lambda: str(uuid.uuid4()),
    )
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ] = Field(description="Current processing status of the request", default="queued")
    scheduler_node_id: int = Field(
        description="ID/rank of the scheduler node handling the request",
        default=-1,
    )
    scheduler_process_id: int = Field(
        description="ID/rank of the node's scheduler process handling the request",
        default=-1,
    )
    scheduler_start_time: float = Field(
        description="Unix timestamp for the local time when scheduler processing began",
        default=-1,
    )

    error: str | None = Field(
        default=None, description="Error message if the request.status is 'errored'"
    )
    scheduler_timings: RequestSchedulerTimings = Field(
        default_factory=RequestSchedulerTimings,
        description="Scheduler-level timing measurements for request lifecycle",
    )
    request_timings: MeasuredRequestTimings | None = Field(
        default=None,
        description="Backend-specific timing measurements for request processing",
    )

    @computed_field  # type: ignore[misc]
    @property
    def started_at(self) -> float | None:
        """
        Get the effective request processing start time.

        :return: Unix timestamp when processing began, or None if not started.
        """
        request_start = (
            self.request_timings.request_start if self.request_timings else None
        )

        return request_start or self.scheduler_timings.resolve_start

    @computed_field  # type: ignore[misc]
    @property
    def completed_at(self) -> float | None:
        """
        Get the effective request processing completion time.

        :return: Unix timestamp when processing completed, or None if not completed.
        """
        request_end = self.request_timings.request_end if self.request_timings else None

        return request_end or self.scheduler_timings.resolve_end

    def model_copy(self, **kwargs) -> ScheduledRequestInfo:  # type: ignore[override]  # noqa: ARG002
        """
        Create a deep copy of the request info with copied timing objects.

        :return: New ScheduledRequestInfo instance with independent timing objects
        """
        return super().model_copy(
            update={
                "scheduler_timings": self.scheduler_timings.model_copy(),
                "request_timings": (
                    self.request_timings.model_copy() if self.request_timings else None
                ),
            },
            deep=False,
        )


class BackendInterface(Protocol, Generic[RequestT, ResponseT]):
    """
    Abstract interface for request processing backends.

    Defines the contract for backend implementations that process requests within
    the scheduler system. Backends handle initialization, validation, processing,
    and shutdown lifecycle management. Must ensure all properties are pickleable
    before process_startup is invoked for multi-process environments.

    Example:
    ::
        from guidellm.scheduler.objects import BackendInterface

        class CustomBackend(BackendInterface):
            @property
            def processes_limit(self) -> int:
                return 4

            async def resolve(self, request, request_info, history=None):
                # Process request and yield responses
                yield response, updated_request_info
    """

    @property
    def processes_limit(self) -> int | None:
        """
        :return: Maximum worker processes supported, or None if unlimited
        """

    @property
    def requests_limit(self) -> int | None:
        """
        :return: Maximum concurrent requests supported, or None if unlimited
        """

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: Backend metadata including model initialization and configuration
        """

    async def process_startup(self) -> None:
        """
        Perform backend initialization and startup procedures.

        :raises: Implementation-specific exceptions for startup failures.
        """

    async def validate(self) -> None:
        """
        Validate backend configuration and operational status.

        :raises: Implementation-specific exceptions for validation failures.
        """

    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        :raises: Implementation-specific exceptions for shutdown failures.
        """

    async def resolve(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo,
        history: list[tuple[RequestT, ResponseT]] | None = None,
    ) -> AsyncIterator[tuple[ResponseT, ScheduledRequestInfo]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process
        :param request_info: Scheduling metadata and timing information
        :param history: Optional conversation history for multi-turn requests
        :yield: Tuples of (response, updated_request_info) for each response chunk
        :raises: Implementation-specific exceptions for processing failures
        """


BackendT = TypeVar("BackendT", bound=BackendInterface)
"""Generic backend interface type for request processing."""


class SchedulerUpdateActionProgress(TypedDict, total=False):
    """
    Progress information for a scheduler update action.

    Optional progress tracking data that provides estimates for remaining work
    in scheduler operations. Used by constraints and monitoring systems to
    track execution progress and make termination decisions.
    """

    remaining_fraction: float | None
    remaining_requests: float | None
    remaining_duration: float | None


class SchedulerUpdateAction(StandardBaseModel):
    """
    Scheduler behavior control directives and actions.

    Encapsulates control signals for scheduler operations including request
    queuing and processing directives. Used by constraints to communicate
    termination conditions and progress information to scheduler components.

    Example:
    ::
        from guidellm.scheduler.objects import SchedulerUpdateAction

        # Signal to stop queuing but continue processing
        action = SchedulerUpdateAction(
            request_queuing="stop",
            request_processing="continue",
            metadata={"reason": "max_requests_reached"}
        )
    """

    request_queuing: Literal["continue", "stop"] = Field(
        default="continue", description="Action to take for request queuing operations"
    )
    request_processing: Literal["continue", "stop_local", "stop_all"] = Field(
        default="continue",
        description="Action to take for request processing operations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context and data for the scheduler action",
    )
    progress: SchedulerUpdateActionProgress = Field(
        default_factory=lambda: SchedulerUpdateActionProgress(),
        description="Progress information for the scheduler action",
    )


class SchedulerState(StandardBaseModel):
    """
    Scheduler operation state tracking and statistics.

    Comprehensive state container for tracking scheduler execution progress,
    request counts, timing information, and constraint enforcement. Central
    to scheduler coordination and provides real-time metrics for monitoring
    and decision-making across distributed worker processes.

    Example:
    ::
        from guidellm.scheduler.objects import SchedulerState

        # Initialize scheduler state
        state = SchedulerState(node_id=0, num_processes=4)

        # Track request processing
        state.created_requests += 1
        state.queued_requests += 1

        # Monitor completion progress
        completion_rate = state.processed_requests / state.created_requests
    """

    node_id: int = Field(
        description="Unique identifier for this scheduler node", default=-1
    )
    num_processes: int = Field(
        description="Number of worker processes in this scheduler", default=-1
    )
    start_time: float = Field(
        description="Unix timestamp when the scheduler started",
        default_factory=time.time,
    )
    end_time: float | None = Field(
        default=None, description="Unix timestamp when the scheduler stopped"
    )
    end_queuing_time: float | None = Field(
        default=None, description="When request queuing stopped, if applicable"
    )
    end_queuing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: float | None = Field(
        default=None, description="When request processing stopped, if applicable"
    )
    end_processing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
    )
    scheduler_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description=(
            "The latest state from all constraints applied during the scheduler run"
        ),
    )

    remaining_fraction: float | None = Field(
        default=None,
        description=(
            "Estimated fraction for the remaining progress of the run, if known"
        ),
    )
    remaining_requests: int | None = Field(
        default=None,
        description="Estimated number of requests remaining to be processed, if known",
    )
    remaining_duration: float | None = Field(
        default=None,
        description=(
            "Estimated time remaining in seconds for the scheduler run, if known"
        ),
    )

    created_requests: int = Field(
        default=0, description="Total number of requests created"
    )
    queued_requests: int = Field(
        default=0, description="Total number of requests queued for processing"
    )
    pending_requests: int = Field(
        default=0, description="Number of requests currently pending processing"
    )
    processing_requests: int = Field(
        default=0, description="Number of requests currently being processed"
    )
    processed_requests: int = Field(
        default=0, description="Total number of requests that completed processing"
    )
    successful_requests: int = Field(
        default=0, description="Number of requests that completed successfully"
    )
    errored_requests: int = Field(
        default=0, description="Number of requests that failed with errors"
    )
    cancelled_requests: int = Field(
        default=0, description="Number of requests that were cancelled"
    )
