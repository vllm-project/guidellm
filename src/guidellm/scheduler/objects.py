"""
Core data structures and interfaces for the GuideLLM scheduler system.

This module defines the fundamental objects, timing models, and interfaces used
throughout the GuideLLM scheduler. It provides type-safe abstractions for
requests, responses, scheduling state, and backend interfaces that enable
distributed request processing and benchmarking.

Classes:
    RequestSchedulerTimings: Timing data for request lifecycle in the scheduler.
    RequestTimings: Base timing data for individual requests.
    ScheduledRequestInfo: Complete information about a scheduled request including
        status, timings, and metadata.
    BackendInterface: Abstract interface for request processing backends.
    SchedulerState: Comprehensive state tracking for scheduler operations.
    SchedulerUpdateAction: Action directives for scheduler behavior control.

Type Variables:
    RequestT: Generic type variable for request objects.
    ResponseT: Generic type variable for response objects.
    RequestTimingsT: Generic type variable for request timing objects.
    BackendT: Generic type variable for backend interface implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
)

from pydantic import Field

from guidellm.objects import StandardBaseModel

__all__ = [
    "BackendInterface",
    "BackendT",
    "RequestSchedulerTimings",
    "RequestT",
    "RequestTimings",
    "RequestTimingsT",
    "ResponseT",
    "ScheduledRequestInfo",
    "SchedulerState",
    "SchedulerUpdateAction",
]

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class RequestSchedulerTimings(StandardBaseModel):
    """
    Timing measurements for request lifecycle within the scheduler system.

    Tracks key timestamps throughout the request processing pipeline from initial
    targeting through final completion. All timing values are Unix timestamps
    (seconds since epoch) for precise temporal analysis and debugging.
    """

    targeted_start: Optional[float] = Field(
        default=None,
        description="When the request was initially targeted for execution",
    )
    queued: Optional[float] = Field(
        default=None,
        description="When the request was placed into the processing queue",
    )
    dequeued: Optional[float] = Field(
        default=None,
        description="When the request was removed from the queue for processing",
    )
    resolve_start: Optional[float] = Field(
        default=None, description="When backend resolution of the request began"
    )
    resolve_end: Optional[float] = Field(
        default=None, description="When backend resolution of the request completed"
    )
    finalized: Optional[float] = Field(
        default=None,
        description="When the request was processed/acknowledged by the scheduler",
    )


class RequestTimings(StandardBaseModel):
    """
    Base timing measurements for individual request processing.

    Provides foundational timing data that can be extended by specific backend
    implementations to track additional timing metrics relevant to their
    processing models.
    """

    request_start: Optional[float] = Field(
        default=None, description="When the backend began processing the request"
    )
    request_end: Optional[float] = Field(
        default=None, description="When the backend completed processing the request"
    )


RequestTimingsT = TypeVar("RequestTimingsT", bound=RequestTimings)


class ScheduledRequestInfo(StandardBaseModel, Generic[RequestTimingsT]):
    """
    Comprehensive information about a scheduled request throughout its lifecycle.

    Encapsulates all metadata, status, timing information, and processing context
    for a request within the scheduler system. Supports generic request timing
    types to accommodate different backend-specific timing requirements.
    """

    request_id: str = Field(description="Unique identifier for the request")
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ] = Field(description="Current processing status of the request")
    error: Optional[str] = Field(
        default=None, description="Error message if the request.status is 'errored'"
    )
    scheduler_node_id: int = Field(
        description="ID/rank of the scheduler node handling the request"
    )
    scheduler_process_id: int = Field(
        description="ID/rank of the node's scheduler process handling the request"
    )
    scheduler_start_time: float = Field(
        description="Unix timestamp for the local time when scheduler processing began"
    )

    scheduler_timings: RequestSchedulerTimings = Field(
        default_factory=RequestSchedulerTimings,
        description="Scheduler-level timing measurements for request lifecycle",
    )
    request_timings: Optional[RequestTimingsT] = Field(
        default=None,
        description="Backend-specific timing measurements for request processing",
    )

    @property
    def started_at(self) -> Optional[float]:
        """
        Get the effective start time for request processing.

        Returns the backend-specific request start time if available, otherwise
        falls back to the scheduler's resolve start time.

        :return: Unix timestamp when request processing effectively began,
            or None if not yet started.
        """
        request_start = (
            self.request_timings.request_start if self.request_timings else None
        )

        return request_start or self.scheduler_timings.resolve_start

    @property
    def completed_at(self) -> Optional[float]:
        """
        Get the effective completion time for request processing.

        Returns the backend-specific request end time if available, otherwise
        falls back to the scheduler's resolve end time.

        :return: Unix timestamp when request processing effectively completed,
            or None if not yet completed.
        """
        request_end = self.request_timings.request_end if self.request_timings else None

        return request_end or self.scheduler_timings.resolve_end


class BackendInterface(ABC, Generic[RequestT, RequestTimingsT, ResponseT]):
    """
    Abstract interface for request processing backends in the scheduler system.

    Defines the contract that all backend implementations must fulfill to integrate
    with the GuideLLM scheduler. Backends handle the actual processing of requests
    and manage their own resource constraints and lifecycle operations.
    """

    @property
    @abstractmethod
    def processes_limit(self) -> Optional[int]:
        """
        :return: Maximum worker processes supported, or None if unlimited.
        """

    @property
    @abstractmethod
    def requests_limit(self) -> Optional[int]:
        """
        :return: Maximum concurrent requests supported, or None if unlimited.
        """

    @abstractmethod
    async def process_startup(self) -> None:
        """
        Perform backend initialization and startup procedures.

        Called once per worker process before any request processing begins.
        Should establish connections, load models, and prepare resources.

        :raises: Implementation-specific exceptions for startup failures.
        """

    @abstractmethod
    async def validate(self) -> None:
        """
        Validate that the backend is properly configured and operational.

        Called after startup to ensure the backend can successfully process
        requests. Should perform health checks and connectivity validation.

        :raises: Implementation-specific exceptions for validation failures.
        """

    @abstractmethod
    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        Called once per worker process after all request processing completes.
        Should clean up resources, close connections, and finalize state.

        :raises: Implementation-specific exceptions for shutdown failures.
        """

    @abstractmethod
    async def resolve(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        history: Optional[list[tuple[RequestT, ResponseT]]] = None,
    ) -> AsyncIterator[tuple[ResponseT, ScheduledRequestInfo[RequestTimingsT]]]:
        """
        Process a request and yield response updates with timing information.

        The primary request processing method that handles individual requests
        and yields incremental responses as they become available. Supports
        streaming responses and multi-turn conversations through history.

        :param request: The request object to process.
        :param request_info: Scheduling metadata and timing information.
        :param history: Optional conversation history for multi-turn requests.
        :yield: Tuples of (response, updated_request_info) for each response chunk.
        :raises: Implementation-specific exceptions for processing failures.
        """


BackendT = TypeVar("BackendT", bound="BackendInterface")


class SchedulerState(StandardBaseModel):
    """
    Comprehensive state tracking for scheduler operations and request processing.

    Maintains detailed statistics and timing information for the entire scheduler
    system, including request counts, processing states, and termination criteria.
    Used for monitoring, debugging, and implementing stopping conditions.
    """

    node_id: int = Field(description="Unique identifier for this scheduler node")
    num_processes: int = Field(
        description="Number of worker processes in this scheduler"
    )
    start_time: float = Field(description="Unix timestamp when the scheduler started")
    end_queuing_time: Optional[float] = Field(
        default=None, description="When request queuing stopped, if applicable"
    )
    end_queuing_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: Optional[float] = Field(
        default=None, description="When request processing stopped, if applicable"
    )
    end_processing_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
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


class SchedulerUpdateAction(StandardBaseModel):
    """
    Action directives for controlling scheduler behavior during operation.

    Provides fine-grained control over scheduler operations, allowing external
    systems to influence queuing and processing behavior based on real-time
    conditions or policy decisions.
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
