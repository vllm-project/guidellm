"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations. Central to
the scheduler architecture, enabling request lifecycle tracking, backend
coordination, and state management across distributed worker processes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import Field
from typing_extensions import TypeAliasType

from guidellm.schemas import RequestInfo, StandardBaseModel
from guidellm.utils import RegistryMixin
from guidellm.utils.registry import RegistryObjT

__all__ = [
    "BackendInterface",
    "BackendT",
    "MultiTurnRequestT",
    "RequestT",
    "ResponseT",
    "SchedulerMessagingPydanticRegistry",
    "SchedulerProgress",
    "SchedulerState",
    "SchedulerUpdateAction",
]

RequestT = TypeVar("RequestT")
"Generic request object type for scheduler processing"

ResponseT = TypeVar("ResponseT")
"Generic response object type returned by backend processing"

MultiTurnRequestT = TypeAliasType(
    "MultiTurnRequestT",
    list[RequestT | tuple[RequestT, float]] | tuple[RequestT | tuple[RequestT, float]],
    type_params=(RequestT,),
)
"Multi-turn request structure supporting conversation history with optional delays"


class SchedulerMessagingPydanticRegistry(RegistryMixin[RegistryObjT]):
    """
    Registry for Pydantic types used in scheduler inter-process messaging.

    Enables generic interface for defining Pydantic class types used for
    communication between distributed scheduler components and worker processes.
    """


class BackendInterface(Protocol, Generic[RequestT, ResponseT]):
    """
    Protocol defining the interface for request processing backends.

    Establishes the contract for backend implementations that process requests
    within the scheduler system. Backends manage initialization, validation,
    processing, and shutdown lifecycle. All properties must be pickleable before
    process_startup is called for multi-process environments.

    Example:
    ::
        class CustomBackend(BackendInterface):
            @property
            def processes_limit(self) -> int:
                return 4

            async def resolve(self, request, request_info, history=None):
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

        :raises Exception: Implementation-specific exceptions for startup failures
        """

    async def validate(self) -> None:
        """
        Validate backend configuration and operational status.

        :raises Exception: Implementation-specific exceptions for validation failures
        """

    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        :raises Exception: Implementation-specific exceptions for shutdown failures
        """

    async def resolve(
        self,
        request: RequestT,
        request_info: RequestInfo,
        history: list[tuple[RequestT, ResponseT]] | None = None,
    ) -> AsyncIterator[tuple[ResponseT, RequestInfo]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process
        :param request_info: Scheduling metadata and timing information
        :param history: Conversation history for multi-turn requests
        :yield: Tuples of (response, updated_request_info) for each response chunk
        :raises Exception: Implementation-specific exceptions for processing failures
        """


BackendT = TypeVar("BackendT", bound=BackendInterface)
"Generic backend interface type for request processing"


class SchedulerProgress(StandardBaseModel):
    """
    Progress tracking data for scheduler operations.

    Provides estimates for remaining work in scheduler operations, including
    fraction complete, request counts, and duration. Used by constraints and
    monitoring systems to track execution progress and make termination decisions.
    """

    remaining_requests: float | None = Field(
        description="Estimated number of remaining requests to process", default=None
    )
    total_requests: float | None = Field(
        description="Total number of requests to process", default=None
    )
    remaining_duration: float | None = Field(
        description="Estimated remaining duration in seconds", default=None
    )
    total_duration: float | None = Field(
        description="Total duration in seconds to process for", default=None
    )
    stop_time: float | None = Field(
        description="The timestamp the processing stopped at", default=None
    )

    @property
    def remaining_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining progress, if known
        """
        fraction: float | None = None

        if (requests_fraction := self.remaining_requests_fraction) is not None:
            fraction = requests_fraction

        if (duration_fraction := self.remaining_duration_fraction) is not None:
            fraction = (
                duration_fraction
                if fraction is None
                else min(fraction, duration_fraction)
            )

        return fraction

    @property
    def remaining_requests_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining requests, if known
        """
        return (
            self.remaining_requests / float(self.total_requests)
            if self.remaining_requests is not None
            and self.total_requests is not None
            and self.total_requests > 0
            else None
        )

    @property
    def remaining_duration_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining duration, if known
        """
        return (
            self.remaining_duration / float(self.total_duration)
            if self.remaining_duration is not None
            and self.total_duration is not None
            and self.total_duration > 0
            else None
        )

    def combine(self, other: SchedulerProgress) -> SchedulerProgress:
        """
        Combine two progress instances, taking the minimum remaining estimates.

        :param other: Another progress instance to combine with
        :return: New progress instance with combined estimates
        """
        if (other_req_fraction := other.remaining_requests_fraction) is not None and (
            (cur_req_fraction := self.remaining_requests_fraction) is None
            or other_req_fraction < cur_req_fraction
        ):
            # Only update if the other is more advanced (lower fraction)
            self.remaining_requests = other.remaining_requests
            self.total_requests = other.total_requests

        if (other_dur_fraction := other.remaining_duration_fraction) is not None and (
            (cur_dur_fraction := self.remaining_duration_fraction) is None
            or other_dur_fraction < cur_dur_fraction
        ):
            # Only update if the other is more advanced (lower fraction)
            self.remaining_duration = other.remaining_duration
            self.total_duration = other.total_duration

        if other.stop_time is not None and (
            self.stop_time is None or other.stop_time < self.stop_time
        ):
            # Only update if the other has an earlier stop time
            self.stop_time = other.stop_time

        return self


class SchedulerUpdateAction(StandardBaseModel):
    """
    Control directives for scheduler behavior and operations.

    Encapsulates control signals for scheduler operations including request
    queuing and processing directives. Used by constraints to communicate
    termination conditions and progress to scheduler components.

    Example:
    ::
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
    progress: SchedulerProgress = Field(
        default_factory=lambda: SchedulerProgress(),
        description="Progress information for the scheduler action",
    )


class SchedulerState(StandardBaseModel):
    """
    Comprehensive state tracking for scheduler execution.

    Tracks scheduler execution progress, request counts, timing information,
    and constraint enforcement. Central to scheduler coordination, providing
    real-time metrics for monitoring and decision-making across distributed
    worker processes.

    Example:
    ::
        state = SchedulerState(node_id=0, num_processes=4)
        state.created_requests += 1
        state.queued_requests += 1
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
    start_requests_time: float | None = Field(
        default=None, description="Unix timestamp of the first sent request"
    )
    end_requests_time: float | None = Field(
        default=None, description="Unix timestamp of the last finalized request"
    )
    end_queuing_time: float | None = Field(
        default=None, description="Unix timestamp when request queuing stopped"
    )
    end_queuing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: float | None = Field(
        default=None, description="Unix timestamp when request processing stopped"
    )
    end_processing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
    )
    scheduler_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Latest state from all constraints applied during scheduler run",
    )

    progress: SchedulerProgress = Field(
        default_factory=lambda: SchedulerProgress(),
        description="Overall progress information for the scheduler run",
    )

    created_requests: int = Field(
        default=0, description="Total number of requests created"
    )
    queued_requests: int = Field(
        default=0, description="Total number of requests queued for processing"
    )
    pending_requests: int = Field(
        default=0,
        description="Number of requests pending processing within a worker",
    )
    processing_requests: int = Field(
        default=0, description="Number of requests currently being processed"
    )
    processed_requests: int = Field(
        default=0, description="Number of requests that completed processing"
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
