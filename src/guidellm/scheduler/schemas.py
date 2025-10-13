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
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
)

from pydantic import Field
from typing_extensions import TypeAliasType, TypedDict

from guidellm.schemas import RequestInfo
from guidellm.utils import (
    RegistryMixin,
    StandardBaseModel,
)
from guidellm.utils.registry import RegistryObjT

__all__ = [
    "BackendInterface",
    "BackendT",
    "MultiTurnRequestT",
    "RequestT",
    "ResponseT",
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
    list[RequestT | tuple[RequestT, float]] | tuple[RequestT | tuple[RequestT, float]],
    type_params=(RequestT,),
)
"""Multi-turn request structure supporting conversation history with optional delays."""


class SchedulerMessagingPydanticRegistry(RegistryMixin[RegistryObjT]):
    """
    Registry for enabling a generic interface to define the pydantic class types used
    for inter-process messaging within the scheduler.
    """


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
        request_info: RequestInfo,
        history: list[tuple[RequestT, ResponseT]] | None = None,
    ) -> AsyncIterator[tuple[ResponseT, RequestInfo]]:
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
        description="Constraints that triggered process ing termination",
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
    remaining_requests: float | None = Field(
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
        default=0,
        description="Total number of requests pending processing within a worker",
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
