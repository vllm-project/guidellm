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
from typing_extensions import TypeAliasType, TypedDict

from guidellm.schemas import RequestInfo
from guidellm.utils import RegistryMixin, StandardBaseModel
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


class SchedulerUpdateActionProgress(TypedDict, total=False):
    """
    Progress tracking data for scheduler operations.

    Provides estimates for remaining work in scheduler operations, including
    fraction complete, request counts, and duration. Used by constraints and
    monitoring systems to track execution progress and make termination decisions.
    """

    remaining_fraction: float | None
    remaining_requests: float | None
    remaining_duration: float | None


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
    progress: SchedulerUpdateActionProgress = Field(
        default_factory=lambda: SchedulerUpdateActionProgress(),
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

    remaining_fraction: float | None = Field(
        default=None,
        description="Estimated fraction of remaining progress, if known",
    )
    remaining_requests: float | None = Field(
        default=None,
        description="Estimated number of remaining requests to process, if known",
    )
    remaining_duration: float | None = Field(
        default=None,
        description="Estimated remaining time in seconds for scheduler run, if known",
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
