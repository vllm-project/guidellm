"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations. Central to
the scheduler architecture, enabling request lifecycle tracking, backend
coordination, and state management across distributed worker processes.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field, computed_field

from guidellm.schemas.base import StandardBaseDict, StandardBaseModel

__all__ = ["RequestInfo", "RequestTimings"]


class RequestTimings(StandardBaseDict):
    """
    Timing measurements for tracking request lifecycle events.

    Provides comprehensive timing data for distributed request processing, capturing
    key timestamps from initial targeting through final completion. Essential for
    performance analysis, SLA monitoring, and debugging request processing bottlenecks
    across scheduler workers and backend systems.
    """

    targeted_start: float | None = Field(
        default=None,
        description="Unix timestamp when request was initially targeted for execution",
    )
    queued: float | None = Field(
        default=None,
        description="Unix timestamp when request was placed into processing queue",
    )
    dequeued: float | None = Field(
        default=None,
        description="Unix timestamp when request was removed from queue for processing",
    )
    scheduled_at: float | None = Field(
        default=None,
        description="Unix timestamp when the request was scheduled for processing",
    )
    resolve_start: float | None = Field(
        default=None,
        description="Unix timestamp when backend resolution of the request began",
    )
    request_start: float | None = Field(
        default=None,
        description="Unix timestamp when the backend began processing the request",
    )
    first_request_iteration: float | None = Field(
        default=None,
    )
    first_token_iteration: float | None = Field(
        default=None,
    )
    last_token_iteration: float | None = Field(
        default=None,
    )
    last_request_iteration: float | None = Field(
        default=None,
    )
    request_iterations: int = Field(
        default=0,
    )
    token_iterations: int = Field(
        default=0,
    )
    request_end: float | None = Field(
        default=None,
        description="Unix timestamp when the backend completed processing the request",
    )
    resolve_end: float | None = Field(
        default=None,
        description="Unix timestamp when backend resolution of the request completed",
    )
    finalized: float | None = Field(
        default=None,
        description="Unix timestamp when request was processed by the scheduler",
    )

    @property
    def last_reported(self) -> float | None:
        """
        Get the most recent timing measurement available.

        :return: The latest Unix timestamp from the timing fields, or None if none
        """
        timing_fields = [
            self.queued,
            self.dequeued,
            self.scheduled_at,
            self.resolve_start,
            self.request_start,
            self.request_end,
            self.resolve_end,
        ]
        valid_timings = [field for field in timing_fields if field is not None]
        return max(valid_timings) if valid_timings else None


class RequestInfo(StandardBaseModel):
    """
    Complete information about a request in the scheduler system.

    Encapsulates all metadata, status tracking, and timing information for requests
    processed through the distributed scheduler. Provides comprehensive lifecycle
    tracking from initial queuing through final completion, including error handling
    and node identification for debugging and performance analysis.

    Example:
    ::
        request = RequestInfo()
        request.status = "in_progress"
        start_time = request.started_at
        completion_time = request.completed_at
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
        description="Unix timestamp when scheduler processing began",
        default=-1,
    )
    timings: RequestTimings = Field(
        default_factory=RequestTimings,
        description="Timing measurements for the request lifecycle",
    )

    error: str | None = Field(
        default=None, description="Error message if the request status is 'errored'"
    )
    traceback: str | None = Field(
        default=None,
        description="Full traceback of the error if the request status is 'errored'",
    )

    @computed_field  # type: ignore[misc]
    @property
    def started_at(self) -> float | None:
        """
        Get the effective request processing start time.

        :return: Unix timestamp when processing began, or None if not started
        """
        return self.timings.request_start or self.timings.resolve_start

    @computed_field  # type: ignore[misc]
    @property
    def completed_at(self) -> float | None:
        """
        Get the effective request processing completion time.

        :return: Unix timestamp when processing completed, or None if not completed
        """
        return self.timings.request_end or self.timings.resolve_end

    def model_copy(self, **_kwargs) -> RequestInfo:  # type: ignore[override]  # noqa: ARG002
        """
        Create a deep copy of the request info with copied timing objects.

        :param kwargs: Additional keyword arguments for model copying
        :return: New RequestInfo instance with independent timing objects
        """
        return super().model_copy(
            update={
                "timings": self.timings.model_copy(),
            },
            deep=False,
        )
