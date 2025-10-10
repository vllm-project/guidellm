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

from guidellm.utils import StandardBaseDict, StandardBaseModel

__all__ = ["RequestInfo", "RequestTimings"]


class RequestTimings(StandardBaseDict):
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
    request_start: float | None = Field(
        default=None, description="When the backend began processing the request"
    )
    first_iteration: float | None = Field(
        default=None,
        description="Unix timestamp when the first generation iteration began.",
    )
    last_iteration: float | None = Field(
        default=None,
        description="Unix timestamp when the last generation iteration completed.",
    )
    iterations: int | None = Field(
        default=None,
        description="Total number of streaming update iterations performed.",
    )
    request_end: float | None = Field(
        default=None, description="When the backend completed processing the request"
    )
    resolve_end: float | None = Field(
        default=None, description="When backend resolution of the request completed"
    )
    finalized: float | None = Field(
        default=None,
        description="When the request was processed/acknowledged by the scheduler",
    )


class RequestInfo(StandardBaseModel):
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
    timings: RequestTimings = Field(
        default_factory=RequestTimings,
        description="Timing measurements for the request lifecycle",
    )

    error: str | None = Field(
        default=None, description="Error message if the request.status is 'errored'"
    )

    @computed_field  # type: ignore[misc]
    @property
    def started_at(self) -> float | None:
        """
        Get the effective request processing start time.

        :return: Unix timestamp when processing began, or None if not started.
        """
        request_start = self.timings.request_start if self.timings else None

        return request_start or self.timings.resolve_start

    @computed_field  # type: ignore[misc]
    @property
    def completed_at(self) -> float | None:
        """
        Get the effective request processing completion time.

        :return: Unix timestamp when processing completed, or None if not completed.
        """
        request_end = self.timings.request_end if self.timings else None

        return request_end or self.timings.resolve_end

    def model_copy(self, **kwargs) -> RequestInfo:  # type: ignore[override]  # noqa: ARG002
        """
        Create a deep copy of the request info with copied timing objects.

        :return: New ScheduledRequestInfo instance with independent timing objects
        """
        return super().model_copy(
            update={
                "timings": self.timings.model_copy(),
            },
            deep=False,
        )
