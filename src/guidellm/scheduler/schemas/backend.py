from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Generic, Protocol, TypeVar

from guidellm.schemas import RequestInfo
from guidellm.utils.registry import RegistryMixin, RegistryObjT

from .types import HistoryT, RequestT, ResponseT

__all__ = [
    "BackendInterface",
    "BackendT",
    "SchedulerMessagingPydanticRegistry",
]


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
        history: HistoryT[RequestT, ResponseT] | None = None,
    ) -> AsyncIterator[tuple[ResponseT | None, RequestInfo]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process
        :param request_info: Scheduling metadata and timing information
        :param history: Conversation history for multi-turn requests
        :yield: Tuples of (response, updated_request_info) for each response chunk.
            Response may be None for intermediate updates (e.g., first token arrival).
        :raises Exception: Implementation-specific exceptions for processing failures
        """


BackendT = TypeVar("BackendT", bound=BackendInterface)
"Generic backend interface type for request processing"
