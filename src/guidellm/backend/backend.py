"""
Backend interface and registry for generative AI model interactions.

Provides the abstract base class for implementing backends that communicate with
generative AI models. Backends handle the lifecycle of generation requests.

Classes:
    Backend: Abstract base class for generative AI backends with registry support.

Type Aliases:
    BackendType: Literal type defining supported backend implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional

from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import BackendInterface, ScheduledRequestInfo
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "Backend",
    "BackendType",
]


BackendType = Literal["openai_http"]


class Backend(
    ABC,
    RegistryMixin["type[Backend]"],
    BackendInterface[GenerationRequest, GenerationRequestTimings, GenerationResponse],
):
    """
    Abstract base class for generative AI backends with registry and lifecycle.

    Provides a standard interface for backends that communicate with generative AI
    models. Combines the registry pattern for automatic discovery with a defined
    lifecycle for process-based distributed execution.

    Backend lifecycle phases:
    1. Creation and configuration
    2. Process startup - Initialize resources in worker process
    3. Validation - Verify backend readiness
    4. Request resolution - Process generation requests
    5. Process shutdown - Clean up resources

    Backend state (excluding process_startup resources) must be pickleable for
    distributed execution across process boundaries.

    Example:
    ::
        @Backend.register("my_backend")
        class MyBackend(Backend):
            def __init__(self, api_key: str):
                super().__init__("my_backend")
                self.api_key = api_key

            async def process_startup(self):
                self.client = MyAPIClient(self.api_key)

        backend = Backend.create("my_backend", api_key="secret")
    """

    @classmethod
    def create(cls, type_: BackendType, **kwargs) -> "Backend":
        """
        Create a backend instance based on the backend type.

        :param type_: The type of backend to create.
        :param kwargs: Additional arguments for backend initialization.
        :return: An instance of a subclass of Backend.
        :raises ValueError: If the backend type is not registered.
        """

        backend = cls.get_registered_object(type_)

        return backend(**kwargs)

    def __init__(self, type_: BackendType):
        """
        Initialize a backend instance.

        :param type_: The backend type identifier.
        """
        self.type_ = type_

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: Maximum number of worker processes supported. None if unlimited.
        """
        return None

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: Maximum number of concurrent requests supported globally.
            None if unlimited.
        """
        return None

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """
        :return: Backend metadata including model information, endpoints, and
            configuration data for reporting and diagnostics.
        """
        ...

    @abstractmethod
    async def process_startup(self):
        """
        Initialize process-specific resources and connections.

        Called when a backend instance is transferred to a worker process.
        Creates connections, clients, and other resources required for request
        processing. Resources created here are process-local and need not be
        pickleable.

        Must be called before validate() or resolve().

        :raises: Exception if startup fails.
        """
        ...

    @abstractmethod
    async def process_shutdown(self):
        """
        Clean up process-specific resources and connections.

        Called when the worker process is shutting down. Cleans up resources
        created during process_startup(). After this method, validate() and
        resolve() should not be used.
        """
        ...

    @abstractmethod
    async def validate(self):
        """
        Validate backend configuration and readiness.

        Verifies the backend is properly configured and can communicate with the
        target model service. Should be called after process_startup() and before
        resolve().

        :raises: Exception if backend is not ready or cannot connect.
        """
        ...

    @abstractmethod
    async def resolve(
        self,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[
        tuple[GenerationResponse, ScheduledRequestInfo[GenerationRequestTimings]]
    ]:
        """
        Process a generation request and yield progressive responses.

        Processes a generation request through the backend's model service,
        yielding intermediate responses as generation progresses. The final
        yielded item contains the complete response and timing data.

        :param request: The generation request with content and parameters.
        :param request_info: Request tracking information updated with timing
            and progress metadata during processing.
        :param history: Optional conversation history for multi-turn requests.
            Each tuple contains a previous request-response pair.
        :yields: Tuples of (response, updated_request_info) as generation
            progresses. Final tuple contains the complete response.
        """
        ...

    @abstractmethod
    async def default_model(self) -> str:
        """
        :return: The default model name or identifier for generation requests.
        """
        ...
