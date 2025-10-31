"""
Backend interface and registry for generative AI model interactions.

Provides the abstract base class for implementing backends that communicate with
generative AI models. Backends handle the lifecycle of generation requests and
provide a standard interface for distributed execution across worker processes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from guidellm.scheduler import BackendInterface
from guidellm.schemas import GenerationRequest, GenerationResponse
from guidellm.utils import RegistryMixin

__all__ = [
    "Backend",
    "BackendType",
]


BackendType = Literal["openai_http"]


class Backend(
    RegistryMixin["type[Backend]"],
    BackendInterface[GenerationRequest, GenerationResponse],
):
    """
    Base class for generative AI backends with registry and lifecycle management.

    Provides a standard interface for backends that communicate with generative AI
    models. Combines the registry pattern for automatic discovery with a defined
    lifecycle for process-based distributed execution. Backend state must be
    pickleable for distributed execution across process boundaries.

    Backend lifecycle phases:
    1. Creation and configuration
    2. Process startup - Initialize resources in worker process
    3. Validation - Verify backend readiness
    4. Request resolution - Process generation requests
    5. Process shutdown - Clean up resources

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
    def create(cls, type_: BackendType, **kwargs) -> Backend:
        """
        Create a backend instance based on the backend type.

        :param type_: The type of backend to create
        :param kwargs: Additional arguments for backend initialization
        :return: An instance of a subclass of Backend
        :raises ValueError: If the backend type is not registered
        """

        backend = cls.get_registered_object(type_)

        if backend is None:
            raise ValueError(
                f"Backend type '{type_}' is not registered. "
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return backend(**kwargs)

    def __init__(self, type_: BackendType):
        """
        Initialize a backend instance.

        :param type_: The backend type identifier
        """
        self.type_ = type_

    @property
    def processes_limit(self) -> int | None:
        """
        :return: Maximum number of worker processes supported, None if unlimited
        """
        return None

    @property
    def requests_limit(self) -> int | None:
        """
        :return: Maximum number of concurrent requests supported globally,
            None if unlimited
        """
        return None

    @abstractmethod
    async def default_model(self) -> str:
        """
        :return: The default model name or identifier for generation requests,
        """
        ...
