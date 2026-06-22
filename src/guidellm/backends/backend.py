"""
Backend interface and registry for generative AI model interactions.

Provides the abstract base class for implementing backends that communicate with
generative AI models. Backends handle the lifecycle of generation requests and
provide a standard interface for distributed execution across worker processes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import Field

from guidellm.scheduler import BackendInterface
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    PydanticClassRegistryMixin,
    standard_model_config,
)
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "Backend",
    "BackendArgs",
]


class BackendArgs(PydanticClassRegistryMixin["BackendArgs"], ABC):
    """
    Base class for backend creation arguments.

    This class serves as a base for defining argument models used in the creation
    of backend instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    backend configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BackendArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base BackendArgs class for schema validation
        """
        if cls.__name__ == "BackendArgs":
            return cls

        return BackendArgs

    kind: str = Field(
        description="Identify the desired backend implementation.",
        examples=["openai_http", "vllm_python"],
    )


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
            def __init__(self, args: MyBackendArgs):
                super().__init__(args)
                self.api_key = args.api_key

            async def process_startup(self):
                self.client = MyAPIClient(self.api_key)

        args = MyBackendArgs(api_key="secret")
        backend = Backend.create(args)
    """

    @classmethod
    def create(cls, args: BackendArgs) -> Backend:
        """
        Create a backend instance based on the backend type.

        :param type_: The type of backend to create
        :param kwargs: Additional arguments for backend initialization
        :return: An instance of a subclass of Backend
        :raises ValueError: If the backend type is not registered
        """
        kind = args.kind

        backend = cls.get_registered_object(kind)

        if backend is None:
            raise ValueError(
                f"Backend type '{kind}' is not registered. "
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return backend(args)

    def __init__(self, args: BackendArgs):
        """
        Initialize a backend instance.

        :param type_: The backend type identifier
        """
        self.kind = args.kind

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
