from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, Optional, Union

from loguru import logger
from PIL import Image

from guidellm.backend.response import ResponseSummary, StreamingTextResponse
from guidellm.settings import settings

__all__ = [
    "Backend",
    "BackendType",
]


BackendType = Literal["openai_http"]


class Backend(ABC):
    """
    Abstract base class for generative AI backends.

    This class provides a common interface for creating and interacting with different
    generative AI backends. Subclasses should implement the abstract methods to
    define specific backend behavior.

    :cvar _registry: A registration dictionary that maps BackendType to backend classes.
    :param type_: The type of the backend.
    """

    _registry: dict[BackendType, "type[Backend]"] = {}

    @classmethod
    def register(cls, backend_type: BackendType):
        """
        A decorator to register a backend class in the backend registry.

        :param backend_type: The type of backend to register.
        :type backend_type: BackendType
        :return: The decorated backend class.
        :rtype: Type[Backend]
        """
        if backend_type in cls._registry:
            raise ValueError(f"Backend type already registered: {backend_type}")

        if not issubclass(cls, Backend):
            raise TypeError("Only subclasses of Backend can be registered")

        def inner_wrapper(wrapped_class: type["Backend"]):
            cls._registry[backend_type] = wrapped_class
            logger.info("Registered backend type: {}", backend_type)
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, type_: BackendType, **kwargs) -> "Backend":
        """
        Factory method to create a backend instance based on the backend type.

        :param type_: The type of backend to create.
        :type type_: BackendType
        :param kwargs: Additional arguments for backend initialization.
        :return: An instance of a subclass of Backend.
        :rtype: Backend
        :raises ValueError: If the backend type is not registered.
        """

        logger.info("Creating backend of type {}", type_)

        if type_ not in cls._registry:
            err = ValueError(f"Unsupported backend type: {type_}")
            logger.error("{}", err)
            raise err

        return Backend._registry[type_](**kwargs)

    def __init__(self, type_: BackendType):
        self._type = type_

    @property
    def type_(self) -> BackendType:
        """
        :return: The type of the backend.
        """
        return self._type

    @property
    @abstractmethod
    def target(self) -> str:
        """
        :return: The target location for the backend.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> Optional[str]:
        """
        :return: The model used for the backend requests.
        """
        ...

    @property
    @abstractmethod
    def info(self) -> dict[str, Any]:
        """
        :return: The information about the backend.
        """
        ...

    @abstractmethod
    async def reset(self) -> None:
        """
        Reset the connection object. This is useful for backends that
        reuse connections or have state that needs to be cleared.
        """
        ...

    async def validate(self):
        """
        Handle final setup and validate the backend is ready for use.
        If not successful, raises the appropriate exception.
        """
        logger.info("{} validating backend {}", self.__class__.__name__, self.type_)
        await self.check_setup()
        models = await self.available_models()
        if not models:
            raise ValueError("No models available for the backend")

        # Use the preferred route defined in the global settings when performing the
        # validation request. This avoids calling an unavailable endpoint (ie
        # /v1/completions) when the deployment only supports the chat completions
        # endpoint.
        if settings.preferred_route == "chat_completions":
            async for _ in self.chat_completions(  # type: ignore[attr-defined]
                content="Test connection", output_token_count=1
            ):
                pass
        else:
            async for _ in self.text_completions(  # type: ignore[attr-defined]
                prompt="Test connection", output_token_count=1
            ):
                pass

        await self.reset()

    @abstractmethod
    async def check_setup(self):
        """
        Check the setup for the backend.
        If unsuccessful, raises the appropriate exception.

        :raises ValueError: If the setup check fails.
        """
        ...

    @abstractmethod
    async def prepare_multiprocessing(self):
        """
        Prepare the backend for use in a multiprocessing environment.
        This is useful for backends that have instance state that can not
        be shared across processes and should be cleared out and re-initialized
        for each new process.
        """
        ...

    @abstractmethod
    async def available_models(self) -> list[str]:
        """
        Get the list of available models for the backend.

        :return: The list of available models.
        :rtype: List[str]
        """
        ...

    @abstractmethod
    async def text_completions(
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        """
        Generate text only completions for the given prompt.
        Does not support multiple modalities, complicated chat interfaces,
        or chat templates. Specifically, it requests with only the prompt.

        :param prompt: The prompt (or list of prompts) to generate a completion for.
            If a list is supplied, these are concatenated and run through the model
            for a single prompt.
        :param request_id: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields a StreamingTextResponse for start,
            a StreamingTextResponse for each received iteration,
            and a ResponseSummary for the final response.
        """
        ...

    @abstractmethod
    async def chat_completions(
        self,
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        request_id: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        """
        Generate chat completions for the given content.
        Supports multiple modalities, complicated chat interfaces, and chat templates.
        Specifically, it requests with the content, which can be any combination of
        text, images, and audio provided the target model supports it,
        and returns the output text. Additionally, any chat templates
        for the model are applied within the backend.

        :param content: The content (or list of content) to generate a completion for.
            This supports any combination of text, images, and audio (model dependent).
            Supported text only request examples:
                content="Sample prompt", content=["Sample prompt", "Second prompt"],
                content=[{"type": "text", "value": "Sample prompt"}.
            Supported text and image request examples:
                content=["Describe the image", PIL.Image.open("image.jpg")],
                content=["Describe the image", Path("image.jpg")],
                content=["Describe the image", {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}].
            Supported text and audio request examples:
                content=["Transcribe the audio", Path("audio.wav")],
                content=["Transcribe the audio", {"type": "input_audio",
                "input_audio": {"data": f"{base64_bytes}", "format": "wav}].
            Additionally, if raw_content=True then the content is passed directly to the
            backend without any processing.
        :param request_id: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields a StreamingTextResponse for start,
            a StreamingTextResponse for each received iteration,
            and a ResponseSummary for the final response.
        """
        ...
