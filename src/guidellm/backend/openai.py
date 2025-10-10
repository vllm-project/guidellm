import base64
import copy
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, Optional, Union

import httpx
from loguru import logger
from PIL import Image

from guidellm.backend.backend import Backend
from guidellm.backend.response import (
    RequestArgs,
    ResponseSummary,
    StreamingTextResponse,
)
from guidellm.config import settings

__all__ = [
    "CHAT_COMPLETIONS",
    "CHAT_COMPLETIONS_PATH",
    "MODELS",
    "TEXT_COMPLETIONS",
    "TEXT_COMPLETIONS_PATH",
    "OpenAIHTTPBackend",
]


TEXT_COMPLETIONS_PATH = "/v1/completions"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

EndpointType = Literal["chat_completions", "models", "text_completions"]
CHAT_COMPLETIONS: EndpointType = "chat_completions"
MODELS: EndpointType = "models"
TEXT_COMPLETIONS: EndpointType = "text_completions"


@Backend.register("openai_http")
class OpenAIHTTPBackend(Backend):
    """
    A HTTP-based backend implementation for requests to an OpenAI compatible server.
    For example, a vLLM server instance or requests to OpenAI's API.

    :param target: The target URL string for the OpenAI server. ex: http://0.0.0.0:8000
    :param model: The model to use for all requests on the target server.
        If none is provided, the first available model will be used.
    :param api_key: The API key to use for requests to the OpenAI server.
        If provided, adds an Authorization header with the value
        "Authorization: Bearer {api_key}".
        If not provided, no Authorization header is added.
    :param organization: The organization to use for requests to the OpenAI server.
        For example, if set to "org_123", adds an OpenAI-Organization header with the
        value "OpenAI-Organization: org_123".
        If not provided, no OpenAI-Organization header is added.
    :param project: The project to use for requests to the OpenAI server.
        For example, if set to "project_123", adds an OpenAI-Project header with the
        value "OpenAI-Project: project_123".
        If not provided, no OpenAI-Project header is added.
    :param timeout: The timeout to use for requests to the OpenAI server.
        If not provided, the default timeout provided from settings is used.
    :param http2: If True, uses HTTP/2 for requests to the OpenAI server.
        Defaults to True.
    :param follow_redirects: If True, the HTTP client will follow redirect responses.
        If not provided, the default value from settings is used.
    :param max_output_tokens: The maximum number of tokens to request for completions.
        If not provided, the default maximum tokens provided from settings is used.
    :param extra_query: Query parameters to include in requests to the OpenAI server.
        If "chat_completions", "models", or "text_completions" are included as keys,
        the values of these keys will be used as the parameters for the respective
        endpoint.
        If not provided, no extra query parameters are added.
    :param extra_body: Body parameters to include in requests to the OpenAI server.
        If "chat_completions", "models", or "text_completions" are included as keys,
        the values of these keys will be included in the body for the respective
        endpoint.
        If not provided, no extra body parameters are added.
    :param remove_from_body: Parameters that should be removed from the body of each
        request.
        If not provided, no parameters are removed from the body.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        http2: Optional[bool] = True,
        follow_redirects: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        extra_query: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        remove_from_body: Optional[list[str]] = None,
        headers: Optional[dict] = None,
        verify: Optional[bool] = None,
    ):
        super().__init__(type_="openai_http")
        self._target = target or settings.openai.base_url

        if not self._target:
            raise ValueError("Target URL must be provided for OpenAI HTTP backend.")

        if self._target.endswith("/v1") or self._target.endswith("/v1/"):
            # backwards compatability, strip v1 off
            self._target = self._target[:-3]

        if self._target.endswith("/"):
            self._target = self._target[:-1]

        self._model = model

        # Start with default headers based on other params
        default_headers: dict[str, str] = {}
        api_key = api_key or settings.openai.api_key
        bearer_token = settings.openai.bearer_token
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        elif bearer_token:
            default_headers["Authorization"] = bearer_token

        self.organization = organization or settings.openai.organization
        if self.organization:
            default_headers["OpenAI-Organization"] = self.organization

        self.project = project or settings.openai.project
        if self.project:
            default_headers["OpenAI-Project"] = self.project

        # User-provided headers from kwargs or settings override defaults
        merged_headers = default_headers.copy()
        merged_headers.update(settings.openai.headers or {})
        if headers:
            merged_headers.update(headers)

        # Remove headers with None values for backward compatibility and convenience
        self.headers = {k: v for k, v in merged_headers.items() if v is not None}

        self.timeout = timeout if timeout is not None else settings.request_timeout
        self.http2 = http2 if http2 is not None else settings.request_http2
        self.follow_redirects = (
            follow_redirects
            if follow_redirects is not None
            else settings.request_follow_redirects
        )
        self.verify = verify if verify is not None else settings.openai.verify
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.openai.max_output_tokens
        )
        self.extra_query = extra_query
        self.extra_body = extra_body
        self.remove_from_body = remove_from_body
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def target(self) -> str:
        """
        :return: The target URL string for the OpenAI server.
        """
        return self._target

    @property
    def model(self) -> Optional[str]:
        """
        :return: The model to use for all requests on the target server.
            If validate hasn't been called yet and no model was passed in,
            this will be None until validate is called to set the default.
        """
        return self._model

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: The information about the backend.
        """
        return {
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
            "http2": self.http2,
            "follow_redirects": self.follow_redirects,
            "headers": self.headers,
            "text_completions_path": TEXT_COMPLETIONS_PATH,
            "chat_completions_path": CHAT_COMPLETIONS_PATH,
        }

    async def reset(self) -> None:
        """
        Reset the connection object. This is useful for backends that
        reuse connections or have state that needs to be cleared.
        For this backend, it closes the async client if it exists.
        """
        if self._async_client is not None:
            await self._async_client.aclose()

    async def check_setup(self):
        """
        Check if the backend is setup correctly and can be used for requests.
        Specifically, if a model is not provided, it grabs the first available model.
        If no models are available, raises a ValueError.
        If a model is provided and not available, raises a ValueError.

        :raises ValueError: If no models or the provided model is not available.
        """
        models = await self.available_models()
        if not models:
            raise ValueError(f"No models available for target: {self.target}")

        if not self.model:
            self._model = models[0]
        elif self.model not in models:
            raise ValueError(
                f"Model {self.model} not found in available models:"
                f"{models} for target: {self.target}"
            )

    async def prepare_multiprocessing(self):
        """
        Prepare the backend for use in a multiprocessing environment.
        Clears out the sync and async clients to ensure they are re-initialized
        for each process.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def available_models(self) -> list[str]:
        """
        Get the available models for the target server using the OpenAI models endpoint:
        /v1/models
        """
        target = f"{self.target}/v1/models"
        headers = self._headers()
        params = self._params(MODELS)
        response = await self._get_async_client().get(
            target, headers=headers, params=params
        )
        response.raise_for_status()

        models = []

        for item in response.json()["data"]:
            models.append(item["id"])

        return models

    async def text_completions(  # type: ignore[override]
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        """
        Generate text completions for the given prompt using the OpenAI
        completions endpoint: /v1/completions.

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
        logger.debug("{} invocation with args: {}", self.__class__.__name__, locals())

        if isinstance(prompt, list):
            raise ValueError(
                "List prompts (batching) is currently not supported for "
                f"text_completions OpenAI pathways. Received: {prompt}"
            )

        headers = self._headers()
        params = self._params(TEXT_COMPLETIONS)
        payload = self._completions_payload(
            endpoint_type=TEXT_COMPLETIONS,
            orig_kwargs=kwargs,
            max_output_tokens=output_token_count,
            prompt=prompt,
        )

        try:
            async for resp in self._iterative_completions_request(
                type_="text_completions",
                request_id=request_id,
                request_prompt_tokens=prompt_token_count,
                request_output_tokens=output_token_count,
                headers=headers,
                params=params,
                payload=payload,
            ):
                yield resp
        except Exception as ex:
            logger.error(
                "{} request with headers: {} and params: {} and payload: {} failed: {}",
                self.__class__.__name__,
                headers,
                params,
                payload,
                ex,
            )
            raise ex

    async def chat_completions(  # type: ignore[override]
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
        Generate chat completions for the given content using the OpenAI
        chat completions endpoint: /v1/chat/completions.

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
        logger.debug("{} invocation with args: {}", self.__class__.__name__, locals())
        headers = self._headers()
        params = self._params(CHAT_COMPLETIONS)
        messages = (
            content if raw_content else self._create_chat_messages(content=content)
        )
        payload = self._completions_payload(
            endpoint_type=CHAT_COMPLETIONS,
            orig_kwargs=kwargs,
            max_output_tokens=output_token_count,
            messages=messages,
        )

        try:
            async for resp in self._iterative_completions_request(
                type_="chat_completions",
                request_id=request_id,
                request_prompt_tokens=prompt_token_count,
                request_output_tokens=output_token_count,
                headers=headers,
                params=params,
                payload=payload,
            ):
                yield resp
        except Exception as ex:
            logger.error(
                "{} request with headers: {} and params: {} and payload: {} failed: {}",
                self.__class__.__name__,
                headers,
                params,
                payload,
                ex,
            )
            raise ex

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Get the async HTTP client for making requests.
        If the client has not been created yet, it will create one.

        :return: The async HTTP client.
        """
        if self._async_client is None or self._async_client.is_closed:
            client = httpx.AsyncClient(
                http2=self.http2,
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
                verify=self.verify,
            )
            self._async_client = client
        else:
            client = self._async_client

        return client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        headers.update(self.headers)
        return headers

    def _params(self, endpoint_type: EndpointType) -> dict[str, str]:
        if self.extra_query is None:
            return {}

        if (
            CHAT_COMPLETIONS in self.extra_query
            or MODELS in self.extra_query
            or TEXT_COMPLETIONS in self.extra_query
        ):
            return self.extra_query.get(endpoint_type, {})

        return self.extra_query

    def _extra_body(self, endpoint_type: EndpointType) -> dict[str, Any]:
        if self.extra_body is None:
            return {}

        if (
            CHAT_COMPLETIONS in self.extra_body
            or MODELS in self.extra_body
            or TEXT_COMPLETIONS in self.extra_body
        ):
            return copy.deepcopy(self.extra_body.get(endpoint_type, {}))

        return copy.deepcopy(self.extra_body)

    def _completions_payload(
        self,
        endpoint_type: EndpointType,
        orig_kwargs: Optional[dict],
        max_output_tokens: Optional[int],
        **kwargs,
    ) -> dict:
        payload = self._extra_body(endpoint_type)
        payload.update(orig_kwargs or {})
        payload.update(kwargs)
        payload["model"] = self.model
        payload["stream"] = True
        payload["stream_options"] = {
            "include_usage": True,
        }

        if max_output_tokens or self.max_output_tokens:
            logger.debug(
                "{} adding payload args for setting output_token_count: {}",
                self.__class__.__name__,
                max_output_tokens or self.max_output_tokens,
            )
            payload["max_tokens"] = max_output_tokens or self.max_output_tokens
            payload["max_completion_tokens"] = payload["max_tokens"]

            if max_output_tokens:
                # only set stop and ignore_eos if max_output_tokens set at request level
                # otherwise the instance value is just the max to enforce we stay below
                payload["stop"] = None
                payload["ignore_eos"] = True

        if self.remove_from_body:
            for key in self.remove_from_body:
                payload.pop(key, None)

        return payload

    @staticmethod
    def _create_chat_messages(
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
    ) -> list[dict]:
        if isinstance(content, str):
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        if isinstance(content, list):
            resolved_content = []

            for item in content:
                if isinstance(item, dict):
                    resolved_content.append(item)
                elif isinstance(item, str):
                    resolved_content.append({"type": "text", "text": item})
                elif isinstance(item, Image.Image) or (
                    isinstance(item, Path) and item.suffix.lower() in [".jpg", ".jpeg"]
                ):
                    image = item if isinstance(item, Image.Image) else Image.open(item)
                    encoded = base64.b64encode(image.tobytes()).decode("utf-8")
                    resolved_content.append(
                        {
                            "type": "image",
                            "image": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                            },
                        }
                    )
                elif isinstance(item, Path) and item.suffix.lower() in [".wav"]:
                    encoded = base64.b64encode(item.read_bytes()).decode("utf-8")
                    resolved_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"{encoded}",
                                "format": "wav",
                            },
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported content item type: {item} in list: {content}"
                    )

            return [
                {
                    "role": "user",
                    "content": resolved_content,
                }
            ]

        raise ValueError(f"Unsupported content type: {content}")

    async def _iterative_completions_request(
        self,
        type_: Literal["text_completions", "chat_completions"],
        request_id: Optional[str],
        request_prompt_tokens: Optional[int],
        request_output_tokens: Optional[int],
        headers: dict[str, str],
        params: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        if type_ == "text_completions":
            target = f"{self.target}{TEXT_COMPLETIONS_PATH}"
        elif type_ == "chat_completions":
            target = f"{self.target}{CHAT_COMPLETIONS_PATH}"
        else:
            raise ValueError(f"Unsupported type: {type_}")

        logger.info(
            "{} making request: {} to target: {} using http2: {} following "
            "redirects: {} for timeout: {} with headers: {} and params: {} and ",
            "payload: {}",
            self.__class__.__name__,
            request_id,
            target,
            self.http2,
            self.follow_redirects,
            self.timeout,
            headers,
            params,
            payload,
        )

        response_value = ""
        response_prompt_count: Optional[int] = None
        response_output_count: Optional[int] = None
        iter_count = 0
        start_time = time.time()
        iter_time = start_time
        first_iter_time: Optional[float] = None
        last_iter_time: Optional[float] = None

        yield StreamingTextResponse(
            type_="start",
            value="",
            start_time=start_time,
            first_iter_time=None,
            iter_count=iter_count,
            delta="",
            time=start_time,
            request_id=request_id,
        )

        # reset start time after yielding start response to ensure accurate timing
        start_time = time.time()

        async with self._get_async_client().stream(
            "POST", target, headers=headers, params=params, json=payload
        ) as stream:
            stream.raise_for_status()

            async for line in stream.aiter_lines():
                iter_time = time.time()
                logger.debug(
                    "{} request: {} recieved iter response line: {}",
                    self.__class__.__name__,
                    request_id,
                    line,
                )

                if not line or not line.strip().startswith("data:"):
                    continue

                if line.strip() == "data: [DONE]":
                    break

                data = json.loads(line.strip()[len("data: ") :])
                if delta := self._extract_completions_delta_content(type_, data):
                    if first_iter_time is None:
                        first_iter_time = iter_time
                    last_iter_time = iter_time

                    iter_count += 1
                    response_value += delta

                    yield StreamingTextResponse(
                        type_="iter",
                        value=response_value,
                        iter_count=iter_count,
                        start_time=start_time,
                        first_iter_time=first_iter_time,
                        delta=delta,
                        time=iter_time,
                        request_id=request_id,
                    )

                if usage := self._extract_completions_usage(data):
                    response_prompt_count = usage["prompt"]
                    response_output_count = usage["output"]

        logger.info(
            "{} request: {} with headers: {} and params: {} and payload: {} completed"
            "with: {}",
            self.__class__.__name__,
            request_id,
            headers,
            params,
            payload,
            response_value,
        )

        yield ResponseSummary(
            value=response_value,
            request_args=RequestArgs(
                target=target,
                headers=headers,
                params=params,
                payload=payload,
                timeout=self.timeout,
                http2=self.http2,
                follow_redirects=self.follow_redirects,
            ),
            start_time=start_time,
            end_time=iter_time,
            first_iter_time=first_iter_time,
            last_iter_time=last_iter_time,
            iterations=iter_count,
            request_prompt_tokens=request_prompt_tokens,
            request_output_tokens=request_output_tokens,
            response_prompt_tokens=response_prompt_count,
            response_output_tokens=response_output_count,
            request_id=request_id,
        )

    @staticmethod
    def _extract_completions_delta_content(
        type_: Literal["text_completions", "chat_completions"], data: dict
    ) -> Optional[str]:
        if "choices" not in data or not data["choices"]:
            return None

        if type_ == "text_completions":
            return data["choices"][0]["text"]

        if type_ == "chat_completions":
            return data.get("choices", [{}])[0].get("delta", {}).get("content")

        raise ValueError(f"Unsupported type: {type_}")

    @staticmethod
    def _extract_completions_usage(
        data: dict,
    ) -> Optional[dict[Literal["prompt", "output"], int]]:
        if "usage" not in data or not data["usage"]:
            return None

        return {
            "prompt": data["usage"]["prompt_tokens"],
            "output": data["usage"]["completion_tokens"],
        }
