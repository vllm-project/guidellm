"""
Mock backend implementation for testing purposes.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from lorem.text import TextLorem

from guidellm.backends import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)


@Backend.register("mock")
class MockBackend(Backend):
    """
    Mock backend for testing that simulates text generation.

    Provides predictable responses with configurable delays and token counts
    for testing the backend interface without requiring an actual LLM service.
    """

    def __init__(
        self,
        target: str = "mock-target",
        model: str = "mock-model",
        iter_delay: float | None = None,
    ):
        """
        Initialize mock backend.

        :param model: Model name to simulate.
        :param target: Target URL to simulate.
        :param iter_delay: Delay between iterations in seconds.
        """
        super().__init__(type_="mock")  # type: ignore [reportCallIssue]
        self._model = model
        self._target = target
        self._iter_delay = iter_delay
        self._in_process = False

    @property
    def target(self) -> str:
        """Target URL for the mock backend."""
        return self._target

    @property
    def model(self) -> str | None:
        """Model name for the mock backend."""
        return self._model

    def info(self) -> dict[str, Any]:
        """
        Return mock backend configuration information.
        """
        return {
            "type": "mock",
            "model": self._model,
            "target": self._target,
            "iter_delay": self._iter_delay,
        }

    async def process_startup(self) -> None:
        """
        Initialize the mock backend process.
        """
        self._in_process = True

    async def process_shutdown(self) -> None:
        """
        Shutdown the mock backend process.
        """
        self._in_process = False

    async def validate(self) -> None:
        """
        Validate the mock backend configuration.
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process")

    async def default_model(self) -> str | None:
        """
        Return the default model for the mock backend.
        """
        return self._model

    async def resolve(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process a generation request and yield progressive responses.

        ### WRITTEN BY AI ###
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process")

        if history is not None:
            raise NotImplementedError(
                "Multi-turn requests not supported in mock backend"
            )

        # Extract token counts from request
        prompt_tokens = request.stats.get("prompt_tokens")
        output_tokens = request.constraints.get("output_tokens")

        # Generate mock tokens
        tokens = self._get_tokens(output_tokens)

        # Initialize response
        response = GenerationResponse(
            request_id=request.request_id,
            request_args={
                "request_type": request.request_type,
                "output_token_count": output_tokens,
                **request.params,
            },
            value="",
            request_prompt_tokens=prompt_tokens,
            request_output_tokens=output_tokens,
        )

        # Initialize timings
        request_info.request_timings = RequestTimings()
        request_info.request_timings.request_start = time.time()

        # Generate response iteratively
        for index, token in enumerate(tokens):
            if self._iter_delay:
                await asyncio.sleep(self._iter_delay)

            if request_info.request_timings.first_iteration is None:
                request_info.request_timings.first_iteration = time.time()

            response.value += token  # type: ignore [reportOperatorIssue]
            response.delta = token
            response.iterations = index + 1
            request_info.request_timings.last_iteration = time.time()

            yield response, request_info

        # Final response with usage stats
        request_info.request_timings.request_end = time.time()
        response.response_prompt_tokens = prompt_tokens or self._estimate_prompt_tokens(
            str(request.content)
        )
        response.response_output_tokens = len(tokens)
        response.delta = None

        yield response, request_info

    @staticmethod
    def _estimate_prompt_tokens(content: str) -> int:
        """
        Estimate prompt tokens from content.
        """
        # Simple word-based token estimation
        return len(str(content).split())

    @staticmethod
    def _get_tokens(token_count: int | None = None) -> list[str]:
        """
        Generate mock tokens for response.
        """
        if token_count is None:
            token_count = random.randint(8, 512)

        words = TextLorem(srange=(token_count, token_count)).sentence().split()
        tokens = []

        for word in words:
            if len(tokens) == token_count - 1:
                tokens.append(".")
                break
            if len(tokens) == token_count - 2:
                tokens.append(word)
                tokens.append(".")
                break
            tokens.append(word)
            tokens.append(" ")

        return tokens
