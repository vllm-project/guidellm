import asyncio
import random
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional, Union

from lorem.text import TextLorem  # type: ignore
from PIL import Image

from guidellm.backends import (
    Backend,
    RequestArgs,
    ResponseSummary,
    StreamingTextResponse,
)


@Backend.register("mock")  # type: ignore
class MockBackend(Backend):
    def __init__(
        self,
        model: Optional[str] = "mock-model",
        target: Optional[str] = "mock-target",
        iter_delay: Optional[float] = None,
    ):
        super().__init__(type_="mock")  # type: ignore
        self._model = model
        self._target = target
        self._iter_delay = iter_delay

    @property
    def target(self) -> str:
        return self._target  # type: ignore

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def info(self) -> dict[str, Any]:
        return {}

    async def reset(self) -> None:
        pass

    async def prepare_multiprocessing(self):
        pass

    async def check_setup(self):
        pass

    async def available_models(self) -> list[str]:
        return [self.model]  # type: ignore

    async def text_completions(  # type: ignore
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Prompt must be a non-empty string")

        async for response in self._text_prompt_response_generator(
            prompt,
            request_id,
            prompt_token_count,
            output_token_count,
        ):
            yield response

    async def chat_completions(  # type: ignore
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
        if not isinstance(content, str) or not content:
            raise ValueError("Content must be a non-empty string")

        async for response in self._text_prompt_response_generator(
            content,
            request_id,
            prompt_token_count,
            output_token_count,
        ):
            yield response

    async def _text_prompt_response_generator(
        self,
        prompt: str,
        request_id: Optional[str],
        prompt_token_count: Optional[int],
        output_token_count: Optional[int],
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        tokens = self._get_tokens(output_token_count)
        start_time = time.time()

        yield StreamingTextResponse(
            type_="start",
            value="",
            start_time=start_time,
            first_iter_time=None,
            iter_count=0,
            delta="",
            time=start_time,
            request_id=request_id,
        )

        first_iter_time = None
        last_iter_time = None

        for index, token in enumerate(tokens):
            if self._iter_delay:
                await asyncio.sleep(self._iter_delay)

            if first_iter_time is None:
                first_iter_time = time.time()

            yield StreamingTextResponse(
                type_="iter",
                value="".join(tokens[: index + 1]),
                start_time=start_time,
                first_iter_time=first_iter_time,
                iter_count=index + 1,
                delta=token,
                time=time.time(),
                request_id=request_id,
            )

            last_iter_time = time.time()

        yield ResponseSummary(
            value="".join(tokens),
            request_args=RequestArgs(
                target=self.target,
                headers={},
                params={},
                payload={"prompt": prompt, "output_token_count": output_token_count},
            ),
            iterations=len(tokens),
            start_time=start_time,
            end_time=time.time(),
            first_iter_time=first_iter_time,
            last_iter_time=last_iter_time,
            request_prompt_tokens=prompt_token_count,
            request_output_tokens=output_token_count,
            response_prompt_tokens=len(prompt.split()) + prompt.count(" "),
            response_output_tokens=len(tokens),
            request_id=request_id,
        )

    @staticmethod
    def _get_tokens(token_count: Optional[int] = None) -> list[str]:
        if token_count is None:
            token_count = random.randint(8, 512)

        words = TextLorem(srange=(token_count, token_count)).sentence().split()
        tokens = []  # type: ignore

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
