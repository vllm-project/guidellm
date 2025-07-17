import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from guidellm.backend.response import ResponseSummary
from guidellm.config import settings
from guidellm.preprocess.item import Item, ItemList
from guidellm.request.request import GenerationRequest
from guidellm.request.types import RequestT, ResponseT

__all__ = ["GenerativeRequestSession", "RequestSession"]


class RequestSession(ABC, Generic[RequestT, ResponseT]):
    """
    A series of requests that build upon each other to
    form a conversion between the user and the model.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_next_request(self) -> RequestT: ...

    @abstractmethod
    def get_next_delay(self) -> float: ...

    @abstractmethod
    def push_response(self, response: ResponseT) -> None: ...

    @property
    @abstractmethod
    def complete(self) -> bool: ...


class GenerativeRequestSession(RequestSession[GenerationRequest, ResponseSummary]):
    def __init__(self, items: ItemList) -> None:
        if len(items) < 1:
            raise ValueError("Prompts cannot be empty")

        self.prompts: Sequence[Item] = items
        self.responses: list[Item] = []

    def __len__(self) -> int:
        return len(self.prompts)

    def get_next_request(self) -> GenerationRequest:
        completed_responses = len(self.responses)

        # FIXME: Can only handle string requests
        content = "".join(
            itertools.chain.from_iterable(
                (x.value, y.value)
                for x, y in zip(self.prompts, self.responses + [Item(value="")])
            )
        )

        prev_prompt_tokens = sum(
            (x.prompt_tokens or 0) + (x.output_tokens or 0) for x in self.responses
        )
        prompt_tokens = (
            self.prompts[completed_responses].prompt_tokens or 0
        ) + prev_prompt_tokens

        output_tokens = self.prompts[completed_responses].output_tokens

        return GenerationRequest(
            request_type=settings.preferred_route,
            content=content,
            stats=(
                {"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {}
            ),
            constraints=(
                {"output_tokens": output_tokens} if output_tokens is not None else {}
            ),
        )

    def get_next_delay(self) -> float:
        return 0.0

    def push_response(self, response: ResponseSummary) -> None:
        if len(self.responses) < len(self.prompts):
            resp = Item(
                value=response.value,
                prompt_tokens=response.response_prompt_tokens
                or response.request_prompt_tokens,
                output_tokens=response.response_output_tokens
                or response.request_output_tokens,
            )
            self.responses.append(resp)
        else:
            raise ValueError("Response list full")

    @property
    def complete(self) -> bool:
        return len(self.responses) >= len(self.prompts)
