import itertools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from guidellm.backend.response import ResponseSummary
from guidellm.request.request import GenerationRequest

__all__ = ["GenerativeRequestSession", "RequestSession"]

# TODO: Replace with specific types that implement needed features
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class RequestSession(ABC, Generic[RequestT, ResponseT]):
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


# FIXME: Bad implementation. Can only handle string requests
class GenerativeRequestSession(RequestSession[GenerationRequest, ResponseSummary]):
    def __init__(self, prompts: list[GenerationRequest]) -> None:
        if not prompts:
            raise ValueError("Prompts cannot be empty")

        self.prompts = prompts
        self.responses: list[str] = []

    def __len__(self) -> int:
        return len(self.prompts)

    def get_next_request(self) -> GenerationRequest:
        completed_responses = len(self.responses)
        base_request = self.prompts[completed_responses].model_copy(deep=True)
        base_request.content = "".join(
            itertools.chain.from_iterable(
                zip((x.content for x in self.prompts), self.responses + [""])
            )
        )
        base_request.stats["prompt_tokens"] = sum(
            x.stats["prompt_tokens"] for x in self.prompts[: completed_responses + 1]
        )
        base_request.constraints["output_tokens"] = sum(
            x.constraints["output_tokens"]
            for x in self.prompts[: completed_responses + 1]
        )

        return base_request

    def get_next_delay(self) -> float:
        return 0.0

    def push_response(self, response: ResponseSummary) -> None:
        if len(self.responses) < len(self.prompts):
            if response.response_output_tokens is not None:
                self.prompts[len(self.responses)].constraints["output_tokens"] = (
                    response.response_output_tokens
                )
            self.responses.append(response.value)
        else:
            raise ValueError("Response list full")

    @property
    def complete(self) -> bool:
        return len(self.responses) >= len(self.prompts)
