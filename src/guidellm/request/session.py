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


# TODO: Implement multiturn support
class GenerativeRequestSession(RequestSession[GenerationRequest, ResponseSummary]):
    def __init__(self, request: GenerationRequest) -> None:
        self.request = request
        self._complete = False

    def __len__(self) -> int:
        return 1

    def get_next_request(self) -> GenerationRequest:
        return self.request

    def get_next_delay(self) -> float:
        return 0.0

    def push_response(self, response: ResponseSummary) -> None:  # noqa: ARG002
        self._complete = True

    @property
    def complete(self) -> bool:
        return self._complete
