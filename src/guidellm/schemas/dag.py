from __future__ import annotations

from collections.abc import Generator
from typing import Generic, Protocol, TypeVar

from pydantic import Field

from guidellm.schemas.base import StandardBaseModel
from guidellm.schemas.info import RequestInfo, RequestSettings

RequestT = TypeVar("RequestT")


class DAGMutator(Protocol[RequestT]):
    def __call__(self, node: DAGNode[RequestT], **kwargs) -> None:
        """
        A callable that mutates a DAGNode.
        """
        ...


class DAGNode(StandardBaseModel, Generic[RequestT]):
    """
    A node in a directed acyclic graph (DAG).
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list, description="The next nodes in the DAG."
    )

    def _apply_to_all(
        self, func: DAGMutator[RequestT], idx: int
    ) -> Generator[int, dict, None]:
        sent = yield idx
        func(self, **sent)
        for child in self.next:
            child._apply_to_all(func, idx + 1)  # noqa: SLF001

    def apply_to_all(self, func: DAGMutator[RequestT]) -> Generator[int, dict, None]:
        """
        Apply a function to this node and all its descendants.
        """
        yield from self._apply_to_all(func, 0)


class StartNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents the start of the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list,
        min_length=1,
        max_length=1,
        description="The next node in the DAG.",
    )


class ForkNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents a fork in the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list, min_length=1, description="The next nodes in the DAG."
    )


class SpawnNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents a spawn point in the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list, min_length=1, description="The next nodes in the DAG."
    )


class JoinNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents a join point in the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list,
        min_length=1,
        max_length=1,
        description="The next nodes in the DAG.",
    )


class EndNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents the end of the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list, max_length=0, description="The next nodes in the DAG."
    )


class RequestNode(DAGNode[RequestT], Generic[RequestT]):
    """
    A node that represents a request in the DAG.
    """

    next: list[DAGNode[RequestT]] = Field(
        default_factory=list,
        min_length=1,
        max_length=1,
        description="The next nodes in the DAG.",
    )
    request: RequestT = Field(..., description="The request associated with this node.")
    settings: RequestSettings = Field(
        default_factory=RequestSettings,
        description="The settings associated with this node.",
    )
    info: RequestInfo | None = Field(
        default_factory=None, description="The info associated with this node."
    )
