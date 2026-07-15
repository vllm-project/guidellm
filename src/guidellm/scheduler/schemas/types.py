from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from typing_extensions import TypeAliasType

from guidellm.schemas import RequestInfo, RequestSettings

__all__ = [
    "ConversationT",
    "DatasetIterT",
    "HistoryT",
    "RequestDataT",
    "RequestT",
    "ResponseT",
]

RequestT = TypeVar("RequestT")
"Generic request object type for scheduler processing"

ResponseT = TypeVar("ResponseT")
"Generic response object type returned by backend processing"

RequestDataT = TypeAliasType(
    "RequestDataT",
    tuple[RequestT, RequestInfo],
    type_params=(RequestT,),
)
"""Request including external metadata and scheduling config."""

ConversationT = TypeAliasType(
    "ConversationT",
    list[RequestDataT[RequestT]],
    type_params=(RequestT,),
)

HistoryT = TypeAliasType(
    "HistoryT",
    list[tuple[RequestT, ResponseT | None]],
    type_params=(RequestT, ResponseT),
)
"""Record of requests + responses in conversation."""

# NOTE: This is the interface between data and scheduler.
DatasetIterT = TypeAliasType(
    "DatasetIterT",
    Iterable[Iterable[tuple[RequestT, RequestSettings]]],
    type_params=(RequestT,),
)
"""
Output of data loader, an iterable of batches,
where each batch is an iterable of (request, timestamp) tuples.
"""
