from __future__ import annotations

from typing import TypeVar

from typing_extensions import TypeAliasType

from guidellm.schemas import RequestInfo

__all__ = [
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

HistoryT = TypeAliasType(
    "HistoryT",
    list[tuple[RequestT, ResponseT | None]],
    type_params=(RequestT, ResponseT),
)
"""Record of requests + responses in conversation."""
