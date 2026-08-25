from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = [
    "InMemoryDictDataArgs",
    "InMemoryDictListDataArgs",
    "InMemoryItemListDataArgs",
]


@DataArgs.register("in_memory_dict")
class InMemoryDictDataArgs(DataArgs):
    """Model for in-memory data deserializer arguments."""

    kind: Literal["in_memory_dict"] = Field(  # type: ignore[assignment]
        default="in_memory_dict",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: dict[str, list] = Field(
        description="In-memory data input for the dataset deserializer.",
        examples=[{"column1": [1, 2, 3], "column2": [4, 5, 6]}],
    )


@DataArgs.register("in_memory_dict_list")
class InMemoryDictListDataArgs(DataArgs):
    kind: Literal["in_memory_dict_list"] = Field(  # type: ignore[assignment]
        default="in_memory_dict_list",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: list[dict[str, Any]] = Field(
        description="In-memory list of dicts input for the dataset deserializer.",
        examples=[{"column1": 1, "column2": 2}, {"column1": 3, "column2": 4}],
    )


@DataArgs.register("in_memory_item_list")
class InMemoryItemListDataArgs(DataArgs):
    kind: Literal["in_memory_item_list"] = Field(  # type: ignore[assignment]
        default="in_memory_item_list",
        description="Type identifier for the in-memory data deserializer.",
    )
    data: list[str | int | float | bool | None] = Field(
        description="In-memory list of primitive items for the dataset deserializer.",
        examples=[[1, 2, 3, 4, 5]],
    )
    column_name: str = Field(
        default="data",
        description="Column name to use when creating the dataset.",
    )
