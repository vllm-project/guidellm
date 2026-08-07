from __future__ import annotations

import contextlib
from typing import Any, Literal

from pydantic import Field, PositiveInt, field_validator

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs
from guidellm.utils.imports import json

__all__ = ["ConcurrentProfileArgs"]


@ProfileArgs.register("concurrent")
class ConcurrentProfileArgs(ProfileArgs):
    """Pydantic model for concurrent profile creation arguments."""

    kind: Literal["concurrent"] = Field(
        default="concurrent",
        description="Profile type discriminator for concurrent scheduling",
    )
    streams: list[PositiveInt] = Field(
        description="Concurrent stream counts to execute",
        examples=[[1, 2, 3], 10],
    )

    @field_validator("streams", mode="before")
    @classmethod
    def _coerce_streams_to_list(cls, value: Any) -> Any:
        """Normalize streams to a list of integers.

        Allow single integer or list of integers.
        """
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                value = json.loads(value)
        if not value:
            raise ValueError("streams requires at least one value")
        if isinstance(value, list | tuple):
            return [int(stream) for stream in value]
        if isinstance(value, int | float):
            return [int(value)]
        raise ValueError(
            "streams must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )
