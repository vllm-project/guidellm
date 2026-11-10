from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, PositiveInt, field_validator

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs, coerce_int_list

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
        """Normalize streams to a list, leaving entry validation to Pydantic.

        Allow a single integer, list, tuple, or JSON-encoded array.
        """
        return coerce_int_list(value, "streams")
