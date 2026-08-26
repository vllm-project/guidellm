from __future__ import annotations

import contextlib
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs
from guidellm.utils.imports import json

__all__ = ["AsyncProfileArgs"]


@ProfileArgs.register(["async", "constant", "poisson"])
class AsyncProfileArgs(ProfileArgs):
    """Pydantic model for asynchronous profile creation arguments."""

    kind: Literal["async", "constant", "poisson"] = Field(
        default="async",
        description="Profile type discriminator for asynchronous scheduling",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rates in requests per second",
        examples=[1.0, [1.0, 2.0, 3.0]],
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
        examples=[10],
    )

    @field_validator("rate", mode="before")
    @classmethod
    def _coerce_rate_to_list(
        cls, value: list[PositiveFloat] | PositiveFloat
    ) -> list[PositiveFloat]:
        """Normalize rate to a list of integers.

        Allow single integer or list of integers.
        """
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                value = json.loads(value)
        if not value:
            raise ValueError("rate requires at least one value")
        if isinstance(value, list | tuple):
            return value
        if isinstance(value, int | float):
            return [value]
        raise ValueError(
            "rate must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )
