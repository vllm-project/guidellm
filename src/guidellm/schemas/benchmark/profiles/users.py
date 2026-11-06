from __future__ import annotations

import contextlib
from typing import Any, Literal

from pydantic import (
    Field,
    PositiveInt,
    field_validator,
)

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs
from guidellm.utils.imports import json

__all__ = ["UsersProfileArgs"]


@ProfileArgs.register("users")
class UsersProfileArgs(ProfileArgs):
    """Pydantic model for users profile creation arguments."""

    kind: Literal["users"] = Field(
        default="users",
        description="Profile type discriminator for users scheduling",
    )
    users: list[PositiveInt] = Field(
        description="Simulated user counts to execute",
        examples=[[8, 16], 8],
    )

    @field_validator("users", mode="before")
    @classmethod
    def _coerce_users_to_list(cls, value: Any) -> Any:
        """Normalize users to a list of integers.

        Allow single integer or list of integers.
        """
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                value = json.loads(value)
        if not value:
            raise ValueError("users requires at least one value")
        if isinstance(value, list | tuple):
            return [int(user) for user in value]
        if isinstance(value, int | float):
            return [int(value)]
        raise ValueError(
            "users must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )
