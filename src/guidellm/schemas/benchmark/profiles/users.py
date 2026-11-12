from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    Field,
    PositiveInt,
    field_validator,
)

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs, coerce_int_list

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
        return coerce_int_list(value, "users")
