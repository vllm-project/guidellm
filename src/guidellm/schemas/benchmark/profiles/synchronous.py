from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["SynchronousProfileArgs"]


@ProfileArgs.register("synchronous")
class SynchronousProfileArgs(ProfileArgs):
    """Pydantic model for synchronous profile creation arguments."""

    kind: Literal["synchronous"] = Field(
        default="synchronous",
        description="Profile type discriminator for synchronous scheduling",
    )
