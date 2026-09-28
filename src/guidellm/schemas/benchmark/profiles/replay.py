from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["ReplayProfileArgs"]


@ProfileArgs.register("replay")
class ReplayProfileArgs(ProfileArgs):
    """Pydantic model for trace replay profile creation arguments."""

    kind: Literal["replay"] = Field(
        default="replay",
        description="Profile type discriminator for trace replay scheduling",
    )
    time_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scheduler scale factor applied to relative timestamps",
    )
    timing: Literal["absolute", "relative"] = Field(
        default="absolute",
        description=(
            "absolute schedules each request at its trace timestamp and waits "
            "only while a prior turn is still running. relative keeps the idle "
            "gap after each request's recorded duration."
        ),
    )
