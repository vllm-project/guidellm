from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["ThroughputProfileArgs"]


@ProfileArgs.register("throughput")
class ThroughputProfileArgs(ProfileArgs):
    """Pydantic model for throughput profile creation arguments."""

    kind: Literal["throughput"] = Field(
        default="throughput",
        description="Profile type discriminator for throughput scheduling",
    )
    max_concurrency: PositiveInt | None = Field(
        description="Maximum concurrent requests to schedule",
        examples=[10],
    )
