from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["SweepProfileArgs"]


@ProfileArgs.register("sweep")
class SweepProfileArgs(ProfileArgs):
    """Pydantic model for sweep profile creation arguments."""

    kind: Literal["sweep"] = Field(
        default="sweep",
        description="Profile type discriminator for sweep scheduling",
    )
    sweep_size: int = Field(
        default=10,
        description="Number of strategies to generate for the sweep",
        ge=2,
    )
    strategy_type: Literal["constant", "poisson"] = Field(
        default="constant",
        description="Type of strategy to use for the asynchronous sweep",
    )
    max_concurrency: PositiveInt | None = Field(
        default=512,
        description="Maximum concurrent requests to schedule",
    )
