from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from guidellm.schemas.data.deserializers.synthetic_image import (
    RESOLUTION_PRESETS,
    SyntheticVisionDataArgs,
    parse_aspect_ratio,
)
from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["SyntheticVideoDataArgs"]


@DataArgs.register("synthetic_video")
class SyntheticVideoDataArgs(SyntheticVisionDataArgs):
    """Model for synthetic video dataset deserializer arguments."""

    kind: Literal["synthetic_video"] = Field(  # type: ignore[assignment]
        default="synthetic_video",
        description="Type identifier for the synthetic video dataset configuration.",
    )
    width: int | None = Field(
        description="Frame width in pixels.",
        gt=0,
        default=None,
    )
    height: int | None = Field(
        description="Frame height in pixels.",
        gt=0,
        default=None,
    )
    resolution: str | None = Field(
        description="Resolution shortcut such as '720p' or '1080p'.",
        default=None,
    )
    aspect_ratio: str | None = Field(
        description="Aspect ratio override, e.g. '16:9' or '4:3'.",
        default=None,
    )
    frames: int = Field(
        description="Number of frames in the clip.",
        ge=1,
    )
    fps: float = Field(
        description="Frames per second.",
        gt=0.0,
        default=1.0,
    )
    format: Literal["mp4"] = Field(
        description="Container / codec. Only mp4 (h264, yuv420p) in v1.",
        default="mp4",
    )
    video_bitrate: str | None = Field(
        description="Optional libx264 bitrate string, e.g. '500k'.",
        default=None,
    )
    content: Literal["gradient", "noise"] = Field(
        description="Frame content to synthesize.",
        default="gradient",
    )

    @model_validator(mode="after")
    def _resolve_dimensions(self) -> SyntheticVideoDataArgs:
        w = self.width
        h = self.height
        if self.resolution is not None:
            preset = RESOLUTION_PRESETS.get(self.resolution.lower())
            if preset is None:
                raise ValueError(
                    f"Unknown resolution '{self.resolution}'. Known: "
                    f"{sorted(RESOLUTION_PRESETS)}"
                )
            preset_w, preset_h = preset
            if h is None:
                h = preset_h
            if w is None:
                w = (
                    int(round(h * parse_aspect_ratio(self.aspect_ratio)))
                    if self.aspect_ratio is not None
                    else preset_w
                )
        elif self.aspect_ratio is not None:
            if h is not None and w is None:
                w = int(round(h * parse_aspect_ratio(self.aspect_ratio)))
            elif w is not None and h is None:
                h = int(round(w / parse_aspect_ratio(self.aspect_ratio)))

        if w is None or h is None:
            raise ValueError(
                "synthetic_video config requires width and height, either "
                "explicitly or via resolution/aspect_ratio."
            )
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved video dims must be positive, got {self.width}x{self.height}"
            )
        return self
