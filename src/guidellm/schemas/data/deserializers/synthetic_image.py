from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = [
    "RESOLUTION_PRESETS",
    "SyntheticImageDataArgs",
    "SyntheticVisionDataArgs",
    "parse_aspect_ratio",
]

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}


def parse_aspect_ratio(aspect: str) -> float:
    """Parse 'W:H' or 'W/H' into a float ratio."""
    sep = ":" if ":" in aspect else "/"
    try:
        w, h = aspect.split(sep)
        return float(w) / float(h)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid aspect_ratio '{aspect}', expected 'W:H' or 'W/H'"
        ) from exc


class SyntheticVisionDataArgs(DataArgs):
    output_tokens: int | None = Field(
        description="The average number of output tokens to request.",
        gt=0,
        default=None,
    )
    output_tokens_stdev: int | None = Field(
        description="Standard deviation of output-token counts per request.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="Minimum number of output tokens per request.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="Maximum number of output tokens per request.",
        gt=0,
        default=None,
    )
    seed: int = Field(
        description="Base random seed for reproducible synthetic payloads.",
        default=42,
    )


@DataArgs.register("synthetic_image")
class SyntheticImageDataArgs(SyntheticVisionDataArgs):
    """Model for synthetic image dataset deserializer arguments."""

    kind: Literal["synthetic_image"] = Field(  # type: ignore[assignment]
        default="synthetic_image",
        description="Type identifier for the synthetic image dataset configuration.",
    )
    width: int | None = Field(
        description="Image width in pixels.",
        gt=0,
        default=None,
    )
    height: int | None = Field(
        description="Image height in pixels.",
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
    format: Literal["jpeg", "png"] = Field(
        description="Encoded image format.",
        default="jpeg",
    )
    jpeg_quality: int = Field(
        description="JPEG quality 1..100. Ignored when format='png'.",
        ge=1,
        le=100,
        default=85,
    )
    content: Literal["gradient", "noise", "solid", "checkerboard"] = Field(
        description="Pixel content to synthesize.",
        default="gradient",
    )
    images_per_request: int = Field(
        description="Number of images per emitted row.",
        ge=1,
        default=1,
    )

    @model_validator(mode="after")
    def _resolve_dimensions(self) -> SyntheticImageDataArgs:
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
                "synthetic_image config requires width and height, either "
                "explicitly or via resolution/aspect_ratio."
            )
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved image dims must be positive, got {self.width}x{self.height}"
            )
        return self
