from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from guidellm.schemas import StandardBaseModel

__all__ = [
    "DataConfig",
    "DataNotSupportedError",
    "GenerativeDatasetColumnType",
    "SyntheticImageDatasetConfig",
    "SyntheticTextDatasetConfig",
    "SyntheticTextPrefixBucketConfig",
    "SyntheticVideoDatasetConfig",
]


# Canonical resolution names. Resolves to (width, height) at the common
# 16:9 aspect ratio. Aspect ratio overrides recompute width from height.
_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
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


def _parse_aspect_ratio(aspect: str) -> float:
    """Parse 'W:H' or 'W/H' into a float ratio."""
    sep = ":" if ":" in aspect else "/"
    try:
        w, h = aspect.split(sep)
        return float(w) / float(h)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid aspect_ratio '{aspect}', expected 'W:H' or 'W/H'"
        ) from exc


GenerativeDatasetColumnType = Literal[
    "prompt_tokens_count_column",
    "output_tokens_count_column",
    "prefix_column",
    "text_column",
    "image_column",
    "video_column",
    "audio_column",
]


class DataNotSupportedError(Exception):
    """
    Exception raised when the data format is not supported by deserializer or config.
    """


class DataConfig(StandardBaseModel):
    """
    A generic parent class for various configs for the data package
    that can be passed in as key-value pairs or JSON.
    """


class PreprocessDatasetConfig(DataConfig):
    prompt_tokens: int = Field(
        description="The average number of text tokens retained or added to prompts.",
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the number of tokens retained in or "
        "added to prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens retained or added to outputs.",
        gt=0,
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the number of tokens retained or "
        "added to outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to outputs.",
        gt=0,
        default=None,
    )
    prefix_tokens_max: int | None = Field(
        description="The maximum number of text tokens left in the prefixes.",
        gt=0,
        default=None,
    )


class SyntheticTextPrefixBucketConfig(StandardBaseModel):
    bucket_weight: int = Field(
        description="Weight of this bucket in the overall distribution.",
        gt=0,
        default=100,
    )
    prefix_count: int = Field(
        description="The number of unique prefixes to generate for this bucket.",
        ge=1,
        default=1,
    )
    prefix_tokens: int = Field(
        description="The number of prefix tokens per-prompt for this bucket.",
        ge=0,
        default=0,
    )


class SyntheticTextDatasetConfig(DataConfig):
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts.",
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int | None = Field(
        description="The average number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    turns: int = Field(
        description="The number of turns in the conversation.",
        gt=0,
        default=1,
    )

    model_config = ConfigDict(
        extra="allow",
    )

    prefix_buckets: list[SyntheticTextPrefixBucketConfig] | None = Field(
        description="Buckets for the prefix tokens distribution.",
        default=None,
    )

    @model_validator(mode="after")
    def check_prefix_options(self) -> SyntheticTextDatasetConfig:
        if self.__pydantic_extra__ is not None:
            prefix_count = self.__pydantic_extra__.get("prefix_count", None)  # type: ignore[attr-defined]
            prefix_tokens = self.__pydantic_extra__.get("prefix_tokens", None)  # type: ignore[attr-defined]

            if prefix_count is not None or prefix_tokens is not None:
                if self.prefix_buckets:
                    raise ValueError(
                        "prefix_buckets is mutually exclusive"
                        " with prefix_count and prefix_tokens"
                    )

                self.prefix_buckets = [
                    SyntheticTextPrefixBucketConfig(
                        prefix_count=prefix_count or 1,
                        prefix_tokens=prefix_tokens or 0,
                    )
                ]

        return self


class _SyntheticMultimodalTextMixin(DataConfig):
    """
    Shared text-shaping fields for the multimodal synthetic configs.

    Canonical field name is ``text_tokens`` (not ``prompt_tokens``) to keep
    the multimodal CLI explicit about which payload is being shaped. The
    pre-existing ``synthetic_text`` config keeps ``prompt_tokens`` unchanged.
    ``prompt_tokens`` is accepted as an alias for ergonomics.
    """

    text_tokens: int = Field(
        description="The average number of text tokens generated for the "
        "text portion of each multimodal prompt.",
        gt=0,
    )
    text_tokens_stdev: int | None = Field(
        description="Standard deviation of text-token counts per prompt.",
        gt=0,
        default=None,
    )
    text_tokens_min: int | None = Field(
        description="Minimum number of text tokens per prompt.",
        gt=0,
        default=None,
    )
    text_tokens_max: int | None = Field(
        description="Maximum number of text tokens per prompt.",
        gt=0,
        default=None,
    )
    output_tokens: int | None = Field(
        description="The average number of output tokens to request.",
        gt=0,
        default=None,
    )
    output_tokens_stdev: int | None = Field(
        description="Standard deviation of output-token counts per prompt.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="Minimum number of output tokens per prompt.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="Maximum number of output tokens per prompt.",
        gt=0,
        default=None,
    )
    seed: int = Field(
        description="Base random seed. Threaded through text generation, "
        "PIL, numpy, and the video encoder so two runs with the same seed "
        "produce byte-identical payloads.",
        default=42,
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_prompt_tokens(cls, data: object) -> object:
        """Accept ``prompt_tokens`` as an alias for ``text_tokens``."""
        if isinstance(data, dict):
            aliases = {
                "prompt_tokens": "text_tokens",
                "prompt_tokens_stdev": "text_tokens_stdev",
                "prompt_tokens_min": "text_tokens_min",
                "prompt_tokens_max": "text_tokens_max",
            }
            for alias, canonical in aliases.items():
                if alias in data and canonical not in data:
                    data[canonical] = data.pop(alias)
        return data


class SyntheticImageDatasetConfig(_SyntheticMultimodalTextMixin):
    """Config for the ``synthetic_image`` deserializer."""

    # Pixel dimensions. Either set width/height directly, or use the
    # ``resolution`` sugar (e.g. ``720p``) optionally combined with
    # ``aspect_ratio`` (e.g. ``16:9``).
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
        description="Resolution shortcut (e.g. '720p', '1080p'). Resolves to "
        "(width, height) at 16:9; pair with aspect_ratio to override.",
        default=None,
    )
    aspect_ratio: str | None = Field(
        description="Aspect ratio override, e.g. '16:9' or '4:3'. Combined "
        "with resolution / height to compute the other dimension.",
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
        description="Pixel content. 'gradient' (default) randomizes base "
        "color and direction per row so payloads are byte-different (defeats "
        "the mm-processor cache) but still compress like real media. 'noise' "
        "is worst-case wire size; 'solid' and 'checkerboard' are opt-in for "
        "cache-sensitivity sweeps.",
        default="gradient",
    )
    images_per_request: int = Field(
        description="Number of images per emitted row.",
        ge=1,
        default=1,
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _resolve_dimensions(self) -> SyntheticImageDatasetConfig:
        # Resolve resolution / aspect_ratio sugar into concrete width/height.
        w = self.width
        h = self.height
        if self.resolution is not None:
            preset = _RESOLUTION_PRESETS.get(self.resolution.lower())
            if preset is None:
                raise ValueError(
                    f"Unknown resolution '{self.resolution}'. Known: "
                    f"{sorted(_RESOLUTION_PRESETS)}"
                )
            preset_w, preset_h = preset
            # Use preset height as the anchor; recompute width from aspect.
            if h is None:
                h = preset_h
            if w is None:
                if self.aspect_ratio is not None:
                    w = int(round(h * _parse_aspect_ratio(self.aspect_ratio)))
                else:
                    w = preset_w
        elif self.aspect_ratio is not None:
            if h is not None and w is None:
                w = int(round(h * _parse_aspect_ratio(self.aspect_ratio)))
            elif w is not None and h is None:
                h = int(round(w / _parse_aspect_ratio(self.aspect_ratio)))

        if w is None or h is None:
            raise ValueError(
                "synthetic_image config requires width and height (either "
                "explicitly, or via 'resolution' / 'aspect_ratio')."
            )
        # Round to even dims so jpeg / downstream yuv420p paths are happy.
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved image dims must be positive, got "
                f"{self.width}x{self.height}"
            )
        return self


class SyntheticVideoDatasetConfig(_SyntheticMultimodalTextMixin):
    """Config for the ``synthetic_video`` deserializer."""

    width: int = Field(
        description="Frame width in pixels.",
        gt=0,
    )
    height: int = Field(
        description="Frame height in pixels.",
        gt=0,
    )
    frames: int = Field(
        description="Number of frames in the clip.",
        ge=1,
    )
    fps: float = Field(
        description="Frames per second. Encoded into the container and used "
        "to derive video_seconds = frames / fps on the emitted row.",
        gt=0.0,
        default=1.0,
    )
    format: Literal["mp4"] = Field(
        description="Container / codec. Only mp4 (h264, yuv420p) in v1.",
        default="mp4",
    )
    video_bitrate: str | None = Field(
        description="Optional libx264 bitrate string, e.g. '500k'. Default "
        "uses the codec's CRF rate control.",
        default=None,
    )
    content: Literal["gradient", "noise"] = Field(
        description="Frame content. 'gradient' (default) randomizes base "
        "color and direction per frame so every row is byte-different. "
        "'noise' is worst-case wire size.",
        default="gradient",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _round_dims(self) -> SyntheticVideoDatasetConfig:
        # libx264 + yuv420p needs even dims. Round down to the nearest even.
        self.width = int(self.width) - (int(self.width) % 2)
        self.height = int(self.height) - (int(self.height) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved video dims must be positive, got "
                f"{self.width}x{self.height}"
            )
        return self
