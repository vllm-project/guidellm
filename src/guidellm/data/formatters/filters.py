from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Callable, Literal

import datasets
import httpx
import librosa
import numpy as np
import soundfile
from PIL import Image as PILImage

from guidellm.utils import RegistryMixin

__all__ = [
    "JinjaFiltersRegistry",
    "download_audio",
    "download_image",
    "download_video",
    "encode_audio",
    "encode_image",
    "encode_image_base64",
    "encode_video",
    "encode_video_base64",
    "get_file_format",
    "is_url",
    "resize_image",
]


class JinjaFiltersRegistry(RegistryMixin[Callable[..., Any]]):
    pass


@JinjaFiltersRegistry.register("is_url")
def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


@JinjaFiltersRegistry.register("encode_image")
def encode_image(
    image: bytes | str | Path | np.ndarray | PILImage.Image | datasets.Image,
    max_size: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    encode_type: Literal["base64", "url"] | None = None,
) -> str:
    """
    Input image types:
    - bytes: raw image bytes, decoded with Pillow
    - str: file path on disk, url, or already base64 encoded image string
    - pathlib.Path: file path on disk
    - np.ndarray: image array, decoded with Pillow
    - PIL.Image.Image: Pillow image
    - datasets.Image: HuggingFace datasets Image object

    max_size: maximum size of the longest edge of the image
    max_width: maximum width of the image
    max_height: maximum height of the image

    encode_type: None to return the supported format
        (url for url, base64 string for others)
        "base64" to return base64 encoded string (or download URL and encode)
        "url" to return url (only if input is url, otherwise fails)

    Returns a str of either:
    - image url
    - "data:image/{type};base64, {data}" string
    """
    url = is_url(image)

    if (
        url
        and (encode_type is None or encode_type == "url")
        and (max_size is not None or max_width is not None or max_height is not None)
    ):
        raise ValueError("Cannot resize image when encode_type is 'url'")
    elif url and (encode_type is None or encode_type == "url"):
        return image
    elif url and encode_type == "base64":
        raise ValueError(f"Cannot convert non-url image to URL {image}")

    return encode_image_base64(
        image=image,
        max_size=max_size,
        max_width=max_width,
        max_height=max_height,
    )


@JinjaFiltersRegistry.register("encode_image_base64")
def encode_image_base64(
    image: bytes | str | Path | np.ndarray | PILImage.Image,
    width: int | None = None,
    height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_size: int | None = None,
) -> str:
    if (
        isinstance(image, str)
        and image.startswith("data:image/")
        and ";base64," in image
    ):
        return image

    if is_url(image):
        image = download_image(image)

    if isinstance(image, bytes):
        image = PILImage.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = PILImage.open(image)
    elif isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)
    elif not isinstance(image, PILImage.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    image = resize_image(
        image,
        width=width,
        height=height,
        max_width=max_width,
        max_height=max_height,
        max_size=max_size,
    )
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{image_base64}"


@JinjaFiltersRegistry.register("resize_image")
def resize_image(
    image: PILImage.Image,
    width: int | None = None,
    height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_size: int | None = None,
) -> PILImage.Image:
    if not isinstance(image, PILImage.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    if width is not None and height is not None:
        return image.resize((width, height), PILImage.Resampling.BILINEAR)

    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    if width is not None:
        target_w = width
        target_h = round(width / aspect)
    elif height is not None:
        target_h = height
        target_w = round(height * aspect)
    else:
        target_w, target_h = orig_w, orig_h

    # Normalize max_size → max_width/max_height
    if max_size is not None:
        max_width = max_width or max_size
        max_height = max_height or max_size

    # Apply max constraints (preserve aspect ratio)
    if max_width or max_height:
        scale_w = max_width / target_w if max_width else 1.0
        scale_h = max_height / target_h if max_height else 1.0
        scale = min(scale_w, scale_h, 1.0)  # never upscale
        target_w = round(target_w * scale)
        target_h = round(target_h * scale)

    if (target_w, target_h) != (orig_w, orig_h):
        image = image.resize((target_w, target_h), PILImage.Resampling.BILINEAR)

    return image


@JinjaFiltersRegistry.register("download_image")
def download_image(url: str) -> bytes:
    response = httpx.get(url)
    response.raise_for_status()
    return response.content


@JinjaFiltersRegistry.register("encode_video")
def encode_video(
    video: bytes | str | Path | datasets.Video,
    encode_type: Literal["base64", "url"] | None = None,
) -> str:
    """
    Input video types:
    - bytes: raw video bytes
    - str: file path on disk, url, or already base64 encoded video string
    - pathlib.Path: file path on disk
    - datasets.Video: HuggingFace datasets Video object

    encode_type: None to return the supported format
        (url for url, base64 string for others)
        "base64" to return base64 encoded string (or download URL and encode)
        "url" to return url (only if input is url, otherwise fails)

    Returns a str of either:
    - video url
    - "data:video/{type};base64, {data}" string
    """
    url = is_url(video)

    if url and (encode_type is None or encode_type == "url"):
        return video
    elif url and encode_type == "base64":
        raise ValueError(f"Cannot encode URL video {video}")

    return encode_video_base64(video=video)


@JinjaFiltersRegistry.register("encode_video_base64")
def encode_video_base64(video: bytes | str | Path) -> str:
    if (
        isinstance(video, str)
        and video.startswith("data:video/")
        and ";base64," in video
    ):
        return video

    video_format = "unknown"

    if is_url(video):
        video, video_format = download_video(video)

    if isinstance(video, (str, Path)):
        path = Path(video)
        video = path.read_bytes()
        video_format = get_file_format(path)
    elif not isinstance(video, bytes):
        raise ValueError(f"Unsupported video type: {type(video)}")

    video_base64 = base64.b64encode(video).decode("utf-8")
    return f"data:video/{video_format};base64,{video_base64}"


@JinjaFiltersRegistry.register("download_video")
def download_video(url: str) -> tuple[bytes, str]:
    response = httpx.get(url)
    response.raise_for_status()
    return response.content, get_file_format(url)


@JinjaFiltersRegistry.register("encode_audio")
def encode_audio(
    audio: bytes | str | Path | dict | np.ndarray,
    sample_rate: int | None = None,
    max_duration: float | None = None,
) -> dict[str, str]:
    """
    Input audio types:
    - bytes: raw audio bytes
    - str: file path on disk or URL
    - pathlib.Path: file path on disk
    - dict: {"data": base64_string, "format": "wav"} format
    - numpy.ndarray: audio array, assumed to be at sample_rate if provided

    sample_rate: sample rate of the input audio if input is np.ndarray
    target_sample_rate: resample to this rate if provided
    duration: limit audio to this duration in seconds if provided

    Returns dict with format:
    {
        "data": base64_encoded_audio_bytes,
        "format": "wav"
    }
    """
    if is_url(audio):
        audio, _ = download_audio(audio)

    if isinstance(audio, dict):
        if "data" not in audio:
            raise ValueError("Audio dict must contain 'data' key")
        audio = base64.b64decode(audio["data"])

    if isinstance(audio, bytes):
        audio_data, sample_rate = librosa.load(io.BytesIO(audio), sr=sample_rate)
    elif isinstance(audio, (str, Path)):
        audio_data, sample_rate = librosa.load(str(audio), sr=sample_rate)
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided for numpy arrays")
        audio_data = audio
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]

    buffer = io.BytesIO()
    soundfile.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")

    return {"data": buffer.getvalue(), "format": "wav"}


@JinjaFiltersRegistry.register("download_audio")
def download_audio(url: str) -> tuple[bytes, str]:
    """Download audio from URL and return bytes with format."""
    response = httpx.get(url)
    response.raise_for_status()
    content = response.content
    audio_format = get_file_format(url)
    return content, audio_format


@JinjaFiltersRegistry.register("get_file_format")
def get_file_format(path: Path | str) -> str:
    """Get file format from path extension."""
    suffix = Path(path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else "unknown"
