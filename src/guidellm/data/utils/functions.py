from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

import datasets
import httpx
import librosa
import numpy as np
import soundfile
from PIL import Image as PILImage
from pydub import AudioSegment
from torch import Tensor

__all__ = [
    "download_audio",
    "download_image",
    "download_video",
    "encode_audio",
    "encode_audio_as_dict",
    "encode_audio_as_file",
    "encode_image",
    "encode_image_base64",
    "encode_video",
    "encode_video_base64",
    "get_file_format",
    "is_url",
    "resize_image",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


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


def download_image(url: str) -> bytes:
    response = httpx.get(url)
    response.raise_for_status()
    return response.content


def encode_video(
    video: bytes | str | Path,
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
    if (
        isinstance(video, str)
        and is_url(video)
        and (encode_type is None or encode_type == "url")
    ):
        return video
    elif isinstance(video, str) and is_url(video) and encode_type == "base64":
        raise ValueError(f"Cannot encode URL video {video}")

    return encode_video_base64(video=video)


def encode_video_base64(video: bytes | str | Path) -> str:
    if (
        isinstance(video, str)
        and video.startswith("data:video/")
        and ";base64," in video
    ):
        return video

    video_format = "unknown"

    if isinstance(video, str) and is_url(video):
        video, video_format = download_video(video)

    if isinstance(video, (str, Path)):
        path = Path(video)
        video = path.read_bytes()
        video_format = get_file_format(path)
    elif not isinstance(video, bytes):
        raise ValueError(f"Unsupported video type: {type(video)}")

    video_base64 = base64.b64encode(video).decode("utf-8")
    return f"data:video/{video_format};base64,{video_base64}"


def download_video(url: str) -> tuple[bytes, str]:
    response = httpx.get(url)
    response.raise_for_status()
    return response.content, get_file_format(url)


def encode_audio_as_dict(
    audio: Any,
    sample_rate: int = 16000,
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str = "mp3",
    bitrate: str = "64k",
) -> dict[Literal["data", "format"], Any]:
    content, _, file_format = encode_audio(
        audio=audio,
        sample_rate=sample_rate,
        encode_sample_rate=encode_sample_rate,
        max_duration=max_duration,
        mono=mono,
        audio_format=audio_format,
        bitrate=bitrate,
    )

    return {
        "data": base64.b64encode(content).decode("utf-8"),
        "format": file_format,
    }


def encode_audio_as_file(
    audio: Any,
    sample_rate: int = 16000,
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str = "mp3",
    bitrate: str = "64k",
) -> tuple[str, bytes, str]:
    content, file_name, file_format = encode_audio(
        audio=audio,
        sample_rate=sample_rate,
        encode_sample_rate=encode_sample_rate,
        max_duration=max_duration,
        mono=mono,
        audio_format=audio_format,
        bitrate=bitrate,
    )

    return file_name, content, f"audio/{file_format}"


def encode_audio(  # noqa: PLR0912, PLR0911, C901
    audio: Any,
    sample_rate: int = 16000,
    file_name: str = "audio.wav",
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str = "mp3",
    bitrate: str = "64k",
) -> tuple[bytes, str, str]:
    audio_buffer: io.BytesIO = io.BytesIO()

    if hasattr(audio, "get_samples_played_in_range"):
        # HF datasets Audio object
        audio_samples = audio.get_samples_played_in_range(
            start_seconds=0.0,
            stop_seconds=None
            if max_duration is None
            else min(max_duration, audio.metadata.duration_seconds_from_header),
        )
        return encode_audio(
            audio=audio_samples.data.numpy(),
            sample_rate=audio_samples.sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if isinstance(audio, Tensor):
        return encode_audio(
            audio=audio.numpy(),
            sample_rate=sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if isinstance(audio, dict):
        sample_rate = audio.get("sample_rate", audio.get("sampling_rate", sample_rate))
        if "data" not in audio and "url" not in audio:
            raise ValueError(
                f"Audio dict must contain either 'data' or 'url' keys, got {audio}"
            )
        return encode_audio(
            audio=audio.get("data") or audio.get("url"),
            sample_rate=sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if isinstance(audio, str) and is_url(audio):
        audio_bytes, file_name, _ = download_audio(audio)
        return encode_audio(
            audio=audio_bytes,
            sample_rate=sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if isinstance(audio, (str, Path)):
        if not Path(audio).exists():
            raise ValueError(f"Audio file does not exist: {audio}")
        file_name = get_file_name(audio)
        data, sample_rate = soundfile.read(str(audio), dtype="float32")

        return encode_audio(
            audio=data,
            sample_rate=sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if isinstance(audio, bytes):
        data, sample_rate = soundfile.read(io.BytesIO(audio), dtype="float32")

        return encode_audio(
            audio=data,
            sample_rate=sample_rate,
            encode_sample_rate=encode_sample_rate,
            max_duration=max_duration,
            mono=mono,
            audio_format=audio_format,
            bitrate=bitrate,
        )

    if not isinstance(audio, np.ndarray):
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    if sample_rate != encode_sample_rate:
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sample_rate, target_sr=encode_sample_rate
        )
        sample_rate = encode_sample_rate

    audio = librosa.to_mono(audio)

    if (
        max_duration is not None
        and max_duration > 0
        and (max_samples := int(max_duration * sample_rate)) < len(audio)
    ):
        audio = audio[:max_samples]

    audio_buffer = io.BytesIO()

    if audio_format.lower() == "mp3":
        wav = io.BytesIO()
        soundfile.write(wav, audio, sample_rate, format="WAV", subtype="PCM_16")
        wav.seek(0)

        sound = AudioSegment.from_wav(wav)
        sound.export(audio_buffer, format="mp3", bitrate=bitrate)
    else:
        soundfile.write(audio_buffer, audio, sample_rate, format=audio_format.upper())

    audio_buffer.seek(0)
    return audio_buffer.read(), file_name, audio_format.lower()


def download_audio(url: str) -> tuple[bytes, str, str]:
    response = httpx.get(url)
    response.raise_for_status()
    content = response.content

    return content, get_file_name(url), get_file_format(url)


def get_file_name(path: Path | str) -> str:
    """Get file name from path."""
    return Path(path).name


def get_file_format(path: Path | str) -> str:
    """Get file format from path extension."""
    suffix = Path(path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else "unknown"
