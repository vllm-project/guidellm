from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

import httpx
import librosa
import numpy as np
import soundfile
from PIL import Image as PILImage
from pydub import AudioSegment
from torch import Tensor

__all__ = [
    "encode_audio",
    "encode_image",
    "encode_video",
    "get_file_format",
    "is_url",
    "resize_image",
    "text_stats",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


def text_stats(
    text: str,
) -> dict[Literal["type", "text", "num_chars", "num_words"], str | int]:
    """Compute basic text statistics."""
    num_chars = len(text)
    num_words = len(text.split())

    return {
        "type": "text",
        "text": text,
        "num_chars": num_chars,
        "num_words": num_words,
    }


def encode_image(
    image: bytes | str | Path | np.ndarray | PILImage.Image,
    width: int | None = None,
    height: int | None = None,
    max_size: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    encode_type: Literal["base64", "url"] | None = "base64",
) -> dict[Literal["type", "image", "image_pixels", "image_bytes"], str | int | None]:
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
    if isinstance(image, str) and is_url(image):
        if encode_type == "base64":
            response = httpx.get(image)
            response.raise_for_status()
            return encode_image(
                image=response.content,
                max_size=max_size,
                max_width=max_width,
                max_height=max_height,
                encode_type="base64",
            )

        if any([width, height, max_size, max_width, max_height]):
            raise ValueError(f"Cannot resize image {image} when encode_type is 'url'")

        return {
            "type": "image_url",
            "image": image,
            "image_pixels": None,
            "image_bytes": None,
        }

    decoded_image: PILImage.Image

    if isinstance(image, bytes):
        decoded_image = PILImage.open(io.BytesIO(image))
    elif isinstance(image, str) and image.startswith("data:image/"):
        _, encoded = image.split(",", 1)
        image_data = base64.b64decode(encoded)
        decoded_image = PILImage.open(io.BytesIO(image_data))
    elif isinstance(image, str | Path):
        decoded_image = PILImage.open(image)
    elif isinstance(image, np.ndarray):
        decoded_image = PILImage.fromarray(image)
    elif isinstance(image, PILImage.Image):
        decoded_image = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)} for {image}")

    output_image = resize_image(
        decoded_image,
        width=width,
        height=height,
        max_width=max_width,
        max_height=max_height,
        max_size=max_size,
    )
    if output_image.mode != "RGB":
        output_image = output_image.convert("RGB")

    buffer = io.BytesIO()
    output_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "type": "image_base64",
        "image": f"data:image/jpeg;base64,{image_base64}",
        "image_pixels": output_image.width * output_image.height,
        "image_bytes": len(image_bytes),
    }


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


def encode_video(
    video: bytes | str | Path,
    encode_type: Literal["base64", "url"] | None = "base64",
) -> dict[
    Literal["type", "video", "video_frames", "video_seconds", "video_bytes"],
    str | int | float | None,
]:
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
    if isinstance(video, str) and is_url(video):
        if encode_type == "base64":
            response = httpx.get(video)
            response.raise_for_status()
            return encode_video(video=response.content, encode_type="base64")

        return {
            "type": "video_url",
            "video": video,
            "video_frames": None,
            "video_seconds": None,
            "video_bytes": None,
        }

    if isinstance(video, str) and video.startswith("data:video/"):
        data_str = video.split(",", 1)[1]

        return {
            "type": "video_base64",
            "video": video,
            "video_frames": None,
            "video_seconds": None,
            "video_bytes": len(data_str) * 3 // 4,  # base64 to bytes
        }

    if isinstance(video, str | Path):
        path = Path(video)
        video_bytes = path.read_bytes()
        video_format = get_file_format(path)
    elif isinstance(video, bytes):
        video_bytes = video
        video_format = "unknown"
    else:
        raise ValueError(f"Unsupported video type: {type(video)} for {video}")

    video_base64 = base64.b64encode(video).decode("utf-8")

    return {
        "type": "video_base64",
        "video": f"data:video/{video_format};base64,{video_base64}",
        "video_frames": None,
        "video_seconds": None,
        "video_bytes": len(video_bytes),
    }


def encode_audio(
    audio: Any,
    b64encode: bool,
    sample_rate: int = 16000,
    file_name: str = "audio.wav",
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str = "mp3",
    bitrate: str = "64k",
) -> dict[
    Literal[
        "type",
        "audio",
        "format",
        "mimetype",
        "audio_samples",
        "audio_seconds",
        "audio_bytes",
    ],
    str | int | float | None,
]:
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

    audio_numpy: np.ndarray

    if hasattr(audio, "get_samples_played_in_range"):
        # HF datasets Audio object
        audio_samples = audio.get_samples_played_in_range(
            start_seconds=0.0,
            stop_seconds=(
                None
                if max_duration is None
                else min(max_duration, audio.metadata.duration_seconds_from_header)
            ),
        )
        audio_numpy = np.array(audio_samples.data)
    elif isinstance(audio, Tensor):
        audio_numpy = audio.numpy()
    elif isinstance(audio, str | Path):
        if is_url(audio):
            response = httpx.get(audio)
            response.raise_for_status()
            audio_stream = response.content
            file_name = get_file_name(audio)
        else:
            if not Path(audio).exists():
                raise ValueError(f"Audio file does not exist: {audio}")
            file_name = get_file_name(audio)
            audio_stream = Path(audio).read_bytes()

        audio_numpy, sample_rate = soundfile.read(
            io.BytesIO(audio_stream), dtype="float32"
        )
    elif isinstance(audio, bytes):
        audio_numpy, sample_rate = soundfile.read(io.BytesIO(audio), dtype="float32")
    elif isinstance(audio, np.ndarray):
        audio_numpy = audio
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    if sample_rate != encode_sample_rate:
        audio_numpy = librosa.resample(
            audio_numpy.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=encode_sample_rate,
        )
        sample_rate = encode_sample_rate

    audio_numpy = librosa.to_mono(audio_numpy)

    if (
        max_duration is not None
        and max_duration > 0
        and (max_samples := int(max_duration * sample_rate)) < len(audio_numpy)
    ):
        audio_numpy = audio_numpy[max_samples:]

    audio_buffer = io.BytesIO()

    if audio_format.lower() == "mp3":
        wav = io.BytesIO()
        soundfile.write(wav, audio_numpy, sample_rate, format="WAV", subtype="PCM_16")
        wav.seek(0)

        sound = AudioSegment.from_wav(wav)
        sound.export(audio_buffer, format="mp3", bitrate=bitrate)
    else:
        soundfile.write(audio_buffer, audio, sample_rate, format=audio_format.upper())

    audio_buffer.seek(0)
    decoded_audio = audio_buffer.read()

    return {
        "type": "audio_base64" if b64encode else "audio_file",
        "audio": (
            base64.b64encode(decoded_audio).decode("utf-8")
            if b64encode
            else decoded_audio
        ),
        "file_name": file_name,
        "format": audio_format,
        "mimetype": f"audio/{audio_format}",
        "audio_samples": len(audio_numpy),
        "audio_seconds": len(audio_numpy) / sample_rate,
        "audio_bytes": len(decoded_audio),
    }


def get_file_name(path: Path | str) -> str:
    """Get file name from path."""
    return Path(path).name


def get_file_format(path: Path | str) -> str:
    """Get file format from path extension."""
    suffix = Path(path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else "unknown"
