from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import torch

try:
    from PIL import Image as PILImage
    from torchcodec import AudioSamples
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
except ImportError as e:
    raise ImportError(
        "Please install guidellm[multimodal] to use audio/image/video features"
    ) from e

__all__ = [
    "encode_audio",
    "encode_image",
    "encode_video",
    "get_file_format",
    "is_url",
    "resize_image",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


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
    audio: AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    b64encode: bool = False,
    sample_rate: int | None = None,
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
    """Decode audio (if necessary) and re-encode to specified format."""
    samples = _decode_audio(audio, sample_rate=sample_rate, max_duration=max_duration)

    bitrate_val = (
        int(bitrate.rstrip("k")) * 1000 if bitrate.endswith("k") else int(bitrate)
    )
    format_val = audio_format.lower()

    encoded_audio = _encode_audio(
        samples=samples,
        resample_rate=encode_sample_rate,
        bitrate=bitrate_val,
        audio_format=format_val,
        mono=mono,
    )

    return {
        "type": "audio_base64" if b64encode else "audio_file",
        "audio": (
            base64.b64encode(encoded_audio).decode("utf-8")
            if b64encode
            else encoded_audio
        ),
        "file_name": get_file_name(audio)
        if isinstance(audio, str | Path)
        else file_name,
        "format": audio_format,
        "mimetype": f"audio/{format_val}",
        "audio_samples": samples.sample_rate,
        "audio_seconds": samples.duration_seconds,
        "audio_bytes": len(encoded_audio),
    }


def _decode_audio(  # noqa: C901, PLR0912
    audio: AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    sample_rate: int | None = None,
    max_duration: float | None = None,
) -> AudioSamples:
    """Decode audio from various input types into AudioSamples."""
    # If input is a dict, unwrap it into a function call
    if isinstance(audio, dict):
        sample_rate = audio.get("sample_rate", audio.get("sampling_rate", sample_rate))
        if "data" not in audio and "url" not in audio:
            raise ValueError(
                f"Audio dict must contain either 'data' or 'url' keys, got {audio}"
            )
        return _decode_audio(
            audio=audio.get("data") or audio.get("url"),
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    # Convert numpy array to torch tensor and re-call
    if isinstance(audio, np.ndarray):
        return _decode_audio(
            audio=torch.from_numpy(audio),
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    samples: AudioSamples

    # HF datasets return AudioDecoder for audio column
    if isinstance(audio, AudioDecoder):
        samples = audio.get_samples_played_in_range(stop_seconds=max_duration)

    elif isinstance(audio, torch.Tensor):
        # If float stream assume decoded audio
        if torch.is_floating_point(audio):
            if sample_rate is None:
                raise ValueError("Sample rate must be set for decoded audio")

            full_duration = audio.shape[1] / sample_rate
            # If max_duration is set, trim the audio to that duration
            if max_duration is not None:
                num_samples = int(max_duration * sample_rate)
                duration = min(max_duration, full_duration)
                data = audio[:, :num_samples]
            else:
                duration = full_duration
                data = audio

            samples = AudioSamples(
                data=data,
                pts_seconds=0.0,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        # If bytes tensor assume encoded audio
        elif audio.dtype == torch.uint8:
            decoder = AudioDecoder(
                source=audio,
                sample_rate=sample_rate,
            )
            samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)

        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    # If bytes, assume encoded audio
    elif isinstance(audio, bytes):
        decoder = AudioDecoder(
            source=audio,
            sample_rate=sample_rate,
        )
        samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)

    # If str or Path, assume file path or URL to encoded audio
    elif isinstance(audio, str | Path):
        if isinstance(audio, str) and is_url(audio):
            response = httpx.get(audio)
            response.raise_for_status()
            data = response.content
        else:
            if not Path(audio).exists():
                raise ValueError(f"Audio file does not exist: {audio}")
            data = Path(audio).read_bytes()
        decoder = AudioDecoder(
            source=data,
        )
        samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    return samples


def _encode_audio(
    samples: AudioSamples,
    resample_rate: int | None = None,
    bitrate: int = 64000,
    audio_format: str = "mp3",
    mono: bool = True,
) -> bytes:
    encoder = AudioEncoder(
        samples=samples.data,
        sample_rate=samples.sample_rate,
    )

    audio_tensor = encoder.to_tensor(
        format=audio_format,
        bit_rate=bitrate if audio_format == "mp3" else None,
        num_channels=1 if mono else None,
        sample_rate=resample_rate,
    )

    return audio_tensor.numpy().tobytes()


def get_file_name(path: Path | str) -> str:
    """Get file name from path."""
    return Path(path).name


def get_file_format(path: Path | str) -> str:
    """Get file format from path extension."""
    suffix = Path(path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else "unknown"
