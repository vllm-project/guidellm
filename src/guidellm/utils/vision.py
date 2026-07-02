from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

# CRITICAL: Use 'import ... as libs' pattern to preserve lazy loading
# This defers errors until attributes are actually accessed
import guidellm.extras.vision as libs

__all__ = [
    "encode_image",
    "encode_video",
    "get_file_format",
    "image_dict_to_pil",
    "is_url",
    "resize_image",
    "synthesize_image",
    "synthesize_video",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


def encode_image(
    image: bytes | str | Path | np.ndarray | libs.Image,
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

    decoded_image: libs.Image

    if isinstance(image, bytes):
        decoded_image = libs.PILImage.open(io.BytesIO(image))
    elif isinstance(image, str) and image.startswith("data:image/"):
        _, encoded = image.split(",", 1)
        image_data = base64.b64decode(encoded)
        decoded_image = libs.PILImage.open(io.BytesIO(image_data))
    elif isinstance(image, str | Path):
        decoded_image = libs.PILImage.open(image)
    elif isinstance(image, np.ndarray):
        decoded_image = libs.PILImage.fromarray(image)
    elif isinstance(image, libs.Image):
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
    image: libs.Image,
    width: int | None = None,
    height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_size: int | None = None,
) -> libs.Image:
    if not isinstance(image, libs.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    if width is not None and height is not None:
        return image.resize((width, height), libs.PILImage.Resampling.BILINEAR)

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
        image = image.resize((target_w, target_h), libs.PILImage.Resampling.BILINEAR)

    return image


def image_dict_to_pil(item: dict[str, Any]) -> libs.Image:
    """
    Decode an encoded image column item to a PIL Image for vLLM multi_modal_data.

    The item must have an "image" key with either a data URL (data:image/...;base64,...)
    or an HTTP(S) URL. For data URLs the image is base64-decoded; for URLs the
    image is fetched with httpx.

    :param item: Dict with "image" key (data URL or URL string)
    :return: PIL Image in RGB if needed
    :raises ValueError: If item has no "image" or unsupported format
    """
    image_spec = item.get("image")
    if not image_spec or not isinstance(image_spec, str):
        raise ValueError(
            "Encoded image item must have an 'image' key with a data URL or URL string."
        )
    if image_spec.startswith("data:image/"):
        _, encoded = image_spec.split(",", 1)
        data = base64.b64decode(encoded)
        decoded_image = libs.PILImage.open(io.BytesIO(data))
    elif image_spec.startswith(("http://", "https://")):
        response = httpx.get(image_spec)
        response.raise_for_status()
        decoded_image = libs.PILImage.open(io.BytesIO(response.content))
    else:
        raise ValueError(
            "Encoded image 'image' value must be a data:image/... URL or "
            f"http(s) URL, got: {image_spec[:80]!r}..."
        )
    if decoded_image.mode != "RGB":
        decoded_image = decoded_image.convert("RGB")  # type: ignore[assignment]
        # convert() returns Image; PILImage.open() may be ImageFile
    return decoded_image


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

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "type": "video_base64",
        "video": f"data:video/{video_format};base64,{video_base64}",
        "video_frames": None,
        "video_seconds": None,
        "video_bytes": len(video_bytes),
    }


def get_file_format(path: Path | str) -> str:
    """Get file format from path extension."""
    suffix = Path(path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else "unknown"


# ---------------------------------------------------------------------------
# Synthetic media generation
# ---------------------------------------------------------------------------

_SYNTHETIC_IMAGE_CONTENT = ("gradient", "noise", "solid", "checkerboard")
_SYNTHETIC_VIDEO_CONTENT = ("gradient", "noise")
_SYNTHETIC_IMAGE_FORMATS = ("jpeg", "png")
_SYNTHETIC_VIDEO_FORMATS = ("mp4",)


def _row_rng(seed: int, row_index: int) -> np.random.Generator:
    """
    Deterministic per-row numpy Generator.

    Uses PCG64 seeded by SeedSequence([seed, row_index]) so two runs with the
    same (seed, row_index) produce byte-identical RNG streams across processes,
    machines, and OS-level RNG state.
    """
    seed_seq = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(row_index)])
    return np.random.Generator(np.random.PCG64(seed_seq))


def _gradient_frame(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a smooth gradient frame with randomized base color and direction.

    Compresses well in JPEG / h264 (similar wire size to real media) but every
    frame is byte-different from the next when ``rng`` is reseeded per row,
    which defeats vLLM's mm-processor cache.
    """
    color_a = rng.integers(0, 256, size=3, dtype=np.int32)
    color_b = rng.integers(0, 256, size=3, dtype=np.int32)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))

    dx, dy = (
        np.asarray(
            libs.PILImage.fromarray(flow, mode="F").resize(
                (width, height), libs.PILImage.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        for flow in rng.uniform(-1.0, 1.0, size=(2, 16, 16)).astype(np.float32)
    )
    sample_xs, sample_ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    sample_xs = np.clip(sample_xs + dx * 80.0, 0.0, width - 1)
    sample_ys = np.clip(sample_ys + dy * 80.0, 0.0, height - 1)
    xs = (sample_xs / max(width - 1, 1) * 2.0 - 1.0).astype(np.float32)
    ys = (sample_ys / max(height - 1, 1) * 2.0 - 1.0).astype(np.float32)
    proj = xs * np.cos(angle) + ys * np.sin(angle)
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    proj = proj[..., None]

    diff = (color_b - color_a).astype(np.float32).reshape(1, 1, 3)
    base = color_a.astype(np.float32).reshape(1, 1, 3)
    frame = base + proj * diff
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _noise_frame(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _solid_frame(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    color = rng.integers(0, 256, size=3, dtype=np.uint8)
    return np.broadcast_to(color, (height, width, 3)).copy()


def _checkerboard_frame(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    color_a = rng.integers(0, 256, size=3, dtype=np.uint8)
    color_b = rng.integers(0, 256, size=3, dtype=np.uint8)
    tile = int(rng.integers(8, 33))
    ys = (np.arange(height) // tile) % 2
    xs = (np.arange(width) // tile) % 2
    mask = (ys.reshape(-1, 1) ^ xs.reshape(1, -1)).astype(bool)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[mask] = color_b
    frame[~mask] = color_a
    return frame


def _generate_image_array(
    height: int,
    width: int,
    content: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if content == "gradient":
        return _gradient_frame(height, width, rng)
    if content == "noise":
        return _noise_frame(height, width, rng)
    if content == "solid":
        return _solid_frame(height, width, rng)
    if content == "checkerboard":
        return _checkerboard_frame(height, width, rng)
    raise ValueError(
        f"Unsupported synthetic image content '{content}', "
        f"expected one of {_SYNTHETIC_IMAGE_CONTENT}"
    )


def synthesize_image(
    width: int,
    height: int,
    *,
    content: str = "gradient",
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
    seed: int = 0,
    row_index: int = 0,
) -> dict[str, Any]:
    """
    Synthesize a single image and return the canonical encoded dict.

    The output shape matches :func:`encode_image` so it flows through the rest
    of the pipeline (column mapper -> finalizer) unchanged.

    :param width: image width in pixels.
    :param height: image height in pixels.
    :param content: ``gradient`` (default, per-row randomized), ``noise``,
        ``solid``, or ``checkerboard``.
    :param image_format: ``jpeg`` (default) or ``png``.
    :param jpeg_quality: JPEG quality 1..100 (ignored for png).
    :param seed: base seed for reproducibility.
    :param row_index: row index used to vary the RNG stream per row so
        successive rows are byte-different even with the same seed.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    fmt = image_format.lower()
    if fmt not in _SYNTHETIC_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported synthetic image format '{image_format}', "
            f"expected one of {_SYNTHETIC_IMAGE_FORMATS}"
        )
    if content not in _SYNTHETIC_IMAGE_CONTENT:
        raise ValueError(
            f"Unsupported synthetic image content '{content}', "
            f"expected one of {_SYNTHETIC_IMAGE_CONTENT}"
        )

    rng = _row_rng(seed, row_index)
    arr = _generate_image_array(height, width, content, rng)
    img = libs.PILImage.fromarray(arr, mode="RGB")

    buffer = io.BytesIO()
    if fmt == "jpeg":
        img.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=False)
        mime = "image/jpeg"
    else:
        img.save(buffer, format="PNG", optimize=False, compress_level=6)
        mime = "image/png"

    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "type": "image_base64",
        "image": f"data:{mime};base64,{image_b64}",
        "image_pixels": width * height,
        "image_bytes": len(image_bytes),
    }


def synthesize_video(
    width: int,
    height: int,
    frames: int,
    *,
    fps: float = 1.0,
    content: str = "gradient",
    video_format: str = "mp4",
    video_bitrate: str | None = None,
    seed: int = 0,
    row_index: int = 0,
) -> dict[str, Any]:
    """
    Synthesize a short video clip and return the canonical encoded dict.

    Matches the shape of :func:`encode_video`. Only ``mp4`` (h264, yuv420p) is
    supported in v1. Encoding uses ``-fflags +bitexact`` so two runs with the
    same seed produce byte-identical mp4 payloads.

    :param width: frame width in pixels (must be > 0).
    :param height: frame height in pixels (must be > 0).
    :param frames: number of frames in the clip (must be >= 1).
    :param fps: frames per second (encoded into the container, drives
        ``video_seconds = frames / fps``).
    :param content: ``gradient`` (default, per-frame randomized) or ``noise``.
    :param video_format: only ``mp4`` is supported in v1.
    :param video_bitrate: optional libx264 bitrate, e.g. ``"500k"``. ``None``
        leaves the codec at its default CRF-based rate control.
    :param seed: base seed for reproducibility.
    :param row_index: row index used to vary the RNG stream per row.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    fmt = video_format.lower()
    if fmt not in _SYNTHETIC_VIDEO_FORMATS:
        raise ValueError(
            f"Unsupported synthetic video format '{video_format}', "
            f"expected one of {_SYNTHETIC_VIDEO_FORMATS}"
        )
    if content not in _SYNTHETIC_VIDEO_CONTENT:
        raise ValueError(
            f"Unsupported synthetic video content '{content}', "
            f"expected one of {_SYNTHETIC_VIDEO_CONTENT}"
        )

    rng = _row_rng(seed, row_index)
    clip = np.empty((frames, height, width, 3), dtype=np.uint8)
    for i in range(frames):
        frame_seed = int(rng.integers(0, 2**31 - 1))
        frame_rng = np.random.Generator(np.random.PCG64(frame_seed))
        if content == "gradient":
            clip[i] = _gradient_frame(height, width, frame_rng)
        else:
            clip[i] = _noise_frame(height, width, frame_rng)

    write_kwargs: dict[str, Any] = {
        "extension": ".mp4",
        "fps": float(fps),
        "codec": "libx264",
        "macro_block_size": 1,
        "ffmpeg_params": [
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
        ],
    }
    if video_bitrate is not None:
        write_kwargs["bitrate"] = str(video_bitrate)

    video_bytes = libs.iio.imwrite("<bytes>", clip, **write_kwargs)  # type: ignore[attr-defined]
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "type": "video_base64",
        "video": f"data:video/mp4;base64,{video_b64}",
        "video_frames": int(frames),
        "video_seconds": float(frames) / float(fps),
        "video_bytes": len(video_bytes),
    }
