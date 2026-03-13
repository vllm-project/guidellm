from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage

    HAS_VISION = True
else:
    from guidellm.utils import ExtrasImporter

    _vision_importer = ExtrasImporter(
        {
            "PILImage": "PIL.Image",
        },
        extras_group="vision",
    )

    # Make imports available at module level for runtime use
    PILImage = _vision_importer.PILImage
    HAS_VISION = _vision_importer.is_available

__all__ = [
    "HAS_VISION",
    "encode_image",
    "encode_video",
    "get_file_format",
    "image_dict_to_pil",
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


def image_dict_to_pil(item: dict[str, Any]) -> PILImage.Image:
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
        decoded_image = PILImage.open(io.BytesIO(data))
    elif image_spec.startswith(("http://", "https://")):
        response = httpx.get(image_spec)
        response.raise_for_status()
        decoded_image = PILImage.open(io.BytesIO(response.content))
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
