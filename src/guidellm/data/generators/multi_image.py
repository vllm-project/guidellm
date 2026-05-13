"""Multi-image synthetic data generation for benchmarking."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError as e:
    raise ImportError(
        "Please install guidellm[vision] to use multi-image features"
    ) from e

__all__ = ["generate_synthetic_images", "ImageSize"]


class ImageSize:
    """Standard image sizes."""

    SIZES = {
        "720p": (1280, 720),
    }


def generate_synthetic_images(
    num_images: int,
    image_size: str = "720p",
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Generate N synthetic JPEG images as base64-encoded strings.

    Args:
        num_images: Number of images to generate
        image_size: Image resolution ("720p" = 1280x720)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (images_list, total_pixels, total_bytes) where:
        - images_list: List of dicts with keys "image", "image_pixels", "image_bytes"
        - total_pixels: Total pixel count across all images
        - total_bytes: Total byte size across all images
    """
    if seed is not None:
        np.random.seed(seed)

    width, height = ImageSize.SIZES.get(image_size, (1280, 720))
    total_pixels = 0
    total_bytes = 0
    images = []

    for _ in range(num_images):
        # Generate random RGB image
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        pil_image = PILImage.fromarray(image_array, mode="RGB")

        # Encode as JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        images.append({
            "image": f"data:image/jpeg;base64,{image_base64}",
            "image_pixels": width * height,
            "image_bytes": len(image_bytes),
        })

        total_pixels += width * height
        total_bytes += len(image_bytes)

    return images, total_pixels, total_bytes
