"""Tests for multi-image generator."""

import base64
import io

import pytest
from PIL import Image as PILImage

from guidellm.data.generators.multi_image import (
    generate_synthetic_images,
    ImageSize,
)


class TestGenerateSyntheticImages:
    """Test image generation."""

    def test_image_size_720p(self):
        """Test 720p images have correct dimensions."""
        width, height = ImageSize.SIZES["720p"]
        assert width == 1280
        assert height == 720

    def test_generates_correct_count(self):
        """Test correct number of images generated."""
        images, _, _ = generate_synthetic_images(3, image_size="720p", seed=42)
        assert len(images) == 3

    def test_single_image(self):
        """Test generating a single image."""
        images, total_pixels, total_bytes = generate_synthetic_images(
            1, image_size="720p", seed=42
        )
        assert len(images) == 1
        assert images[0]["image"].startswith("data:image/jpeg;base64,")
        assert total_pixels == 1280 * 720
        assert total_bytes > 0

    def test_image_base64_encoding(self):
        """Test image is valid base64-encoded JPEG."""
        images, _, _ = generate_synthetic_images(1, image_size="720p", seed=42)
        image_data = images[0]["image"]

        # Extract base64 content
        _, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Verify it's a valid JPEG
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        assert pil_image.format == "JPEG"
        assert pil_image.size == (1280, 720)
        assert pil_image.mode == "RGB"

    def test_total_pixels_calculation(self):
        """Test total pixels is sum of all image pixels."""
        images, total_pixels, _ = generate_synthetic_images(5, image_size="720p")
        expected_pixels = len(images) * 1280 * 720
        assert total_pixels == expected_pixels

    def test_total_bytes_calculation(self):
        """Test total bytes is sum of all image bytes."""
        images, _, total_bytes = generate_synthetic_images(3, image_size="720p")
        expected_bytes = sum(img["image_bytes"] for img in images)
        assert total_bytes == expected_bytes

    def test_reproducible_with_seed(self):
        """Test same seed produces same images."""
        images1, _, _ = generate_synthetic_images(2, image_size="720p", seed=42)
        images2, _, _ = generate_synthetic_images(2, image_size="720p", seed=42)

        assert len(images1) == len(images2)
        # Same seed = same base64 strings
        assert images1[0]["image"] == images2[0]["image"]
        assert images1[1]["image"] == images2[1]["image"]

    def test_different_seed_produces_different_images(self):
        """Test different seeds produce different images."""
        images1, _, _ = generate_synthetic_images(1, image_size="720p", seed=42)
        images2, _, _ = generate_synthetic_images(1, image_size="720p", seed=99)

        assert images1[0]["image"] != images2[0]["image"]
