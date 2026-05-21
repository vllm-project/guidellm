"""Tests for MultiImageDatasetConfig."""

import pytest
from pydantic import ValidationError

from guidellm.data.schemas import MultiImageDatasetConfig


class TestMultiImageDatasetConfig:
    """Test MultiImageDatasetConfig validation and defaults."""

    def test_defaults(self):
        """Test default values."""
        config = MultiImageDatasetConfig(
            prompt_tokens=256,
            output_tokens=128,
        )
        assert config.images_per_request == 1
        assert config.image_size == "720p"

    def test_explicit_image_count(self):
        """Test setting explicit image count."""
        config = MultiImageDatasetConfig(
            prompt_tokens=256,
            output_tokens=128,
            images_per_request=5,
        )
        assert config.images_per_request == 5

    def test_invalid_image_count_zero(self):
        """Test that 0 images is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MultiImageDatasetConfig(
                prompt_tokens=256,
                output_tokens=128,
                images_per_request=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_invalid_image_count_exceeds_max(self):
        """Test that > 10 images is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MultiImageDatasetConfig(
                prompt_tokens=256,
                output_tokens=128,
                images_per_request=11,
            )
        assert "less than or equal to 10" in str(exc_info.value)

    def test_image_size_fixed_720p(self):
        """Test image_size is 720p."""
        config = MultiImageDatasetConfig(
            prompt_tokens=256,
            output_tokens=128,
            image_size="720p",
        )
        assert config.image_size == "720p"

    def test_inherits_synthetic_text_fields(self):
        """Test that it inherits SyntheticTextDatasetConfig fields."""
        config = MultiImageDatasetConfig(
            prompt_tokens=256,
            output_tokens=128,
            prompt_tokens_stdev=50,
            output_tokens_max=256,
        )
        assert config.prompt_tokens == 256
        assert config.output_tokens == 128
        assert config.prompt_tokens_stdev == 50
        assert config.output_tokens_max == 256
