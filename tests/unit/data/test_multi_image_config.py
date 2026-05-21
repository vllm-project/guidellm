"""Tests for MultiImageDatasetConfig."""

import pytest
from pydantic import ValidationError

from guidellm.data.deserializers.multi_image import MultiImageDataArgs as MultiImageDatasetConfig


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

    @pytest.mark.parametrize("size", ["480p", "720p", "1080p", "1440p", "4k"])
    def test_valid_image_sizes(self, size):
        """Test all supported image resolution keys are accepted."""
        config = MultiImageDatasetConfig(
            prompt_tokens=256,
            output_tokens=128,
            image_size=size,
        )
        assert config.image_size == size

    def test_invalid_image_size(self):
        """Test that an unknown image size key is rejected."""
        with pytest.raises(ValidationError):
            MultiImageDatasetConfig(
                prompt_tokens=256,
                output_tokens=128,
                image_size="8k",
            )

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
