"""
Unit tests for guidellm.data.finalizers module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.finalizers import (
    FinalizerRegistry,
    GenerativeRequestFinalizer,
)
from guidellm.schemas import GenerationRequest


class TestGenerativeRequestFinalizerTokenAggregation:
    """Test cases for GenerativeRequestFinalizer token aggregation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer()

    @pytest.mark.smoke
    def test_finalize_single_turn_prompt_tokens(self, valid_instances):
        """Test finalize with single prompt token count.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [100]}

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.input_metrics.text_tokens == 100

    @pytest.mark.smoke
    def test_finalize_multi_turn_prompt_tokens(self, valid_instances):
        """Test finalize with multiple prompt token counts (sums them).

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [50, 75, 100]}

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.input_metrics.text_tokens == 225  # 50 + 75 + 100

    @pytest.mark.smoke
    def test_finalize_multi_turn_output_tokens(self, valid_instances):
        """Test finalize with multiple output token counts (sums them).

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"output_tokens_count_column": [20, 30, 40]}

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.output_metrics.text_tokens == 90  # 20 + 30 + 40

    @pytest.mark.sanity
    def test_finalize_with_none_values_in_list(self, valid_instances):
        """Test finalize skips None values when summing.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [50, None, 100]}

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.input_metrics.text_tokens == 150  # 50 + 100, skips None

    @pytest.mark.regression
    def test_finalize_with_empty_column_lists(self, valid_instances):
        """Test finalize with empty column lists.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "prompt_tokens_count_column": [],
            "output_tokens_count_column": [],
        }

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.input_metrics.text_tokens is None
        assert result.output_metrics.text_tokens is None


class TestGenerativeRequestFinalizerMultimodal:
    """Test cases for GenerativeRequestFinalizer multimodal aggregation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer()

    @pytest.mark.sanity
    def test_finalize_multi_value_text_columns(self, valid_instances):
        """Test finalize accumulates text metrics for multiple text values.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": [
                "Hello world",
                "How are you?",
                "I am fine",
            ],
        }

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        # Should accumulate metrics from all text values
        assert result.input_metrics.text_words > 0
        assert result.input_metrics.text_characters > 0

    @pytest.mark.sanity
    def test_finalize_multi_value_image_columns(self, valid_instances):
        """Test finalize sums image pixels and bytes across multiple images.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "image_column": [
                {"image_pixels": 1000, "image_bytes": 5000},
                {"image_pixels": 2000, "image_bytes": 10000},
                {"image_pixels": 1500, "image_bytes": 7500},
            ],
        }

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        assert result.input_metrics.image_pixels == 4500  # 1000 + 2000 + 1500
        assert result.input_metrics.image_bytes == 22500  # 5000 + 10000 + 7500

    @pytest.mark.regression
    def test_finalize_preserves_columns(self, valid_instances):
        """Test finalize preserves input columns in result.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": ["Hello"],
            "prompt_tokens_count_column": [50],
            "output_tokens_count_column": [25],
        }

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        # Original columns should be preserved
        assert result.columns == columns
        # And metrics should be set
        assert result.input_metrics.text_tokens == 50
        assert result.output_metrics.text_tokens == 25


class TestFinalizerTopLevel:
    """Test cases for top-level finalizer interface.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer()

    @pytest.mark.smoke
    def test_finalizer_returns_list(self, valid_instances):
        """Test __call__ returns list of GenerationRequest objects.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        items = [
            {"prompt_tokens_count_column": [50]},
            {"prompt_tokens_count_column": [75]},
            {"prompt_tokens_count_column": [100]},
        ]

        result = instance(items)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, GenerationRequest) for r in result)

    @pytest.mark.sanity
    def test_finalizer_handles_empty_list(self, valid_instances):
        """Test __call__ handles empty list.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        items = []

        result = instance(items)

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.sanity
    def test_finalizer_aggregates_multimodal_metrics(self, valid_instances):
        """Test finalize aggregates all multimodal metrics correctly.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": ["Hello world"],
            "image_column": [{"image_pixels": 1920 * 1080, "image_bytes": 50000}],
            "video_column": [
                {"video_frames": 120, "video_seconds": 4.0, "video_bytes": 1000000}
            ],
            "audio_column": [
                {"audio_samples": 48000, "audio_seconds": 1.0, "audio_bytes": 96000}
            ],
        }

        result = instance.finalize_turn(columns)

        assert isinstance(result, GenerationRequest)
        # Text metrics
        assert result.input_metrics.text_words == 2
        assert result.input_metrics.text_characters == 11

        # Image metrics
        assert result.input_metrics.image_pixels == 1920 * 1080
        assert result.input_metrics.image_bytes == 50000

        # Video metrics
        assert result.input_metrics.video_frames == 120
        assert result.input_metrics.video_seconds == 4.0
        assert result.input_metrics.video_bytes == 1000000

        # Audio metrics
        assert result.input_metrics.audio_samples == 48000
        assert result.input_metrics.audio_seconds == 1.0
        assert result.input_metrics.audio_bytes == 96000


class TestFinalizerRegistry:
    """Test cases for FinalizerRegistry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_registry_has_generative(self):
        """Test registry has 'generative' finalizer registered.

        ### WRITTEN BY AI ###
        """
        finalizer_cls = FinalizerRegistry.get_registered_object("generative")

        assert finalizer_cls is not None
        assert finalizer_cls == GenerativeRequestFinalizer

    @pytest.mark.sanity
    def test_protocol_conformance(self):
        """Test GenerativeRequestFinalizer conforms to DatasetFinalizer protocol.

        ### WRITTEN BY AI ###
        """
        instance = GenerativeRequestFinalizer()

        # Should have __call__ method
        assert callable(instance)

        # Test it works as expected
        result = instance([{"text_column": ["test"]}])
        assert isinstance(result, list)
