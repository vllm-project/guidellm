"""Unit tests for EmbeddingsRequestFinalizer."""

from __future__ import annotations

import pytest

from guidellm.data.finalizers import EmbeddingsRequestFinalizer
from guidellm.schemas import GenerationRequest, UsageMetrics


class TestEmbeddingsRequestFinalizer:
    """Tests for EmbeddingsRequestFinalizer."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that finalizer is properly registered."""
        from guidellm.data.finalizers import FinalizerRegistry

        assert "embeddings" in FinalizerRegistry.registry
        assert FinalizerRegistry.registry["embeddings"] == EmbeddingsRequestFinalizer

    @pytest.mark.sanity
    def test_initialization(self):
        """Test finalizer initialization."""
        finalizer = EmbeddingsRequestFinalizer()
        assert finalizer is not None

    @pytest.mark.sanity
    def test_single_text_input(self):
        """Test finalizer with single text input."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {"text_column": ["Hello world"]}

        request = finalizer([columns])

        assert isinstance(request, GenerationRequest)
        assert request.columns == columns
        assert request.input_metrics.text_words == 2
        assert request.input_metrics.text_characters == 11
        # Note: text_tokens is not automatically populated, only words and chars
        assert request.output_metrics.text_tokens is None

    @pytest.mark.sanity
    def test_multiple_text_inputs(self):
        """Test finalizer with multiple text inputs."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {
            "text_column": [
                "First sentence here",
                "Second sentence here",
            ]
        }

        request = finalizer([columns])

        assert isinstance(request, GenerationRequest)
        # Should aggregate metrics from both texts
        assert request.input_metrics.text_words == 6  # 3 + 3
        assert request.input_metrics.text_characters == 39

    @pytest.mark.regression
    def test_empty_text_column_raises(self):
        """Test that empty text_column raises ValueError."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {"text_column": []}

        with pytest.raises(ValueError, match="No text found in dataset row"):
            finalizer([columns])

    @pytest.mark.regression
    def test_missing_text_column_raises(self):
        """Test that missing text_column raises ValueError."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {"other_column": ["Some text"]}

        with pytest.raises(ValueError, match="No text found in dataset row"):
            finalizer([columns])

    @pytest.mark.regression
    def test_text_column_with_none_values(self):
        """Test handling of None values in text_column."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {
            "text_column": [
                None,
                "Valid text here",
                None,
            ]
        }

        request = finalizer([columns])

        # Should skip None values and only process valid text
        assert request.input_metrics.text_words == 3
        assert request.input_metrics.text_characters > 0

    @pytest.mark.regression
    def test_text_column_with_empty_strings(self):
        """Test handling of empty strings in text_column."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {
            "text_column": [
                "",
                "Valid text",
                "",
            ]
        }

        request = finalizer([columns])

        # Should skip empty strings
        assert request.input_metrics.text_words == 2
        assert request.input_metrics.text_characters > 0

    @pytest.mark.regression
    def test_all_none_or_empty_raises(self):
        """Test that all None/empty values raises ValueError."""
        finalizer = EmbeddingsRequestFinalizer()

        # All None
        columns_none = {"text_column": [None, None]}
        with pytest.raises(ValueError, match="No text found in dataset row"):
            finalizer([columns_none])

        # All empty strings
        columns_empty = {"text_column": ["", ""]}
        with pytest.raises(ValueError, match="No text found in dataset row"):
            finalizer([columns_empty])

    @pytest.mark.sanity
    def test_preserves_original_columns(self):
        """Test that finalizer preserves original column data."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {
            "text_column": ["Test text"],
        }

        request = finalizer([columns])

        assert request.columns == columns
        assert "text_column" in request.columns

    @pytest.mark.sanity
    def test_output_metrics_always_empty(self):
        """Test that output metrics are always empty for embeddings."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {"text_column": ["Short", "Long text with many words"]}

        request = finalizer([columns])

        # Embeddings have no output
        assert request.output_metrics.text_tokens is None
        assert request.output_metrics.text_words is None
        assert request.output_metrics.text_characters is None

    @pytest.mark.sanity
    def test_word_and_character_counting(self):
        """Test that word and character counting works correctly."""
        finalizer = EmbeddingsRequestFinalizer()

        # Known text with predictable word count
        text = "The quick brown fox jumps over the lazy dog"
        columns = {"text_column": [text]}

        request = finalizer([columns])

        assert request.input_metrics.text_words == 9
        assert request.input_metrics.text_characters == len(text)

    @pytest.mark.regression
    def test_long_text_input(self):
        """Test finalizer with very long text input."""
        finalizer = EmbeddingsRequestFinalizer()

        # Create a long text
        long_text = " ".join(["word"] * 1000)
        columns = {"text_column": [long_text]}

        request = finalizer([columns])

        assert request.input_metrics.text_words == 1000
        assert request.input_metrics.text_characters > 0

    @pytest.mark.regression
    def test_special_characters_handling(self):
        """Test handling of special characters in text."""
        finalizer = EmbeddingsRequestFinalizer()

        text_with_special = "Hello! @user #hashtag $price €100 50%"
        columns = {"text_column": [text_with_special]}

        request = finalizer([columns])

        # Should handle special characters without error
        assert isinstance(request, GenerationRequest)
        assert request.input_metrics.text_characters > 0

    @pytest.mark.regression
    def test_unicode_text_handling(self):
        """Test handling of Unicode text."""
        finalizer = EmbeddingsRequestFinalizer()

        unicode_text = "Hello 世界 مرحبا мир 🌍"
        columns = {"text_column": [unicode_text]}

        request = finalizer([columns])

        # Should handle Unicode without error
        assert isinstance(request, GenerationRequest)
        assert request.input_metrics.text_characters > 0

    @pytest.mark.sanity
    def test_whitespace_text(self):
        """Test handling of whitespace-only text."""
        finalizer = EmbeddingsRequestFinalizer()

        # Whitespace only should be treated as empty
        columns = {"text_column": ["   ", "\t\n", "Valid text"]}

        request = finalizer([columns])

        # Should only count the valid text
        assert request.input_metrics.text_words == 2

    @pytest.mark.regression
    def test_batch_like_input(self):
        """Test finalizer with batch-like input (multiple texts)."""
        finalizer = EmbeddingsRequestFinalizer()

        # Simulate batch processing
        columns = {
            "text_column": [
                "First embedding text",
                "Second embedding text",
                "Third embedding text",
            ]
        }

        request = finalizer([columns])

        # Should aggregate all texts
        assert request.input_metrics.text_words == 9  # 3 * 3
        assert request.input_metrics.text_characters > 0

    @pytest.mark.sanity
    def test_usage_metrics_type(self):
        """Test that metrics are correct type."""
        finalizer = EmbeddingsRequestFinalizer()
        columns = {"text_column": ["Test"]}

        request = finalizer([columns])

        assert isinstance(request.input_metrics, UsageMetrics)
        assert isinstance(request.output_metrics, UsageMetrics)

    @pytest.mark.regression
    def test_multiple_calls_independent(self):
        """Test that multiple calls are independent."""
        finalizer = EmbeddingsRequestFinalizer()

        request1 = finalizer([{"text_column": ["First"]}])
        request2 = finalizer([{"text_column": ["Second call"]}])

        # Each call should produce independent results
        assert request1.input_metrics.text_words == 1
        assert request2.input_metrics.text_words == 2
        assert request1.columns != request2.columns

    @pytest.mark.regression
    def test_newlines_and_formatting(self):
        """Test handling of newlines and formatting in text."""
        finalizer = EmbeddingsRequestFinalizer()

        text_with_newlines = """This is a
        multi-line
        text with
        various formatting"""

        columns = {"text_column": [text_with_newlines]}

        request = finalizer([columns])

        # Should handle newlines correctly
        assert isinstance(request, GenerationRequest)
        assert request.input_metrics.text_words == 8
