"""
Unit tests for guidellm.data.builders module, specifically process_dataset function.
"""

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
import yaml
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from guidellm.data.builders import (
    PromptTooShortError,
    ShortPromptStrategy,
    ShortPromptStrategyHandler,
    push_dataset_to_hub,
)
from guidellm.data.entrypoints import (
    process_dataset,
)


@pytest.fixture
def tokenizer_mock():
    """Fixture to provide a mocked tokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

    # Simple tokenizer: each character is a token
    def encode_side_effect(text):
        if not text:
            return []
        # Count tokens as roughly one per character for simplicity
        return list(range(len(text)))

    def decode_side_effect(tokens, skip_special_tokens=False):
        if not tokens:
            return ""
        # Simple decode: return a string representation
        return "".join(chr(65 + (t % 26)) for t in tokens[:100])

    tokenizer.encode.side_effect = encode_side_effect
    tokenizer.decode.side_effect = decode_side_effect
    return tokenizer


@pytest.fixture
def sample_dataset_default_columns():
    """Sample dataset with default column names."""
    return Dataset.from_dict({
        "prompt": [
            (
                "This is a very long prompt that should be sufficient for "
                "testing purposes. "
            ) * 10,
            "Short.",
            (
                "Another very long prompt for testing the dataset processing "
                "functionality. "
            ) * 10,
        ],
    })


@pytest.fixture
def sample_dataset_custom_columns():
    """Sample dataset with custom column names requiring mapping."""
    return Dataset.from_dict({
        "question": [
            (
                "What is the meaning of life? This is a longer question that "
                "should work for testing. "
            ) * 10,
            (
                "How does this work? Let me explain in detail how this system "
                "functions. "
            ) * 10,
            (
                "Tell me about machine learning. Machine learning is a "
                "fascinating field. "
            ) * 10,
        ],
    })


@pytest.fixture
def sample_dataset_with_prefix():
    """Sample dataset with prefix column."""
    return Dataset.from_dict({
        "prompt": [
            (
                "This is a long prompt that should be sufficient for testing "
                "purposes. "
            ) * 10,
            "Another long prompt here that will work for testing. " * 10,
            "Yet another long prompt for testing purposes. " * 10,
        ],
        "system_prompt": [
            "You are a helpful assistant.",
            "You are a helpful assistant.",
            "You are a helpful assistant.",
        ],
    })


@pytest.fixture
def sample_config_json():
    """Sample config as JSON string."""
    return '{"prompt_tokens": 50, "output_tokens": 30}'


@pytest.fixture
def sample_config_key_value():
    """Sample config as key-value pairs."""
    return "prompt_tokens=50,output_tokens=30"


@pytest.fixture
def temp_output_path(tmp_path):
    """Temporary file path for output."""
    return tmp_path / "output.json"


class TestProcessDatasetShortPromptStrategies:
    """Test cases for different ShortPromptStrategy types."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_ignore_strategy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test IGNORE strategy filters out short prompts.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
            short_prompt_strategy=ShortPromptStrategy.IGNORE,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify that short prompts were filtered out
        # The second prompt "Short." is only 6 characters, which is less than 50 tokens
        # So it should be filtered out
        assert len(saved_dataset) <= 2  # At most 2 prompts should remain

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_concatenate_strategy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test that the CONCATENATE strategy merges short prompts with subsequent rows.
        ## WRITTEN BY AI ##
        """
        # Create a dataset with short prompts that can be concatenated to reach target
        # Use a lower target (15 tokens) so concatenation is achievable
        short_config = '{"prompt_tokens": 15, "output_tokens": 10}'
        short_prompts_dataset = Dataset.from_dict({
            "prompt": [
                "A",  # 1 char = 1 token
                "B",  # 1 char = 1 token
                "C",  # 1 char = 1 token
                "D",  # 1 char = 1 token
                "E",  # 1 char = 1 token
                "F",  # 1 char = 1 token
                "G",  # 1 char = 1 token
                "H",  # 1 char = 1 token
                "I",  # 1 char = 1 token
                "J",  # 1 char = 1 token
                "K",  # 1 char = 1 token
                "L",  # 1 char = 1 token
                "M",  # 1 char = 1 token
                "N",  # 1 char = 1 token
                "O",  # 1 char = 1 token
                "P",  # 1 char = 1 token
                "Q",  # 1 char = 1 token
                "R",  # 1 char = 1 token
                "S",  # 1 char = 1 token
                "T",  # 1 char = 1 token
            ],
        })

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = short_prompts_dataset

        # Run process_dataset with the `concatenate` strategy
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=short_config,
            short_prompt_strategy=ShortPromptStrategy.CONCATENATE,
            concat_delimiter="\n",
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed
        assert len(saved_dataset) > 0

        # Verify concatenation occurred: check for delimiter and that prompts
        # meet minimum token count
        concatenated_found = False
        for row in saved_dataset:
            prompt_text = row["prompt"]
            # Check that delimiter is present (indicating concatenation)
            if "\n" in prompt_text:
                concatenated_found = True
                # Verify that multiple single-character prompts are present
                # The concatenated prompt should contain multiple letters
                # separated by newlines
                parts = prompt_text.split("\n")
                assert len(parts) >= 2, (
                    f"Concatenated prompt should contain multiple parts "
                    f"separated by delimiter, got: {prompt_text[:100]}..."
                )
            # Verify token counts meet minimum requirements
            actual_tokens = len(tokenizer_mock.encode(prompt_text))
            assert actual_tokens >= 15, (
                f"Concatenated prompt should have at least 15 tokens, "
                f"got {actual_tokens}"
            )
            assert row["prompt_tokens_count"] == actual_tokens
            assert row["prompt_tokens_count"] >= 15

        # Verify that at least some concatenation occurred
        # (Short single-character prompts should have been concatenated with
        # subsequent rows)
        assert concatenated_found, (
            "Expected to find concatenated prompts with delimiter"
        )

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_pad_strategy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test PAD strategy adds padding to short prompts.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with pad strategy
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
            short_prompt_strategy=ShortPromptStrategy.PAD,
            pad_char="X",
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed
        assert len(saved_dataset) > 0

        # Get original prompts for comparison
        sample_dataset_default_columns["prompt"]

        # Check that prompts have been padded (they should be longer)
        for row in saved_dataset:
            assert "prompt" in row
            assert len(row["prompt"]) > 0

            # Verify that prompts meet minimum token count requirements
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens >= 50, \
                f"Padded prompt should have at least 50 tokens, got {actual_tokens}"
            assert row["prompt_tokens_count"] == actual_tokens

            # For the "Short." prompt (index 1), verify it was padded
            # The original "Short." is only 6 characters, so if it was
            # processed, it should have been padded to meet the minimum token
            # requirement
            if "Short." in row["prompt"] or len(row["prompt"]) > 10:
                # If this is the short prompt, verify it was padded
                assert actual_tokens >= 50, (
                    f"Short prompt should have been padded to at least 50 "
                    f"tokens, got {actual_tokens}"
                )

    @pytest.mark.sanity
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_error_strategy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test that the `ERROR` strategy raises PromptTooShortError for short prompts.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with error strategy - should raise exception
        with pytest.raises(PromptTooShortError):
            process_dataset(
                data="test_data",
                output_path=temp_output_path,
                processor=tokenizer_mock,
                config=sample_config_json,
                short_prompt_strategy=ShortPromptStrategy.ERROR,
            )


class TestProcessDatasetColumnNames:
    """Test cases for different column name scenarios."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_default_columns(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test process_dataset works with default column names (no mapping required).
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset without column mapping
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed correctly
        assert len(saved_dataset) > 0
        for row in saved_dataset:
            assert "prompt" in row
            assert "prompt_tokens_count" in row
            assert "output_tokens_count" in row

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_custom_columns_with_mapping(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_custom_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test process_dataset works with custom column names via explicit mapping.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_custom_columns
        )

        # Run process_dataset with column mapping
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
            data_column_mapper={"text_column": "question"},
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed correctly
        assert len(saved_dataset) > 0
        for row in saved_dataset:
            assert "question" in row
            assert "prompt_tokens_count" in row
            assert "output_tokens_count" in row

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_with_prefix_column(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_with_prefix,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test process_dataset handles prefix column correctly.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_with_prefix
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed correctly
        assert len(saved_dataset) > 0
        for row in saved_dataset:
            assert "prompt" in row
            assert "system_prompt" in row
            assert "prompt_tokens_count" in row
            assert "output_tokens_count" in row

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_with_instruction_column(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test process_dataset works with 'instruction' column (default text_column).
        ## WRITTEN BY AI ##
        """
        # Create dataset with 'instruction' column (one of the default
        # text_column names)
        dataset = Dataset.from_dict({
            "instruction": [
                "Follow these instructions carefully. " * 20,
                "Complete the task as described. " * 20,
            ],
        })

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = dataset

        # Run process_dataset without column mapping (should auto-detect 'instruction')
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed correctly
        assert len(saved_dataset) > 0


class TestProcessDatasetConfigFormats:
    """Test cases for different config format inputs."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_config_json(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test process_dataset accepts config as JSON string.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )
        # Run process_dataset with JSON config
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_config_key_value(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_key_value,
        temp_output_path,
    ):
        """
        Test process_dataset accepts config as key-value pairs.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with key-value config
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_key_value,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_config_file_json(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        tmp_path,
    ):
        """
        Test process_dataset accepts config from JSON file.
        ## WRITTEN BY AI ##
        """
        # Create a temporary JSON config file
        config_file = tmp_path / "config.json"
        config_data = {"prompt_tokens": 50, "output_tokens": 30}
        config_file.write_text(json.dumps(config_data))

        output_path = tmp_path / "output.json"

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with JSON file config
        process_dataset(
            data="test_data",
            output_path=output_path,
            processor=tokenizer_mock,
            config=str(config_file),
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_config_file_yaml(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        tmp_path,
    ):
        """
        Test process_dataset accepts config from YAML file.
        ## WRITTEN BY AI ##
        """
        # Create a temporary YAML config file
        config_file = tmp_path / "config.yaml"
        config_data = {"prompt_tokens": 50, "output_tokens": 30}
        config_file.write_text(yaml.dump(config_data))

        output_path = tmp_path / "output.json"

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with YAML file config
        process_dataset(
            data="test_data",
            output_path=output_path,
            processor=tokenizer_mock,
            config=str(config_file),
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_config_file_config_extension(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        tmp_path,
    ):
        """
        Test process_dataset accepts config from .config file.
        ## WRITTEN BY AI ##
        """
        # Create a temporary .config file
        config_file = tmp_path / "config.config"
        config_data = {"prompt_tokens": 50, "output_tokens": 30}
        config_file.write_text(yaml.dump(config_data))

        output_path = tmp_path / "output.json"

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset with .config file
        process_dataset(
            data="test_data",
            output_path=output_path,
            processor=tokenizer_mock,
            config=str(config_file),
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called


class TestProcessDatasetIntegration:
    """Integration tests for process_dataset function."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_successful_processing(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test successful processing with valid dataset.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
        )

        # Verify all expected calls were made
        mock_check_processor.assert_called_once()
        mock_deserializer_factory_class.deserialize.assert_called_once()
        assert mock_save_to_file.called

        # Verify the saved dataset structure
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]
        assert len(saved_dataset) > 0

        # Verify each row has required fields
        for row in saved_dataset:
            assert "prompt" in row
            assert "prompt_tokens_count" in row
            assert "output_tokens_count" in row
            assert isinstance(row["prompt_tokens_count"], int)
            assert isinstance(row["output_tokens_count"], int)

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_empty_after_filtering(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_config_json,
        temp_output_path,
    ):
        """
        Test handling of empty dataset after filtering.
        ## WRITTEN BY AI ##
        """
        # Create dataset with only very short prompts that will be filtered out
        dataset = Dataset.from_dict({
            # Very short prompts (1 char each, less than 50 tokens)
            "prompt": ["A", "B", "C"],
        })

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = dataset

        # Run process_dataset with IGNORE strategy
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
            short_prompt_strategy=ShortPromptStrategy.IGNORE,
        )

        # Verify all expected calls were made (even though dataset is empty)
        mock_check_processor.assert_called_once()
        mock_deserializer_factory_class.deserialize.assert_called_once()
        # When all prompts are filtered out, save_dataset_to_file is not called
        # (the function returns early in _finalize_processed_dataset)
        # This is expected behavior - the function handles empty datasets gracefully
        assert not mock_save_to_file.called

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_with_prefix_tokens(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_with_prefix,
        temp_output_path,
    ):
        """
        Test process_dataset handles trimming prefix tokens correctly.
        ## WRITTEN BY AI ##
        """
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_with_prefix
        )
        config = '{"prompt_tokens": 50, "output_tokens": 30, "prefix_tokens_max": 10}'

        # Run process_dataset with prefix_tokens
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed
        assert len(saved_dataset) > 0

        # Verify that prefix lengths are correct (trimmed to 10 tokens)
        for row in saved_dataset:
            assert "system_prompt" in row
            prefix_text = row["system_prompt"]

            # Verify prefix is trimmed to exactly 10 tokens
            prefix_tokens = len(tokenizer_mock.encode(prefix_text))
            assert prefix_tokens == 10, (
                f"Prefix should be trimmed to 10 tokens, got {prefix_tokens} "
                f"for prefix: {prefix_text[:50]}..."
            )

            # Verify prompt and output token counts are present
            assert "prompt_tokens_count" in row
            assert "output_tokens_count" in row

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_include_prefix_in_token_count(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_with_prefix,
        sample_config_json,
        temp_output_path,
    ):
        """Test process_dataset with include_prefix_in_token_count flag."""
        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_with_prefix
        )

        # Run process_dataset with include_prefix_in_token_count
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=sample_config_json,
            include_prefix_in_token_count=True,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed
        assert len(saved_dataset) > 0

        # Verify that the token count accounts for the prefix
        # When include_prefix_in_token_count=True, the prefix tokens are subtracted from
        # the target prompt length, so prompt_tokens_count is just the prompt part,
        # but the total effective tokens (prefix + prompt) should equal the target
        for row in saved_dataset:
            assert "system_prompt" in row
            assert "prompt" in row

            prefix_text = row["system_prompt"]
            prompt_text = row["prompt"]

            # Calculate token counts
            prefix_tokens = len(tokenizer_mock.encode(prefix_text))
            prompt_tokens = len(tokenizer_mock.encode(prompt_text))
            stored_count = row["prompt_tokens_count"]

            # Verify stored count matches actual prompt token count
            assert stored_count == prompt_tokens, (
                f"prompt_tokens_count should match actual prompt tokens. "
                f"Expected {prompt_tokens}, got {stored_count}"
            )

            # Verify that the prompt was adjusted to account for prefix
            # The total effective tokens (prefix + prompt) should be close to
            # the target (50). The prompt should have been reduced by the
            # prefix token count
            total_effective_tokens = prefix_tokens + prompt_tokens
            # Allow some variance due to sampling, but total should be around
            # target
            assert 40 <= total_effective_tokens <= 60, (
                f"Total effective tokens (prefix: {prefix_tokens} + prompt: "
                f"{prompt_tokens} = {total_effective_tokens}) should be close "
                f"to target of 50 when include_prefix_in_token_count=True"
            )

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_with_different_config_values(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        sample_dataset_default_columns,
        temp_output_path,
    ):
        """
        Test process_dataset with different config values (min, max, stdev).
        ## WRITTEN BY AI ##
        """
        # Create config with min, max, and stdev
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 50, '
            '"prompt_tokens_max": 150, "prompt_tokens_stdev": 10, '
            '"output_tokens": 50, "output_tokens_min": 25, '
            '"output_tokens_max": 75, "output_tokens_stdev": 5}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            sample_dataset_default_columns
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Verify save_dataset_to_file was called
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify dataset was processed
        assert len(saved_dataset) > 0


@pytest.fixture
def large_dataset_for_validation():
    """Large dataset with many rows for statistical validation."""
    # Create 20 rows with long prompts to ensure they pass filtering
    prompts = [
        f"This is a very long prompt number {i} for testing purposes. " * 15
        for i in range(20)
    ]
    return Dataset.from_dict({"prompt": prompts})


class TestProcessDatasetConfigValidation:
    """Test cases for validating config settings by verifying actual token counts."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_fixed_prompt_token_count(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that fixed prompt token counts (min=max) are respected.
        ## WRITTEN BY AI ##
        """
        # Config with fixed prompt tokens (min=max=100)
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 100, '
            '"prompt_tokens_max": 100, "output_tokens": 50}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all prompts have exactly 100 tokens
        for row in saved_dataset:
            assert row["prompt_tokens_count"] == 100
            # Verify actual tokenized length matches
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens == 100, f"Expected 100 tokens, got {actual_tokens}"

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_fixed_output_token_count(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that fixed output token counts (min=max) are respected.
        ## WRITTEN BY AI ##
        """
        # Config with fixed output tokens (min=max=75)
        config = (
            '{"prompt_tokens": 100, "output_tokens": 75, '
            '"output_tokens_min": 75, "output_tokens_max": 75}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all outputs have exactly 75 tokens
        for row in saved_dataset:
            assert row["output_tokens_count"] == 75

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_prompt_min_max_constraints(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that prompt token counts respect min/max constraints.
        ## WRITTEN BY AI ##
        """
        # Config with prompt min=80, max=120
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 80, '
            '"prompt_tokens_max": 120, "output_tokens": 50}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all prompt token counts are within bounds
        prompt_counts = [row["prompt_tokens_count"] for row in saved_dataset]
        assert len(prompt_counts) > 0
        assert min(prompt_counts) >= 80, (
            f"Found prompt count {min(prompt_counts)} below min 80"
        )
        assert max(prompt_counts) <= 120, (
            f"Found prompt count {max(prompt_counts)} above max 120"
        )

        # Verify actual tokenized lengths match stored counts
        for row in saved_dataset:
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens == row["prompt_tokens_count"], (
                f"Stored count {row['prompt_tokens_count']} doesn't match "
                f"actual {actual_tokens}"
            )

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_output_min_max_constraints(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that output token counts respect min/max constraints.
        ## WRITTEN BY AI ##
        """
        # Config with output min=40, max=60
        config = (
            '{"prompt_tokens": 100, "output_tokens": 50, '
            '"output_tokens_min": 40, "output_tokens_max": 60}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all output token counts are within bounds
        output_counts = [row["output_tokens_count"] for row in saved_dataset]
        assert len(output_counts) > 0
        assert min(output_counts) >= 40, (
            f"Found output count {min(output_counts)} below min 40"
        )
        assert max(output_counts) <= 60, (
            f"Found output count {max(output_counts)} above max 60"
        )

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_prompt_stdev_distribution(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that prompt token counts follow expected distribution with stdev.
        ## WRITTEN BY AI ##
        """
        # Config with prompt average=100, stdev=10, min=70, max=130
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_stdev": 10, '
            '"prompt_tokens_min": 70, "prompt_tokens_max": 130, '
            '"output_tokens": 50}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
            random_seed=42,  # Fixed seed for reproducibility
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify distribution properties
        prompt_counts = [row["prompt_tokens_count"] for row in saved_dataset]
        assert len(prompt_counts) > 0

        # Check bounds
        assert min(prompt_counts) >= 70
        assert max(prompt_counts) <= 130

        # Check mean is close to average (within 2 stdev of the mean of means)
        mean_count = sum(prompt_counts) / len(prompt_counts)
        # With enough samples, mean should be close to 100
        assert 90 <= mean_count <= 110, f"Mean {mean_count} not close to expected 100"

        # Verify actual tokenized lengths match stored counts
        for row in saved_dataset:
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens == row["prompt_tokens_count"]

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_output_stdev_distribution(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that output token counts follow expected distribution with stdev.
        ## WRITTEN BY AI ##
        """
        # Config with output average=50, stdev=5, min=35, max=65
        config = (
            '{"prompt_tokens": 100, "output_tokens": 50, '
            '"output_tokens_stdev": 5, "output_tokens_min": 35, '
            '"output_tokens_max": 65}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
            random_seed=42,  # Fixed seed for reproducibility
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify distribution properties
        output_counts = [row["output_tokens_count"] for row in saved_dataset]
        assert len(output_counts) > 0

        # Check bounds
        assert min(output_counts) >= 35
        assert max(output_counts) <= 65

        # Check mean is close to average
        mean_count = sum(output_counts) / len(output_counts)
        assert 45 <= mean_count <= 55, f"Mean {mean_count} not close to expected 50"

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_token_count_accuracy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that stored token counts match actual tokenized lengths.
        ## WRITTEN BY AI ##
        """
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 80, '
            '"prompt_tokens_max": 120, "output_tokens": 50}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify stored counts match actual tokenized lengths
        for row in saved_dataset:
            prompt_text = row["prompt"]
            stored_count = row["prompt_tokens_count"]
            actual_count = len(tokenizer_mock.encode(prompt_text))

            assert actual_count == stored_count, (
                f"Stored count {stored_count} doesn't match actual tokenized "
                f"length {actual_count} for prompt: {prompt_text[:50]}..."
            )

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_prompt_trimming_accuracy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that prompts exceeding target length are trimmed correctly.
        ## WRITTEN BY AI ##
        """
        # Use a small max to force trimming
        config = (
            '{"prompt_tokens": 50, "prompt_tokens_min": 50, '
            '"prompt_tokens_max": 50, "output_tokens": 30}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all prompts are trimmed to exactly 50 tokens
        for row in saved_dataset:
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens == 50, \
                f"Prompt not trimmed correctly: expected 50 tokens, got {actual_tokens}"

    @pytest.mark.sanity
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_prompt_padding_accuracy(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        temp_output_path,
    ):
        """
        Test that prompts below target length are padded correctly with PAD strategy.
        ## WRITTEN BY AI ##
        """
        # Create dataset with short prompts
        short_prompts = ["Short", "Tiny", "Small prompt"] * 5
        dataset = Dataset.from_dict({"prompt": short_prompts})

        # Use a large target to force padding
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 100, '
            '"prompt_tokens_max": 100, "output_tokens": 30}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = dataset

        # Run process_dataset with PAD strategy
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
            short_prompt_strategy=ShortPromptStrategy.PAD,
            pad_char="X",
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify all prompts are padded to exactly 100 tokens
        pad_char_found = False
        for row in saved_dataset:
            prompt_text = row["prompt"]
            actual_tokens = len(tokenizer_mock.encode(prompt_text))
            assert actual_tokens == 100, \
                f"Prompt not padded correctly: expected 100 tokens, got {actual_tokens}"
            assert row["prompt_tokens_count"] == 100

            # Verify that pad_char "X" appears in the padded prompts
            # Since original prompts are short ("Short", "Tiny", "Small prompt"),
            # they should all be padded with "X" characters
            if "X" in prompt_text:
                pad_char_found = True
                # Verify that X characters appear at the end (where padding would be)
                # or verify that the prompt contains X characters indicating
                # padding
                assert prompt_text.count("X") > 0, (
                    f"Expected pad_char 'X' in padded prompt, but not found: "
                    f"{prompt_text[:100]}..."
                )

        # Verify that at least some prompts contain the pad character
        assert pad_char_found, (
            "Expected to find pad_char 'X' in at least some padded prompts, "
            "but none were found"
        )

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_combined_config_constraints(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test config with all parameters set (average, min, max, stdev) for
        both prompt and output.
        ## WRITTEN BY AI ##
        """
        # Config with all parameters
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 80, '
            '"prompt_tokens_max": 120, "prompt_tokens_stdev": 10, '
            '"output_tokens": 50, "output_tokens_min": 40, '
            '"output_tokens_max": 60, "output_tokens_stdev": 5}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
            random_seed=42,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Verify prompt constraints
        prompt_counts = [row["prompt_tokens_count"] for row in saved_dataset]
        assert min(prompt_counts) >= 80
        assert max(prompt_counts) <= 120

        # Verify output constraints
        output_counts = [row["output_tokens_count"] for row in saved_dataset]
        assert min(output_counts) >= 40
        assert max(output_counts) <= 60

        # Verify token count accuracy
        for row in saved_dataset:
            actual_tokens = len(tokenizer_mock.encode(row["prompt"]))
            assert actual_tokens == row["prompt_tokens_count"]

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_edge_cases_token_counts(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test edge cases: very small token counts and min=max=1 token counts.
        ## WRITTEN BY AI ##
        """
        # Test 1: Very small token counts (use PAD strategy to ensure prompts
        # are processed)
        config_small = (
            '{"prompt_tokens": 7, "prompt_tokens_min": 5, '
            '"prompt_tokens_max": 10, "output_tokens": 5, '
            '"output_tokens_min": 3, "output_tokens_max": 8}'
        )

        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config_small,
            short_prompt_strategy=ShortPromptStrategy.PAD,
            pad_char="X",
        )

        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        prompt_counts = [row["prompt_tokens_count"] for row in saved_dataset]
        output_counts = [row["output_tokens_count"] for row in saved_dataset]

        assert min(prompt_counts) >= 5
        assert max(prompt_counts) <= 10
        assert min(output_counts) >= 3
        assert max(output_counts) <= 8

        # Test 2: min=max=1 (minimum valid value) - use PAD strategy to
        # ensure processing
        config_min = (
            '{"prompt_tokens": 1, "prompt_tokens_min": 1, '
            '"prompt_tokens_max": 1, "output_tokens": 1, '
            '"output_tokens_min": 1, "output_tokens_max": 1}'
        )

        mock_save_to_file.reset_mock()
        # Create a dataset with very short prompts for this test
        short_dataset = Dataset.from_dict({"prompt": ["A"] * 5})
        mock_deserializer_factory_class.deserialize.return_value = short_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config_min,
            short_prompt_strategy=ShortPromptStrategy.PAD,
            pad_char="X",
        )

        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        for row in saved_dataset:
            assert row["prompt_tokens_count"] == 1
            assert row["output_tokens_count"] == 1

    @pytest.mark.regression
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_no_stdev_behavior(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        large_dataset_for_validation,
        temp_output_path,
    ):
        """
        Test that when stdev is not specified, values are uniformly
        distributed within min/max.
        ## WRITTEN BY AI ##
        """
        # Config without stdev (omitted entirely) - should use uniform
        # distribution
        config = (
            '{"prompt_tokens": 100, "prompt_tokens_min": 90, '
            '"prompt_tokens_max": 110, "output_tokens": 50, '
            '"output_tokens_min": 45, "output_tokens_max": 55}'
        )

        # Setup mocks
        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = (
            large_dataset_for_validation
        )

        # Run process_dataset
        process_dataset(
            data="test_data",
            output_path=temp_output_path,
            processor=tokenizer_mock,
            config=config,
            random_seed=42,
        )

        # Extract saved dataset
        assert mock_save_to_file.called
        call_args = mock_save_to_file.call_args
        saved_dataset = call_args[0][0]

        # Without stdev, values should be uniformly distributed within min/max
        prompt_counts = [row["prompt_tokens_count"] for row in saved_dataset]
        output_counts = [row["output_tokens_count"] for row in saved_dataset]

        assert min(prompt_counts) >= 90
        assert max(prompt_counts) <= 110
        assert min(output_counts) >= 45
        assert max(output_counts) <= 55


class TestShortPromptStrategyHandlers:
    """Unit tests for individual short prompt strategy handler functions."""

    @pytest.mark.sanity
    def test_handle_ignore_strategy_too_short(self, tokenizer_mock):
        """Test handle_ignore returns None for short prompts."""
        result = ShortPromptStrategyHandler.handle_ignore("short", 10, tokenizer_mock)
        assert result is None
        tokenizer_mock.encode.assert_called_with("short")

    @pytest.mark.sanity
    def test_handle_ignore_strategy_sufficient_length(self, tokenizer_mock):
        """Test handle_ignore returns prompt for sufficient length."""
        result = ShortPromptStrategyHandler.handle_ignore(
            "long prompt", 5, tokenizer_mock
        )
        assert result == "long prompt"
        tokenizer_mock.encode.assert_called_with("long prompt")

    @pytest.mark.sanity
    def test_handle_concatenate_strategy_enough_prompts(self, tokenizer_mock):
        """Test handle_concatenate with enough prompts."""
        dataset_iter = iter([{"prompt": "longer"}])
        result = ShortPromptStrategyHandler.handle_concatenate(
            "short", 10, dataset_iter, "prompt", tokenizer_mock, "\n"
        )
        assert result == "short\nlonger"

    @pytest.mark.sanity
    def test_handle_concatenate_strategy_not_enough_prompts(self, tokenizer_mock):
        """Test handle_concatenate without enough prompts."""
        dataset_iter: Iterator = iter([])
        result = ShortPromptStrategyHandler.handle_concatenate(
            "short", 10, dataset_iter, "prompt", tokenizer_mock, ""
        )
        assert result is None

    @pytest.mark.sanity
    def test_handle_pad_strategy(self, tokenizer_mock):
        """Test handle_pad pads short prompts."""
        result = ShortPromptStrategyHandler.handle_pad("short", 10, tokenizer_mock, "p")
        assert result.startswith("shortppppp")

    @pytest.mark.sanity
    def test_handle_error_strategy_valid_prompt(self, tokenizer_mock):
        """Test handle_error returns prompt for valid length."""
        result = ShortPromptStrategyHandler.handle_error(
            "valid prompt", 5, tokenizer_mock
        )
        assert result == "valid prompt"
        tokenizer_mock.encode.assert_called_with("valid prompt")

    @pytest.mark.sanity
    def test_handle_error_strategy_too_short_prompt(self, tokenizer_mock):
        """Test handle_error raises error for short prompts."""
        with pytest.raises(PromptTooShortError):
            ShortPromptStrategyHandler.handle_error("short", 10, tokenizer_mock)


class TestProcessDatasetPushToHub:
    """Test cases for push_to_hub functionality."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.push_dataset_to_hub")
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_push_to_hub_called(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        mock_push,
        tokenizer_mock,
        tmp_path,
    ):
        """Test that push_to_hub is called when push_to_hub=True."""
        # Create a dataset with prompts long enough to be processed
        sample_dataset = Dataset.from_dict({
            "prompt": ["abc " * 50],  # Long enough
        })

        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = sample_dataset

        output_path = tmp_path / "output.json"
        config = '{"prompt_tokens": 10, "output_tokens": 5}'

        process_dataset(
            data="input",
            output_path=output_path,
            processor=tokenizer_mock,
            config=config,
            push_to_hub=True,
            hub_dataset_id="id123",
        )

        # Verify push_to_hub was called with the correct arguments
        assert mock_push.called
        call_args = mock_push.call_args
        assert call_args[0][0] == "id123"
        assert isinstance(call_args[0][1], Dataset)

    @pytest.mark.sanity
    @patch("guidellm.data.builders.push_dataset_to_hub")
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_process_dataset_push_to_hub_not_called(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        mock_push,
        tokenizer_mock,
        tmp_path,
    ):
        """Test that push_to_hub is not called when push_to_hub=False."""
        # Create a dataset with prompts long enough to be processed
        sample_dataset = Dataset.from_dict({
            "prompt": ["abc " * 50],  # Long enough
        })

        mock_check_processor.return_value = tokenizer_mock
        mock_deserializer_factory_class.deserialize.return_value = sample_dataset

        output_path = tmp_path / "output.json"
        config = '{"prompt_tokens": 10, "output_tokens": 5}'

        process_dataset(
            data="input",
            output_path=output_path,
            processor=tokenizer_mock,
            config=config,
            push_to_hub=False,
        )

        # Verify push_to_hub was not called
        mock_push.assert_not_called()

    @pytest.mark.regression
    def test_push_dataset_to_hub_success(self):
        """Test push_dataset_to_hub success case."""
        os.environ["HF_TOKEN"] = "token"
        mock_dataset = MagicMock(spec=Dataset)
        push_dataset_to_hub("dataset_id", mock_dataset)
        mock_dataset.push_to_hub.assert_called_once_with("dataset_id", token="token")

    @pytest.mark.regression
    def test_push_dataset_to_hub_error_no_env(self):
        """Test push_dataset_to_hub raises error when HF_TOKEN is missing."""
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        mock_dataset = MagicMock(spec=Dataset)
        with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
            push_dataset_to_hub("dataset_id", mock_dataset)

    @pytest.mark.regression
    def test_push_dataset_to_hub_error_no_id(self):
        """Test push_dataset_to_hub raises error when hub_dataset_id is missing."""
        os.environ["HF_TOKEN"] = "token"
        mock_dataset = MagicMock(spec=Dataset)
        with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
            push_dataset_to_hub(None, mock_dataset)


class TestProcessDatasetStrategyHandlerIntegration:
    """Test cases for strategy handler integration with process_dataset."""

    @pytest.mark.smoke
    @patch("guidellm.data.builders.save_dataset_to_file")
    @patch("guidellm.data.builders.DatasetDeserializerFactory")
    @patch("guidellm.data.builders.check_load_processor")
    def test_strategy_handler_called(
        self,
        mock_check_processor,
        mock_deserializer_factory_class,
        mock_save_to_file,
        tokenizer_mock,
        tmp_path,
    ):
        """Test that strategy handlers are called during dataset processing."""
        from guidellm.data.builders import STRATEGY_HANDLERS
        mock_handler = MagicMock(return_value="processed_prompt")
        with patch.dict(STRATEGY_HANDLERS, {ShortPromptStrategy.IGNORE: mock_handler}):
            # Create a dataset with prompts that need processing
            sample_dataset = Dataset.from_dict({
                "prompt": [
                    "abc" * 20,  # Long enough to pass
                    "def" * 20,  # Long enough to pass
                ],
            })

            mock_check_processor.return_value = tokenizer_mock
            mock_deserializer_factory_class.deserialize.return_value = sample_dataset

            output_path = tmp_path / "output.json"
            config = '{"prompt_tokens": 10, "output_tokens": 5}'

            process_dataset(
                data="input",
                output_path=output_path,
                processor=tokenizer_mock,
                config=config,
                short_prompt_strategy=ShortPromptStrategy.IGNORE,
            )

            # Verify that the handler was called during processing
            # The handler is called for each row that needs processing
            mock_deserializer_factory_class.deserialize.assert_called_once()
            mock_check_processor.assert_called_once()
            assert mock_save_to_file.called
            # Verify handler was called (at least once if there are rows to process)
            if len(sample_dataset) > 0:
                assert mock_handler.called
