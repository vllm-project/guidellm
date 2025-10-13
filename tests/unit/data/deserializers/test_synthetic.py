"""
Unit tests for guidellm.data.deserializers.synthetic module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from datasets import IterableDataset

from guidellm.data.deserializers.deserializer import DataNotSupportedError
from guidellm.data.deserializers.synthetic import (
    SyntheticTextDatasetConfig,
    SyntheticTextDatasetDeserializer,
    SyntheticTextGenerator,
    SyntheticTextPrefixBucketConfig,
)


class TestPrefixBucketConfig:
    """Test cases for PrefixBucketConfig class.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_creation_with_valid_params(self):
        """Test creating PrefixBucketConfig with valid parameters.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextPrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=5
        )

        assert config.bucket_weight == 100
        assert config.prefix_count == 1
        assert config.prefix_tokens == 5

    @pytest.mark.sanity
    def test_creation_with_negative_values(self):
        """Test creating PrefixBucketConfig with negative values raises ValueError.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=-10, prefix_count=1, prefix_tokens=5
            )

        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=100, prefix_count=-1, prefix_tokens=5
            )

        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=100, prefix_count=1, prefix_tokens=-5
            )

    @pytest.mark.regression
    def test_prefix_bucket_zero_weight_error(self):
        """Test that zero total weight raises an error.

        ### WRITTEN BY AI ###
        """
        # Test validation error for creating PrefixBucketConfig with weight=0
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=0, prefix_count=1, prefix_tokens=2
            )

    @pytest.mark.sanity
    def test_prefix_bucket_config_validation(self):
        """Test PrefixBucketConfig validation.

        ### WRITTEN BY AI ###
        """
        # Test valid config
        valid_config = SyntheticTextPrefixBucketConfig(
            bucket_weight=50, prefix_count=2, prefix_tokens=3
        )
        assert valid_config.bucket_weight == 50
        assert valid_config.prefix_count == 2
        assert valid_config.prefix_tokens == 3

        # Test invalid bucket_weight
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=0, prefix_count=1, prefix_tokens=2
            )

        # Test invalid prefix_count
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=100, prefix_count=0, prefix_tokens=2
            )

        # Test invalid prefix_tokens
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(
                bucket_weight=100, prefix_count=1, prefix_tokens=-1
            )


class TestSyntheticDatasetConfig:
    """Test cases for SyntheticDatasetConfig class.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_config_creation_with_all_params(self):
        """Test creating config with all parameters specified.

        ### WRITTEN BY AI ###
        """
        prefix_bucket = SyntheticTextPrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=5
        )

        config = SyntheticTextDatasetConfig(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=100,
            prompt_tokens_stdev=10,
            prompt_tokens_min=50,
            prompt_tokens_max=150,
            output_tokens=30,
            output_tokens_stdev=5,
            output_tokens_min=20,
            output_tokens_max=40,
            source="custom_text.txt",
        )

        assert config.prefix_buckets[0].prefix_tokens == 5  # type: ignore [index]
        assert config.prompt_tokens == 100
        assert config.prompt_tokens_stdev == 10
        assert config.prompt_tokens_min == 50
        assert config.prompt_tokens_max == 150
        assert config.output_tokens == 30
        assert config.output_tokens_stdev == 5
        assert config.output_tokens_min == 20
        assert config.output_tokens_max == 40
        assert config.source == "custom_text.txt"

    @pytest.mark.regression
    def test_parse_json_string(self):
        """Test parsing JSON string configuration.

        ### WRITTEN BY AI ###
        """
        json_str = json.dumps(
            {
                "prompt_tokens": 75,
                "output_tokens": 25,
                "source": "test.txt",
                "prefix_buckets": [
                    {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 10}
                ],
            }
        )

        config = SyntheticTextDatasetConfig.model_validate_json(json_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 25
        assert config.source == "test.txt"
        assert config.prefix_buckets[0].prefix_tokens == 10  # type: ignore [index]

    @pytest.mark.sanity
    def test_validation_positive_values(self):
        """Test that negative or zero values are rejected.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(prompt_tokens=0, output_tokens=20)

        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(prompt_tokens=20, output_tokens=0)

        # Test negative prefix tokens via PrefixBucketConfig validation
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(prefix_tokens=-1)

    @pytest.mark.regression
    def test_validation_optional_positive_values(self):
        """Test that optional parameters reject negative values.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(
                prompt_tokens=20, output_tokens=10, prompt_tokens_stdev=-1
            )

        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(
                prompt_tokens=20, output_tokens=10, prompt_tokens_min=-1
            )

        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(
                prompt_tokens=20, output_tokens=10, output_tokens_max=0
            )


class TestSyntheticTextGenerator:
    """Test cases for SyntheticTextGenerator class.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture to provide a mocked tokenizer.

        ### WRITTEN BY AI ###
        """
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.decode.side_effect = (
            lambda tokens, skip_special_tokens=False: " ".join(
                f"token_{t}" for t in tokens[:5]
            )
        )
        return tokenizer

    @pytest.fixture
    def simple_config(self):
        """Fixture for simple configuration.

        ### WRITTEN BY AI ###
        """
        return SyntheticTextDatasetConfig(
            prompt_tokens=15,
            output_tokens=10,
            source="The quick brown fox jumps over the lazy dog.",
        )

    @pytest.fixture
    def config_with_prefix(self):
        """Fixture for configuration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        prefix_bucket = SyntheticTextPrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=3
        )

        return SyntheticTextDatasetConfig(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=15,
            output_tokens=10,
            source="The quick brown fox jumps over the lazy dog.",
        )

    @pytest.mark.smoke
    def test_generator_initialization(self, simple_config, mock_tokenizer):
        """Test generator initialization.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        assert generator.config == simple_config
        assert generator.processor == mock_tokenizer
        assert generator.random_seed == 42

    @pytest.mark.smoke
    def test_basic_iteration(self, simple_config, mock_tokenizer):
        """Test basic iteration functionality.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        items = []
        for i, item in enumerate(generator):
            items.append(item)
            if i >= 4:  # Only get 5 items
                break

        # Verify we get the expected number of items
        assert len(items) == 5

        # Verify each item has the required keys
        for item in items:
            assert "prefix" in item
            assert "prompt" in item
            assert "prompt_tokens_count" in item
            assert "output_tokens_count" in item
            assert isinstance(item["prefix"], str)
            assert isinstance(item["prompt"], str)
            assert isinstance(item["prompt_tokens_count"], int)
            assert isinstance(item["output_tokens_count"], int)

    @pytest.mark.sanity
    def test_create_prompt_method(self, simple_config, mock_tokenizer):
        """Test _create_prompt method.

        ### WRITTEN BY AI ###
        """
        from faker import Faker

        generator = SyntheticTextGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )
        faker = Faker()
        faker.seed_instance(42)

        # Test normal case
        result = generator._create_prompt(5, faker, "unique_prefix ")
        assert isinstance(result, str)
        # The result should be the decoded tokens (token_0 token_1 etc.) due to our mock
        assert "token_" in result

        # Test zero tokens
        result = generator._create_prompt(0, faker)
        assert result == ""

    @pytest.mark.regression
    def test_prefix_tokens_integration(self, config_with_prefix, mock_tokenizer):
        """Test integration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextGenerator(
            config_with_prefix, mock_tokenizer, random_seed=42
        )

        items = []
        for i, item in enumerate(generator):
            items.append(item)
            if i >= 2:  # Only get 3 items
                break

        # Verify prefix is present in items
        for item in items:
            assert isinstance(item["prefix"], str)

    @pytest.mark.regression
    def test_random_seeding_consistency(self, simple_config, mock_tokenizer):
        """Test that same seed produces consistent results.

        ### WRITTEN BY AI ###
        """
        # Create two generators with same seed
        generator1 = SyntheticTextGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )
        generator2 = SyntheticTextGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        items1 = []
        items2 = []
        for i, (item1, item2) in enumerate(zip(generator1, generator2, strict=False)):
            items1.append(item1)
            items2.append(item2)
            if i >= 2:  # Only get 3 items
                break

        # With same seed and deterministic mocks, results should be identical
        assert len(items1) == len(items2)
        for item1, item2 in zip(items1, items2, strict=False):
            assert item1["prompt_tokens_count"] == item2["prompt_tokens_count"]
            assert item1["output_tokens_count"] == item2["output_tokens_count"]


class TestSyntheticDatasetDeserializer:
    """Test cases for SyntheticDatasetDeserializer class.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture to provide a mocked tokenizer.

        ### WRITTEN BY AI ###
        """
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.decode.side_effect = (
            lambda tokens, skip_special_tokens=False: " ".join(
                f"token_{t}" for t in tokens[:5]
            )
        )
        return tokenizer

    @pytest.mark.sanity
    def test_load_config_file_yaml(self):
        """Test loading YAML config file.

        ### WRITTEN BY AI ###
        """
        config_data = {
            "prompt_tokens": 60,
            "output_tokens": 15,
            "source": "yaml_test.txt",
            "prefix_buckets": [
                {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 3}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            deserializer = SyntheticTextDatasetDeserializer()
            config = deserializer._load_config_file(yaml_path)

            assert config.prompt_tokens == 60
            assert config.output_tokens == 15
            assert config.source == "yaml_test.txt"
            assert config.prefix_buckets[0].prefix_tokens == 3  # type: ignore [index]
        finally:
            Path(yaml_path).unlink()

    @pytest.mark.sanity
    def test_load_config_file_config_extension(self):
        """Test loading .config file.

        ### WRITTEN BY AI ###
        """
        config_data = {
            "prompt_tokens": 90,
            "output_tokens": 35,
            "prefix_buckets": [
                {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 2}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".config", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            deserializer = SyntheticTextDatasetDeserializer()
            config = deserializer._load_config_file(config_path)

            assert config.prompt_tokens == 90
            assert config.output_tokens == 35
            assert config.prefix_buckets[0].prefix_tokens == 2  # type: ignore [index]
        finally:
            Path(config_path).unlink()

    @pytest.mark.smoke
    def test_load_config_str_json(self):
        """Test loading JSON string config.

        ### WRITTEN BY AI ###
        """
        json_str = '{"prompt_tokens": 50, "output_tokens": 25}'
        deserializer = SyntheticTextDatasetDeserializer()
        config = deserializer._load_config_str(json_str)

        assert config.prompt_tokens == 50
        assert config.output_tokens == 25

    @pytest.mark.smoke
    def test_load_config_str_key_value(self):
        """Test loading key-value string config.

        ### WRITTEN BY AI ###
        """
        kv_str = "prompt_tokens=50,output_tokens=25"
        deserializer = SyntheticTextDatasetDeserializer()
        config = deserializer._load_config_str(kv_str)

        assert config.prompt_tokens == 50
        assert config.output_tokens == 25

    @pytest.mark.sanity
    def test_load_config_str_invalid_format(self):
        """Test loading invalid format raises DataNotSupportedError.

        ### WRITTEN BY AI ###
        """
        deserializer = SyntheticTextDatasetDeserializer()
        with pytest.raises(DataNotSupportedError, match="Unsupported string data"):
            deserializer._load_config_str("invalid_format_string")

    @pytest.mark.regression
    def test_load_config_file_non_existent(self):
        """Test loading non-existent file returns None.

        ### WRITTEN BY AI ###
        """
        deserializer = SyntheticTextDatasetDeserializer()
        config = deserializer._load_config_file("/non/existent/path.config")
        assert config is None

    @pytest.mark.regression
    def test_load_config_str_non_string(self):
        """Test loading non-string returns None.

        ### WRITTEN BY AI ###
        """
        deserializer = SyntheticTextDatasetDeserializer()
        config = deserializer._load_config_str(123)
        assert config is None

    @pytest.mark.smoke
    def test_call_with_config_object(self, mock_tokenizer):
        """Test calling deserializer with SyntheticTextDatasetConfig.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(prompt_tokens=50, output_tokens=25)
        deserializer = SyntheticTextDatasetDeserializer()

        result = deserializer(
            data=config,
            data_kwargs={},
            processor_factory=lambda: mock_tokenizer,
            random_seed=42,
        )

        assert isinstance(result, IterableDataset)

    @pytest.mark.regression
    def test_call_with_unsupported_data(self, mock_tokenizer):
        """Test calling deserializer with unsupported data raises error.

        ### WRITTEN BY AI ###
        """
        deserializer = SyntheticTextDatasetDeserializer()

        with pytest.raises(DataNotSupportedError, match="Unsupported data"):
            deserializer(
                data=123,
                data_kwargs={},
                processor_factory=lambda: mock_tokenizer,
                random_seed=42,
            )

    @pytest.mark.regression
    def test_call_with_json_string(self, mock_tokenizer):
        """Test calling deserializer with JSON string.

        ### WRITTEN BY AI ###
        """
        json_str = '{"prompt_tokens": 50, "output_tokens": 25}'
        deserializer = SyntheticTextDatasetDeserializer()

        result = deserializer(
            data=json_str,
            data_kwargs={},
            processor_factory=lambda: mock_tokenizer,
            random_seed=42,
        )

        assert isinstance(result, IterableDataset)

    @pytest.mark.regression
    def test_call_with_config_file(self, mock_tokenizer):
        """Test calling deserializer with config file.

        ### WRITTEN BY AI ###
        """
        config_data = {"prompt_tokens": 65, "output_tokens": 45}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            deserializer = SyntheticTextDatasetDeserializer()
            result = deserializer(
                data=config_path,
                data_kwargs={},
                processor_factory=lambda: mock_tokenizer,
                random_seed=42,
            )
            assert isinstance(result, IterableDataset)
        finally:
            Path(config_path).unlink()
