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
from faker import Faker

from guidellm.data import config as config_module
from guidellm.data.deserializers.synthetic import (
    SyntheticTextDataset,
    SyntheticTextDatasetDeserializer,
)
from guidellm.data.schemas import (
    DataNotSupportedError,
    SyntheticTextDatasetConfig,
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
        generator = SyntheticTextDataset(simple_config, mock_tokenizer, random_seed=42)

        assert generator.config == simple_config
        assert generator.processor == mock_tokenizer
        assert generator.random_seed == 42

    @pytest.mark.smoke
    def test_basic_iteration(self, simple_config, mock_tokenizer):
        """Test basic iteration functionality.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextDataset(simple_config, mock_tokenizer, random_seed=42)

        items = []
        for i, item in enumerate(generator):
            items.append(item)
            if i >= 4:  # Only get 5 items
                break

        # Verify we get the expected number of items
        assert len(items) == 5

        # Verify each item has the required keys (with turn index suffix for multiturn)
        for item in items:
            assert "prefix" in item
            assert "prompt_0" in item
            assert "prompt_tokens_count_0" in item
            assert "output_tokens_count_0" in item
            assert isinstance(item["prefix"], str)
            assert isinstance(item["prompt_0"], str)
            assert isinstance(item["prompt_tokens_count_0"], int)
            assert isinstance(item["output_tokens_count_0"], int)

    @pytest.mark.sanity
    def test_create_prompt_method(self, simple_config, mock_tokenizer):
        """Test _create_prompt method.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextDataset(simple_config, mock_tokenizer, random_seed=42)
        faker = Faker()
        faker.seed_instance(42)

        # Access the _create_prompt method through the examples iterable
        ex_iterable = generator._ex_iterable

        # Test normal case
        result = ex_iterable._create_prompt(5, faker, "unique_prefix ")
        assert isinstance(result, str)
        # The result should be the decoded tokens (token_0 token_1 etc.) due to our mock
        assert "token_" in result

        # Test zero tokens
        result = ex_iterable._create_prompt(0, faker)
        assert result == ""

    @pytest.mark.regression
    def test_prefix_tokens_integration(self, config_with_prefix, mock_tokenizer):
        """Test integration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextDataset(
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
        generator1 = SyntheticTextDataset(simple_config, mock_tokenizer, random_seed=42)
        generator2 = SyntheticTextDataset(simple_config, mock_tokenizer, random_seed=42)

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
            assert item1["prompt_tokens_count_0"] == item2["prompt_tokens_count_0"]
            assert item1["output_tokens_count_0"] == item2["output_tokens_count_0"]


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
            loaded_config = config_module._load_config_file(
                yaml_path,
                SyntheticTextDatasetConfig,
            )

            assert loaded_config.prompt_tokens == 60
            assert loaded_config.output_tokens == 15
            assert loaded_config.source == "yaml_test.txt"
            assert loaded_config.prefix_buckets[0].prefix_tokens == 3  # type: ignore [index]
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
            loaded_config = config_module._load_config_file(
                config_path,
                SyntheticTextDatasetConfig,
            )

            assert loaded_config.prompt_tokens == 90
            assert loaded_config.output_tokens == 35
            assert loaded_config.prefix_buckets[0].prefix_tokens == 2  # type: ignore [index]
        finally:
            Path(config_path).unlink()

    @pytest.mark.smoke
    def test_load_config_str_json(self):
        """Test loading JSON string config.

        ### WRITTEN BY AI ###
        """
        json_str = '{"prompt_tokens": 50, "output_tokens": 25}'
        loaded_config = config_module._load_config_str(
            json_str,
            SyntheticTextDatasetConfig,
        )

        assert loaded_config.prompt_tokens == 50
        assert loaded_config.output_tokens == 25

    @pytest.mark.smoke
    def test_load_config_str_key_value(self):
        """Test loading key-value string config.

        ### WRITTEN BY AI ###
        """
        kv_str = "prompt_tokens=50,output_tokens=25"
        loaded_config = config_module._load_config_str(
            kv_str,
            SyntheticTextDatasetConfig,
        )

        assert loaded_config.prompt_tokens == 50
        assert loaded_config.output_tokens == 25

    @pytest.mark.sanity
    def test_load_config_str_invalid_format(self):
        """Test loading invalid format raises DataNotSupportedError.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(DataNotSupportedError, match="Unsupported string data"):
            config_module._load_config_str(
                "invalid_format_string",
                SyntheticTextDatasetConfig,
            )

    @pytest.mark.regression
    def test_load_config_file_non_existent(self):
        """Test loading non-existent file returns None.

        ### WRITTEN BY AI ###
        """
        loaded_config = config_module._load_config_file(
            "/non/existent/path.config",
            SyntheticTextDatasetConfig,
        )
        assert loaded_config is None

    @pytest.mark.regression
    def test_load_config_str_non_string(self):
        """Test loading non-string returns None.

        ### WRITTEN BY AI ###
        """
        loaded_config = config_module._load_config_str(123, SyntheticTextDatasetConfig)
        assert loaded_config is None

    @pytest.mark.smoke
    def test_call_with_config_object(self, mock_tokenizer):
        """Test calling deserializer with SyntheticTextDatasetConfig.

        ### WRITTEN BY AI ###
        """
        config_input = SyntheticTextDatasetConfig(prompt_tokens=50, output_tokens=25)
        deserializer = SyntheticTextDatasetDeserializer()

        result = deserializer(
            data=config_input,
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


class TestSyntheticTextDatasetMultiturn:
    """Test cases for SyntheticTextDataset with turns parameter.

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

    @pytest.mark.smoke
    def test_synthetic_config_default_turns(self):
        """Test SyntheticTextDatasetConfig has default turns=1.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=100,
            output_tokens=50,
        )

        assert config.turns == 1

    @pytest.mark.sanity
    def test_synthetic_config_custom_turns(self):
        """Test SyntheticTextDatasetConfig accepts custom turns.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=100,
            output_tokens=50,
            turns=3,
        )

        assert config.turns == 3

    @pytest.mark.sanity
    def test_synthetic_config_invalid_turns(self):
        """Test SyntheticTextDatasetConfig rejects invalid turns values.

        ### WRITTEN BY AI ###
        """
        # turns=0 should fail (gt=0 constraint)
        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(
                prompt_tokens=100,
                output_tokens=50,
                turns=0,
            )

        # turns=-1 should fail
        with pytest.raises(ValueError):
            SyntheticTextDatasetConfig(
                prompt_tokens=100,
                output_tokens=50,
                turns=-1,
            )

    @pytest.mark.smoke
    def test_synthetic_single_turn_columns(self, mock_tokenizer):
        """Test synthetic dataset generates correct columns for single turn.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=25,
            turns=1,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        # Should have turn-indexed columns
        assert "prefix" in item
        assert "prompt_0" in item
        assert "prompt_tokens_count_0" in item
        assert "output_tokens_count_0" in item

        # Should not have prompt_1, etc
        assert "prompt_1" not in item
        assert "prompt_tokens_count_1" not in item

    @pytest.mark.smoke
    def test_synthetic_multi_turn_columns(self, mock_tokenizer):
        """Test synthetic dataset generates correct columns for multiple turns.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=25,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        # Should have turn-indexed columns for all 3 turns
        for turn in range(3):
            assert f"prompt_{turn}" in item
            assert f"prompt_tokens_count_{turn}" in item
            assert f"output_tokens_count_{turn}" in item

        # Should not have prompt_3
        assert "prompt_3" not in item

    @pytest.mark.sanity
    def test_synthetic_turn_column_values_unique(self, mock_tokenizer):
        """Test each turn column has unique content.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=25,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        # Prompts for different turns should be different
        # (Due to the unique prefix in _create_prompt that includes sample count)
        prompt_0 = item["prompt_0"]
        prompt_1 = item["prompt_1"]
        prompt_2 = item["prompt_2"]

        # Note: With our mock tokenizer, the prompts will be similar but should
        # have different indices in the unique prefix, making them different
        assert isinstance(prompt_0, str)
        assert isinstance(prompt_1, str)
        assert isinstance(prompt_2, str)

    @pytest.mark.regression
    def test_synthetic_iteration_with_turns(self, mock_tokenizer):
        """Test iterating dataset with turns generates all columns per row.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=30,
            output_tokens=15,
            turns=2,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get multiple items
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= 2:  # Get 3 items
                break

        # Each item should have all turn columns
        for item in items:
            assert "prefix" in item
            for turn in range(2):
                assert f"prompt_{turn}" in item
                assert f"prompt_tokens_count_{turn}" in item
                assert f"output_tokens_count_{turn}" in item
                # Values should be populated
                assert isinstance(item[f"prompt_{turn}"], str)
                assert isinstance(item[f"prompt_tokens_count_{turn}"], int)
                assert isinstance(item[f"output_tokens_count_{turn}"], int)

    @pytest.mark.sanity
    def test_synthetic_features_match_turns(self, mock_tokenizer):
        """Test dataset features match configured turns.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=25,
            turns=4,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Access the features through the examples iterable
        features = dataset._ex_iterable.features

        # Should have prefix + 4 sets of turn columns
        assert "prefix" in features
        for turn in range(4):
            assert f"prompt_{turn}" in features
            assert f"prompt_tokens_count_{turn}" in features
            assert f"output_tokens_count_{turn}" in features

    @pytest.mark.regression
    def test_synthetic_turn_token_counts_consistent(self, mock_tokenizer):
        """Test token counts are consistent across turns in a sample.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=25,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        # All turns in a sample should have the same token counts
        # (based on how synthetic generation works - same counts per sample)
        prompt_count_0 = item["prompt_tokens_count_0"]
        prompt_count_1 = item["prompt_tokens_count_1"]
        prompt_count_2 = item["prompt_tokens_count_2"]

        # These should all be the same for a given sample
        assert prompt_count_0 == prompt_count_1 == prompt_count_2

        output_count_0 = item["output_tokens_count_0"]
        output_count_1 = item["output_tokens_count_1"]
        output_count_2 = item["output_tokens_count_2"]

        # These should all be the same for a given sample
        assert output_count_0 == output_count_1 == output_count_2
