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
    DEFAULT_SYNTHETIC_TOOLS,
    SyntheticTextDataArgs,
    SyntheticTextDataset,
    SyntheticTextDatasetDeserializer,
    SyntheticTextPrefixBucketConfig,
    _SyntheticTextExamplesIterable,
)
from guidellm.data.schemas import DataNotSupportedError


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

        config = SyntheticTextDataArgs(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=100,
            prompt_tokens_stdev=10,
            prompt_tokens_min=50,
            prompt_tokens_max=150,
            output_tokens=30,
            output_tokens_stdev=5,
            output_tokens_min=20,
            output_tokens_max=40,
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

    @pytest.mark.regression
    def test_parse_json_string(self):
        """Test parsing JSON string configuration.

        ### WRITTEN BY AI ###
        """
        json_str = json.dumps(
            {
                "prompt_tokens": 75,
                "output_tokens": 25,
                "prefix_buckets": [
                    {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 10}
                ],
            }
        )

        config = SyntheticTextDataArgs.model_validate_json(json_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 25
        assert config.prefix_buckets[0].prefix_tokens == 10  # type: ignore [index]

    @pytest.mark.sanity
    def test_validation_positive_values(self):
        """Test that negative or zero values are rejected.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticTextDataArgs(prompt_tokens=0, output_tokens=20)

        with pytest.raises(ValueError):
            SyntheticTextDataArgs(prompt_tokens=20, output_tokens=0)

        # Test negative prefix tokens via PrefixBucketConfig validation
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(prefix_tokens=-1)

    @pytest.mark.regression
    def test_validation_optional_positive_values(self):
        """Test that optional parameters reject negative values.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticTextDataArgs(
                prompt_tokens=20, output_tokens=10, prompt_tokens_stdev=-1
            )

        with pytest.raises(ValueError):
            SyntheticTextDataArgs(
                prompt_tokens=20, output_tokens=10, prompt_tokens_min=-1
            )

        with pytest.raises(ValueError):
            SyntheticTextDataArgs(
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
        return SyntheticTextDataArgs(
            prompt_tokens=15,
            output_tokens=10,
        )

    @pytest.fixture
    def config_with_prefix(self):
        """Fixture for configuration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        prefix_bucket = SyntheticTextPrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=3
        )

        return SyntheticTextDataArgs(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=15,
            output_tokens=10,
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
                SyntheticTextDataArgs,
            )

            assert loaded_config.prompt_tokens == 60
            assert loaded_config.output_tokens == 15
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
                SyntheticTextDataArgs,
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
            SyntheticTextDataArgs,
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
            SyntheticTextDataArgs,
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
                SyntheticTextDataArgs,
            )

    @pytest.mark.regression
    def test_load_config_file_non_existent(self):
        """Test loading non-existent file returns None.

        ### WRITTEN BY AI ###
        """
        loaded_config = config_module._load_config_file(
            "/non/existent/path.config",
            SyntheticTextDataArgs,
        )
        assert loaded_config is None

    @pytest.mark.regression
    def test_load_config_str_non_string(self):
        """Test loading non-string returns None.

        ### WRITTEN BY AI ###
        """
        loaded_config = config_module._load_config_str(123, SyntheticTextDataArgs)
        assert loaded_config is None

    @pytest.mark.smoke
    def test_call_with_config_object(self, mock_tokenizer):
        """Test calling deserializer with SyntheticTextDataArgs config.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(prompt_tokens=50, output_tokens=25)
        deserializer = SyntheticTextDatasetDeserializer()

        result = deserializer(
            config=config,
            processor_factory=lambda: mock_tokenizer,
            random_seed=42,
        )

        assert isinstance(result, IterableDataset)


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
        config = SyntheticTextDataArgs(
            prompt_tokens=100,
            output_tokens=50,
        )

        assert config.turns == 1

    @pytest.mark.sanity
    def test_synthetic_config_custom_turns(self):
        """Test SyntheticTextDatasetConfig accepts custom turns.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(
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
            SyntheticTextDataArgs(
                prompt_tokens=100,
                output_tokens=50,
                turns=0,
            )

        # turns=-1 should fail
        with pytest.raises(ValueError):
            SyntheticTextDataArgs(
                prompt_tokens=100,
                output_tokens=50,
                turns=-1,
            )

    @pytest.mark.smoke
    def test_synthetic_single_turn_columns(self, mock_tokenizer):
        """Test synthetic dataset generates correct columns for single turn.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(
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
        config = SyntheticTextDataArgs(
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
        config = SyntheticTextDataArgs(
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
        config = SyntheticTextDataArgs(
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
        config = SyntheticTextDataArgs(
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
        config = SyntheticTextDataArgs(
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


class TestSyntheticTextDatasetConfigToolCallFields:
    """Validate tool_call_turns and tools fields on SyntheticTextDataArgs.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_defaults_no_tool_calling(self):
        """Default config has no tool calling enabled.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=50, output_tokens=50)
        assert config.tool_call_turns == []
        assert config.tools is None

    @pytest.mark.smoke
    def test_tool_call_turns_less_than_turns(self):
        """tool_call_turns int is normalized to a list of indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=2
        )
        assert config.tool_call_turns == [0, 1]

    @pytest.mark.sanity
    def test_tool_call_turns_equal_to_turns_accepted(self):
        """tool_call_turns == turns is valid (all turns are tool-call turns).

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=3
        )
        assert config.tool_call_turns == [0, 1, 2]

    @pytest.mark.sanity
    def test_custom_tools_accepted(self):
        """Custom tools with valid tool_call_turns are accepted.

        ## WRITTEN BY AI ##
        """
        custom_tools = [{"type": "function", "function": {"name": "my_func"}}]
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=50,
            turns=3,
            tool_call_turns=1,
            tools=custom_tools,
        )
        assert config.tools == custom_tools
        assert config.tool_call_turns == [0]

    @pytest.mark.smoke
    def test_list_tool_call_turns_accepted(self):
        """Explicit list of turn indices is accepted and sorted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=4, tool_call_turns=[2, 0]
        )
        assert config.tool_call_turns == [0, 2]

    @pytest.mark.sanity
    def test_list_tool_call_turns_validation_out_of_range(self):
        """List indices must be within [0, turns).

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="out of range"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=[0, 3]
            )

    @pytest.mark.sanity
    def test_list_tool_call_turns_validation_duplicates(self):
        """Duplicate indices in the list are rejected.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="duplicates"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=[0, 0]
            )

    @pytest.mark.sanity
    def test_int_tool_call_turns_exceeds_turns_rejected(self):
        """An int greater than turns is rejected.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="out of range"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=2, tool_call_turns=3
            )


class TestSyntheticDataToolColumns:
    """Verify synthetic data emits tools_{turn} columns for tool_call_turns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def processor(self):
        """Minimal mock processor for token encoding/decoding.

        ## WRITTEN BY AI ##
        """
        proc = Mock()
        proc.encode.return_value = list(range(100))
        proc.decode.return_value = "mock text"
        return proc

    @pytest.mark.smoke
    def test_no_tools_columns_when_tool_call_turns_zero(self, processor):
        """With tool_call_turns=0, no tools columns are emitted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=10, output_tokens=10, turns=3)
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" not in row
        assert "tools_1" not in row
        assert "tools_2" not in row

    @pytest.mark.smoke
    def test_tools_columns_emitted_for_tool_call_turns(self, processor):
        """With tool_call_turns=2 and turns=3, tools_0 and tools_1 are emitted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" in row
        assert "tools_1" in row
        assert "tools_2" not in row

        tools_0 = json.loads(row["tools_0"])
        assert tools_0 == DEFAULT_SYNTHETIC_TOOLS

    @pytest.mark.smoke
    def test_non_contiguous_tool_call_turns_list(self, processor):
        """With tool_call_turns=[0, 2] and turns=4, only turns 0 and 2 get tools.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=4, tool_call_turns=[0, 2]
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" in row
        assert "tools_1" not in row
        assert "tools_2" in row
        assert "tools_3" not in row

        tools_0 = json.loads(row["tools_0"])
        assert tools_0 == DEFAULT_SYNTHETIC_TOOLS

    @pytest.mark.sanity
    def test_custom_tools_used_in_synthetic_data(self, processor):
        """User-provided tools are used instead of the default placeholder.

        ## WRITTEN BY AI ##
        """
        custom_tools = [{"type": "function", "function": {"name": "custom_fn"}}]
        config = SyntheticTextDataArgs(
            prompt_tokens=10,
            output_tokens=10,
            turns=2,
            tool_call_turns=1,
            tools=custom_tools,
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        tools_0 = json.loads(row["tools_0"])
        assert tools_0 == custom_tools

    @pytest.mark.sanity
    def test_features_include_tools_columns(self, processor):
        """Features property includes tools_{i} entries for tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "tools_0" in features
        assert "tools_1" in features
        assert "tools_2" not in features

    @pytest.mark.sanity
    def test_features_non_contiguous_tool_call_turns(self, processor):
        """Features property includes tools_{i} only for listed turn indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=4, tool_call_turns=[1, 3]
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "tools_0" not in features
        assert "tools_1" in features
        assert "tools_2" not in features
        assert "tools_3" in features


class TestSyntheticTextDatasetConfigServerToolCallFields:
    """Validate server_tool_call_turns field on SyntheticTextDataArgs.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_defaults_no_server_tool_calling(self):
        """Default config has no server tool calling enabled.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=50, output_tokens=50)
        assert config.server_tool_call_turns == []

    @pytest.mark.smoke
    def test_server_tool_call_turns_int_coercion(self):
        """server_tool_call_turns int is normalized to a list of indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns=2
        )
        assert config.server_tool_call_turns == [0, 1]

    @pytest.mark.smoke
    def test_server_tool_call_turns_list_sorted(self):
        """Explicit list of turn indices is sorted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=4, server_tool_call_turns=[2, 0]
        )
        assert config.server_tool_call_turns == [0, 2]

    @pytest.mark.sanity
    def test_server_tool_call_turns_out_of_range_rejected(self):
        """Indices must be within [0, turns).

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="server_tool_call_turns index"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                server_tool_call_turns=[0, 3],
            )

    @pytest.mark.sanity
    def test_server_tool_call_turns_duplicates_rejected(self):
        """Duplicate indices are rejected.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="duplicates"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                server_tool_call_turns=[1, 1],
            )

    @pytest.mark.sanity
    def test_overlap_with_tool_call_turns_rejected(self):
        """server_tool_call_turns and tool_call_turns must not overlap.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must not overlap"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=4,
                tool_call_turns=[0, 1],
                server_tool_call_turns=[1, 2],
            )

    @pytest.mark.sanity
    def test_no_overlap_accepted(self):
        """Non-overlapping tool_call_turns and server_tool_call_turns are accepted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=50,
            turns=4,
            tool_call_turns=[0, 1],
            server_tool_call_turns=[2, 3],
        )
        assert config.tool_call_turns == [0, 1]
        assert config.server_tool_call_turns == [2, 3]

    @pytest.mark.sanity
    def test_all_turns_server_tool_call(self):
        """All turns can be server_tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns=3
        )
        assert config.server_tool_call_turns == [0, 1, 2]

    @pytest.mark.smoke
    def test_server_tool_call_turns_all_string(self):
        """
        The string "all" expands to all turn indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns="all"
        )
        assert config.server_tool_call_turns == [0, 1, 2]

    @pytest.mark.smoke
    def test_server_tool_call_turns_all_single_turn(self):
        """
        The string "all" works with a single turn.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=1, server_tool_call_turns="all"
        )
        assert config.server_tool_call_turns == [0]

    @pytest.mark.sanity
    def test_server_tool_call_turns_all_rejects_overlap(self):
        """
        Using "all" for server_tool_call_turns rejects overlap with tool_call_turns.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must not overlap"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                tool_call_turns=[0],
                server_tool_call_turns="all",
            )

    @pytest.mark.sanity
    def test_tool_call_turns_all_string(self):
        """
        The string "all" also works for tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns="all"
        )
        assert config.tool_call_turns == [0, 1, 2]

    @pytest.mark.sanity
    def test_invalid_string_rejected(self):
        """
        Strings other than "all" are rejected.

        ## WRITTEN BY AI ##
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be 'all'"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                server_tool_call_turns="none",
            )


class TestSyntheticDataServerToolCallColumnsAll:
    """Verify synthetic data emits correct columns when server_tool_call_turns="all".

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def processor(self):
        """
        Minimal mock processor for token encoding/decoding.

        ## WRITTEN BY AI ##
        """
        proc = Mock()
        proc.encode.return_value = list(range(100))
        proc.decode.return_value = "mock text"
        return proc

    @pytest.mark.smoke
    def test_all_turns_emit_turn_type_columns(self, processor):
        """
        All turns emit turn_type_N = "server_tool_call" when "all" is used.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns="all"
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert row["turn_type_0"] == "server_tool_call"
        assert row["turn_type_1"] == "server_tool_call"
        assert row["turn_type_2"] == "server_tool_call"

    @pytest.mark.sanity
    def test_all_turns_features_include_all_turn_types(self, processor):
        """
        Features property includes turn_type_{i} for all turns when "all" is used.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns="all"
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "turn_type_0" in features
        assert "turn_type_1" in features
        assert "turn_type_2" in features


class TestSyntheticDataServerToolCallColumns:
    """Verify synthetic data emits turn_type_{turn} columns for server_tool_call_turns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def processor(self):
        """Minimal mock processor for token encoding/decoding.

        ## WRITTEN BY AI ##
        """
        proc = Mock()
        proc.encode.return_value = list(range(100))
        proc.decode.return_value = "mock text"
        return proc

    @pytest.mark.smoke
    def test_no_turn_type_columns_when_no_server_tool_call_turns(self, processor):
        """With no server_tool_call_turns, no turn_type columns are emitted.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=10, output_tokens=10, turns=3)
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "turn_type_0" not in row
        assert "turn_type_1" not in row
        assert "turn_type_2" not in row

    @pytest.mark.smoke
    def test_turn_type_columns_emitted_for_server_tool_call_turns(self, processor):
        """Server tool call turns emit turn_type_N = "server_tool_call".

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert row["turn_type_0"] == "server_tool_call"
        assert row["turn_type_1"] == "server_tool_call"
        assert "turn_type_2" not in row

    @pytest.mark.smoke
    def test_server_tool_call_turns_do_not_emit_tools_columns(self, processor):
        """Server tool call turns do not emit tools_N or tool_response_N columns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=[0, 1]
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" not in row
        assert "tools_1" not in row
        assert "tool_response_0" not in row
        assert "tool_response_1" not in row

    @pytest.mark.sanity
    def test_mixed_client_and_server_tool_call_turns(self, processor):
        """Client and server tool call turns emit different columns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10,
            output_tokens=10,
            turns=4,
            tool_call_turns=[0],
            server_tool_call_turns=[2, 3],
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        # Client tool call turn 0: tools + tool_response, no turn_type
        assert "tools_0" in row
        assert "tool_response_0" in row
        assert "turn_type_0" not in row

        # Standard turn 1: no tools, no turn_type
        assert "tools_1" not in row
        assert "turn_type_1" not in row

        # Server tool call turns 2 and 3: turn_type, no tools
        assert "turn_type_2" in row
        assert row["turn_type_2"] == "server_tool_call"
        assert "tools_2" not in row
        assert "turn_type_3" in row
        assert row["turn_type_3"] == "server_tool_call"
        assert "tools_3" not in row

    @pytest.mark.sanity
    def test_features_include_turn_type_columns(self, processor):
        """Features property includes turn_type_{i} for server_tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10,
            output_tokens=10,
            turns=3,
            server_tool_call_turns=[0, 2],
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "turn_type_0" in features
        assert "turn_type_1" not in features
        assert "turn_type_2" in features


class TestSyntheticTextDatasetConfigToolResponseFields:
    """Validate tool_response_tokens fields on SyntheticTextDataArgs.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_tool_response_tokens_defaults_to_none(self):
        """Default config has no tool_response_tokens.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=50, output_tokens=50)
        assert config.tool_response_tokens is None
        assert config.tool_response_tokens_stdev is None
        assert config.tool_response_tokens_min is None
        assert config.tool_response_tokens_max is None

    @pytest.mark.smoke
    def test_tool_response_tokens_accepted_with_tool_call_turns(self):
        """tool_response_tokens is valid when tool_call_turns > 0.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=50,
            turns=3,
            tool_call_turns=2,
            tool_response_tokens=50,
        )
        assert config.tool_response_tokens == 50

    @pytest.mark.sanity
    def test_tool_response_tokens_variance_fields(self):
        """All variance fields are accepted together.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=50,
            turns=3,
            tool_call_turns=2,
            tool_response_tokens=100,
            tool_response_tokens_stdev=20,
            tool_response_tokens_min=50,
            tool_response_tokens_max=150,
        )
        assert config.tool_response_tokens == 100
        assert config.tool_response_tokens_stdev == 20
        assert config.tool_response_tokens_min == 50
        assert config.tool_response_tokens_max == 150


class TestSyntheticDataToolResponseColumns:
    """Verify synthetic data emits tool_response_{turn} columns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def processor(self):
        """Minimal mock processor for token encoding/decoding.

        ## WRITTEN BY AI ##
        """
        proc = Mock()
        proc.encode.return_value = list(range(100))
        proc.decode.return_value = "mock text"
        return proc

    @pytest.mark.smoke
    def test_default_tool_response_columns_emitted(self, processor):
        """When tool_response_tokens is None, placeholder responses are used.

        ## WRITTEN BY AI ##
        """
        from guidellm.settings import settings

        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert row["tool_response_0"] == settings.default_synthetic_tool_response
        assert row["tool_response_1"] == settings.default_synthetic_tool_response
        assert "tool_response_2" not in row

    @pytest.mark.smoke
    def test_variable_length_tool_response_columns(self, processor):
        """When tool_response_tokens is set, generated JSON responses are used.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10,
            output_tokens=10,
            turns=3,
            tool_call_turns=2,
            tool_response_tokens=30,
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        parsed_0 = json.loads(row["tool_response_0"])
        parsed_1 = json.loads(row["tool_response_1"])
        assert "result" in parsed_0
        assert "result" in parsed_1
        assert "tool_response_2" not in row

    @pytest.mark.sanity
    def test_features_include_tool_response_columns(self, processor):
        """Features property includes tool_response_{i} for tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "tool_response_0" in features
        assert "tool_response_1" in features
        assert "tool_response_2" not in features
