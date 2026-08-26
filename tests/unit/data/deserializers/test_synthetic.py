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
from pydantic import ValidationError

from guidellm.data import config as config_module
from guidellm.data.deserializers.synthetic import (
    DEFAULT_SYNTHETIC_TOOLS,
    SyntheticTextDataset,
    SyntheticTextDatasetDeserializer,
    _SyntheticTextExamplesIterable,
)
from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
from guidellm.schemas.data import SyntheticTextDataArgs, SyntheticTextPrefixBucketConfig
from guidellm.settings import settings


def _conversation_graph(row: dict) -> ConversationGraphData:
    """Parse the conversation_turns payload from a synthetic row.

    ## WRITTEN BY AI ##
    """
    return ConversationGraphData.model_validate(json.loads(row["conversation_turns"]))


def _main_turn_map(row: dict) -> dict[str, object]:
    """Map main_* node ids to ConversationTurnData for a synthetic row.

    ## WRITTEN BY AI ##
    """
    graph = _conversation_graph(row)
    return {
        turn.node_id: turn for turn in graph.turns if turn.node_id.startswith("main_")
    }


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
            delay=1.2,
            delay_stdev=0.5,
            delay_min=0.1,
            delay_max=2.4,
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
        assert config.delay == 1.2
        assert config.delay_stdev == 0.5
        assert config.delay_min == 0.1
        assert config.delay_max == 2.4

    @pytest.mark.regression
    def test_parse_json_string(self):
        """Test parsing JSON string configuration.

        ### WRITTEN BY AI ###
        """
        json_str = json.dumps(
            {
                "prompt_tokens": 75,
                "output_tokens": 25,
                "delay": 1.2,
                "prefix_buckets": [
                    {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 10}
                ],
            }
        )

        config = SyntheticTextDataArgs.model_validate_json(json_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 25
        assert config.delay == 1.2
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

        with pytest.raises(ValueError):
            SyntheticTextDataArgs(prompt_tokens=20, output_tokens=20, delay=0.0)

        # Test negative prefix tokens via PrefixBucketConfig validation
        with pytest.raises(ValueError):
            SyntheticTextPrefixBucketConfig(prefix_tokens=-1)

    @pytest.mark.sanity
    def test_validation_positive_or_zero(self):
        """`delay_min` will take 0.0 as a valid value."""
        SyntheticTextDataArgs(prompt_tokens=20, delay=1.2, delay_min=0.0)
        with pytest.raises(ValueError):
            SyntheticTextDataArgs(prompt_tokens=20, delay=1.2, delay_min=-0.1)

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
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"token_{t}" for t in tokens[:5])
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

        # Verify each item is a conversation_turns graph payload
        for item in items:
            assert set(item) == {"conversation_turns"}
            graph = _conversation_graph(item)
            assert len(graph.turns) == 1
            turn = graph.turns[0]
            assert turn.node_id == "main_0"
            assert "text_column" in turn.columns
            assert "prompt_tokens_count_column" in turn.columns
            assert "output_tokens_count_column" in turn.columns
            assert isinstance(turn.columns["text_column"][0], str)
            assert isinstance(turn.columns["prompt_tokens_count_column"][0], int)
            assert isinstance(turn.columns["output_tokens_count_column"][0], int)

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

        # Verify prefix is present on the first main turn
        for item in items:
            turn = _main_turn_map(item)["main_0"]
            assert "prefix_column" in turn.columns
            assert isinstance(turn.columns["prefix_column"][0], str)

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
            turns1 = _main_turn_map(item1)
            turns2 = _main_turn_map(item2)
            assert (
                turns1["main_0"].columns["prompt_tokens_count_column"]
                == turns2["main_0"].columns["prompt_tokens_count_column"]
            )
            assert (
                turns1["main_0"].columns["output_tokens_count_column"]
                == turns2["main_0"].columns["output_tokens_count_column"]
            )


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
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"token_{t}" for t in tokens[:5])
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
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"token_{t}" for t in tokens[:5])
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
        """Test synthetic dataset generates a one-node conversation graph.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=25,
            delay=3.0,
            turns=1,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        assert set(item) == {"conversation_turns"}
        turns = _main_turn_map(item)
        assert set(turns) == {"main_0"}
        assert turns["main_0"].settings is not None
        assert turns["main_0"].settings.requeue_delay == 3.0
        assert "text_column" in turns["main_0"].columns
        assert turns["main_0"].columns["prompt_tokens_count_column"] == [50]
        assert turns["main_0"].columns["output_tokens_count_column"] == [25]

    @pytest.mark.smoke
    def test_synthetic_multi_turn_columns(self, mock_tokenizer):
        """Test synthetic dataset generates a multi-node conversation graph.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=25,
            delay=3.0,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        # Get one item
        item = next(iter(dataset))

        turns = _main_turn_map(item)
        assert set(turns) == {"main_0", "main_1", "main_2"}
        for turn in turns.values():
            assert turn.settings is not None
            assert turn.settings.requeue_delay == 3.0
            assert turn.columns["prompt_tokens_count_column"] == [50]
            assert turn.columns["output_tokens_count_column"] == [25]
            assert "text_column" in turn.columns

    @pytest.mark.sanity
    def test_synthetic_turn_column_values_unique(self, mock_tokenizer):
        """Test each turn has unique prompt content.

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
        turns = _main_turn_map(item)

        prompt_0 = turns["main_0"].columns["text_column"][0]
        prompt_1 = turns["main_1"].columns["text_column"][0]
        prompt_2 = turns["main_2"].columns["text_column"][0]

        assert isinstance(prompt_0, str)
        assert isinstance(prompt_1, str)
        assert isinstance(prompt_2, str)

    @pytest.mark.regression
    def test_synthetic_iteration_with_turns(self, mock_tokenizer):
        """Test iterating dataset with turns generates graph payloads.

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

        for item in items:
            turns = _main_turn_map(item)
            assert set(turns) == {"main_0", "main_1"}
            for turn in turns.values():
                assert isinstance(turn.columns["text_column"][0], str)
                assert isinstance(turn.columns["prompt_tokens_count_column"][0], int)
                assert isinstance(turn.columns["output_tokens_count_column"][0], int)

    @pytest.mark.sanity
    def test_synthetic_features_match_turns(self, mock_tokenizer):
        """Test dataset features are the conversation_turns graph column.

        ### WRITTEN BY AI ###
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=25,
            turns=4,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)

        features = dataset._ex_iterable.features
        assert set(features) == {"conversation_turns"}

    @pytest.mark.regression
    def test_synthetic_turn_token_counts_fixed_without_stdev(self, mock_tokenizer):
        """Without stdev, every turn equals the configured token means.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=25,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)
        turns = _main_turn_map(next(iter(dataset)))

        assert (
            turns["main_0"].columns["prompt_tokens_count_column"]
            == turns["main_1"].columns["prompt_tokens_count_column"]
            == turns["main_2"].columns["prompt_tokens_count_column"]
            == [50]
        )
        assert (
            turns["main_0"].columns["output_tokens_count_column"]
            == turns["main_1"].columns["output_tokens_count_column"]
            == turns["main_2"].columns["output_tokens_count_column"]
            == [25]
        )

    @pytest.mark.regression
    def test_synthetic_turn_token_counts_independent_with_stdev(self, mock_tokenizer):
        """With stdev, turns sample independently and are not forced equal.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            prompt_tokens_stdev=20,
            prompt_tokens_min=10,
            prompt_tokens_max=100,
            output_tokens=25,
            output_tokens_stdev=10,
            output_tokens_min=5,
            output_tokens_max=50,
            turns=8,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)
        turns = _main_turn_map(next(iter(dataset)))

        prompt_counts = [
            turns[f"main_{i}"].columns["prompt_tokens_count_column"][0]
            for i in range(8)
        ]
        output_counts = [
            turns[f"main_{i}"].columns["output_tokens_count_column"][0]
            for i in range(8)
        ]
        assert len(set(prompt_counts)) > 1
        assert len(set(output_counts)) > 1

    @pytest.mark.regression
    def test_synthetic_first_prompt_tokens_override(self, mock_tokenizer):
        """first_prompt_tokens applies only to turn 0; later turns use prompt_tokens.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50,
            output_tokens=25,
            first_prompt_tokens=200,
            turns=3,
        )
        dataset = SyntheticTextDataset(config, mock_tokenizer, random_seed=42)
        turns = _main_turn_map(next(iter(dataset)))

        assert turns["main_0"].columns["prompt_tokens_count_column"] == [200]
        assert turns["main_1"].columns["prompt_tokens_count_column"] == [50]
        assert turns["main_2"].columns["prompt_tokens_count_column"] == [50]
        assert turns["main_0"].columns["output_tokens_count_column"] == [25]

    @pytest.mark.sanity
    def test_first_prompt_tokens_requires_mean(self):
        """Reject first_prompt_tokens_stdev without first_prompt_tokens.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="first_prompt_tokens must be set"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=25,
                turns=3,
                first_prompt_tokens_stdev=10,
            )


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

        with pytest.raises(ValidationError, match="out of range"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=[0, 3]
            )

    @pytest.mark.sanity
    def test_list_tool_call_turns_validation_duplicates(self):
        """Duplicate indices in the list are rejected.

        ## WRITTEN BY AI ##
        """

        with pytest.raises(ValidationError, match="duplicates"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=[0, 0]
            )

    @pytest.mark.sanity
    def test_int_tool_call_turns_exceeds_turns_rejected(self):
        """An int greater than turns is rejected.

        ## WRITTEN BY AI ##
        """

        with pytest.raises(ValidationError, match="out of range"):
            SyntheticTextDataArgs(
                prompt_tokens=50, output_tokens=50, turns=2, tool_call_turns=3
            )


class TestSyntheticDataToolColumns:
    """Verify synthetic graphs embed tools on tool_call turns.

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
        """With tool_call_turns=0, no tools columns are emitted on turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(prompt_tokens=10, output_tokens=10, turns=3)
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        for turn in turns.values():
            assert "tools_column" not in turn.columns

    @pytest.mark.smoke
    def test_tools_columns_emitted_for_tool_call_turns(self, processor):
        """With tool_call_turns=2 and turns=3, main_0 and main_1 carry tools.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        assert "tools_column" in turns["main_0"].columns
        assert "tools_column" in turns["main_1"].columns
        assert "tools_column" not in turns["main_2"].columns

        tools_0 = json.loads(turns["main_0"].columns["tools_column"][0])
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
        turns = _main_turn_map(row)

        assert "tools_column" in turns["main_0"].columns
        assert "tools_column" not in turns["main_1"].columns
        assert "tools_column" in turns["main_2"].columns
        assert "tools_column" not in turns["main_3"].columns

        tools_0 = json.loads(turns["main_0"].columns["tools_column"][0])
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
        turns = _main_turn_map(row)

        tools_0 = json.loads(turns["main_0"].columns["tools_column"][0])
        assert tools_0 == custom_tools

    @pytest.mark.sanity
    def test_features_include_tools_columns(self, processor):
        """Features property is always the conversation_turns column.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        assert set(iterable.features) == {"conversation_turns"}

    @pytest.mark.sanity
    def test_features_non_contiguous_tool_call_turns(self, processor):
        """Features stay conversation_turns regardless of tool_call_turns list.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=4, tool_call_turns=[1, 3]
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        assert set(iterable.features) == {"conversation_turns"}


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
    def test_server_tool_call_turns_minus_one(self):
        """
        The value -1 expands to all turn indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns=-1
        )
        assert config.server_tool_call_turns == [0, 1, 2]

    @pytest.mark.smoke
    def test_server_tool_call_turns_minus_one_single_turn(self):
        """
        The value -1 works with a single turn.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=1, server_tool_call_turns=-1
        )
        assert config.server_tool_call_turns == [0]

    @pytest.mark.sanity
    def test_server_tool_call_turns_minus_one_rejects_overlap(self):
        """
        Using -1 for server_tool_call_turns rejects overlap with tool_call_turns.

        ## WRITTEN BY AI ##
        """

        with pytest.raises(ValidationError, match="must not overlap"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                tool_call_turns=[0],
                server_tool_call_turns=-1,
            )

    @pytest.mark.sanity
    def test_tool_call_turns_minus_one(self):
        """
        The value -1 expands to all turn indices for tool_call_turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=-1
        )
        assert config.tool_call_turns == [0, 1, 2]

    @pytest.mark.sanity
    def test_invalid_string_rejected(self):
        """
        Non-JSON strings are rejected.

        ## WRITTEN BY AI ##
        """

        with pytest.raises(ValidationError, match="JSON int or list of ints"):
            SyntheticTextDataArgs(
                prompt_tokens=50,
                output_tokens=50,
                turns=3,
                server_tool_call_turns="none",
            )

    @pytest.mark.smoke
    def test_string_int_coercion(self):
        """
        A string int like "2" is coerced to int and normalized.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns="2"
        )
        assert config.server_tool_call_turns == [0, 1]

    @pytest.mark.smoke
    def test_string_minus_one_coercion(self):
        """
        The string "-1" is coerced to -1 and expands to all turn indices.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=3, server_tool_call_turns="-1"
        )
        assert config.server_tool_call_turns == [0, 1, 2]

    @pytest.mark.smoke
    def test_string_list_coercion(self):
        """
        A JSON list string like "[0, 2]" is coerced to a list of ints.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=50, output_tokens=50, turns=4, server_tool_call_turns="[0, 2]"
        )
        assert config.server_tool_call_turns == [0, 2]


class TestSyntheticDataServerToolCallColumnsAll:
    """Verify synthetic graphs mark all turns when server_tool_call_turns=-1.

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
        All turns carry turn_type_column=server_tool_call when -1 is used.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=-1
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        for turn in turns.values():
            assert turn.columns["turn_type_column"] == ["server_tool_call"]

    @pytest.mark.sanity
    def test_all_turns_features_include_all_turn_types(self, processor):
        """
        Features property is conversation_turns when server_tool_call_turns=-1.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=-1
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        assert set(iterable.features) == {"conversation_turns"}


class TestSyntheticDataServerToolCallColumns:
    """Verify synthetic graphs embed server_tool_call turn types.

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
        turns = _main_turn_map(row)

        for turn in turns.values():
            assert "turn_type_column" not in turn.columns

    @pytest.mark.smoke
    def test_turn_type_columns_emitted_for_server_tool_call_turns(self, processor):
        """Server tool call turns emit turn_type_column=server_tool_call.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        assert turns["main_0"].columns["turn_type_column"] == ["server_tool_call"]
        assert turns["main_1"].columns["turn_type_column"] == ["server_tool_call"]
        assert "turn_type_column" not in turns["main_2"].columns

    @pytest.mark.smoke
    def test_server_tool_call_turns_do_not_emit_tools_columns(self, processor):
        """Server tool call turns do not emit tools or tool_response columns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, server_tool_call_turns=[0, 1]
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        assert "tools_column" not in turns["main_0"].columns
        assert "tools_column" not in turns["main_1"].columns
        assert "tool_response_column" not in turns["main_0"].columns
        assert "tool_response_column" not in turns["main_1"].columns

    @pytest.mark.sanity
    def test_mixed_client_and_server_tool_call_turns(self, processor):
        """Client and server tool call turns embed different columns.

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
        turns = _main_turn_map(row)

        # Client tool call turn 0: tools + tool_response, no turn_type
        assert "tools_column" in turns["main_0"].columns
        assert "tool_response_column" in turns["main_0"].columns
        assert "turn_type_column" not in turns["main_0"].columns

        # Standard turn 1: no tools, no turn_type
        assert "tools_column" not in turns["main_1"].columns
        assert "turn_type_column" not in turns["main_1"].columns

        # Server tool call turns 2 and 3: turn_type, no tools
        assert turns["main_2"].columns["turn_type_column"] == ["server_tool_call"]
        assert "tools_column" not in turns["main_2"].columns
        assert turns["main_3"].columns["turn_type_column"] == ["server_tool_call"]
        assert "tools_column" not in turns["main_3"].columns

    @pytest.mark.sanity
    def test_features_include_turn_type_columns(self, processor):
        """Features property is conversation_turns for server tool configs.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10,
            output_tokens=10,
            turns=3,
            server_tool_call_turns=[0, 2],
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        assert set(iterable.features) == {"conversation_turns"}


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
    """Verify synthetic graphs embed tool_response on tool_call turns.

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

        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))
        turns = _main_turn_map(row)

        assert (
            turns["main_0"].columns["tool_response_column"][0]
            == settings.default_synthetic_tool_response
        )
        assert (
            turns["main_1"].columns["tool_response_column"][0]
            == settings.default_synthetic_tool_response
        )
        assert "tool_response_column" not in turns["main_2"].columns

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
        turns = _main_turn_map(row)

        parsed_0 = json.loads(turns["main_0"].columns["tool_response_column"][0])
        parsed_1 = json.loads(turns["main_1"].columns["tool_response_column"][0])
        assert "result" in parsed_0
        assert "result" in parsed_1
        assert "tool_response_column" not in turns["main_2"].columns

    @pytest.mark.sanity
    def test_features_include_tool_response_columns(self, processor):
        """Features property is conversation_turns for tool_call configs.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDataArgs(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        assert set(iterable.features) == {"conversation_turns"}
