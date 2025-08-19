"""
Unit tests for guidellm.dataset.synthetic module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from guidellm.dataset.synthetic import (
    PrefixBucketConfig,
    SyntheticDatasetConfig,
    SyntheticDatasetCreator,
    SyntheticTextItemsGenerator,
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
        prefix_bucket = PrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=5
        )

        config = SyntheticDatasetConfig(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=100,
            prompt_tokens_stdev=10,
            prompt_tokens_min=50,
            prompt_tokens_max=150,
            output_tokens=30,
            output_tokens_stdev=5,
            output_tokens_min=20,
            output_tokens_max=40,
            samples=500,
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
        assert config.samples == 500
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
                "samples": 200,
                "source": "test.txt",
                "prefix_buckets": [
                    {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 10}
                ],
            }
        )

        config = SyntheticDatasetConfig.parse_str(json_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 25
        assert config.samples == 200
        assert config.source == "test.txt"
        assert config.prefix_buckets[0].prefix_tokens == 10  # type: ignore [index]

    @pytest.mark.regression
    def test_parse_key_value_pairs(self):
        """Test parsing key-value pairs configuration.

        ### WRITTEN BY AI ###
        """
        kv_str = "prompt_tokens=80,output_tokens=30,samples=300,source=data.txt"

        config = SyntheticDatasetConfig.parse_str(kv_str)

        assert config.prompt_tokens == 80
        assert config.output_tokens == 30
        assert config.samples == 300
        assert config.source == "data.txt"
        assert config.prefix_buckets is None

    @pytest.mark.sanity
    def test_parse_yaml_file(self):
        """Test parsing YAML file configuration.

        ### WRITTEN BY AI ###
        """
        config_data = {
            "prompt_tokens": 60,
            "output_tokens": 15,
            "samples": 100,
            "source": "yaml_test.txt",
            "prefix_buckets": [
                {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 3}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            config = SyntheticDatasetConfig.parse_str(yaml_path)

            assert config.prompt_tokens == 60
            assert config.output_tokens == 15
            assert config.samples == 100
            assert config.source == "yaml_test.txt"
            assert config.prefix_buckets[0].prefix_tokens == 3  # type: ignore [index]
        finally:
            Path(yaml_path).unlink()

    @pytest.mark.sanity
    def test_parse_config_file(self):
        """Test parsing .config file.

        ### WRITTEN BY AI ###
        """
        config_data = {
            "prompt_tokens": 90,
            "output_tokens": 35,
            "samples": 150,
            "prefix_buckets": [
                {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 2}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".config", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = SyntheticDatasetConfig.parse_str(config_path)

            assert config.prompt_tokens == 90
            assert config.output_tokens == 35
            assert config.samples == 150
            assert config.prefix_buckets[0].prefix_tokens == 2  # type: ignore [index]
        finally:
            Path(config_path).unlink()

    @pytest.mark.regression
    def test_parse_path_object(self):
        """Test parsing with Path object.

        ### WRITTEN BY AI ###
        """
        config_data = {"prompt_tokens": 45, "output_tokens": 25}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)

        try:
            config = SyntheticDatasetConfig.parse_str(yaml_path)
            assert config.prompt_tokens == 45
            assert config.output_tokens == 25
        finally:
            yaml_path.unlink()

    @pytest.mark.sanity
    def test_parse_invalid_format(self):
        """Test parsing invalid format raises ValueError.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError, match="Unsupported data format"):
            SyntheticDatasetConfig.parse_str("invalid_format_string")

    @pytest.mark.sanity
    def test_validation_positive_values(self):
        """Test that negative or zero values are rejected.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=0, output_tokens=20)

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=20, output_tokens=0)

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=20, output_tokens=10, samples=0)

        # Test negative prefix tokens via PrefixBucketConfig validation
        with pytest.raises(ValueError):
            PrefixBucketConfig(prefix_tokens=-1)

    @pytest.mark.regression
    def test_validation_optional_positive_values(self):
        """Test that optional parameters reject negative values.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            SyntheticDatasetConfig(
                prompt_tokens=20, output_tokens=10, prompt_tokens_stdev=-1
            )

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(
                prompt_tokens=20, output_tokens=10, prompt_tokens_min=-1
            )

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(
                prompt_tokens=20, output_tokens=10, output_tokens_max=0
            )

    @pytest.mark.regression
    def test_parse_json_method_directly(self):
        """Test parse_json static method directly.

        ### WRITTEN BY AI ###
        """
        json_data = {"prompt_tokens": 100, "output_tokens": 50}
        json_str = json.dumps(json_data)

        config = SyntheticDatasetConfig.parse_json(json_str)

        assert config.prompt_tokens == 100
        assert config.output_tokens == 50

    @pytest.mark.regression
    def test_parse_key_value_pairs_method_directly(self):
        """Test parse_key_value_pairs static method directly.

        ### WRITTEN BY AI ###
        """
        kv_str = "prompt_tokens=75,output_tokens=35"

        config = SyntheticDatasetConfig.parse_key_value_pairs(kv_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 35

    @pytest.mark.regression
    def test_parse_config_file_method_directly(self):
        """Test parse_config_file static method directly.

        ### WRITTEN BY AI ###
        """
        config_data = {"prompt_tokens": 65, "output_tokens": 45}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = SyntheticDatasetConfig.parse_config_file(config_path)
            assert config.prompt_tokens == 65
            assert config.output_tokens == 45
        finally:
            Path(config_path).unlink()


class TestSyntheticTextItemsGenerator:
    """Test cases for SyntheticTextItemsGenerator class.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture to provide a mocked tokenizer.

        ### WRITTEN BY AI ###
        """
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {f"token_{i}": i for i in range(1000)}
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        tokenizer.decode.side_effect = (
            lambda tokens, skip_special_tokens=False: " ".join(
                f"token_{t}" for t in tokens[:5]
            )
        )
        return tokenizer

    @pytest.fixture
    def mock_integer_range_sampler(self):
        """Fixture to provide a mocked IntegerRangeSampler.

        ### WRITTEN BY AI ###
        """
        with patch("guidellm.dataset.synthetic.IntegerRangeSampler") as mock_sampler:
            # Default side effect for basic iteration
            def mock_sampler_side_effect(*args, **kwargs):
                mock_instance = Mock()
                mock_instance.__iter__ = Mock(return_value=iter([15, 15, 15, 15, 15]))
                return mock_instance

            mock_sampler.side_effect = mock_sampler_side_effect
            yield mock_sampler

    @pytest.fixture
    def simple_config(self):
        """Fixture for simple configuration.

        ### WRITTEN BY AI ###
        """
        return SyntheticDatasetConfig(
            prompt_tokens=15,
            output_tokens=10,
            samples=5,
            source="The quick brown fox jumps over the lazy dog.",
        )

    @pytest.fixture
    def config_with_prefix(self):
        """Fixture for configuration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        prefix_bucket = PrefixBucketConfig(
            bucket_weight=100, prefix_count=1, prefix_tokens=3
        )

        return SyntheticDatasetConfig(
            prefix_buckets=[prefix_bucket],
            prompt_tokens=15,
            output_tokens=10,
            samples=5,
            source="The quick brown fox jumps over the lazy dog.",
        )

    @pytest.fixture
    def complex_config(self):
        """Fixture for complex configuration with variance.

        ### WRITTEN BY AI ###
        """
        return SyntheticDatasetConfig(
            prompt_tokens=20,
            prompt_tokens_stdev=5,
            prompt_tokens_min=10,
            prompt_tokens_max=30,
            output_tokens=15,
            output_tokens_stdev=3,
            output_tokens_min=10,
            output_tokens_max=20,
            samples=10,
            source="The quick brown fox jumps over the lazy dog.",
        )

    @pytest.mark.smoke
    @patch("guidellm.dataset.synthetic.EndlessTextCreator")
    def test_generator_initialization(
        self, mock_text_creator, simple_config, mock_tokenizer
    ):
        """Test generator initialization.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        assert generator.config == simple_config
        assert generator.processor == mock_tokenizer
        assert generator.random_seed == 42
        mock_text_creator.assert_called_once_with(data=simple_config.source)

    @pytest.mark.smoke
    def test_basic_iteration(
        self,
        mock_integer_range_sampler,
        simple_config,
        mock_tokenizer,
    ):
        """Test basic iteration functionality.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        items = list(generator)

        # Verify we get the expected number of items
        assert len(items) == simple_config.samples

        # Verify each item has the required keys
        for item in items:
            assert "prompt" in item
            assert "prompt_tokens_count" in item
            assert "output_tokens_count" in item
            assert isinstance(item["prompt"], str)
            assert isinstance(item["prompt_tokens_count"], int)
            assert isinstance(item["output_tokens_count"], int)

    @pytest.mark.sanity
    def test_create_prompt_method(self, simple_config, mock_tokenizer):
        """Test _create_prompt method.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        # Test normal case
        result = generator._create_prompt(5, 0, 42)
        assert result[0] == 42  # Unique prefix token
        assert len(result) == 5

        # Test zero tokens
        result = generator._create_prompt(0, 0, 42)
        assert result == []

        # Test without unique prefix
        result = generator._create_prompt(3, 0)
        assert len(result) == 3

    @pytest.mark.regression
    def test_create_prompt_binary_search(self, simple_config, mock_tokenizer):
        """Test binary search logic in _create_prompt.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        # Test that binary search finds appropriate length
        result = generator._create_prompt(5, 0, 42)
        assert len(result) >= 4  # Should include prefix + some tokens

    @pytest.mark.sanity
    def test_prefix_tokens_integration(
        self, mock_integer_range_sampler, config_with_prefix, mock_tokenizer
    ):
        """Test integration with prefix tokens.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            config_with_prefix, mock_tokenizer, random_seed=42
        )

        items = list(generator)

        # Verify prompt_tokens_count includes prefix
        for item in items:
            assert (
                item["prompt_tokens_count"]
                == config_with_prefix.prefix_buckets[0].prefix_tokens + 15
            )

    @pytest.mark.regression
    def test_random_seeding_consistency(
        self, mock_integer_range_sampler, simple_config, mock_tokenizer
    ):
        """Test that same seed produces consistent results.

        ### WRITTEN BY AI ###
        """
        # Create two generators with same seed
        generator1 = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )
        generator2 = SyntheticTextItemsGenerator(
            simple_config, mock_tokenizer, random_seed=42
        )

        items1 = list(generator1)
        items2 = list(generator2)

        # With same seed and deterministic mocks, results should be identical
        assert len(items1) == len(items2)
        for item1, item2 in zip(items1, items2):
            assert item1["prompt"] == item2["prompt"]
            assert item1["prompt_tokens_count"] == item2["prompt_tokens_count"]
            assert item1["output_tokens_count"] == item2["output_tokens_count"]

    @pytest.mark.regression
    def test_variance_configuration(
        self, mock_integer_range_sampler, complex_config, mock_tokenizer
    ):
        """Test that variance configuration is properly used.

        ### WRITTEN BY AI ###
        """
        generator = SyntheticTextItemsGenerator(
            complex_config, mock_tokenizer, random_seed=42
        )

        # Initialize the generator to trigger sampler creation
        generator_iter = iter(generator)
        next(generator_iter)

        # Verify that IntegerRangeSampler is called with correct parameters
        assert mock_integer_range_sampler.call_count == 2

        # Check prompt tokens sampler call
        prompt_call = mock_integer_range_sampler.call_args_list[0]
        assert prompt_call[1]["average"] == complex_config.prompt_tokens
        assert prompt_call[1]["variance"] == complex_config.prompt_tokens_stdev
        assert prompt_call[1]["min_value"] == complex_config.prompt_tokens_min
        assert prompt_call[1]["max_value"] == complex_config.prompt_tokens_max
        assert prompt_call[1]["random_seed"] == 42

        # Check output tokens sampler call
        output_call = mock_integer_range_sampler.call_args_list[1]
        assert output_call[1]["average"] == complex_config.output_tokens
        assert output_call[1]["variance"] == complex_config.output_tokens_stdev
        assert output_call[1]["min_value"] == complex_config.output_tokens_min
        assert output_call[1]["max_value"] == complex_config.output_tokens_max
        assert output_call[1]["random_seed"] == 43  # 42 + 1

    @pytest.mark.regression
    def test_unique_prefix_generation(self, simple_config, mock_tokenizer):
        """Test that unique prefixes are generated for each request.

        ### WRITTEN BY AI ###
        """
        # Mock the cycle to return predictable values
        with patch("guidellm.dataset.synthetic.cycle") as mock_cycle:
            mock_cycle.return_value = iter([100, 101, 102, 103, 104])

            generator = SyntheticTextItemsGenerator(
                simple_config, mock_tokenizer, random_seed=42
            )

            # Access the iterator to trigger the cycle creation
            generator_iter = iter(generator)
            next(generator_iter)

            # Verify cycle was called with vocab values
            mock_cycle.assert_called_once()


class TestSyntheticDatasetCreator:
    """Test cases for SyntheticDatasetCreator class.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.sanity
    def test_is_supported_path_config_file(self):
        """Test is_supported with config file paths.

        ### WRITTEN BY AI ###
        """
        with tempfile.NamedTemporaryFile(suffix=".config", delete=False) as f:
            config_path = Path(f.name)

        try:
            assert SyntheticDatasetCreator.is_supported(config_path, None)
        finally:
            config_path.unlink()

    @pytest.mark.sanity
    def test_is_supported_path_yaml_file(self):
        """Test is_supported with YAML file paths.

        ### WRITTEN BY AI ###
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)

        try:
            assert SyntheticDatasetCreator.is_supported(yaml_path, None)
        finally:
            yaml_path.unlink()

    @pytest.mark.smoke
    def test_is_supported_json_string(self):
        """Test is_supported with JSON string.

        ### WRITTEN BY AI ###
        """
        json_str = '{"prompt_tokens": 50, "output_tokens": 25}'
        assert SyntheticDatasetCreator.is_supported(json_str, None)

    @pytest.mark.smoke
    def test_is_supported_key_value_string(self):
        """Test is_supported with key-value string.

        ### WRITTEN BY AI ###
        """
        kv_str = "prompt_tokens=50,output_tokens=25"
        assert SyntheticDatasetCreator.is_supported(kv_str, None)

    @pytest.mark.sanity
    def test_is_supported_config_filename_string(self):
        """Test is_supported with config filename string.

        ### WRITTEN BY AI ###
        """
        assert SyntheticDatasetCreator.is_supported("config.yaml", None)
        assert SyntheticDatasetCreator.is_supported("settings.config", None)

    @pytest.mark.sanity
    def test_is_not_supported_regular_string(self):
        """Test is_supported returns False for regular strings.

        ### WRITTEN BY AI ###
        """
        assert not SyntheticDatasetCreator.is_supported("regular string", None)
        assert not SyntheticDatasetCreator.is_supported("single=pair", None)

    @pytest.mark.regression
    def test_is_not_supported_non_existent_path(self):
        """Test is_supported returns False for non-existent paths.

        ### WRITTEN BY AI ###
        """
        non_existent_path = Path("/non/existent/path.config")
        assert not SyntheticDatasetCreator.is_supported(non_existent_path, None)

    @pytest.mark.regression
    def test_is_not_supported_other_types(self):
        """Test is_supported returns False for other data types.

        ### WRITTEN BY AI ###
        """
        assert not SyntheticDatasetCreator.is_supported(123, None)
        assert not SyntheticDatasetCreator.is_supported(["list"], None)
        assert not SyntheticDatasetCreator.is_supported({"dict": "value"}, None)

    @pytest.mark.smoke
    @patch("guidellm.dataset.synthetic.check_load_processor")
    @patch("guidellm.dataset.synthetic.SyntheticTextItemsGenerator")
    @patch("guidellm.dataset.synthetic.Dataset")
    def test_handle_create_basic(
        self, mock_dataset, mock_generator, mock_check_processor
    ):
        """Test handle_create basic functionality.

        ### WRITTEN BY AI ###
        """
        # Setup mocks
        mock_processor = Mock()
        mock_check_processor.return_value = mock_processor

        mock_generator_instance = Mock()
        mock_generator_instance.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "prompt": "test",
                        "prompt_tokens_count": 10,
                        "output_tokens_count": 5,
                    }
                ]
            )
        )
        mock_generator.return_value = mock_generator_instance

        mock_dataset_instance = Mock()
        mock_dataset.from_list.return_value = mock_dataset_instance

        # Test
        data = '{"prompt_tokens": 50, "output_tokens": 25}'
        result = SyntheticDatasetCreator.handle_create(
            data=data,
            data_args=None,
            processor="gpt2",
            processor_args=None,
            random_seed=42,
        )

        # Verify
        mock_check_processor.assert_called_once_with(
            "gpt2",
            None,
            error_msg="Processor/tokenizer required for synthetic dataset generation.",
        )
        mock_generator.assert_called_once()
        mock_dataset.from_list.assert_called_once()
        assert result == mock_dataset_instance

    @pytest.mark.sanity
    @patch("guidellm.dataset.synthetic.check_load_processor")
    def test_handle_create_processor_required(self, mock_check_processor):
        """Test handle_create requires processor.

        ### WRITTEN BY AI ###
        """
        mock_check_processor.side_effect = ValueError("Processor required")

        data = '{"prompt_tokens": 50, "output_tokens": 25}'

        with pytest.raises(ValueError, match="Processor required"):
            SyntheticDatasetCreator.handle_create(
                data=data,
                data_args=None,
                processor=None,
                processor_args=None,
                random_seed=42,
            )

    @pytest.mark.regression
    @patch("guidellm.dataset.synthetic.check_load_processor")
    @patch("guidellm.dataset.synthetic.SyntheticTextItemsGenerator")
    @patch("guidellm.dataset.synthetic.Dataset")
    def test_handle_create_with_data_args(
        self, mock_dataset, mock_generator, mock_check_processor
    ):
        """Test handle_create with data_args.

        ### WRITTEN BY AI ###
        """
        # Setup mocks
        mock_processor = Mock()
        mock_check_processor.return_value = mock_processor

        mock_generator_instance = Mock()
        mock_generator_instance.__iter__ = Mock(return_value=iter([]))
        mock_generator.return_value = mock_generator_instance

        mock_dataset_instance = Mock()
        mock_dataset.from_list.return_value = mock_dataset_instance

        # Test with data_args
        data = '{"prompt_tokens": 50, "output_tokens": 25}'
        data_args = {"features": "custom_features"}

        SyntheticDatasetCreator.handle_create(
            data=data,
            data_args=data_args,
            processor="gpt2",
            processor_args=None,
            random_seed=42,
        )

        # Verify data_args are passed to Dataset.from_list
        mock_dataset.from_list.assert_called_once_with([], **data_args)

    @pytest.mark.sanity
    def test_extract_args_column_mappings_empty(self):
        """Test extract_args_column_mappings with empty data_args.

        ### WRITTEN BY AI ###
        """
        result = SyntheticDatasetCreator.extract_args_column_mappings(None)

        expected = {
            "prompt_column": "prompt",
            "prompt_tokens_count_column": "prompt_tokens_count",
            "output_tokens_count_column": "output_tokens_count",
        }
        assert result == expected

    @pytest.mark.regression
    def test_extract_args_column_mappings_with_parent_mappings(self):
        """Test extract_args_column_mappings rejects column mappings.

        ### WRITTEN BY AI ###
        """
        with (
            patch.object(
                SyntheticDatasetCreator.__bases__[0],
                "extract_args_column_mappings",
                return_value={"prompt_column": "custom_prompt"},
            ),
            pytest.raises(ValueError, match="Column mappings are not supported"),
        ):
            SyntheticDatasetCreator.extract_args_column_mappings({"some": "args"})

    @pytest.mark.regression
    def test_extract_args_column_mappings_no_parent_mappings(self):
        """Test extract_args_column_mappings with no parent mappings.

        ### WRITTEN BY AI ###
        """
        with patch.object(
            SyntheticDatasetCreator.__bases__[0],
            "extract_args_column_mappings",
            return_value={},
        ):
            result = SyntheticDatasetCreator.extract_args_column_mappings(
                {"some": "args"}
            )

            expected = {
                "prompt_column": "prompt",
                "prompt_tokens_count_column": "prompt_tokens_count",
                "output_tokens_count_column": "output_tokens_count",
            }
            assert result == expected
