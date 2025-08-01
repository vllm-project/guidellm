"""
Unit tests for guidellm.dataset.synthetic module.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from transformers import AutoTokenizer

from guidellm.dataset.synthetic import (
    SyntheticDatasetConfig,
    SyntheticDatasetCreator,
    SyntheticTextItemsGenerator,
)


class TestSyntheticDatasetConfig:
    """Test cases for SyntheticDatasetConfig class."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = SyntheticDatasetConfig(prompt_tokens=50, output_tokens=20)

        assert config.prompt_tokens == 50
        assert config.output_tokens == 20
        assert config.samples == 1000  # default
        assert config.source == "data:prideandprejudice.txt.gz"  # default
        assert config.prompt_tokens_stdev is None
        assert config.prompt_tokens_min is None
        assert config.prompt_tokens_max is None

    def test_config_creation_with_all_params(self):
        """Test creating config with all parameters specified."""
        config = SyntheticDatasetConfig(
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

    def test_parse_json_string(self):
        json_str = json.dumps(
            {
                "prompt_tokens": 75,
                "output_tokens": 25,
                "samples": 200,
                "source": "test.txt",
            }
        )

        config = SyntheticDatasetConfig.parse_str(json_str)

        assert config.prompt_tokens == 75
        assert config.output_tokens == 25
        assert config.samples == 200
        assert config.source == "test.txt"

    def test_parse_key_value_pairs(self):
        kv_str = "prompt_tokens=80,output_tokens=30,samples=300,source=data.txt"

        config = SyntheticDatasetConfig.parse_str(kv_str)

        assert config.prompt_tokens == 80
        assert config.output_tokens == 30
        assert config.samples == 300
        assert config.source == "data.txt"

    def test_parse_yaml_file(self):
        config_data = {
            "prompt_tokens": 60,
            "output_tokens": 15,
            "samples": 100,
            "source": "yaml_test.txt",
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
        finally:
            Path(yaml_path).unlink()

    def test_parse_config_file(self):
        config_data = {"prompt_tokens": 90, "output_tokens": 35, "samples": 150}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".config", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = SyntheticDatasetConfig.parse_str(config_path)

            assert config.prompt_tokens == 90
            assert config.output_tokens == 35
            assert config.samples == 150
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported data format"):
            SyntheticDatasetConfig.parse_str("invalid_format_string")

    def test_validation_positive_values(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=-1, output_tokens=20)

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=20, output_tokens=-1)

        with pytest.raises(ValueError):
            SyntheticDatasetConfig(prompt_tokens=20, output_tokens=10, samples=-1)


class TestSyntheticTextItemsGenerator:
    @pytest.fixture
    def tokenizer(self):
        """Fixture to provide a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def simple_config(self):
        return SyntheticDatasetConfig(
            prompt_tokens=15,
            output_tokens=10,
            samples=5,
            source=(
                "The quick brown fox jumps over the lazy dog. Machine learning models "
                "require diverse training data."
            ),
        )

    @pytest.fixture
    def complex_config(self):
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
            source=(
                "The quick brown fox jumps over the lazy dog. Machine learning models "
                "require diverse training data."
            ),
        )

    def test_generator_initialization(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        assert generator.config == simple_config
        assert generator.processor == tokenizer
        assert generator.random_seed == 42
        assert generator.request_counter == 0
        assert generator.text_creator is not None

    def test_basic_prompt_generation(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items = list(generator)

        # Verify we get the expected number of items
        assert len(items) == simple_config.samples

        # Verify each item has the required keys
        for item in items:
            assert "prompt" in item
            assert "prompt_tokens_count" in item
            assert "output_tokens_count" in item

            # Verify types
            assert isinstance(item["prompt"], str)
            assert isinstance(item["prompt_tokens_count"], int)
            assert isinstance(item["output_tokens_count"], int)

            # Verify non-empty prompt
            assert len(item["prompt"]) > 0

    def test_unique_prefix_generation(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items = list(generator)
        prompts = [str(item["prompt"]) for item in items]

        # Verify each prompt starts with a unique request ID
        for i, prompt in enumerate(prompts, 1):
            assert prompt.startswith(f"{i}: "), (
                f"Prompt {i} should start with '{i}: ', got '{prompt[:10]}...'"
            )

        # Verify no two prompts are identical
        assert len(set(prompts)) == len(prompts), "All prompts should be unique"

    def test_prefix_caching_prevention(self, simple_config, tokenizer):
        """Test that prefix caching is effectively prevented."""
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items = list(generator)
        prompts = [str(item["prompt"]) for item in items]

        # Test that no prompt is a prefix of another
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts):
                if i != j:
                    assert not prompt1.startswith(prompt2), (
                        f"Prompt {i} starts with prompt {j}"
                    )
                    assert not prompt2.startswith(prompt1), (
                        f"Prompt {j} starts with prompt {i}"
                    )

        # Test that first characters are all different
        first_chars = [prompt[0] for prompt in prompts]
        assert len(set(first_chars)) == len(first_chars), (
            "First characters should all be different"
        )

    def test_token_count_accuracy(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items = list(generator)

        for item in items:
            actual_tokens = len(tokenizer.tokenize(str(item["prompt"])))
            target_tokens = int(item["prompt_tokens_count"])

            # Allow small variance due to tokenization differences
            assert abs(actual_tokens - target_tokens) <= 2, (
                f"Token count mismatch: expected ~{target_tokens}, got {actual_tokens}"
            )

    def test_variance_in_token_counts(self, complex_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            complex_config, tokenizer, random_seed=42
        )

        items = list(generator)

        prompt_token_counts = [int(item["prompt_tokens_count"]) for item in items]
        output_token_counts = [int(item["output_tokens_count"]) for item in items]

        # With variance, we should see different token counts
        assert len(set(prompt_token_counts)) > 1, (
            "Should have variance in prompt token counts"
        )
        assert len(set(output_token_counts)) > 1, (
            "Should have variance in output token counts"
        )

        # Verify bounds are respected
        assert all(
            complex_config.prompt_tokens_min
            <= count
            <= complex_config.prompt_tokens_max
            for count in prompt_token_counts
        ), "Prompt tokens should be within bounds"
        assert all(
            complex_config.output_tokens_min
            <= count
            <= complex_config.output_tokens_max
            for count in output_token_counts
        ), "Output tokens should be within bounds"

    def test_reproducibility_with_same_seed(self, simple_config, tokenizer):
        generator1 = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )
        generator2 = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items1 = list(generator1)
        items2 = list(generator2)

        # Results should be identical with same seed
        assert len(items1) == len(items2)
        for item1, item2 in zip(items1, items2):
            assert str(item1["prompt"]) == str(item2["prompt"])
            assert int(item1["prompt_tokens_count"]) == int(
                item2["prompt_tokens_count"]
            )
            assert int(item1["output_tokens_count"]) == int(
                item2["output_tokens_count"]
            )

    def test_different_seeds_produce_different_results(self, simple_config, tokenizer):
        """Test that different seeds produce different results."""
        generator1 = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )
        generator2 = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=123
        )

        items1 = list(generator1)
        items2 = list(generator2)

        # Results should be different with different seeds
        prompts1 = [str(item["prompt"]) for item in items1]
        prompts2 = [str(item["prompt"]) for item in items2]

        different_content = False
        for p1, p2 in zip(prompts1, prompts2):
            # Remove the prefix and compare content
            content1 = p1.split(": ", 1)[1] if ": " in p1 else p1
            content2 = p2.split(": ", 1)[1] if ": " in p2 else p2
            if content1 != content2:
                different_content = True
                break

        assert different_content, "Different seeds should produce different content"

    def test_create_prompt_method_directly(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        # Test normal prompt creation
        prompt = generator._create_prompt(10, 0, 5)
        assert prompt.startswith("5: "), "Prompt should start with request ID"

        actual_tokens = len(tokenizer.tokenize(prompt))
        assert abs(actual_tokens - 10) <= 1, (
            "Token count should be approximately correct"
        )

        # Test empty prompt
        empty_prompt = generator._create_prompt(0, 0, 3)
        assert empty_prompt == "3: ", "Empty prompt should just be the prefix"

    def test_request_counter_increments_correctly(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        # Initially should be 0
        assert generator.request_counter == 0

        # Get items one by one and check counter
        items = []
        for i, item in enumerate(generator, 1):
            items.append(item)
            # Counter should increment for each item
            assert generator.request_counter == i
            if i >= 3:  # Just test first 3
                break

        # Verify prompts have correct prefixes
        for i, item in enumerate(items, 1):
            assert str(item["prompt"]).startswith(f"{i}: ")

    def test_prefix_format_consistency(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        items = list(generator)

        for i, item in enumerate(items, 1):
            prompt = str(item["prompt"])

            # Should start with number followed by colon and space
            assert prompt.startswith(f"{i}: "), f"Prompt should start with '{i}: '"

            # Should be able to split on ': ' to get request ID and content
            parts = prompt.split(": ", 1)
            assert len(parts) == 2, "Prompt should have exactly one ': ' separator"
            assert parts[0] == str(i), f"First part should be request ID {i}"

            # Content part should not be empty (unless it's a zero-token prompt)
            if int(item["prompt_tokens_count"]) > 0:
                assert len(parts[1]) > 0, (
                    "Content part should not be empty for non-zero token prompts"
                )

    def test_binary_search_token_accuracy(self, simple_config, tokenizer):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        # Test various token counts
        test_cases = [5, 10, 15, 20, 25]

        for target_tokens in test_cases:
            prompt = generator._create_prompt(target_tokens, 0, 999)
            actual_tokens = len(tokenizer.tokenize(prompt))

            # Should be very close to target
            # (allowing for small tokenization differences)
            assert abs(actual_tokens - target_tokens) <= 1, (
                f"Target: {target_tokens}, Actual: {actual_tokens}, "
                f"Prompt: '{prompt[:50]}...'"
            )

    def test_vllm_cache_simulation_comprehensive(self, simple_config, tokenizer):
        # Use larger sample for more thorough testing
        config = SyntheticDatasetConfig(
            prompt_tokens=20, output_tokens=10, samples=20, source=simple_config.source
        )

        generator = SyntheticTextItemsGenerator(config, tokenizer, random_seed=42)
        items = list(generator)
        prompts = [str(item["prompt"]) for item in items]

        # Simulate vLLM cache with different granularities
        cache_scenarios = [
            ("Character-level", 1),
            ("Token-level", 4),
            ("Word-level", 10),
        ]

        for scenario_name, granularity in cache_scenarios:
            cache_hits = 0
            total_comparisons = 0

            for i, prompt1 in enumerate(prompts):
                for _, prompt2 in enumerate(prompts[i + 1 :], i + 1):
                    total_comparisons += 1

                    # Check for common prefix at specified granularity
                    min_len = min(len(prompt1), len(prompt2))
                    common_prefix_len = 0

                    for k in range(0, min_len, granularity):
                        chunk1 = prompt1[k : k + granularity]
                        chunk2 = prompt2[k : k + granularity]
                        if chunk1 == chunk2:
                            common_prefix_len += len(chunk1)
                        else:
                            break

                    # If meaningful common prefix exists, it's a cache hit
                    if common_prefix_len > granularity:
                        cache_hits += 1

            cache_hit_rate = (
                (cache_hits / total_comparisons) * 100 if total_comparisons > 0 else 0
            )

            # All scenarios should have 0% cache hit rate
            assert cache_hit_rate == 0.0, (
                f"{scenario_name} caching: Expected 0% hit rate, "
                f"got {cache_hit_rate:.1f}%"
            )

    def test_edge_case_very_short_prompts(self, tokenizer):
        config = SyntheticDatasetConfig(
            prompt_tokens=1,
            output_tokens=5,
            samples=5,
            source="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        )

        generator = SyntheticTextItemsGenerator(config, tokenizer, random_seed=42)
        items = list(generator)

        for i, item in enumerate(items, 1):
            # Even very short prompts should have unique prefixes
            assert str(item["prompt"]).startswith(f"{i}: ")

            # Should have at least the prefix
            assert len(str(item["prompt"])) >= len(f"{i}: ")

    def test_create_prompt_method_signature_and_documentation(
        self, simple_config, tokenizer
    ):
        generator = SyntheticTextItemsGenerator(
            simple_config, tokenizer, random_seed=42
        )

        # Test method exists and is callable
        assert hasattr(generator, "_create_prompt")
        assert callable(generator._create_prompt)

        # Test method signature by calling with expected parameters
        prompt = generator._create_prompt(prompt_tokens=10, start_index=0, request_id=1)

        # Should return a string
        assert isinstance(prompt, str)

        # Should start with the request ID
        assert prompt.startswith("1: ")

        # Test that docstring exists and mentions key concepts
        docstring = generator._create_prompt.__doc__
        assert docstring is not None
        assert "prefix" in docstring.lower()
        assert "cache" in docstring.lower() or "caching" in docstring.lower()
        assert "request_id" in docstring


class TestIntegration:
    """Integration tests for the complete synthetic dataset workflow."""

    @pytest.fixture
    def tokenizer(self):
        """Fixture to provide a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def test_end_to_end_workflow(self, tokenizer):
        """Test the complete workflow from config to dataset."""
        # Create configuration
        config_dict = {
            "prompt_tokens": 20,
            "output_tokens": 15,
            "samples": 10,
            "source": (
                "The quick brown fox jumps over the lazy dog. Machine learning models "
                "require diverse training data to perform well across different tasks "
                "and domains."
            ),
        }

        config_str = json.dumps(config_dict)

        # Create dataset
        dataset = SyntheticDatasetCreator.handle_create(
            data=config_str,
            data_args=None,
            processor=tokenizer,
            processor_args=None,
            random_seed=42,
        )

        # Verify dataset properties
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 10

        # Verify all prompts are unique and have correct prefixes
        prompts = dataset["prompt"]
        for i, prompt in enumerate(prompts, 1):
            assert prompt.startswith(f"{i}: "), f"Prompt {i} should start with '{i}: '"

        # Verify no cache hits would occur
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts):
                if i != j:
                    assert not prompt1.startswith(prompt2)
                    assert not prompt2.startswith(prompt1)

        # Verify token counts are reasonable
        for i, row in enumerate(dataset):
            actual_tokens = len(tokenizer.tokenize(row["prompt"]))
            target_tokens = row["prompt_tokens_count"]
            assert abs(actual_tokens - target_tokens) <= 2, (
                f"Row {i}: token count mismatch"
            )

    def test_cache_prevention_effectiveness(self, tokenizer):
        """Test that the cache prevention is effective across larger datasets."""
        config = SyntheticDatasetConfig(
            prompt_tokens=25,
            output_tokens=20,
            samples=50,
            source=(
                "The quick brown fox jumps over the lazy dog. Machine learning models "
                "require diverse training data to perform well across different tasks "
                "and domains. Natural language processing has advanced significantly "
                "in recent years."
            ),
        )

        generator = SyntheticTextItemsGenerator(config, tokenizer, random_seed=42)
        items = list(generator)
        prompts = [str(item["prompt"]) for item in items]

        prefixes = [prompt.split(": ", 1)[0] for prompt in prompts]
        assert len(set(prefixes)) == len(prefixes), "All prefixes should be unique"

        for i, prefix in enumerate(prefixes, 1):
            assert prefix == str(i), f"Prefix should be '{i}', got '{prefix}'"

        # Test that no prompt starts with the same prefix as another
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts):
                if i != j:
                    prefix1 = prompt1.split(": ", 1)[0] + ": "
                    prefix2 = prompt2.split(": ", 1)[0] + ": "
                    assert not prompt1.startswith(prefix2), (
                        f"Prompt {i} starts with prefix from prompt {j}"
                    )
                    assert not prompt2.startswith(prefix1), (
                        f"Prompt {j} starts with prefix from prompt {i}"
                    )

        # Test that all prompts are unique
        assert len(set(prompts)) == len(prompts), "All prompts should be unique"
