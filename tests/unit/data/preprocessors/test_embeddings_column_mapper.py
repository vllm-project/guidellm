"""Unit tests for EmbeddingsColumnMapper preprocessor."""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.preprocessors.embeddings_mapper import EmbeddingsColumnMapper


class TestEmbeddingsColumnMapper:
    """Tests for EmbeddingsColumnMapper preprocessor."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that mapper is properly registered."""
        from guidellm.data.preprocessors.preprocessor import PreprocessorRegistry

        assert "embeddings_column_mapper" in PreprocessorRegistry.registry
        assert (
            PreprocessorRegistry.registry["embeddings_column_mapper"]
            == EmbeddingsColumnMapper
        )

    @pytest.mark.sanity
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = EmbeddingsColumnMapper()
        assert mapper is not None
        assert mapper.input_mappings is None
        assert mapper.datasets_column_mappings is None

    @pytest.mark.sanity
    def test_initialization_with_mappings(self):
        """Test mapper initialization with custom mappings."""
        mappings = {"text_column": "content"}
        mapper = EmbeddingsColumnMapper(column_mappings=mappings)
        assert mapper.input_mappings == mappings

    @pytest.mark.sanity
    def test_defaults_contain_common_text_columns(self):
        """Test that defaults include common text column names."""
        expected_columns = [
            "text",
            "input",
            "content",
            "prompt",
            "prompt_0",
            "sentence",
            "document",
            "passage",
            "query",
            "body",
            "message",
        ]

        for col in expected_columns:
            assert col in EmbeddingsColumnMapper.defaults["text_column"]

    @pytest.mark.sanity
    def test_datasets_default_mappings_single_dataset(self):
        """Test auto-detection with single dataset."""
        dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        datasets = [dataset]

        mappings = EmbeddingsColumnMapper.datasets_default_mappings(datasets)

        assert "text_column" in mappings
        assert len(mappings["text_column"]) == 1
        assert mappings["text_column"][0] == (0, "text")

    @pytest.mark.sanity
    def test_datasets_default_mappings_case_insensitive(self):
        """Test auto-detection with different case variations."""
        # Test uppercase
        dataset_upper = Dataset.from_dict({"TEXT": ["Hello"]})
        mappings_upper = EmbeddingsColumnMapper.datasets_default_mappings(
            [dataset_upper]
        )
        assert mappings_upper["text_column"][0][1] == "TEXT"

        # Test capitalized
        dataset_cap = Dataset.from_dict({"Text": ["World"]})
        mappings_cap = EmbeddingsColumnMapper.datasets_default_mappings([dataset_cap])
        assert mappings_cap["text_column"][0][1] == "Text"

    @pytest.mark.sanity
    def test_datasets_default_mappings_multiple_datasets(self):
        """Test auto-detection with multiple datasets."""
        dataset1 = Dataset.from_dict({"text": ["Hello"]})
        dataset2 = Dataset.from_dict({"content": ["World"]})
        datasets = [dataset1, dataset2]

        mappings = EmbeddingsColumnMapper.datasets_default_mappings(datasets)

        # Should find text column in first dataset
        assert "text_column" in mappings
        assert len(mappings["text_column"]) == 1
        assert mappings["text_column"][0] == (0, "text")

    @pytest.mark.sanity
    def test_datasets_default_mappings_prefers_text_over_others(self):
        """Test that 'text' column is preferred when multiple options exist."""
        dataset = Dataset.from_dict(
            {
                "text": ["Hello"],
                "content": ["World"],
                "prompt": ["Test"],
            }
        )

        mappings = EmbeddingsColumnMapper.datasets_default_mappings([dataset])

        # Should prefer 'text' since it appears first in defaults
        assert mappings["text_column"][0][1] == "text"

    @pytest.mark.sanity
    def test_datasets_default_mappings_empty_datasets(self):
        """Test auto-detection with empty dataset list."""
        mappings = EmbeddingsColumnMapper.datasets_default_mappings([])
        assert mappings == {}

    @pytest.mark.sanity
    def test_datasets_mappings_single_column(self):
        """Test user-specified mapping for single column."""
        dataset = Dataset.from_dict({"my_text": ["Hello", "World"]})
        datasets = [dataset]
        input_mappings = {"text_column": "my_text"}

        mappings = EmbeddingsColumnMapper.datasets_mappings(datasets, input_mappings)

        assert "text_column" in mappings
        assert mappings["text_column"] == [(0, "my_text")]

    @pytest.mark.sanity
    def test_datasets_mappings_multiple_columns(self):
        """Test user-specified mapping for multiple columns from one dataset."""
        dataset = Dataset.from_dict(
            {
                "title": ["Title"],
                "body": ["Content"],
            }
        )
        datasets = [dataset]
        input_mappings = {"text_column": ["title", "body"]}

        mappings = EmbeddingsColumnMapper.datasets_mappings(datasets, input_mappings)

        assert mappings["text_column"] == [(0, "title"), (0, "body")]

    @pytest.mark.sanity
    def test_datasets_mappings_with_dataset_prefix(self):
        """Test mapping with dataset index prefix."""
        dataset1 = Dataset.from_dict({"text1": ["Hello"]})
        dataset2 = Dataset.from_dict({"text2": ["World"]})
        datasets = [dataset1, dataset2]
        input_mappings = {"text_column": ["0.text1", "1.text2"]}

        mappings = EmbeddingsColumnMapper.datasets_mappings(datasets, input_mappings)

        assert mappings["text_column"] == [(0, "text1"), (1, "text2")]

    @pytest.mark.regression
    def test_datasets_mappings_invalid_dataset_index(self):
        """Test that invalid dataset index raises error."""
        dataset = Dataset.from_dict({"text": ["Hello"]})
        datasets = [dataset]
        input_mappings = {"text_column": "5.text"}  # Dataset 5 doesn't exist

        with pytest.raises(ValueError, match="Dataset '5.text' not found"):
            EmbeddingsColumnMapper.datasets_mappings(datasets, input_mappings)

    @pytest.mark.regression
    def test_datasets_mappings_invalid_column_name(self):
        """Test that invalid column name raises error."""
        dataset = Dataset.from_dict({"text": ["Hello"]})
        datasets = [dataset]
        input_mappings = {"text_column": "nonexistent"}

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            EmbeddingsColumnMapper.datasets_mappings(datasets, input_mappings)

    @pytest.mark.sanity
    def test_setup_data_auto_detect(self):
        """Test setup_data with auto-detection."""
        dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        datasets = [dataset]
        mapper = EmbeddingsColumnMapper()

        mapper.setup_data(datasets, [{}])

        assert mapper.datasets_column_mappings is not None
        assert "text_column" in mapper.datasets_column_mappings
        assert mapper.datasets_column_mappings["text_column"] == [(0, "text")]

    @pytest.mark.sanity
    def test_setup_data_with_custom_mappings(self):
        """Test setup_data with custom mappings."""
        dataset = Dataset.from_dict({"my_column": ["Hello"]})
        datasets = [dataset]
        mapper = EmbeddingsColumnMapper(column_mappings={"text_column": "my_column"})

        mapper.setup_data(datasets, [{}])

        assert mapper.datasets_column_mappings["text_column"] == [(0, "my_column")]

    @pytest.mark.sanity
    def test_call_single_text(self):
        """Test __call__ with single text column."""
        dataset = Dataset.from_dict({"text": ["Hello world"]})
        datasets = [dataset]
        mapper = EmbeddingsColumnMapper()
        mapper.setup_data(datasets, [{}])

        # Simulate what the data loader provides
        items = [{"dataset": {"text": "Hello world"}}]
        result = mapper(items)

        # Should return list of turns (single turn for embeddings)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "text_column" in result[0]
        assert result[0]["text_column"] == ["Hello world"]

    @pytest.mark.sanity
    def test_call_multiple_texts(self):
        """Test __call__ with multiple text entries."""
        dataset1 = Dataset.from_dict({"text": ["First"]})
        dataset2 = Dataset.from_dict({"text": ["Second"]})
        datasets = [dataset1, dataset2]
        mapper = EmbeddingsColumnMapper(
            column_mappings={"text_column": ["0.text", "1.text"]}
        )
        mapper.setup_data(datasets, [{}, {}])

        items = [
            {"dataset": {"text": "First"}},
            {"dataset": {"text": "Second"}},
        ]
        result = mapper(items)

        # Should return list with single turn
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text_column"] == ["First", "Second"]

    @pytest.mark.regression
    def test_call_without_setup_raises(self):
        """Test that calling without setup_data raises error."""
        mapper = EmbeddingsColumnMapper()
        items = [{"dataset": {"text": "Hello"}}]

        with pytest.raises(ValueError, match="not setup with data"):
            mapper(items)

    @pytest.mark.regression
    def test_call_returns_list_of_dict_not_defaultdict(self):
        """Test that __call__ returns list of regular dict, not defaultdict."""
        dataset = Dataset.from_dict({"text": ["Hello"]})
        datasets = [dataset]
        mapper = EmbeddingsColumnMapper()
        mapper.setup_data(datasets, [{}])

        items = [{"dataset": {"text": "Hello"}}]
        result = mapper(items)

        # Should be list of dict, not defaultdict
        assert isinstance(result, list)
        assert len(result) == 1
        assert type(result[0]) is dict
        assert result[0] == {"text_column": ["Hello"]}

    @pytest.mark.sanity
    def test_support_for_synthetic_data_format(self):
        """Test that prompt_0 (synthetic data format) is supported."""
        dataset = Dataset.from_dict({"prompt_0": ["Synthetic prompt"]})
        datasets = [dataset]
        mapper = EmbeddingsColumnMapper()

        mapper.setup_data(datasets, [{}])

        assert mapper.datasets_column_mappings["text_column"] == [(0, "prompt_0")]

    @pytest.mark.regression
    def test_multiple_datasets_with_mixed_column_names(self):
        """Test complex scenario with mixed column names across datasets."""
        dataset1 = Dataset.from_dict({"prompt": ["Q1"]})
        dataset2 = Dataset.from_dict({"sentence": ["S1"]})
        dataset3 = Dataset.from_dict({"text": ["T1"]})
        datasets = [dataset1, dataset2, dataset3]

        mapper = EmbeddingsColumnMapper(
            column_mappings={
                "text_column": ["0.prompt", "1.sentence", "2.text"],
            }
        )
        mapper.setup_data(datasets, [{}, {}, {}])

        items = [
            {"dataset": {"prompt": "Q1"}},
            {"dataset": {"sentence": "S1"}},
            {"dataset": {"text": "T1"}},
        ]
        result = mapper(items)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text_column"] == ["Q1", "S1", "T1"]

    @pytest.mark.regression
    def test_ignores_global_kwargs(self):
        """Test that mapper ignores unexpected kwargs."""
        # Should not raise even with extra kwargs
        mapper = EmbeddingsColumnMapper(
            column_mappings={"text_column": "text"},
            some_other_arg="ignored",
            another_arg=123,
        )
        assert mapper.input_mappings == {"text_column": "text"}
