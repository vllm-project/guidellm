from unittest.mock import MagicMock, patch

import pytest

from guidellm.dataset.file import FileDatasetCreator
from guidellm.dataset.hf_datasets import HFDatasetsCreator
from guidellm.dataset.in_memory import InMemoryDatasetCreator
from guidellm.dataset.synthetic import SyntheticDatasetCreator
from guidellm.presentation.data_models import Bucket, Dataset
from tests.unit.mock_benchmark import mock_generative_benchmark


@pytest.mark.smoke
def test_bucket_from_data():
    buckets, bucket_width = Bucket.from_data([8, 8, 8, 8, 8, 8], 1)
    assert len(buckets) == 1
    assert buckets[0].value == 8.0
    assert buckets[0].count == 6
    assert bucket_width == 1

    buckets, bucket_width = Bucket.from_data([8, 8, 8, 8, 8, 7], 1)
    assert len(buckets) == 2
    assert buckets[0].value == 7.0
    assert buckets[0].count == 1
    assert buckets[1].value == 8.0
    assert buckets[1].count == 5
    assert bucket_width == 1


def mock_processor(cls):
    return mock_generative_benchmark().request_loader.processor


def new_handle_create(cls, *args, **kwargs):
    return MagicMock()


def new_extract_dataset_name(cls, *args, **kwargs):
    return "data:prideandprejudice.txt.gz"


@pytest.mark.smoke
def test_dataset_from_data_uses_extracted_dataset_name():
    mock_benchmark = mock_generative_benchmark()
    with (
        patch.object(SyntheticDatasetCreator, "handle_create", new=new_handle_create),
        patch.object(
            SyntheticDatasetCreator,
            "extract_dataset_name",
            new=new_extract_dataset_name,
        ),
    ):
        dataset = Dataset.from_data(mock_benchmark.request_loader)
        assert dataset.name == "data:prideandprejudice.txt.gz"


def new_is_supported(cls, *args, **kwargs):
    return True


@pytest.mark.smoke
def test_dataset_from_data_with_in_memory_dataset():
    mock_benchmark = mock_generative_benchmark()
    with patch.object(InMemoryDatasetCreator, "is_supported", new=new_is_supported):
        dataset = Dataset.from_data(mock_benchmark.request_loader)
        assert dataset.name == "In-memory"


def hardcoded_isnt_supported(cls, *args, **kwargs):
    return False


def new_extract_dataset_name_none(cls, *args, **kwargs):
    return None


@pytest.mark.smoke
def test_dataset_from_data_with_synthetic_dataset():
    mock_benchmark = mock_generative_benchmark()
    with (
        patch.object(SyntheticDatasetCreator, "handle_create", new=new_handle_create),
        patch.object(
            InMemoryDatasetCreator, "is_supported", new=hardcoded_isnt_supported
        ),
        patch.object(SyntheticDatasetCreator, "is_supported", new=new_is_supported),
        patch.object(
            SyntheticDatasetCreator,
            "extract_dataset_name",
            new=new_extract_dataset_name_none,
        ),
    ):
        dataset = Dataset.from_data(mock_benchmark.request_loader)
        assert dataset.name == "data:prideandprejudice.txt.gz"


@pytest.mark.smoke
def test_dataset_from_data_with_file_dataset():
    mock_benchmark = mock_generative_benchmark()
    mock_benchmark.request_loader.data = "dataset.yaml"
    with (
        patch.object(FileDatasetCreator, "handle_create", new=new_handle_create),
        patch.object(
            InMemoryDatasetCreator, "is_supported", new=hardcoded_isnt_supported
        ),
        patch.object(
            SyntheticDatasetCreator, "is_supported", new=hardcoded_isnt_supported
        ),
        patch.object(FileDatasetCreator, "is_supported", new=new_is_supported),
        patch.object(
            FileDatasetCreator,
            "extract_dataset_name",
            new=new_extract_dataset_name_none,
        ),
    ):
        dataset = Dataset.from_data(mock_benchmark.request_loader)
        assert dataset.name == "dataset.yaml"


@pytest.mark.smoke
def test_dataset_from_data_with_hf_dataset():
    mock_benchmark = mock_generative_benchmark()
    mock_benchmark.request_loader.data = "openai/gsm8k"
    with (
        patch.object(HFDatasetsCreator, "handle_create", new=new_handle_create),
        patch.object(
            InMemoryDatasetCreator, "is_supported", new=hardcoded_isnt_supported
        ),
        patch.object(
            SyntheticDatasetCreator, "is_supported", new=hardcoded_isnt_supported
        ),
        patch.object(FileDatasetCreator, "is_supported", new=hardcoded_isnt_supported),
        patch.object(HFDatasetsCreator, "is_supported", new=new_is_supported),
        patch.object(
            HFDatasetsCreator, "extract_dataset_name", new=new_extract_dataset_name_none
        ),
    ):
        dataset = Dataset.from_data(mock_benchmark.request_loader)
        assert dataset.name == "openai/gsm8k"
