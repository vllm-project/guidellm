import pytest
from datasets import Dataset

from guidellm.data.deserializers.huggingface import (
    HuggingFaceDatasetDeserializer,
)


@pytest.fixture
def processor_factory():
    return None


@pytest.fixture
def deserializer():
    return HuggingFaceDatasetDeserializer()


def test_hf_dataset_direct_return(deserializer, processor_factory):
    # build one simple HF dataset
    data = Dataset.from_dict({"text": ["hello", "world"]})
    result = deserializer(data, processor_factory, random_seed=42)
    assert result is data, "return original Dataset object"


def test_local_hf_directory_dataset(deserializer, processor_factory, tmp_path):
    # --- 1. build one simple HF dataset ---
    dataset = Dataset.from_dict({"id": [1, 2], "text": ["a", "b"]})
    #  --- 2.  Save to a local directory ---
    dataset_dir = tmp_path / "local_hf_dataset"
    dataset.save_to_disk(dataset_dir)

    # --- 3. call HF DatasetDeserializer  ---
    result = deserializer(
        dataset_dir,
        processor_factory,
        random_seed=123,
    )

    # --- 4. assertion ---
    assert isinstance(result, Dataset)
    assert result["text"] == ["a", "b"]


@pytest.mark.parametrize(
    "internal_ds_name",
    [
        "mnist",
        "imdb",
    ],
)
def test_hf_internal_dataset(deserializer, processor_factory, internal_ds_name):
    result = deserializer(internal_ds_name, processor_factory, random_seed=42)

    assert isinstance(result, (Dataset | dict)), "HF dataset loading failed"
    assert "train" in result or isinstance(result, Dataset), (
        "Expected 'train' split in the loaded dataset"
    )
    assert "test" in result or isinstance(result, Dataset), (
        "Expected 'test' split in the loaded dataset"
    )
