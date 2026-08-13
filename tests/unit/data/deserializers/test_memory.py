"""
Unit tests for guidellm.data.deserializers.memory module.

### WRITTEN BY AI ###
"""

import pytest
from datasets import Dataset

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
)
from guidellm.data.deserializers.memory import (
    InMemoryDictDataArgs,
    InMemoryDictDatasetDeserializer,
    InMemoryDictListDataArgs,
    InMemoryDictListDatasetDeserializer,
    InMemoryItemListDataArgs,
    InMemoryItemListDatasetDeserializer,
)


@pytest.fixture
def processor_factory():
    return None  # Dummy processor factory for testing


###################
# Tests dict in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_dict_deserializer_success(processor_factory):
    """Dict deserializer returns correct Dataset.

    ### WRITTEN BY AI ###
    """
    deserializer = InMemoryDictDatasetDeserializer()
    config = InMemoryDictDataArgs(data={"text": ["hello", "world"], "id": [1, 2]})

    dataset = deserializer(
        config=config, processor_factory=processor_factory, random_seed=42
    )

    assert isinstance(dataset, Dataset)
    assert dataset["text"] == ["hello", "world"]
    assert dataset["id"] == [1, 2]
    assert len(dataset) == 2


@pytest.mark.smoke
def test_in_memory_dict_deserializer_empty_dict(processor_factory):
    """Dict deserializer raises DataNotSupportedError for empty dict.

    ### WRITTEN BY AI ###
    """
    deserializer = InMemoryDictDatasetDeserializer()
    config = InMemoryDictDataArgs(data={})

    with pytest.raises(DataNotSupportedError):
        deserializer(config=config, processor_factory=processor_factory, random_seed=42)


@pytest.mark.smoke
def test_in_memory_dict_deserializer_list_length_mismatch(processor_factory):
    """Dict deserializer raises DataNotSupportedError for mismatched list lengths.

    ### WRITTEN BY AI ###
    """
    deserializer = InMemoryDictDatasetDeserializer()
    config = InMemoryDictDataArgs(
        data={"text": ["hello", "world"], "id": [1]}  # different lengths
    )

    with pytest.raises(DataNotSupportedError):
        deserializer(config=config, processor_factory=processor_factory, random_seed=42)


###################
# Tests dict list in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_dict_list_deserializer_success(processor_factory):
    """Dict list deserializer returns correct Dataset.

    ### WRITTEN BY AI ###
    """
    data = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
        {"id": 3, "text": "guidellm"},
    ]
    config = InMemoryDictListDataArgs(data=data)
    deserializer = InMemoryDictListDatasetDeserializer()

    dataset = deserializer(
        config=config, processor_factory=processor_factory, random_seed=42
    )

    assert isinstance(dataset, Dataset)
    assert dataset["id"] == [1, 2, 3]
    assert dataset["text"] == ["hello", "world", "guidellm"]
    assert len(dataset) == 3


@pytest.mark.smoke
def test_in_memory_dict_list_deserializer_key_mismatch(processor_factory):
    """Dict list deserializer raises DataNotSupportedError for mismatched keys.

    ### WRITTEN BY AI ###
    """
    deserializer = InMemoryDictListDatasetDeserializer()
    config = InMemoryDictListDataArgs(
        data=[
            {"id": 1, "text": "hello"},
            {"id": 2, "msg": "world"},  # key mismatch
        ]
    )

    with pytest.raises(DataNotSupportedError):
        deserializer(config=config, processor_factory=processor_factory, random_seed=42)


###################
# Tests list in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_item_list_deserializer_success(processor_factory):
    """Item list deserializer returns correct Dataset.

    ### WRITTEN BY AI ###
    """
    data = ["a", "b", "c"]
    config = InMemoryItemListDataArgs(data=data)
    deserializer = InMemoryItemListDatasetDeserializer()

    dataset = deserializer(
        config=config, processor_factory=processor_factory, random_seed=42
    )

    assert isinstance(dataset, Dataset)
    assert dataset["data"] == data
    assert len(dataset) == 3


@pytest.mark.smoke
def test_in_memory_item_list_custom_column_name(processor_factory):
    """Item list deserializer respects custom column_name.

    ### WRITTEN BY AI ###
    """
    config = InMemoryItemListDataArgs(data=[1, 2, 3], column_name="numbers")
    deserializer = InMemoryItemListDatasetDeserializer()

    dataset = deserializer(
        config=config, processor_factory=processor_factory, random_seed=123
    )

    assert list(dataset.column_names) == ["numbers"]
    assert dataset["numbers"] == [1, 2, 3]
