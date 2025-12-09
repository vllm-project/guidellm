import pytest
from datasets import Dataset

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
)
from guidellm.data.deserializers.memory import (
    InMemoryCsvDatasetDeserializer,
    InMemoryDictDatasetDeserializer,
    InMemoryDictListDatasetDeserializer,
    InMemoryItemListDatasetDeserializer,
    InMemoryJsonStrDatasetDeserializer,
)


@pytest.fixture
def processor_factory():
    return None  # Dummy processor factory for testing


###################
# Tests dict in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_dict_deserializer_success(processor_factory):
    deserializer = InMemoryDictDatasetDeserializer()

    data = {
        "text": ["hello", "world"],
        "id": [1, 2],
    }

    dataset = deserializer(
        data=data,
        processor_factory=processor_factory,
        random_seed=42,
    )

    assert isinstance(dataset, Dataset)
    assert dataset["text"] == ["hello", "world"]
    assert dataset["id"] == [1, 2]
    assert len(dataset) == 2


@pytest.mark.smoke
def test_in_memory_dict_deserializer_invalid_not_dict(processor_factory):
    deserializer = InMemoryDictDatasetDeserializer()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data="not a dict",
            processor_factory=processor_factory,
            random_seed=42,
        )


@pytest.mark.smoke
def test_in_memory_dict_deserializer_empty_dict(processor_factory):
    deserializer = InMemoryDictDatasetDeserializer()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data={},
            processor_factory=processor_factory,
            random_seed=42,
        )


@pytest.mark.smoke
def test_in_memory_dict_deserializer_value_not_list(processor_factory):
    deserializer = InMemoryDictDatasetDeserializer()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data={"text": "hello"},  # value 不是 list
            processor_factory=processor_factory,
            random_seed=42,
        )


@pytest.mark.smoke
def test_in_memory_dict_deserializer_list_length_mismatch(processor_factory):
    deserializer = InMemoryDictDatasetDeserializer()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data={
                "text": ["hello", "world"],
                "id": [1],  # diferent length
            },
            processor_factory=processor_factory,
            random_seed=42,
        )


###################
# Tests dict list in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_dict_list_deserializer_success(processor_factory):
    # Arrange
    data = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
        {"id": 3, "text": "guidellm"},
    ]

    deserializer = InMemoryDictListDatasetDeserializer()

    # Act
    dataset = deserializer(
        data=data,
        processor_factory=processor_factory,
        random_seed=42,
    )

    # Assert
    assert isinstance(dataset, Dataset)
    assert dataset["id"] == [1, 2, 3]
    assert dataset["text"] == ["hello", "world", "guidellm"]
    assert len(dataset) == 3


@pytest.mark.smoke
def test_in_memory_dict_list_deserializer_key_mismatch(processor_factory):
    deserializer = InMemoryDictListDatasetDeserializer()

    wrong_data = [
        {"id": 1, "text": "hello"},
        {"id": 2, "msg": "world"},  # key mismatch
    ]

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=wrong_data,
            processor_factory=processor_factory,
            random_seed=42,
        )


###################
# Tests list in memory deserializer
###################


@pytest.mark.smoke
def test_in_memory_item_list_deserializer_key_mismatch(processor_factory):
    data = ["a", "b", "c"]

    deserializer = InMemoryItemListDatasetDeserializer()

    # Act
    dataset = deserializer(
        data=data,
        processor_factory=processor_factory,
        random_seed=42,
    )

    # Assert
    assert isinstance(dataset, Dataset)
    assert dataset["data"] == data
    assert len(dataset) == 3


@pytest.mark.smoke
def test_in_memory_item_list_custom_column_name(processor_factory):
    deserializer = InMemoryItemListDatasetDeserializer()
    data = [1, 2, 3]

    dataset = deserializer(
        data=data,
        processor_factory=processor_factory,
        random_seed=123,
        column_name="numbers",
    )

    assert list(dataset.column_names) == ["numbers"]
    assert dataset["numbers"] == [1, 2, 3]


###################
# Tests json in memory deserializer
###################


@pytest.mark.parametrize(
    ("json_input"),
    [
        '{"text": ["hello", "world"], "id": [1, 2]}',
        '[{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]',
        '["a", "b", "c"]',
    ],
)
@pytest.mark.smoke
def test_in_memory_json_deserializer_success(processor_factory, json_input):
    deserializer = InMemoryJsonStrDatasetDeserializer()

    dataset = deserializer(
        data=json_input,
        processor_factory=processor_factory,
        random_seed=42,
    )

    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0


###################
# Tests csv in memory deserializer
###################


@pytest.mark.smoke
def test_csv_file_deserializer_success(processor_factory):
    csv_str = "id,text\n1,hello\n2,world\n"

    deserializer = InMemoryCsvDatasetDeserializer()

    dataset = deserializer(
        data=csv_str,
        processor_factory=processor_factory,
        random_seed=43,
    )

    assert isinstance(dataset, Dataset)
    assert {"id", "text"}.issubset(set(dataset.column_names))
    assert dataset["id"] == ["1", "2"]
    assert dataset["text"] == ["hello", "world"]
    assert len(dataset) == 2
