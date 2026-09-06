import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from guidellm.utils.hf_datasets import load_dataset_from_file, save_dataset_to_file


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_csv(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.csv")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_csv.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_csv_capitalized(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.CSV")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_csv.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_json(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.json")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_json.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_json_capitalized(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.JSON")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_json.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_jsonl(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.jsonl")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_json.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_jsonl_capitalized(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.JSONL")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_json.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_parquet(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.parquet")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_parquet.assert_called_once_with(output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
@patch.object(Path, "mkdir")
def test_save_dataset_to_file_unsupported_type(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.txt")
    with pytest.raises(ValueError, match=r"Unsupported file suffix '.txt'.*"):
        save_dataset_to_file(mock_dataset, output_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.regression
def test_load_jsonl_unions_optional_nested_fields(tmp_path: Path):
    """JSONL rows that add nested fields after earlier records still load.

    HuggingFace's json builder infers schema from the first chunk and
    cannot cast later structs that include extra keys such as ``ttft``.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "mixed.jsonl"
    rows = [
        {"id": "a", "requests": [{"t": 1.0, "in": 10, "out": 5}]},
        {"id": "b", "requests": [{"t": 2.0, "in": 11, "out": 6, "ttft": 0.05}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset = load_dataset_from_file(path)
    assert len(dataset) == 2
    assert dataset[0]["requests"][0]["in"] == 10
    assert dataset[1]["requests"][0]["ttft"] == pytest.approx(0.05)


@pytest.mark.sanity
def test_load_json_array_file(tmp_path: Path):
    """A ``.json`` file containing a top-level array of records still loads.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            [
                {"timestamp": 1, "input_length": 10, "output_length": 1},
                {"timestamp": 2, "input_length": 20, "output_length": 2},
            ]
        )
    )
    dataset = load_dataset_from_file(path)
    assert len(dataset) == 2
    assert dataset[1]["input_length"] == 20
