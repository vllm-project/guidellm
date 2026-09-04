from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from guidellm.utils.hf_datasets import save_dataset_to_file


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
