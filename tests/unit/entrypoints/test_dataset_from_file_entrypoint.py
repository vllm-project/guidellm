import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from guidellm.preprocess.dataset_from_file import (
    DatasetCreationError,
    create_dataset_from_file,
    extract_dataset_from_benchmark_report,
    print_dataset_statistics,
    save_dataset_from_benchmark,
    validate_benchmark_file,
)

REGENERATE_ARTIFACTS = False


@pytest.fixture
def get_test_asset_dir():
    def _() -> Path:
        return Path(__file__).parent / "assets"

    return _


@pytest.fixture
def cleanup():
    to_delete: list[Path] = []
    yield to_delete
    for item in to_delete:
        if item.exists():
            item.unlink()  # Deletes the file


@pytest.fixture
def temp_file():
    """Create a temporary file that gets cleaned up automatically."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


def test_create_dataset_from_valid_benchmark_json(get_test_asset_dir, cleanup):
    """Test creating dataset from a valid benchmark JSON file."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"
    output_file = asset_dir / "test_dataset_output.json"
    cleanup.append(output_file)

    create_dataset_from_file(
        benchmark_file=source_file,
        output_path=output_file,
        show_stats=False,
        enable_console=False,
    )

    assert output_file.exists()

    with output_file.open() as f:
        dataset = json.load(f)

    assert "version" in dataset
    assert "description" in dataset
    assert "data" in dataset
    assert isinstance(dataset["data"], list)
    assert len(dataset["data"]) > 0

    for item in dataset["data"]:
        assert "prompt" in item
        assert "output_tokens_count" in item
        assert "prompt_tokens_count" in item
        assert isinstance(item["prompt"], str)
        assert isinstance(item["output_tokens_count"], int)
        assert isinstance(item["prompt_tokens_count"], int)
        assert len(item["prompt"]) > 0
        assert item["output_tokens_count"] > 0


def test_create_dataset_from_valid_benchmark_yaml(get_test_asset_dir, cleanup):
    """Test creating dataset from a valid benchmark YAML file."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.yaml"
    output_file = asset_dir / "test_dataset_yaml_output.json"
    cleanup.append(output_file)

    create_dataset_from_file(
        benchmark_file=source_file,
        output_path=output_file,
        show_stats=False,
        enable_console=False,
    )

    assert output_file.exists()

    with output_file.open() as f:
        dataset = json.load(f)

    assert "data" in dataset
    assert len(dataset["data"]) > 0


def test_create_dataset_with_stats_output(capfd, get_test_asset_dir, cleanup):
    """Test creating dataset with statistics output enabled."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"
    output_file = asset_dir / "test_dataset_stats_output.json"
    cleanup.append(output_file)

    create_dataset_from_file(
        benchmark_file=source_file,
        output_path=output_file,
        show_stats=True,
        enable_console=True,
    )

    out, err = capfd.readouterr()
    assert "Validating benchmark report file" in out
    assert "Valid benchmark report with" in out
    assert "Dataset saved to" in out
    assert "Success, Created dataset with" in out
    assert "Dataset Statistics:" in out
    assert "Total items:" in out
    assert "Prompt tokens - Min:" in out
    assert "Output tokens - Min:" in out


def test_create_dataset_with_console_disabled(capfd, get_test_asset_dir, cleanup):
    """Test creating dataset with console output disabled."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"
    output_file = asset_dir / "test_dataset_no_console.json"
    cleanup.append(output_file)

    create_dataset_from_file(
        benchmark_file=source_file,
        output_path=output_file,
        show_stats=True,
        enable_console=False,
    )

    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""

    assert output_file.exists()


def test_validate_benchmark_file_valid_file(get_test_asset_dir):
    """Test validation with a valid benchmark file."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"

    report = validate_benchmark_file(source_file)
    assert report is not None
    assert len(report.benchmarks) > 0


def test_validate_benchmark_file_invalid_json(temp_file):
    """Test validation with invalid JSON."""
    temp_file.write_text("This is not JSON")

    with pytest.raises(DatasetCreationError) as exc_info:
        validate_benchmark_file(temp_file)

    assert "Invalid benchmark report file" in str(exc_info.value)
    assert "Expecting value" in str(exc_info.value)


def test_validate_benchmark_file_invalid_structure(temp_file):
    """Test validation with valid JSON but invalid benchmark structure."""
    temp_file.write_text('{"invalid": "structure"}')

    with pytest.raises(DatasetCreationError) as exc_info:
        validate_benchmark_file(temp_file)

    assert "Invalid benchmark report file" in str(exc_info.value)


def test_validate_benchmark_file_no_benchmarks(temp_file):
    """Test validation with valid structure but no benchmarks."""
    temp_file.write_text('{"benchmarks": []}')

    with pytest.raises(DatasetCreationError) as exc_info:
        validate_benchmark_file(temp_file)

    assert "Benchmark report contains no benchmark data" in str(exc_info.value)


def test_extract_dataset_from_benchmark_report(get_test_asset_dir):
    """Test extracting dataset from a validated benchmark report."""
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"

    report = validate_benchmark_file(source_file)

    dataset_items = extract_dataset_from_benchmark_report(report)

    assert len(dataset_items) > 0

    for item in dataset_items:
        assert "prompt" in item
        assert "output_tokens" in item
        assert "prompt_tokens" in item
        assert len(item["prompt"]) > 0
        assert item["output_tokens"] > 0
        assert item["prompt_tokens"] > 0


def test_save_dataset_from_benchmark(cleanup):
    """Test saving dataset to file."""
    dataset_items = [
        {
            "prompt": "Test prompt 1",
            "output_tokens": 100,
            "prompt_tokens": 50,
        },
        {
            "prompt": "Test prompt 2",
            "output_tokens": 200,
            "prompt_tokens": 75,
        },
    ]

    output_file = Path("test_save_dataset.json")
    cleanup.append(output_file)

    save_dataset_from_benchmark(dataset_items, output_file)

    assert output_file.exists()

    with output_file.open() as f:
        saved_data = json.load(f)

    assert "version" in saved_data
    assert "description" in saved_data
    assert "data" in saved_data
    assert len(saved_data["data"]) == 2

    for item in saved_data["data"]:
        assert "prompt" in item
        assert "output_tokens_count" in item
        assert "prompt_tokens_count" in item


def test_print_dataset_statistics_with_data(capfd):
    """Test printing statistics with valid dataset."""
    dataset_items = [
        {"prompt": "Test 1", "output_tokens": 100, "prompt_tokens": 50},
        {"prompt": "Test 2", "output_tokens": 200, "prompt_tokens": 75},
        {"prompt": "Test 3", "output_tokens": 150, "prompt_tokens": 60},
    ]

    print_dataset_statistics(dataset_items, enable_console=True)

    out, err = capfd.readouterr()
    assert "Dataset Statistics:" in out
    assert "Total items: 3" in out
    assert "Prompt tokens - Min: 50, Max: 75, Mean: 61.7" in out
    assert "Output tokens - Min: 100, Max: 200, Mean: 150.0" in out


def test_print_dataset_statistics_empty_dataset(capfd):
    """Test printing statistics with empty dataset."""
    dataset_items: list[dict[str, Any]] = []

    print_dataset_statistics(dataset_items, enable_console=True)

    out, err = capfd.readouterr()
    assert "No valid items found in dataset" in err


def test_print_dataset_statistics_console_disabled(capfd):
    """Test printing statistics with console disabled."""
    dataset_items = [
        {"prompt": "Test", "output_tokens": 100, "prompt_tokens": 50},
    ]

    print_dataset_statistics(dataset_items, enable_console=False)

    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_create_dataset_from_file_nonexistent_file():
    """Test error handling for nonexistent file."""
    nonexistent_file = Path("does_not_exist.json")
    output_file = Path("output.json")

    with pytest.raises(DatasetCreationError):
        create_dataset_from_file(
            benchmark_file=nonexistent_file,
            output_path=output_file,
            show_stats=False,
            enable_console=False,
        )


def test_create_dataset_from_file_no_successful_requests(temp_file):
    """Test handling of benchmark with no successful requests."""
    benchmark_data: dict[str, Any] = {
        "benchmarks": [
            {"requests": {"successful": [], "errored": [], "incomplete": []}}
        ]
    }
    temp_file.write_text(json.dumps(benchmark_data))

    output_file = Path("output.json")

    with pytest.raises(DatasetCreationError) as exc_info:
        create_dataset_from_file(
            benchmark_file=temp_file,
            output_path=output_file,
            show_stats=False,
            enable_console=False,
        )

    assert "Invalid benchmark report file" in str(exc_info.value)


if __name__ == "__main__":
    unittest.main()
