import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from guidellm.benchmark import (
    GenerativeBenchmarksReport,
)
from guidellm.benchmark.output import GenerativeBenchmarksConsole
from tests.unit.mock_benchmark import mock_generative_benchmark


def test_generative_benchmark_initilization():
    report = GenerativeBenchmarksReport()
    assert len(report.benchmarks) == 0

    mock_benchmark = mock_generative_benchmark()
    report_with_benchmarks = GenerativeBenchmarksReport(benchmarks=[mock_benchmark])
    assert len(report_with_benchmarks.benchmarks) == 1
    assert report_with_benchmarks.benchmarks[0] == mock_benchmark


def test_generative_benchmark_invalid_initilization():
    with pytest.raises(ValidationError):
        GenerativeBenchmarksReport(benchmarks="invalid_type")  # type: ignore[arg-type]


def test_generative_benchmark_marshalling():
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(benchmarks=[mock_benchmark])

    serialized = report.model_dump()
    deserialized = GenerativeBenchmarksReport.model_validate(serialized)
    deserialized_benchmark = deserialized.benchmarks[0]

    for field in mock_benchmark.model_fields:
        assert getattr(mock_benchmark, field) == getattr(deserialized_benchmark, field)


def test_file_json():
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.json")
    report.save_file(mock_path)

    with mock_path.open("r") as file:
        saved_data = json.load(file)
    assert saved_data == report.model_dump()

    loaded_report = GenerativeBenchmarksReport.load_file(mock_path)
    loaded_benchmark = loaded_report.benchmarks[0]

    for field in mock_benchmark.model_fields:
        assert getattr(mock_benchmark, field) == getattr(loaded_benchmark, field)

    mock_path.unlink()


def test_file_yaml():
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.yaml")
    report.save_file(mock_path)

    with mock_path.open("r") as file:
        saved_data = yaml.safe_load(file)
    assert saved_data == report.model_dump()

    loaded_report = GenerativeBenchmarksReport.load_file(mock_path)
    loaded_benchmark = loaded_report.benchmarks[0]

    for field in mock_benchmark.model_fields:
        assert getattr(mock_benchmark, field) == getattr(loaded_benchmark, field)

    mock_path.unlink()


def test_file_csv():
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.csv")
    report.save_csv(mock_path)

    with mock_path.open("r") as file:
        reader = csv.reader(file)
        headers = next(reader)
        rows = list(reader)

    assert "Type" in headers
    assert len(rows) == 1

    mock_path.unlink()


def test_console_benchmarks_profile_str():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    assert (
        console.benchmarks_profile_str == "type=synchronous, strategies=['synchronous']"
    )


def test_console_benchmarks_args_str():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    assert console.benchmarks_args_str == (
        "max_number=None, max_duration=10.0, max_error=0.05, warmup_number=None, "
        "warmup_duration=None, cooldown_number=None, cooldown_duration=None"
    )


def test_console_benchmarks_worker_desc_str():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    assert console.benchmarks_worker_desc_str == str(mock_benchmark.worker)


def test_console_benchmarks_request_loader_desc_str():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    assert console.benchmarks_request_loader_desc_str == str(
        mock_benchmark.request_loader
    )


def test_console_benchmarks_extras_str():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    assert console.benchmarks_extras_str == "None"


def test_console_print_section_header():
    console = GenerativeBenchmarksConsole(enabled=True)
    with patch.object(console.console, "print") as mock_print:
        console.print_section_header("Test Header")
        mock_print.assert_called_once()


def test_console_print_labeled_line():
    console = GenerativeBenchmarksConsole(enabled=True)
    with patch.object(console.console, "print") as mock_print:
        console.print_labeled_line("Label", "Value")
        mock_print.assert_called_once()


def test_console_print_line():
    console = GenerativeBenchmarksConsole(enabled=True)
    with patch.object(console.console, "print") as mock_print:
        console.print_line("Test Line")
        mock_print.assert_called_once()


def test_console_print_table():
    console = GenerativeBenchmarksConsole(enabled=True)
    headers = ["Header1", "Header2"]
    rows = [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    with (
        patch.object(console, "print_section_header") as mock_header,
        patch.object(console, "print_table_divider") as mock_divider,
        patch.object(console, "print_table_row") as mock_row,
    ):
        console.print_table(headers, rows, "Test Table")
        mock_header.assert_called_once()
        mock_divider.assert_called()
        mock_row.assert_called()


def test_console_print_benchmarks_metadata():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    with (
        patch.object(console, "print_section_header") as mock_header,
        patch.object(console, "print_labeled_line") as mock_labeled,
    ):
        console.print_benchmarks_metadata()
        mock_header.assert_called_once()
        mock_labeled.assert_called()


def test_console_print_benchmarks_info():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    with patch.object(console, "print_table") as mock_table:
        console.print_benchmarks_info()
        mock_table.assert_called_once()


def test_console_print_benchmarks_stats():
    console = GenerativeBenchmarksConsole(enabled=True)
    mock_benchmark = mock_generative_benchmark()
    console.benchmarks = [mock_benchmark]
    with patch.object(console, "print_table") as mock_table:
        console.print_benchmarks_stats()
        mock_table.assert_called_once()
