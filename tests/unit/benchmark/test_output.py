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
from guidellm.benchmark.output import (
    GenerativeBenchmarkerConsole,
    GenerativeBenchmarkerCSV,
)
from guidellm.benchmark.schemas import BenchmarkGenerativeTextArgs
from tests.unit.mock_benchmark import mock_generative_benchmark


def test_generative_benchmark_initilization():
    args = BenchmarkGenerativeTextArgs(target="http://localhost:8000", data=["test"])
    report = GenerativeBenchmarksReport(args=args)
    assert len(report.benchmarks) == 0

    mock_benchmark = mock_generative_benchmark()
    report_with_benchmarks = GenerativeBenchmarksReport(
        args=args, benchmarks=[mock_benchmark]
    )
    assert len(report_with_benchmarks.benchmarks) == 1
    assert report_with_benchmarks.benchmarks[0] == mock_benchmark


def test_generative_benchmark_invalid_initilization():
    with pytest.raises(ValidationError):
        GenerativeBenchmarksReport(benchmarks="invalid_type")  # type: ignore[arg-type]


def test_generative_benchmark_marshalling():
    args = BenchmarkGenerativeTextArgs(target="http://localhost:8000", data=["test"])
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(args=args, benchmarks=[mock_benchmark])

    serialized = report.model_dump()
    deserialized = GenerativeBenchmarksReport.model_validate(serialized)
    deserialized_benchmark = deserialized.benchmarks[0]

    # model_dump as workaround for duplicate fields for computed fields.
    assert mock_benchmark.model_dump() == deserialized_benchmark.model_dump()


def test_file_json():
    args = BenchmarkGenerativeTextArgs(target="http://localhost:8000", data=["test"])
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(args=args, benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.json")
    report.save_file(mock_path)

    with mock_path.open("r") as file:
        saved_data = json.load(file)
    assert saved_data == report.model_dump()

    loaded_report = GenerativeBenchmarksReport.load_file(mock_path)
    loaded_benchmark = loaded_report.benchmarks[0]

    # model_dump as workaround for duplicate fields for computed fields.
    assert mock_benchmark.model_dump() == loaded_benchmark.model_dump()

    mock_path.unlink()


def test_file_yaml():
    args = BenchmarkGenerativeTextArgs(target="http://localhost:8000", data=["test"])
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(args=args, benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.yaml")
    report.save_file(mock_path)

    with mock_path.open("r") as file:
        saved_data = yaml.safe_load(file)
    assert saved_data == report.model_dump()

    loaded_report = GenerativeBenchmarksReport.load_file(mock_path)
    loaded_benchmark = loaded_report.benchmarks[0]

    # model_dump as workaround for duplicate fields for computed fields.
    assert mock_benchmark.model_dump() == loaded_benchmark.model_dump()

    mock_path.unlink()


@pytest.mark.asyncio
async def test_file_csv():
    args = BenchmarkGenerativeTextArgs(target="http://localhost:8000", data=["test"])
    mock_benchmark = mock_generative_benchmark()
    report = GenerativeBenchmarksReport(args=args, benchmarks=[mock_benchmark])

    mock_path = Path("mock_report.csv")
    csv_benchmarker = GenerativeBenchmarkerCSV(output_path=mock_path)
    await csv_benchmarker.finalize(report)

    with mock_path.open("r") as file:  # noqa: ASYNC230  # This is a test.
        reader = csv.reader(file)
        headers = next(reader)
        rows = list(reader)

    assert "Type" in headers
    assert "Profile" in headers
    assert len(rows) == 1

    mock_path.unlink()


def test_console_benchmarks_profile_str():
    console = GenerativeBenchmarkerConsole()
    mock_benchmark = mock_generative_benchmark()
    profile_str = console._get_profile_str(mock_benchmark)
    # The profile string should contain the profile type information
    assert "synchronous" in profile_str


def test_console_print_section_header():
    console = GenerativeBenchmarkerConsole()
    with patch.object(console.console, "print") as mock_print:
        console._print_section_header("Test Header")
        mock_print.assert_called_once()


def test_console_print_labeled_line():
    console = GenerativeBenchmarkerConsole()
    with patch.object(console.console, "print") as mock_print:
        console._print_labeled_line("Label", "Value")
        mock_print.assert_called_once()


def test_console_print_line():
    console = GenerativeBenchmarkerConsole()
    with patch.object(console.console, "print") as mock_print:
        console._print_line("Test Line")
        mock_print.assert_called_once()


def test_console_print_table():
    console = GenerativeBenchmarkerConsole()
    headers = ["Header1", "Header2"]
    rows = [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    with (
        patch.object(console, "_print_section_header") as mock_header,
        patch.object(console, "_print_table_divider") as mock_divider,
        patch.object(console, "_print_table_row") as mock_row,
    ):
        console._print_table(headers, rows, "Test Table")
        mock_header.assert_called_once()
        mock_divider.assert_called()
        mock_row.assert_called()


def test_console_print_benchmarks_metadata():
    console = GenerativeBenchmarkerConsole()
    mock_benchmark = mock_generative_benchmark()
    with (
        patch.object(console, "_print_section_header") as mock_header,
        patch.object(console, "_print_labeled_line") as mock_labeled,
    ):
        console._print_benchmarks_metadata([mock_benchmark])
        mock_header.assert_called_once()
        mock_labeled.assert_called()


def test_console_print_benchmarks_info():
    console = GenerativeBenchmarkerConsole()
    mock_benchmark = mock_generative_benchmark()
    with patch.object(console, "_print_table") as mock_table:
        console._print_benchmarks_info([mock_benchmark])
        mock_table.assert_called_once()


def test_console_print_benchmarks_stats():
    console = GenerativeBenchmarkerConsole()
    mock_benchmark = mock_generative_benchmark()
    with patch.object(console, "_print_table") as mock_table:
        console._print_benchmarks_stats([mock_benchmark])
        mock_table.assert_called_once()
