"""
Module for creating datasets from saved benchmark report files.

This module provides functionality to extract prompts and their corresponding
output token counts from benchmark results to create datasets for future
'apples-to-apples' comparisons.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from guidellm.benchmark.output import GenerativeBenchmarksReport

__all__ = [
    "DatasetCreationError",
    "create_dataset_from_file",
    "extract_dataset_from_benchmark_report",
    "print_dataset_statistics",
    "save_dataset_from_benchmark",
    "validate_benchmark_file",
]


class DatasetCreationError(Exception):
    """Exception raised when dataset creation fails."""


def validate_benchmark_file(filepath: Path) -> GenerativeBenchmarksReport:
    """
    Validate that the file is a proper GuideLLM benchmark report.

    Args:
        filepath: Path to the benchmark report file

    Returns:
        GenerativeBenchmarksReport: The validated and loaded report

    Raises:
        DatasetCreationError: If file validation fails
    """
    try:
        report = GenerativeBenchmarksReport.load_file(filepath)
        if not report.benchmarks:
            raise DatasetCreationError("Benchmark report contains no benchmark data")
        return report
    except Exception as e:
        error_msg = f"Invalid benchmark report file: {e}"
        raise DatasetCreationError(error_msg) from e


def extract_dataset_from_benchmark_report(
    report: GenerativeBenchmarksReport,
) -> list[dict[str, Any]]:
    """
    Extract prompts and output tokens from a validated benchmark report.

    Args:
        report: A validated GenerativeBenchmarksReport instance

    Returns:
        List of dataset items with prompt and token information
    """
    dataset_items = []

    for benchmark in report.benchmarks:
        # Access the StatusBreakdown properties directly
        requests_breakdown = benchmark.requests

        # Get successful requests (these are the ones we want)
        successful_requests = requests_breakdown.successful

        for request in successful_requests:
            # Extract the needed data - these are Request objects
            prompt = request.prompt
            output_tokens = request.output_tokens
            prompt_tokens = request.prompt_tokens

            # Only include items with valid data
            if prompt and output_tokens > 0:
                dataset_items.append(
                    {
                        "prompt": prompt,
                        "output_tokens": output_tokens,
                        "prompt_tokens": prompt_tokens,
                    }
                )

    return dataset_items


def save_dataset_from_benchmark(
    dataset_items: list[dict[str, Any]], output_file: Path
) -> None:
    """Save the dataset to a JSON file."""
    # Convert to the format expected by guidellm documentation
    formatted_items = []
    for item in dataset_items:
        formatted_items.append(
            {
                "prompt": item["prompt"],
                "output_tokens_count": item["output_tokens"],
                "prompt_tokens_count": item["prompt_tokens"],
            }
        )

    dataset_data = {
        "version": "1.0",
        "description": (
            "Dataset created from benchmark results for apples-to-apples comparisons"
        ),
        "data": formatted_items,
    }

    with output_file.open("w") as f:
        json.dump(dataset_data, f, indent=2)


def print_dataset_statistics(
    dataset_items: list[dict[str, Any]], enable_console: bool = True
) -> None:
    """Print statistics about the dataset."""
    if not enable_console:
        return

    console = Console()
    console_err = Console(stderr=True)

    if not dataset_items:
        console_err.print("No valid items found in dataset")
        return

    total_items = len(dataset_items)
    prompt_tokens = [item["prompt_tokens"] for item in dataset_items]
    output_tokens = [item["output_tokens"] for item in dataset_items]

    console.print("\nDataset Statistics:")
    console.print(f"Total items: {total_items}")
    console.print(
        f"Prompt tokens - Min: {min(prompt_tokens)}, "
        f"Max: {max(prompt_tokens)}, "
        f"Mean: {sum(prompt_tokens) / len(prompt_tokens):.1f}"
    )
    console.print(
        f"Output tokens - Min: {min(output_tokens)}, "
        f"Max: {max(output_tokens)}, "
        f"Mean: {sum(output_tokens) / len(output_tokens):.1f}"
    )


def create_dataset_from_file(
    benchmark_file: Path,
    output_path: Path,
    show_stats: bool = False,
    enable_console: bool = True,
) -> None:
    """
    Create a dataset from a saved benchmark report file.

    This function validates the benchmark file format, loads it using the same
    validation as the 'from-file' command, then extracts prompts and their
    corresponding output token counts from successful requests.

    Args:
        benchmark_file: Path to the benchmark results JSON/YAML file
        output_path: Path where the dataset should be saved
        show_stats: Whether to display dataset statistics
        enable_console: Whether to enable console output

    Raises:
        DatasetCreationError: If validation fails or no valid requests found
    """
    console = Console()
    console_err = Console(stderr=True)

    if enable_console:
        console.print(f"Validating benchmark report file: {benchmark_file}")

    try:
        report = validate_benchmark_file(benchmark_file)

        if enable_console:
            console.print(
                f"Valid benchmark report with {len(report.benchmarks)} benchmark(s)"
            )
            console.print("Loading and extracting dataset from benchmark results...")

        dataset_items = extract_dataset_from_benchmark_report(report)

        if not dataset_items:
            error_msg = (
                "No valid requests with prompts and output tokens "
                "found in benchmark report"
            )
            if enable_console:
                console_err.print(f"Error: {error_msg}")
            raise DatasetCreationError(error_msg)

        save_dataset_from_benchmark(dataset_items, output_path)

        if enable_console:
            console.print(f"Dataset saved to: {output_path}")
            console.print(f"Success, Created dataset with {len(dataset_items)} items")
            console.print(
                f"You can now use this dataset for future guidellm runs "
                f"by specifying: --data {output_path}"
            )

        if show_stats:
            print_dataset_statistics(dataset_items, enable_console)

    except DatasetCreationError:
        raise
    except Exception as e:
        if enable_console:
            console_err.print(f"Unexpected error: {e}")
        raise DatasetCreationError(f"Failed to process benchmark file: {e}") from e
